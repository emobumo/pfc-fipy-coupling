# -*- coding: utf-8 -*-
"""Section 4.8, step 2B (solver part): five single-hole cases on the cropped 0.5 m grid.

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\run_cases.py --check     (crop vs full reach)
    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\run_cases.py --run       (S1..S5, serial, resumable)
    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\run_cases.py --summary   (metrics -> cases.txt)
    ... run_cases.py --cases S1,S4 [--cap-hours H]   (one batch, own ledger wall_S1-S4.json; batches run in
        parallel through scripts/xz_solver/launch_batches.ps1, one thread each, flush from the first step)

Geometry configuration from scripts/xz_config ($XZ_CONFIG, default 40m: the paths, case list and lower-end
threshold below; another configuration reads and writes the directories with its suffix, e.g. outputs/xz_solver_30m).
Input: data/xz_prejudge/base_geometry_K1.25_v4.npz (scripts/xz_prejudge/step2b.py), scenario 1.

0.5 m grid (aggregation as the step-2A timing probe): a 0.5 m cell is rubble when >= 2 of its 2 x 2 fine cells are
rubble, n = mean n1 of those; other cells n = 1e-3 and take the most frequent non-rubble class of their fine cells
(ties: lowest code). S, T, X, Z = means of the fine values (linear, so the coarse-cell centre values). Sources = the
rubble 0.5 m cells containing a fine grouting cell of the hole. Zones on the 0.5 m grid: basal s <= 3 m; hanging
side by prejudge.zones (0.5 m t-bins); lower end of the basal layer = basal cells with t <= -3.37 m (step 1 v2).

Domain: bounding rectangle of the 0.5 m rubble plus 3 m (6 cells) of rock each side. v0.6, gravity, constant
p0 on the whole grouting section (interior Dirichlet), one pass; dt_cap 5 s; stop rule (4-29) with eps 0.1,
beta 0.1, n_ref 0.20, OR (main line, 2026-10-18) the filled area outside the reach domain of the same hole and
p0 (0.5 m crop) reaching 10% of the reach domain -- the regularised Bingham law creeps past the stall boundary.
Snapshots when the reach-domain fill fraction first reaches 0.90 / 0.95 / 0.99 (S field saved). All fill and
loss metrics count reach-domain cells only; cells filled outside are reported apart as numerical creep. S1 was run
before these rules and is closed from its last checkpoint (close_s1). Checkpoint every 200 steps (re-run the same command to resume); a wall-time ledger stops the
queue after the case during which the cumulative wall time passes 48 h.
"""
from __future__ import print_function

import imp
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

S2A = imp.load_source("xz_step2a", os.path.join(REPO, "scripts", "xz_prejudge", "step2a.py"))
DG = imp.load_source("xz_diag", os.path.join(REPO, "scripts", "xz_solver", "diag.py"))
FZ = imp.load_source("xz_ftz", os.path.join(REPO, "scripts", "xz_solver", "ftz.py"))
FZ.install()
P, Z = S2A.P, S2A.Z
from fipy import CellVariable
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.analysis.fill_diagnostics import reachable_domain
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport.equations import solve_transport_step
from src.models.slurry_transport.variables import initialize_slurry_variables

XC, CFG = P.XC, P.CFG           # geometry configuration ($XZ_CONFIG, default 40m)
SRC = os.path.join(P.DATA, "base_geometry_K1.25_v4.npz")
OUT = XC.out(CFG, "outputs", "xz_solver")
CELL = 0.5
FAC = 2
PAD = CFG["solver"]["pad"]
DT_CAP = 5.0
MAX_STEPS = 60000
CKPT_EVERY = 200
RECORD_EVERY = 20
N_REF = 0.20
LOW_T = CFG["solver"]["low_t"]  # None: min t of the 0.25 m basal layer + low_end_dt (set in load())
LOSS_DT = 5.0
CAP_S = 48 * 3600.0
# Subnormal flush (scripts/xz_solver/ftz.py; main line 2026-10-18): first step it is active in, per case
# (None = never). 40 m: S2 from its step-8000 checkpoint, S1 (re-run) and S3-S5 from the first step; any new end
# state: every case from the first step (main line 2026-10-20).
FTZ_FROM = dict(CFG["solver"]["ftz_from"])
# S1 re-run (main line 2026-10-19, --rerun-s1): current settings, flush from step 1, own directory, 10 h wall cap
RERUN_DIR = os.path.join(OUT, "s1_rerun")
CASE_CAP = [None]
LEDGER = ["wall.json"]         # --cases runs keep their own ledger (parallel processes must not share one)
SNAPS = (0.90, 0.95, 0.99)     # reach-domain fill fractions for the snapshots (main line, 2026-10-18)
CREEP_MAX = 0.10                # stop when the filled area outside the reach domain reaches 10% of it
RUB, VOID = P.RUB, P.VOID
# tag, label, source key (fan index or "s2"), p0
CASES = [tuple(c_) for c_ in CFG["solver"]["cases"]]
# stage-2 cases of a new end state (main line 2026-10-21; --cases only): dicts with tag, label, key (fan index or
# "s2"), p0, grid (a stored 0.5 m grid, outputs/xz_solver<suffix>/grid_<grid>.npz, used unchanged) and optionally
# outlet = {"ref": closed case, "factor": f}: the lower end of the basal layer is a pressure-0 outflow boundary
# (kept at S = 0 after every step, see outlet_trial.py) and the run ends when V_in reaches f x V_in of the closed
# case at its 0.95 snapshot (rate and creep rules only recorded)
CASES2 = dict((c_["tag"], c_) for c_ in CFG["solver"].get("cases2", []))
CORNER_BOX = tuple(CFG["solver"]["corner_box"])   # old box definition of the lower-end corner (not cited)


def coarse(D):
    cls, ny, nx = D["cls"], D["cls"].shape[0], D["cls"].shape[1]
    cy, cx = ny // FAC, nx // FAC

    def blocks(a):
        return a[:cy * FAC, :cx * FAC].reshape(cy, FAC, cx, FAC)
    b = blocks(cls)
    nr = (b == RUB).sum(axis=(1, 3))
    rub = nr * 2 >= FAC * FAC
    n1 = np.where(rub, np.where(b == RUB, blocks(D["n1"]), 0.0).sum(axis=(1, 3)) / np.maximum(nr, 1), P.ROCK_N)
    counts = np.array([(b == q).sum(axis=(1, 3)) for q in range(1, 6)])
    ccls = np.where(rub, RUB, np.argmax(counts, axis=0) + 1)
    mean = lambda a: blocks(a).mean(axis=(1, 3))
    S, T, X, Zc = mean(D["S"]), mean(D["T"]), mean(D["X"]), mean(D["Z"])
    basal = rub & (S <= P.BASAL)
    hang = P.zones(rub, S, T)[2]
    low = basal & (T <= LOW_T)
    return {"rub": rub, "n": n1, "cls": ccls, "S": S, "T": T, "X": X, "Z": Zc, "basal": basal, "hang": hang,
            "low": low, "cy": cy, "cx": cx}


def sources(fine_cells, nx_f, C):
    fy, fx = np.asarray(fine_cells) // nx_f, np.asarray(fine_cells) % nx_f
    return sorted(set((j // FAC, i // FAC) for j, i in zip(fy, fx) if C["rub"][j // FAC, i // FAC]))


def crop_box(C):
    jj, ii = np.where(C["rub"])
    j0, j1 = max(jj.min() - PAD, 0), min(jj.max() + 1 + PAD, C["cy"])
    i0, i1 = max(ii.min() - PAD, 0), min(ii.max() + 1 + PAD, C["cx"])
    return j0, j1, i0, i1


def build(n_field, src_ji, box, p0):
    """State on box = (j0, j1, i0, i1) of the 0.5 m grid (None = whole domain)."""
    ny, nx = n_field.shape
    j0, j1, i0, i1 = box if box is not None else (0, ny, 0, nx)
    dom = (P.DOM[0] + i0 * CELL, P.DOM[0] + i1 * CELL, P.DOM[2] + j0 * CELL, P.DOM[2] + j1 * CELL)
    mesh, x, y, fx, fy = build_mesh_for_domain(dom[0], dom[1], dom[2], dom[3], CELL)
    params = Z.slurry_params()
    params["inlet_pressure_core_value"] = p0
    st = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    st.update(initialize_slurry_variables(mesh, params=params))
    Z.set_porosity(st, n_field[j0:j1, i0:i1].ravel())
    cnx = i1 - i0
    src = np.array(sorted((j - j0) * cnx + (i - i0) for j, i in src_ji), int)
    m = np.zeros(mesh.numberOfCells)
    m[src] = 1.0
    st["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=m)
    st["interior_dirichlet_value"] = p0
    st["saturation"].setValue(m.copy())
    st["pressure"].setValue(np.where(m > 0.5, p0, 0.0))
    return st, src, (j0, j1, i0, i1)


def load():
    global LOW_T
    D = np.load(SRC)
    if LOW_T is None:
        LOW_T = float(D["T"][D["basal"]].min()) + P.LOW_DT
    C = coarse(D)
    nx_f = D["cls"].shape[1]
    fc, fp = D["fan_cells"], D["fan_ptr"]
    srcs = {}
    for tag, _, key, p0 in CASES:
        cells = D["s2_cells"] if key == "s2" else fc[fp[key]:fp[key + 1]]
        srcs[tag] = sources(cells, nx_f, C)
    collar_t = {tag: float(D["fan_collar_t"][1 if key == "s2" else key]) for tag, _, key, _ in CASES}
    return D, C, srcs, collar_t


def cost_on(st, src, shape):
    return np.asarray(reachable_domain(st, src, p0=2.0e6)["cost"], float).reshape(shape)


def check():
    D, C, srcs, _ = load()
    box = crop_box(C)
    j0, j1, i0, i1 = box
    L = ["Crop check: 0.5 m grid %d x %d; crop rows %d..%d, cols %d..%d -> %d x %d cells (X %.2f..%.2f, Z %.2f..%.2f m)"
         % (C["cx"], C["cy"], j0, j1, i0, i1, i1 - i0, j1 - j0, P.DOM[0] + i0 * CELL - P.ORIGIN[0],
            P.DOM[0] + i1 * CELL - P.ORIGIN[0], P.DOM[2] + j0 * CELL - P.ORIGIN[1], P.DOM[2] + j1 * CELL - P.ORIGIN[1])]
    L.append("  rubble cells %d, all inside the crop: %s" % (C["rub"].sum(), C["rub"][j0:j1, i0:i1].sum() == C["rub"].sum()))
    for tag in ("S1", "S4"):
        sf, srcf, _ = build(C["n"], srcs[tag], None, 2.0e6)
        sc, srcc, _ = build(C["n"], srcs[tag], box, 2.0e6)
        cf = cost_on(sf, srcf, C["n"].shape)[j0:j1, i0:i1]
        cc = cost_on(sc, srcc, (j1 - j0, i1 - i0))
        r = C["rub"][j0:j1, i0:i1]
        both = r & np.isfinite(cf) & np.isfinite(cc)
        d = np.abs(cc - cf)[both]
        lo = both & (cf <= 2.0e6)
        L.append("  %s (%d source cells): rubble cells with finite cost full / crop %d / %d; max |diff| over rubble %.6g Pa"
                 " (cells with full-domain cost <= 2 MPa: %.6g Pa); cells differing %d"
                 % (tag, len(srcc), (r & np.isfinite(cf)).sum(), (r & np.isfinite(cc)).sum(), d.max() if d.size else 0.0,
                    np.abs(cc - cf)[lo].max() if lo.any() else 0.0, (d > 0).sum()))
        if (d > 0).any():
            k = np.argmax(np.abs(cc - cf) * both)
            jj, ii = np.unravel_index(k, cc.shape)
            L.append("     largest at X %.2f, Z %.2f: full %.4f MPa, crop %.4f MPa"
                     % (C["X"][j0 + jj, i0 + ii], C["Z"][j0 + jj, i0 + ii], cf[jj, ii] / 1e6, cc[jj, ii] / 1e6))
    txt = "\n".join(L) + "\n"
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "crop_check.txt"), "w").write(txt)
    print(txt)


def ledger_path():
    return os.path.join(OUT, LEDGER[0])


def read_ledger():
    p = ledger_path()
    return json.load(open(p)) if os.path.isfile(p) else {}


def write_ledger(led):
    json.dump(led, open(ledger_path(), "w"), indent=1, sort_keys=True)


def case_masks(tag, p0, C, srcs, collar_t, box, st=None, src=None):
    """Flat masks on the crop for one case; reach = rubble cells with p_req <= p0 (same hole, 0.5 m crop)."""
    j0, j1, i0, i1 = box
    crop = lambda a: a[j0:j1, i0:i1].ravel()
    if st is None:
        st, src, _ = build(C["n"], srcs[tag], box, p0)
    cost = np.asarray(reachable_domain(st, src, p0=p0)["cost"], float)
    M = {"rub": crop(C["rub"]), "low": crop(C["low"]), "basal": crop(C["basal"]), "hang": crop(C["hang"]),
         "n": crop(C["n"]), "T": crop(C["T"])}
    M["reach"] = M["rub"] & (cost <= p0)
    M["loss"] = M["basal"] & (M["T"] <= collar_t[tag] - LOSS_DT)
    cls = C["cls"][j0:j1, i0:i1]
    M["vbf"] = P.neighbours_count(C["rub"][j0:j1, i0:i1], cls == VOID).ravel()
    M["vols"] = np.asarray(st["mesh"].cellVolumes, dtype=float)
    M["nreach"] = int(M["reach"].sum())
    return M, st, src


def metrics(s, M):
    """Fill metrics counted INSIDE the reach domain only (main-line rule); cells filled outside are reported apart."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    f = (s >= 0.5) & M["rub"]
    fin = f & M["reach"]
    out = f & ~M["reach"]
    stored = np.where(M["reach"], M["n"] * s * M["vols"], 0.0)
    tot = float(stored.sum())
    loss = float(stored[M["loss"]].sum())
    return {"frac_in": fin.sum() / float(M["nreach"]), "filled_in": int(fin.sum()), "outside": int(out.sum()),
            "outside_area": float(out.sum()) * CELL * CELL, "stored_in": tot, "loss": loss,
            "loss_share": loss / tot if tot > 0 else 0.0,
            "hang_fill": (fin & M["hang"]).sum() / float(M["hang"].sum()),
            "void_contact": float((M["vbf"] * fin).sum()) * CELL,
            "low_in": bool(np.any(fin & M["low"])), "low_any": bool(np.any(f & M["low"]))}


def run_case(tag, label, p0, D, C, srcs, collar_t, box, opts=None):
    opts = opts or {}
    outlet = opts.get("outlet")
    st, src, _ = build(C["n"], srcs[tag], box, p0)
    j0, j1, i0, i1 = box
    shp = (j1 - j0, i1 - i0)
    M, _, _ = case_masks(tag, p0, C, srcs, collar_t, box, st, src)
    rub, reach, low = M["rub"], M["reach"], M["low"]
    vols = M["vols"]
    hm = np.zeros(vols.size, bool)
    hm[src] = True
    sp = st["slurry_parameters"]
    qref = stop_rule.q_ref(N_REF, sp["calibrated_permeability_coefficient"], p0, sp["plastic_viscosity"])
    fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % (tag, p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(N_REF)
    if opts:
        fp += "|grid=%s|outlet=%s" % (opts.get("grid"), json.dumps(outlet, sort_keys=True))
    om = low.copy() if outlet else None
    v_target = None
    if outlet:
        ref = json.load(open(os.path.join(OUT, "result_%s.json" % outlet["ref"])))
        v_target = outlet["factor"] * float(ref["snaps"]["0.95"]["v_in"])
        v_eq = float(ref["snaps"]["0.95"]["v_in"])
    keeper = Checkpointer(os.path.join(OUT, "ckpt_" + tag), every=CKPT_EVERY, fingerprint=state_fingerprint(st, fp))
    res = keeper.restore(st)
    led = read_ledger()
    wall_prev = float(led.get(tag, {}).get("wall", 0.0))
    if res:
        step0, t, v_in = int(res["step"]), float(res["t"]), float(res["v_in"])
        t_hist, v_hist = list(np.atleast_1d(res["t_hist"])), list(np.atleast_1d(res["v_hist"]))
        arr = json.loads(str(res["arr_json"]))
        series = json.loads(str(res["series_json"]))
        snaps = json.loads(str(res["snaps_json"]))
        if "ftz_json" in res:
            FZ.STATE.update(json.loads(str(res["ftz_json"])))
        ox = json.loads(str(res["out_json"])) if "out_json" in res else {}
        print("  [%s] resuming at step %d (t = %.2f s)" % (tag, step0, t))
    else:
        step0, t, v_in, t_hist, v_hist, arr, series, snaps = 0, 0.0, 0.0, [0.0], [0.0], {}, [], {}
        ox = {}
        print("  [%s] %s: %d cells, %d sources, reach %d cells, Q_ref %.4e" % (tag, label, vols.size, len(src),
                                                                             M["nreach"], qref))
    ox.setdefault("v_out", 0.0)
    print("  [%s] threads: MKL_NUM_THREADS=%s OMP_NUM_THREADS=%s" % (tag, os.environ.get("MKL_NUM_THREADS", "default"),
                                                                 os.environ.get("OMP_NUM_THREADS", "default")))
    sys.stdout.flush()
    w0 = time.time()
    reason = "max_steps"
    DG.install()
    DG.take()
    dlog = open(os.path.join(OUT, "diag_%s.csv" % tag), "a")
    dlog.write("# step,wall_s,dt_s,attempts,linear_solves,solve_time_s,picard_last,probe_s (resumed at step %d, %s)\n"
               % (step0, time.strftime("%Y-%m-%d %H:%M:%S")))
    for step in range(step0, MAX_STEPS):
        FZ.STATE["step"] = step + 1
        FZ.set_active(FTZ_FROM.get(tag, 0) is not None and step >= FTZ_FROM.get(tag, 0))
        ws = time.time()
        dt = solve_transport_step(st, dt_cap=DT_CAP)
        dgc = DG.take()
        dlog.write("%d,%.4f,%.6g,%d,%d,%.4f,%d,%s\n" % (step + 1, time.time() - ws, dt, dgc["attempts"], dgc["solves"],
                                                  dgc["solve_time"], st.get("picard_iterations_last", 0),
                                                  ("%.4f" % DG.probe()) if (step + 1) % 100 == 0 else ""))
        if (step + 1) % 100 == 0:
            dlog.flush()
        t += float(dt)
        dq = np.asarray(st["last_div_q"], dtype=float)
        v_in += float(np.sum(dq[hm] * vols[hm])) * float(dt)
        s = np.array(st["saturation"].value, copy=True)
        s[hm] = 1.0
        if outlet:
            o_step = float(np.sum(-dq[om] * vols[om])) * float(dt)
            ox["v_out"] += o_step
            if "first_out" not in ox and o_step > 0.0:
                ox["first_out"] = {"step": step + 1, "t": t, "v_in": v_in}
            s[om] = 0.0
        st["saturation"].setValue(s)
        t_hist.append(t)
        v_hist.append(v_in)
        f = (s >= 0.5) & rub
        if "t" not in arr and np.any(f & low & reach):
            arr = {"t": t, "v_in": v_in, "step": step + 1}
        frac = (f & reach).sum() / float(M["nreach"])
        n_out = int((f & ~reach).sum())
        for thr in SNAPS:
            key = "%.2f" % thr
            if key not in snaps and frac >= thr:
                m = metrics(s, M)
                m.update({"t": t, "v_in": v_in, "step": step + 1, "arrival": dict(arr),
                          "rate_ratio": stop_rule.window_rate(t_hist, v_hist) / (stop_rule.EPS * qref)})
                snaps[key] = m
                np.savez_compressed(os.path.join(OUT, "snap_%s_%s.npz" % (tag, key)), s=s.reshape(shp))
                print("    [%s] snapshot %s at step %d, t %.2f s, V_in %.3f" % (tag, key, step + 1, t, v_in))
        if outlet and "eq" not in ox and v_in >= v_eq:
            m = metrics(s, M)
            m.update({"t": t, "v_in": v_in, "step": step + 1, "v_out": ox["v_out"], "frac": frac,
                      "rate_ratio": stop_rule.window_rate(t_hist, v_hist) / (stop_rule.EPS * qref)})
            ox["eq"] = m
            np.savez_compressed(os.path.join(OUT, "snap_%s_eq.npz" % tag), s=s.reshape(shp))
            print("    [%s] V_in reached the closed case's 0.95 value at step %d, t %.2f s; outflow %.3f"
                  % (tag, step + 1, t, ox["v_out"]))
        if (step + 1) % RECORD_EVERY == 0:
            series.append([step + 1, t, float(dt), v_in, int(f.sum()), int((f & reach).sum()), n_out]
                          + ([ox["v_out"]] if outlet else []))
        if (step + 1) % 500 == 0:
            print("    [%s] step %6d  t %9.2f s  dt %.2e  V_in %8.3f  reach filled %.4f  outside %d  wall %.0f s"
                  % (tag, step + 1, t, dt, v_in, frac, n_out, wall_prev + time.time() - w0))
            sys.stdout.flush()
        if outlet:
            if "rate_at" not in ox and stop_rule.stop_rule_physical(t_hist, v_hist, qref):
                ox["rate_at"] = {"step": step + 1, "t": t, "v_in": v_in}
            if "creep_at" not in ox and n_out >= CREEP_MAX * M["nreach"]:
                ox["creep_at"] = {"step": step + 1, "t": t, "v_in": v_in}
            if v_in >= v_target:
                reason = "V_in target %.4f (%.1f x the 0.95 snapshot of %s)" % (v_target, outlet["factor"], outlet["ref"])
                break
        elif stop_rule.stop_rule_physical(t_hist, v_hist, qref):
            reason = "rate (4-29)"
            break
        elif n_out >= CREEP_MAX * M["nreach"]:
            reason = "creep: filled outside the reach domain >= %.0f%% of it" % (100 * CREEP_MAX)
            break
        if CASE_CAP[0] is not None and wall_prev + time.time() - w0 >= CASE_CAP[0]:
            reason = "wall cap %.0f h" % (CASE_CAP[0] / 3600.0)
            break
        if (step + 1) % CKPT_EVERY == 0:
            keeper.maybe_save(st, step=step + 1, t=t, v_in=v_in, t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist),
                              arr_json=json.dumps(arr), series_json=json.dumps(series), snaps_json=json.dumps(snaps),
                              ftz_json=json.dumps(dict((k, FZ.STATE[k]) for k in ("calls", "diag_subnormal",
                                                                                   "diag_steps", "flushed"))),
                              out_json=json.dumps(ox))
            led = read_ledger()
            led[tag] = {"wall": wall_prev + time.time() - w0, "done": False, "step": step + 1}
            write_ledger(led)
    dlog.close()
    wall = wall_prev + time.time() - w0
    s = np.asarray(st["saturation"].value, dtype=float)
    p = np.asarray(st["pressure"].value, dtype=float)
    np.savez_compressed(os.path.join(OUT, "final_%s.npz" % tag), s=s.reshape(shp), p=p.reshape(shp), box=np.array(box),
                        src=src, series=np.array(series), t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    fm = metrics(s, M)
    fm.update({"rate_ratio": stop_rule.window_rate(t_hist, v_hist) / (stop_rule.EPS * qref)})
    out = {"tag": tag, "label": label, "p0": p0, "reason": reason, "steps": step + 1, "t": t, "v_in": v_in,
           "wall": wall, "arrival": arr, "qref": qref, "snaps": snaps, "final": fm, "nreach": M["nreach"],
           "approx": False, "ftz_from": FTZ_FROM.get(tag, 0), "opts": opts, "outlet": ox if outlet else None,
           "ftz": dict((k, FZ.STATE[k]) for k in ("calls", "diag_subnormal", "diag_steps", "flushed"))}
    FZ.set_active(False)
    FZ.STATE.update({"calls": 0, "diag_subnormal": 0, "diag_steps": [], "flushed": 0})
    json.dump(out, open(os.path.join(OUT, "result_%s.json" % tag), "w"), indent=1, sort_keys=True)
    led = read_ledger()
    led[tag] = {"wall": wall, "done": True, "step": step + 1}
    write_ledger(led)
    keeper.finish()
    print("  [%s] done: %s at step %d, t %.2f s, V_in %.3f m3/m, wall %.1f h" % (tag, reason, step + 1, t, v_in,
                                                                                wall / 3600.0))
    sys.stdout.flush()
    return out


def close_s1(C, srcs, collar_t, box):
    """S1 was run before the snapshot / creep rules (main line, 2026-10-18: end it now). End state = the last
    checkpoint (copied to outputs/xz_solver/s1_end); snapshots approximated from the 20-step records, in which the
    filled count includes cells outside the reach domain -- so the approximate reach-fill fraction is an upper bound
    and the snapshot times / volumes are lower bounds."""
    tag, p0 = "S1", 2.0e6
    M, st, src = case_masks(tag, p0, C, srcs, collar_t, box)
    j0, j1, i0, i1 = box
    shp = (j1 - j0, i1 - i0)
    from src.analysis.run_checkpoint import load as ck_load
    sp = st["slurry_parameters"]
    qref = stop_rule.q_ref(N_REF, sp["calibrated_permeability_coefficient"], p0, sp["plastic_viscosity"])
    fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % (tag, p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(N_REF)
    res = ck_load(os.path.join(OUT, "s1_end", "ckpt_S1"), st, fingerprint=state_fingerprint(st, fp))
    s = np.asarray(st["saturation"].value, dtype=float)
    t_hist, v_hist = list(np.atleast_1d(res["t_hist"])), list(np.atleast_1d(res["v_hist"]))
    series = json.loads(str(res["series_json"]))
    arr = json.loads(str(res["arr_json"]))
    fm = metrics(s, M)
    fm.update({"rate_ratio": stop_rule.window_rate(t_hist, v_hist) / (stop_rule.EPS * qref)})
    np.savez_compressed(os.path.join(OUT, "final_S1.npz"), s=s.reshape(shp), box=np.array(box), src=src,
                        series=np.array(series), t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    snaps = {}
    for thr in SNAPS:
        hit = [r for r in series if r[4] >= thr * M["nreach"]]
        if not hit:
            continue
        r = hit[0]
        k = int(r[0])
        loss = r[5]
        snaps["%.2f" % thr] = {"t": r[1], "v_in": r[3], "step": k, "loss": loss, "loss_share": loss / r[3],
                               "arrival": arr if arr and arr["step"] <= k else {},
                               "rate_ratio": stop_rule.window_rate(t_hist[:k + 1], v_hist[:k + 1]) / (stop_rule.EPS * qref)}
    lb = [r for r in series if r[4] >= (1 + CREEP_MAX) * M["nreach"]]
    out = {"tag": tag, "label": CASES[0][1], "p0": p0, "reason": "stopped by the main line (creep rule exceeded)",
           "steps": int(res["step"]), "t": float(res["t"]), "v_in": float(res["v_in"]),
           "wall": float(read_ledger()["S1"]["wall"]), "arrival": arr, "qref": qref, "snaps": snaps, "final": fm,
           "nreach": M["nreach"], "approx": True,
           "creep_rule_by": ({"step": lb[0][0], "t": lb[0][1], "v_in": lb[0][3]} if lb else {})}
    json.dump(out, open(os.path.join(OUT, "result_S1.json"), "w"), indent=1, sort_keys=True)
    led = read_ledger()
    led["S1"]["done"] = True
    write_ledger(led)
    print("  [S1] closed at step %d (t %.2f s, V_in %.3f)" % (out["steps"], out["t"], out["v_in"]))


def run():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    D, C, srcs, collar_t = load()
    box = crop_box(C)
    if not os.path.isfile(os.path.join(OUT, "result_S1.json")) and os.path.isdir(os.path.join(OUT, "s1_end")):
        close_s1(C, srcs, collar_t, box)
    for tag, label, _, p0 in CASES:
        led = read_ledger()
        total = sum(float(v.get("wall", 0.0)) for v in led.values())
        if led.get(tag, {}).get("done"):
            print("  [%s] already done" % tag)
            continue
        if total >= CAP_S:
            print("  cumulative wall time %.1f h >= 48 h: stopping before %s" % (total / 3600.0, tag))
            break
        run_case(tag, label, p0, D, C, srcs, collar_t, box)


def grid_setup(c2):
    """Stored 0.5 m grid of a stage-2 case (grids.py output, used unchanged): C, sources, collar t, crop box. The
    sources are re-cut from the K's v4 hole cells exactly as grids.py did and must equal the stored ones."""
    G = np.load(os.path.join(OUT, "grid_%s.npz" % c2["grid"]))
    K = float(G["K"])
    Dk = np.load(os.path.join(P.DATA, "base_geometry_K%.2f_v4.npz" % K))
    C = dict((q, G[q]) for q in ("n", "rub", "cls", "S", "T", "X", "Z", "basal", "hang", "low"))
    C["cy"], C["cx"] = C["rub"].shape
    if abs(float(G["low_t"]) - LOW_T) > 1e-12:
        raise SystemExit("grid %s: lower-end threshold %.6f differs from %.6f" % (c2["grid"], float(G["low_t"]), LOW_T))
    key = c2["key"]
    if key == "s2":
        cells, ct = Dk["s2_cells"], float(Dk["fan_collar_t"][1])
    else:
        cells, ct = Dk["fan_cells"][Dk["fan_ptr"][key]:Dk["fan_ptr"][key + 1]], float(Dk["fan_collar_t"][key])
    src = sources(cells, Dk["cls"].shape[1], C)
    stored = [k_ for k_ in G.files if k_.startswith("src_")]
    same = [k_ for k_ in stored if np.array_equal(G[k_], np.array(src, int).reshape(-1, 2))]
    if not same:
        raise SystemExit("case %s: sources do not match any stored hole set of grid %s" % (c2["tag"], c2["grid"]))
    return C, {c2["tag"]: src}, {c2["tag"]: ct}, tuple(int(v) for v in G["box"])


def run_batch(tags):
    """Run the listed cases serially in this process (one process per batch; batches run in parallel, each with its
    own ledger LEDGER[0] and the per-case wall cap CASE_CAP). Stage-2 cases (CASES2) run on their stored grids; an
    outlet case reads the result of its closed reference case, so it goes after that case in the same batch."""
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    D, C, srcs, collar_t = load()
    box = crop_box(C)
    known = dict((c_[0], c_) for c_ in CASES)
    for tag in tags:
        if tag not in known and tag not in CASES2:
            raise SystemExit("unknown case %s (configuration %s)" % (tag, CFG["name"]))
    if CASES2:
        # the base cases must run on the stored base grid unchanged (main line 2026-10-21)
        G = np.load(os.path.join(OUT, "grid_K%.2f_s1.npz" % P.K_BASE))
        if tuple(int(v) for v in G["box"]) != tuple(box) or not all(np.array_equal(G[q], C[q]) for q in
                                                                    ("n", "rub", "cls", "basal", "hang", "low")):
            raise SystemExit("base grid differs from the stored grid_K%.2f_s1.npz" % P.K_BASE)
    for tag in tags:
        if read_ledger().get(tag, {}).get("done"):
            print("  [%s] already done" % tag)
            continue
        if tag in known:
            _, label, _, p0 = known[tag]
            run_case(tag, label, p0, D, C, srcs, collar_t, box)
        else:
            c2 = CASES2[tag]
            C2, srcs2, ct2, box2 = grid_setup(c2)
            run_case(tag, c2["label"], c2["p0"], D, C2, srcs2, ct2, box2,
                     opts=dict((k_, c2[k_]) for k_ in ("grid", "outlet") if k_ in c2))


def fmt_m(m, nreach):
    a = m.get("arrival") or {}
    return ("t %.2f s, V_in %.3f m3/m, step %d; lower end %s; loss %.3f m3/m (%.4f of the stored in the reach domain);"
            " hanging side filled %s; void contact %s; filled outside the reach domain %s; rate / threshold %.3f"
            % (m["t"], m["v_in"], m["step"], ("reached at t %.2f s, V_in %.3f" % (a["t"], a["v_in"])) if a else "not reached",
               m["loss"], m["loss_share"], ("%.4f" % m["hang_fill"]) if "hang_fill" in m else "-",
               ("%.2f m" % m["void_contact"]) if "void_contact" in m else "-",
               ("%d cells" % m["outside"]) if "outside" in m else "-", m["rate_ratio"]))


def summary():
    D, C, srcs, collar_t = load()
    box = crop_box(C)
    j0, j1, i0, i1 = box
    a2 = CELL * CELL
    cr = lambda a: a[j0:j1, i0:i1]
    rub, basal, hang, low, cls = (cr(C[k]) for k in ("rub", "basal", "hang", "low", "cls"))
    L = ["Solver cases (scenario 1, K 1.25 v4, 0.5 m, cropped domain %d x %d; v0.6; stop: rule (4-29) eps %.2g beta %.2g"
         " n_ref %.2f, or filled outside the reach domain >= %.0f%% of it)" % (i1 - i0, j1 - j0, stop_rule.EPS,
                                                                            stop_rule.BETA, N_REF, 100 * CREEP_MAX),
         "  zones on the 0.5 m grid: rubble %.2f m2, basal %.2f m2, hanging side %.2f m2; lower end of the basal layer"
         " (t <= %.2f m) %d cells; all fill metrics count reach-domain cells only" % (
             rub.sum() * a2, basal.sum() * a2, hang.sum() * a2, LOW_T, low.sum())]
    reach = {}
    for tag, label, _, p0 in CASES:
        M, st, src = case_masks(tag, p0, C, srcs, collar_t, box)
        reach[tag] = M["reach"].reshape(rub.shape)
        fn = os.path.join(OUT, "result_%s.json" % tag)
        L.append("  %s (%s): reach domain %d cells = %.2f m2 (pore %.2f m3/m); lower end in the reach domain: %s;"
                 " loss zone = basal cells with t <= %.2f m (%d cells, %d in the reach domain)"
                 % (tag, label, M["nreach"], M["nreach"] * a2, (M["n"] * M["vols"])[M["reach"]].sum(),
                    bool(np.any(M["low"] & M["reach"])), collar_t[tag] - LOSS_DT, M["loss"].sum(),
                    (M["loss"] & M["reach"]).sum()))
        if not os.path.isfile(fn):
            L.append("     not run")
            continue
        r = json.load(open(fn))
        L.append("     end: %s; step %d, t %.2f s, V_in %.3f m3/m, wall %.2f h%s" % (
            r["reason"], r["steps"], r["t"], r["v_in"], r["wall"] / 3600.0,
            ("; creep rule would have stopped it by step %d (t %.2f s, V_in %.3f; lower bound from the total filled"
             " count)" % (r["creep_rule_by"]["step"], r["creep_rule_by"]["t"], r["creep_rule_by"]["v_in"]))
            if r.get("creep_rule_by") else ""))
        for key in ("0.90", "0.95", "0.99"):
            if key in r["snaps"]:
                L.append("     snapshot %s%s: %s" % (key, " (APPROXIMATE, 20-step records)" if r["approx"] else "",
                                                    fmt_m(r["snaps"][key], r["nreach"])))
            else:
                L.append("     snapshot %s: not reached" % key)
        # lower-end corner of the rubble (old box X < -5 m, Z < 20 m; the largest uncovered block since step 1)
        corner = rub & (cr(C["X"]) < CORNER_BOX[0]) & (cr(C["Z"]) < CORNER_BOX[1])
        sp_ = os.path.join(OUT, "snap_%s_0.95.npz" % tag)
        s95 = np.load(sp_)["s"] if os.path.isfile(sp_) else np.load(os.path.join(OUT, "final_%s.npz" % tag))["s"]
        f95 = (s95 >= 0.5) & rub & reach[tag]
        L.append("     lower-end corner (X < %g, Z < %g m; %.2f m2, hanging-side part %.2f m2) filled at %s: %.4f (hanging-side"
                 " part %.4f); in the reach domain %.4f" % (
                     CORNER_BOX[0], CORNER_BOX[1], corner.sum() * a2, (corner & hang).sum() * a2, "0.95" if os.path.isfile(sp_) else "the end state",
                     (f95 & corner).sum() / float(corner.sum()), (f95 & corner & hang).sum() / float((corner & hang).sum()),
                     (reach[tag] & corner).sum() / float(corner.sum())))
        fm = dict(r["final"], t=r["t"], v_in=r["v_in"], step=r["steps"], arrival=r["arrival"])
        L.append("     end state: reach filled %.4f (%d of %d); %s" % (fm["frac_in"], fm["filled_in"], r["nreach"],
                                                                     fmt_m(fm, r["nreach"])))
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "cases.txt"), "w").write(txt)
    np.savez_compressed(os.path.join(OUT, "summary.npz"), box=np.array(box), rub=rub, cls=cls, hang=hang, basal=basal,
                        low=low, X=cr(C["X"]), Z=cr(C["Z"]), tags=np.array([c_[0] for c_ in CASES]),
                        reach=np.array([reach[c_[0]] for c_ in CASES]), p0=np.array([c_[3] for c_ in CASES]))
    print(txt)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        check()
    if "--run" in a:
        run()
    if "--summary" in a:
        summary()
    if "--cases" in a:
        # parallel batches (main line 2026-10-20): one thread, flush from step 1, never changed mid-run
        if CFG["name"] != "40m" and (os.environ.get("MKL_NUM_THREADS") != "1" or os.environ.get("OMP_NUM_THREADS") != "1"):
            raise SystemExit("set MKL_NUM_THREADS=1 and OMP_NUM_THREADS=1 (scripts/xz_solver/launch_batches.ps1 does)")
        if CFG["name"] != "40m" and any(v != 0 for v in FTZ_FROM.values()):
            raise SystemExit("every case must flush subnormals from the first step (solver.ftz_from = 0)")
        tags_ = a[a.index("--cases") + 1].split(",")
        if "--cap-hours" in a:
            CASE_CAP[0] = float(a[a.index("--cap-hours") + 1]) * 3600.0
        LEDGER[0] = "wall_%s.json" % "-".join(tags_)
        run_batch(tags_)
    if "--rerun-s1" in a:
        OUT = RERUN_DIR
        if not os.path.isdir(OUT):
            os.makedirs(OUT)
        FTZ_FROM["S1"] = 0
        CASE_CAP[0] = 10 * 3600.0
        D_, C_, srcs_, collar_t_ = load()
        run_case("S1", CASES[0][1], CASES[0][3], D_, C_, srcs_, collar_t_, crop_box(C_))
