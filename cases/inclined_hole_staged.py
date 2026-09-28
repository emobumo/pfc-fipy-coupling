# -*- coding: utf-8 -*-
"""
APPLICATION case: staged advancing grouting versus a single pass.

Same hole, parameters, mesh and per-stage stop rule as inclined_hole_channel
(which borrows them from inclined_hole_gradient, the line-source baseline).
The hole is drilled in passes; at stage k the perforated pipe bleeds along the
whole hole so far, [0, DEPTH_k] along-hole; when the stage stops, the grout
sets (apply_stage_closure: n_{k+1} = n_k(1-S_k), k and lambda follow) and the
next pass drills deeper into a medium the previous one has partly cemented.

    single      one pass, full hole, the whole budget
    staged      passes at DEPTHS, the SAME total budget, unused quota carried
                forward (the crew moves on when a hole will not take grout;
                the volume is not thrown away)

BUDGET: the volume the uniform baseline needed to stall, 46.14 m^3/m -- so
single and staged inject the same grout into the same hole, and the question
is purely what the sequence did with it.

WHAT TO READ
------------
Cumulative fill is what the sequence cemented: occupied = (n_0 - n_final)/n_0
per cell. It is classified against the reachable domain of the VIRGIN field
from the FULL hole -- what a single pass could have reached -- so that a
reachable, unfilled pocket sealed in by earlier cement shows as a BYPASS VOID.
That is the sequence-induced residual: ground the design could have grouted
and the staging walled off. Per stage, the reach from that stage's source in
the CURRENT (cemented) field is also recorded, to see what each pass could
still get to.

Two things the ladder's step 7 predicts and this case can test:
  - a drilling increment shorter than the local L_max leaves the new hole
    section inside the previous pass's cemented zone, so the pass delivers
    little (the reference project's 7 -> 10 -> 17.5 m has increments of 3
    and 7.5 m against a stall length of ~8 m under the old parameter set and
    ~18-20 m under v0.6);
  - the cemented pipe length self-blocks, so no packer is needed.

    powershell -File scripts\run_local.ps1 cases\inclined_hole_staged.py [--field base|collar|random] [--plot]

Outputs under outputs/inclined_hole_staged/<field>/<run>/.
"""
import imp
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.run_checkpoint import (
    Checkpointer, state_fingerprint, STAGE_STATE_KEYS,
)
from src.models.slurry_transport.stage_update import (
    initialize_stage_ledger, apply_stage_closure, stage_ledger_report,
)
from src.analysis.fill_diagnostics import (
    FILLED, UNREACHABLE, FRONT_SHORTFALL, BYPASS_VOID, OVERSHOOT,
    reachable_domain, classify_unfilled, injection_rate_diagnostics,
)

C = imp.load_source("inclined_hole_channel",
                    os.path.join(REPO, "cases", "inclined_hole_channel.py"))
G = C.G

# Drilling passes, along-hole depth [m]. The reference project's proportions
# (7 / 10 / 17.5 on a 17.5 m hole) mapped onto this 17 m hole.
DEPTHS_REFERENCE = (7.0, 10.0, 17.0)
# A two-pass alternative. Under the old parameter set its 8.5 m increment
# exceeded the ~8 m stall length; under v0.6 (L_max ~18-20 m) NO increment
# on a 17 m hole can, which is itself the point to measure.

DEPTHS_SPACED = (8.5, 17.0)   # NOTE: at 2.5 m cells 7.0 and 8.5 m select the same source cells
SEQUENCES = (("staged_7_10_17", DEPTHS_REFERENCE), ("staged_8.5_17", DEPTHS_SPACED))

OUT_ROOT = os.path.join(REPO, "outputs", "inclined_hole_staged")
_CKPT_DIR = os.path.join(REPO, "outputs", "checkpoints")
# Crash-resilience: see src/analysis/run_checkpoint.py. 0 disables.
CKPT_EVERY = 200


def hole_cells_to_depth(mx, my, nx, ny, depth):
    """Cells the hole passes through between the mouth and `depth` along it,
    sampled the way G.hole_cells samples the full hole -- so the full depth
    returns exactly the baseline's source set."""
    import math
    end, d, _ = G.hole_geometry()
    idx, dx, dy = G._cell_index_map(mx, my, nx, ny)
    cells = set()
    nsamp = 2000
    for s in range(nsamp + 1):
        t = float(s) / nsamp * float(depth) / G.HOLE_LEN
        x = G.MOUTH[0] + t * (end[0] - G.MOUTH[0])
        y = G.MOUTH[1] + t * (end[1] - G.MOUTH[1])
        ix = int(math.floor((x - G.X_MIN) / dx))
        iy = int(math.floor((y - G.Y_MIN) / dy))
        if (ix, iy) in idx:
            cells.add(idx[(ix, iy)])
    return np.array(sorted(cells), dtype=int)


def make_field(kind, mx, my):
    if kind == "base":
        return C.field("base", mx, my)
    if kind == "collar":
        # NOTE (2026-09-27): x = -13.75 was the collar column of the old 30 x 30 m
        # domain (mouth at x = -15). On the v0.6 60 x 60 m domain (mouth at -30,
        # toe at x = -16.07) this band sits just PAST THE TOE, so results
        # labelled "collar" describe a near-toe channel. Kept unchanged so the
        # old anchors still reproduce; use "collar_v06" for the real collar.
        return C.field("band", mx, my, phi_band=C.BAND_PHI_POS, band_x=-13.75)
    if kind == "collar_v06":
        # the same band the channel case runs as band_collar_x-28.75
        return C.field("band", mx, my, phi_band=C.BAND_PHI_POS, band_x=C.BAND_X_COLLAR)
    if kind == "random":
        return C.field("random", mx, my, seed=1)
    raise ValueError(kind)


def install_source(state, cells, p0):
    from fipy import CellVariable
    mask = np.zeros(state["mesh"].numberOfCells, dtype=float)
    mask[cells] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=state["mesh"], value=mask)
    state["interior_dirichlet_value"] = float(p0)
    s = np.zeros(mask.size)
    s[cells] = 1.0
    state["saturation"].setValue(s)
    state["pressure"].setValue(np.where(mask > 0.5, float(p0), 0.0))


def run_sequence(label, phi, depths, budget, reach_virgin, full_cells, plot,
                 ckpt_every=CKPT_EVERY, n_ref=None):
    """n_ref: the stop rule's reference porosity (default: C.PHI_BASE). Each
    pass restarts its own clock (the march starts at t = 0), Q_ref is fixed."""
    from src.analysis import stop_rule
    rtag = stop_rule.rate_tag(C.PHI_BASE if n_ref is None else n_ref)
    state, mx, my, nx, ny, phi_used, _, _ = C.build_from_phi(phi)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    n0 = np.array(state["porosity"].value, copy=True)
    initialize_stage_ledger(state)

    remaining = float(budget) if budget is not None else None
    stages = []
    history_all = []
    t_offset = 0.0
    filled_by_stage = np.zeros(mx.size, dtype=int) - 1     # -1 = never
    src_prev = np.zeros(mx.size, dtype=bool)
    k0 = 0

    # The sequence checkpoint carries what between-stage solidification
    # rewrites (porosity, permeability, mobilities, ledger) on top of the
    # per-step state, so a crash in stage 3 resumes at stage 3 instead of
    # replaying stages 1 and 2. The fingerprint is taken on the VIRGIN state,
    # before any closure, so it identifies the sequence rather than a stage.
    seq_keeper = None
    if ckpt_every > 0:
        if not os.path.isdir(_CKPT_DIR):
            os.makedirs(_CKPT_DIR)
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", "staged_" + label)
        seq_keeper = Checkpointer(
            os.path.join(_CKPT_DIR, tag + "_seq"), every=1,
            also=STAGE_STATE_KEYS,
            fingerprint=state_fingerprint(
                state, "staged|%s|depths=%s|budget=%s%s" % (label, depths, budget, rtag)))
        resumed = seq_keeper.restore(state)
        if resumed:
            k0 = int(resumed["stage"])
            t_offset = float(resumed["t_offset"])
            remaining = (None if budget is None else float(resumed["remaining"]))
            filled_by_stage = np.array(resumed["filled_by_stage"], dtype=int)
            src_prev = np.array(resumed["src_prev"], dtype=bool)
            history_all = [(float(a), float(b), int(c))
                           for a, b, c in np.atleast_2d(resumed["history_all"])]
            stages = json.loads(str(resumed["stages_json"]))
            state["stage_ledger"] = json.loads(str(resumed["ledger_json"]))
            state["stage_history"] = json.loads(str(resumed["history_json"]))
            print("  [%s] resuming after stage %d" % (label, k0))
            sys.stdout.flush()

    for k, depth in enumerate(depths):
        if k < k0:
            continue                      # already done, restored from disk
        src = hole_cells_to_depth(mx, my, nx, ny, depth)
        src_mask = np.zeros(mx.size, dtype=bool)
        src_mask[src] = True
        if k == 0:
            install_source(state, src, G.P0)
        # (later stages: apply_stage_closure already installed src_mask)

        # What this pass can still reach, in the ground as it now is.
        reach_now = reachable_domain(state, src, p0=G.P0, use_gravity=True)
        n_at_start = np.array(state["porosity"].value, copy=True)
        floor = float(state["slurry_parameters"].get("stage_porosity_floor", 1.0e-3))
        buried = int(np.sum(n_at_start[src] <= floor * (1.0 + 1e-9)))

        stage_label = "%s / stage %d (to %.1f m, %d src cells, %d buried)" % (
            label, k + 1, depth, src.size, buried)
        quota = None if remaining is None else (remaining if remaining > 0 else 1e-12)
        stage_keeper = None
        if ckpt_every > 0:
            tag = re.sub(r"[^A-Za-z0-9._-]+", "_",
                         "staged_%s_stage%d" % (label, k + 1))
            stage_keeper = Checkpointer(
                os.path.join(_CKPT_DIR, tag), every=ckpt_every,
                fingerprint=state_fingerprint(
                    state, "staged|%s|stage=%d|depth=%.3f%s" % (label, k, depth, rtag)))
        m = C.march(state, src, vols, stage_label, v_quota=quota,
                    keeper=stage_keeper, n_ref=n_ref)
        for (t, q) in m["history"]:
            history_all.append((t_offset + t, q, k))
        t_offset += m["t"]
        if remaining is not None:
            remaining -= m["v_in"]

        s_end = np.asarray(state["saturation"].value, dtype=float)
        newly = (s_end >= 0.5) & (filled_by_stage < 0) & np.logical_not(src_mask)
        filled_by_stage[newly] = k
        q = injection_rate_diagnostics(m["history"], tail_fraction=0.2, stall_ratio=0.05)
        stages.append({
            "k": k + 1, "depth": depth, "src_cells": int(src.size), "buried": buried,
            "reason": m["reason"], "t": m["t"], "v_in": m["v_in"],
            "q_decay": q["decay_ratio"] if "decay_ratio" in q else 0.0,
            "reach_now": int(np.sum(reach_now["reachable"])),
            "filled_now": int(np.sum(newly)),
        })

        next_src = None
        if k + 1 < len(depths):
            next_src = hole_cells_to_depth(mx, my, nx, ny, depths[k + 1])
        apply_stage_closure(state, source_mask=next_src)
        src_prev = src_prev | src_mask
        if seq_keeper is not None:
            seq_keeper.save_now(
                state, stage=k + 1, t_offset=t_offset,
                remaining=(-1.0 if remaining is None else remaining),
                filled_by_stage=filled_by_stage, src_prev=src_prev,
                history_all=np.asarray(history_all, dtype=float),
                stages_json=json.dumps(stages),
                ledger_json=json.dumps(state["stage_ledger"]),
                history_json=json.dumps(state["stage_history"]))

    # Cumulative fill = what set. Read it as a saturation for the classifier.
    n_final = np.asarray(state["porosity"].value, dtype=float)
    occupied = np.clip((n0 - n_final) / np.maximum(n0, 1e-12), 0.0, 1.0)
    state["saturation"].setValue(occupied)
    # What the LAST pass could still reach in the ground as the sequence
    # left it. Cemented source cells sit at the porosity floor and emit
    # nothing, so this is effectively the reach of the final fresh section.
    # reach_virgin \ reach_final, unfilled, is the sequence's permanent
    # residual: ground a single pass could have grouted and no further pass
    # from this hole can -- earlier cement now stands between. Enclosed or
    # not, it is what staging cost.
    last_src = hole_cells_to_depth(mx, my, nx, ny, depths[-1])
    reach_final = reachable_domain(state, last_src, p0=G.P0, use_gravity=True)["reachable"]
    cls = classify_unfilled(state, full_cells, reach_virgin, s_c=0.5)
    unfilled_reach = (occupied < 0.5) & reach_virgin & np.logical_not(np.in1d(np.arange(mx.size), full_cells))
    shadowed = unfilled_reach & np.logical_not(reach_final)
    ledger = stage_ledger_report(state)
    if seq_keeper is not None:
        seq_keeper.finish()

    cat = cls["category"]
    row = {
        "label": label, "stages": stages,
        "v_total": sum(st["v_in"] for st in stages),
        "t_total": t_offset,
        "filled": int(np.sum((occupied >= 0.5) & ~np.in1d(np.arange(mx.size), full_cells))),
        "reach_virgin": int(np.sum(reach_virgin)),
        "bypass_void": int(np.sum(cat == BYPASS_VOID)), "n_voids": cls["n_bypass_voids"],
        "front_shortfall": int(np.sum(cat == FRONT_SHORTFALL)),
        "overshoot": int(np.sum(cat == OVERSHOOT)),
        "shadowed": int(np.sum(shadowed)),
        "reach_final": int(np.sum(reach_final)),
        "fill_ratio": ledger["fill_ratio_cumulative"],
        "over_reach_filled": cls["over_reachable"]["filled_fraction"],
    }

    out = os.path.join(OUT_ROOT, label)
    if not os.path.isdir(out):
        os.makedirs(out)
    np.savetxt(os.path.join(out, "cells.csv"),
               np.column_stack([mx, my, phi_used, n_final, occupied, filled_by_stage,
                                reach_virgin.astype(int), reach_final.astype(int),
                                shadowed.astype(int), cat]),
               delimiter=",", comments="",
               header="x,y,phi0,phi_final,occupied,filled_by_stage(-1 never;0..),reachable_virgin,reachable_final,shadowed,category")
    np.savetxt(os.path.join(out, "injection_rate.csv"), np.array(history_all),
               delimiter=",", comments="", header="t_s,Q_m3_per_m_s,stage")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        for key in sorted(row):
            if key != "stages":
                fh.write("%s = %s\n" % (key, row[key]))
        for st in stages:
            fh.write("stage %s\n" % st)
    if plot:
        _figure(label, mx, my, nx, ny, full_cells, phi_used, filled_by_stage, reach_virgin,
                cat, history_all, depths, out)
    return row


def _figure(label, mx, my, nx, ny, hcells, phi, by_stage, reach, cat, hist, depths, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    xe = np.linspace(G.X_MIN, G.X_MAX, nx + 1)
    ye = np.linspace(G.Y_MIN, G.Y_MAX, ny + 1)
    xc, yc = G.grid_axes(nx, ny)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    nst = len(depths)
    cm = ListedColormap(["#e6e6e6"] + ["#9ecae1", "#3182bd", "#08519c", "#041f4a"][:nst])
    nm = BoundaryNorm(np.arange(-1.5, nst + 0.5, 1.0), cm.N)
    im = ax.pcolormesh(xe, ye, G.to_grid(by_stage.astype(float), mx, my, nx, ny), cmap=cm, norm=nm)
    ax.contour(xc, yc, G.to_grid(reach.astype(float), mx, my, nx, ny), levels=[0.5], colors="k", linewidths=1.6)
    G.draw_hole(ax, hcells, mx, my)
    cb = fig.colorbar(im, ax=ax, ticks=range(-1, nst))
    cb.ax.set_yticklabels(["never"] + ["stage %d" % (i + 1) for i in range(nst)])
    ax.set_aspect("equal"); ax.set_title("%s\nwhich pass filled each cell; black = virgin reach" % label)

    ax = axes[1]
    cmap = ListedColormap(["#4a90d9", "#d9d9d9", "#f5c56b", "#d0403f", "#7b4fa3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    im = ax.pcolormesh(xe, ye, G.to_grid(cat.astype(float), mx, my, nx, ny), cmap=cmap, norm=norm)
    ax.contour(xc, yc, G.to_grid(reach.astype(float), mx, my, nx, ny), levels=[0.5], colors="k", linewidths=1.6)
    G.draw_hole(ax, hcells, mx, my)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["filled", "unreachable", "front shortfall", "bypass void", "overshoot"])
    ax.set_aspect("equal"); ax.set_title("cumulative fill vs virgin reach")

    ax = axes[2]
    h = np.array(hist)
    for k in range(nst):
        sel = h[:, 2] == k
        if np.any(sel):
            ax.semilogy(h[sel, 0], np.maximum(h[sel, 1], 1e-30), "-", label="stage %d" % (k + 1))
    ax.set_xlabel("t [s] (stages concatenated)"); ax.set_ylabel("Q [m^3/m/s]")
    ax.set_title("injection rate per stage"); ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "staged_diagnostics.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print("    wrote %s" % path)


def main(argv):
    fields = ["base"]
    plot = False
    i = 0
    while i < len(argv):
        if argv[i] == "--field":
            fields = argv[i + 1].split(","); i += 2
        elif argv[i] == "--plot":
            plot = True; i += 1
        else:
            i += 1

    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx = int(round((G.X_MAX - G.X_MIN) / G.CELL))
    ny = int(round((G.Y_MAX - G.Y_MIN) / G.CELL))
    full_cells = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)

    # Budget: the uniform baseline's stalled volume. The baseline single pass
    # runs first with no quota; what it took to stall is the budget for the
    # staged runs on that field (and for the structured fields too, so every
    # sequence injects the same grout as the uniform baseline).

    budget = None

    all_rows = []
    for fld in fields:
        phi = make_field(fld, mx, my)
        # Virgin reach from the full hole: what one pass could have reached.
        st0, _, _, _, _, _, _, _ = C.build_from_phi(phi)
        reach_virgin = reachable_domain(st0, full_cells, p0=G.P0, use_gravity=True)["reachable"]
        print("=== field %s: virgin reach from full hole = %d cells ===" % (fld, int(reach_virgin.sum())))
        sys.stdout.flush()

        rows = []
        if budget is None:
            # First field's single pass runs to stall and sets the budget.
            base_row = run_sequence("%s/single" % fld, phi, (G.HOLE_LEN,), None,
                                    reach_virgin, full_cells, plot)
            budget = base_row["v_total"]
            print("  budget for every later run = V(single pass at stall) = %.3f m^3/m" % budget)
            sys.stdout.flush()
            rows.append(base_row)
        else:
            rows.append(run_sequence("%s/single" % fld, phi, (G.HOLE_LEN,), budget,
                                     reach_virgin, full_cells, plot))
        for name, depths in SEQUENCES:
            rows.append(run_sequence("%s/%s" % (fld, name), phi, depths, budget,
                                     reach_virgin, full_cells, plot))
        all_rows.extend(rows)

    hdr = "%-26s %7s %7s %6s %6s %5s %5s %6s %5s %7s %7s" % (
        "run", "V_tot", "t_tot", "fill", "reach", "void", "short", "shadow", "over", "fill%", "reach%")
    lines = ["", hdr, "-" * len(hdr)]
    for r in all_rows:
        lines.append("%-26s %7.2f %7.0f %6d %6d %5d %5d %6d %5d %6.1f%% %6.1f%%" % (
            r["label"], r["v_total"], r["t_total"], r["filled"], r["reach_virgin"],
            r["bypass_void"], r["front_shortfall"], r["shadowed"], r["overshoot"],
            100 * r["fill_ratio"], 100 * r["over_reach_filled"]))
        for st in r["stages"]:
            lines.append("    stage %d to %4.1f m: %2d src (%2d buried) | %-5s t=%6.0f s V=%6.2f Qtail=%.3f | reach now %2d, newly filled %2d"
                         % (st["k"], st["depth"], st["src_cells"], st["buried"], st["reason"],
                            st["t"], st["v_in"], st["q_decay"], st["reach_now"], st["filled_now"]))
    lines.append("")
    lines.append("fill = cells cemented (excl. hole); reach = virgin reach from the full hole; void/short/over =")
    lines.append("bypass void / front shortfall / overshoot vs virgin reach; shadow = reachable in the virgin")
    lines.append("field, unfilled, and UNREACHABLE from the last pass in the cemented field: the sequence's")
    lines.append("permanent residual; fill% = cumulative pore capacity")
    lines.append("occupied (stage ledger); reach% = share of virgin-reachable cells cemented.")
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT_ROOT):
        os.makedirs(OUT_ROOT)
    with open(os.path.join(OUT_ROOT, "comparison.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
