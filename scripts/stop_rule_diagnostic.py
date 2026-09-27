# -*- coding: utf-8 -*-
"""
Stop-rule diagnostic (step 1 of replacing the step-count stall window).

    powershell -File scripts\run_local.ps1 scripts\stop_rule_diagnostic.py run D1 2.5
    powershell -File scripts\run_local.ps1 scripts\stop_rule_diagnostic.py evaluate

The current stall rule ("60 steps without a new cell crossing S = 0.5") is
grid dependent twice over: steps shrink with the cell, and "a new cell" is a
front-speed threshold dx / T_w. Under constant pressure there is no finite-
time stall anyway -- the front only approaches the reach -- so "takes no
more" is a stop RULE. The candidate form, like the field's grouting codes:

    stop when the mean injection rate over a window T_w,  Q_bar / Q_ref < eps.

run: one single pass with stall detection and quota OFF, to model time
T_END (or Q / Q_ref(i) < 1e-4). Every step records t, dt, Q, V_in, V_store
and the count of cells with S >= 0.5; saturation snapshots are kept at
log-spaced model times (40 per decade). Its own march loop, a copy of the
cases' (solve_transport_step, sources held at S = 1); no default and no
existing loop is changed. Checkpointed.

    D1 uniform 0.18, full 17 m hole (the channel baseline: quota, target)
    D2 gradient 0.12 -> 0.30, full hole (the flip anchor)
    D3 uniform 0.10, full hole (where the old rule broke at 1.25 m)
    D4 uniform 0.10, 7 m source (first pass of 7 -> 17)
    D5 uniform 0.18, 7 m source

evaluate: replays candidate rules on the records, no re-run. Q_bar over a
window is taken from the cumulative injection, (V_in(t) - V_in(t - T_w)) / T_w
-- exact and immune to single-step spikes. Q_ref (i) = k(n_ref) p0 / mu_p
(n_ref: the porosity at the hole, the gradient field's at the hole's
mid-point); (ii) = peak Q after the first 1% of the model time.
Outputs: outputs/stop_rule_diagnostic/.
"""
from __future__ import print_function

import imp
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.fill_diagnostics import reachable_domain
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
C, G = S.C, S.G
OUT = os.path.join(REPO, "outputs", "stop_rule_diagnostic")
T_END = float(os.environ.get("STOPRULE_T_END", "3.0e4"))   # env override for smoke tests only
Q_FLOOR = 1.0e-4            # stop early once Q / Q_ref(i) is below this (after T_MIN)
T_MIN = 100.0
SNAP_PER_DECADE = 40
MAX_STEPS = 400000          # this script's own bound; the cases' MAX_STEPS is untouched

CASES = {  # kind, porosity (uniform) or None for the gradient, source depth along the hole
    "D1": ("uniform", 0.18, 17.0),
    "D2": ("gradient", None, 17.0),
    "D3": ("uniform", 0.10, 17.0),
    "D4": ("uniform", 0.10, 7.0),
    "D5": ("uniform", 0.18, 7.0),
}


def k_of(n):
    return G.A_CAL * n ** 3 / (1.0 - n) ** 2


def l_max(n):
    return 83.33 * n / (1.0 - n)


def build(case, cell):
    kind, n, depth = CASES[case]
    G.CELL = cell
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, cell)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx = int(round((G.X_MAX - G.X_MIN) / cell))
    ny = int(round((G.Y_MAX - G.Y_MIN) / cell))
    if kind == "gradient":
        st, mx, my, nx, ny, phi, hc, k = G.build_case("gradient")
        end, d, nrm = G.hole_geometry()
        n_ref = float(G.PHI_BOTTOM + (G.PHI_TOP - G.PHI_BOTTOM) * (0.5 * (G.MOUTH[1] + end[1]) / G.H))
    else:
        st, mx, my, nx, ny, phi, hc, k = C.build_from_phi(np.full(mx.size, n))
        n_ref = n
    if depth < G.HOLE_LEN:
        hc = S.hole_cells_to_depth(mx, my, nx, ny, depth)
        S.install_source(st, hc, G.P0)
    return st, mx, my, np.asarray(hc, dtype=int), n_ref, depth


def run(case, cell):
    st, mx, my, src, n_ref, depth = build(case, cell)
    q_ref = k_of(n_ref) * G.P0 / G.MU_P
    vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
    hm = np.zeros(mx.size, dtype=bool)
    hm[src] = True
    phi = np.clip(np.asarray(st["porosity"].value, dtype=float), 1e-6, 1.0)
    s0 = np.array(st["saturation"].value, copy=True)
    reach = reachable_domain(st, src, p0=G.P0, use_gravity=True)["reachable"]
    targets = 10.0 ** (np.arange(-1.0, math.log10(T_END) + 1e-9, 1.0 / SNAP_PER_DECADE))

    tag = "%s_%g" % (case, cell)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    keeper = Checkpointer(os.path.join(OUT, "ckpt_" + tag), every=500,
                          fingerprint=state_fingerprint(st, "stoprule|%s|%g" % (case, cell)))
    t, v_in, step0 = 0.0, 0.0, 0
    hist, snap_t, snap_s = [], [], []
    got = keeper.restore(st)
    if got:
        step0 = int(got["step"])
        t, v_in = float(got["t"]), float(got["v_in"])
        hist = [list(r) for r in np.atleast_2d(got["hist"])]
        snap_t = list(np.atleast_1d(got["snap_t"]))
        snap_s = [np.asarray(r) for r in np.atleast_2d(got["snap_s"])] if len(snap_t) else []
        print("  [%s] resuming at step %d, t = %.1f s" % (tag, step0, t))
    else:
        print("  [%s] n_ref %.4f, Q_ref %.4e m3/m/s, %d cells, %d source cells, reach %d"
              % (tag, n_ref, q_ref, mx.size, len(src), int(reach.sum())))
    sys.stdout.flush()
    reason = "max_steps"
    for step in range(step0, MAX_STEPS):
        dt = float(solve_transport_step(st, dt_cap=G.DT_CAP))
        t += dt
        dq = np.asarray(st["last_div_q"], dtype=float)
        q = float(np.sum(dq[hm] * vols[hm]))
        v_in += q * dt
        s = np.array(st["saturation"].value, copy=True)
        s[hm] = 1.0
        st["saturation"].setValue(s)
        v_store = float(np.sum(phi * (s - s0) * vols))
        count = int(np.sum((s >= 0.5) & ~hm))
        hist.append([t, dt, q, v_in, v_store, count])
        k = len(snap_t)
        if k < len(targets) and t >= targets[k]:
            while k < len(targets) and t >= targets[k]:
                k += 1
            snap_t.append(t)
            snap_s.append(s.astype(np.float32))
        if step % 2000 == 0:
            print("    step %6d  t=%9.1f  dt=%.3g  Q/Qref=%.3e  V=%.2f  cells=%d"
                  % (step, t, dt, q / q_ref, v_in, count))
            sys.stdout.flush()
        if t >= T_END:
            reason = "t_end"
            break
        if t > T_MIN and q / q_ref < Q_FLOOR:
            reason = "q_floor"
            break
        keeper.maybe_save(st, step=step + 1, t=t, v_in=v_in, hist=np.asarray(hist),
                          snap_t=np.asarray(snap_t), snap_s=np.asarray(snap_s))
    keeper.finish()
    np.savez(os.path.join(OUT, "series_%s.npz" % tag), hist=np.asarray(hist), snap_t=np.asarray(snap_t),
             snap_s=np.asarray(snap_s), x=mx, y=my, src=src, reach=reach, n_ref=n_ref, q_ref=q_ref,
             cell=cell, depth=depth, case=case, reason=reason)
    print("  [%s] %s after %d steps, t = %.1f s, V_in = %.3f" % (tag, reason, len(hist), t, v_in))
    return 0


# --- evaluation ---------------------------------------------------------------

BETAS = (0.05, 0.1, 0.2)
FIXED = (100.0, 300.0, 1000.0)
EPS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
CELLS = (2.5, 1.25)


def load(case, cell):
    p = os.path.join(OUT, "series_%s_%g.npz" % (case, cell))
    return np.load(p) if os.path.exists(p) else None


def old_rule(count, patience=60):
    best, run_ = -1, 0
    for i, v in enumerate(count):
        if v > best:
            best, run_ = v, 0
        else:
            run_ += 1
        if run_ >= patience:
            return i
    return None


def stop_index(t, v, q_ref, window, eps):
    """First step where (V(t) - V(t - T_w)) / T_w < eps * Q_ref. window(t) gives T_w."""
    tw = np.array([window(x) for x in t]) if not isinstance(window(t[0]), np.ndarray) else window(t)
    valid = (t - tw) >= t[0]
    v0 = np.interp(t - tw, t, v)
    cond = valid & ((v - v0) / tw < eps * q_ref)
    return int(np.argmax(cond)) if np.any(cond) else None


def field_metrics(d, ts):
    """Metrics from the last snapshot at or before ts."""
    snap_t = d["snap_t"]
    j = int(np.searchsorted(snap_t, ts, side="right")) - 1
    j = max(j, 0)
    s = d["snap_s"][j].astype(float)
    mx, my = d["x"], d["y"]
    src = np.zeros(mx.size, dtype=bool)
    src[d["src"]] = True
    filled = (s >= 0.5) & ~src
    reach = d["reach"] & ~src
    end, dd, nrm = G.hole_geometry()
    perp = (mx - G.MOUTH[0]) * nrm[0] + (my - G.MOUTH[1]) * nrm[1]
    along = (mx - G.MOUTH[0]) * dd[0] + (my - G.MOUTH[1]) * dd[1]
    out = {"t_snap": float(snap_t[j]), "cover": float(np.sum(filled & reach)) / max(int(reach.sum()), 1)}
    if np.any(filled):
        out["updown"] = float(-perp[filled].min() / perp[filled].max()) if perp[filled].max() > 0 else float("nan")
        near = filled & (np.abs(perp) < float(d["cell"]))
        out["x_f"] = float(along[near].max() - float(d["depth"])) if np.any(near) else float("nan")
    else:
        out["updown"], out["x_f"] = float("nan"), float("nan")
    return out


def rule_label(kind, par, eps):
    return ("beta%.2f" % par if kind == "beta" else "Tw%gs" % par) + "_eps%g" % eps


def evaluate(argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    data = dict(((c, cell), load(c, cell)) for c in CASES for cell in CELLS)
    lines = ["Stop-rule diagnostic: evaluation (records only)", ""]

    # --- per-run facts: Q_ref, old rule, noise, asymptotic slope
    lines.append("%-4s %5s %8s %10s %10s | %6s %9s %11s %11s | %8s %8s | %7s | %s" % (
        "case", "cell", "n_ref", "Qref(i)", "Qref(ii)", "steps", "t_end", "old60 step", "old60 t",
        "Q/Qi", "Qbar/Qi", "slope", "noise med/p99 |dQ/Q|"))
    facts = {}
    for (c, cell), d in sorted(data.items()):
        if d is None:
            continue
        h = d["hist"]
        t, q, v, cnt = h[:, 0], h[:, 2], h[:, 3], h[:, 5]
        qi = float(d["q_ref"])
        late = t > 0.01 * t[-1]
        qii = float(q[late].max())
        i60 = old_rule(cnt)
        if i60 is not None:
            j0 = max(i60 - 60, 0)
            qbar60 = (v[i60] - v[j0]) / max(t[i60] - t[j0], 1e-12)
        sel = (t > t[-1] / 30.0) & (q > 0)
        slope = np.polyfit(np.log(t[sel]), np.log(q[sel]), 1)[0] if np.sum(sel) > 10 else float("nan")
        rel = np.abs(np.diff(q[t > T_MIN])) / np.maximum(np.abs(q[t > T_MIN][:-1]), 1e-30)
        facts[(c, cell)] = {"qi": qi, "qii": qii, "i60": i60, "slope": slope}
        lines.append("%-4s %5.2f %8.4f %10.3e %10.3e | %6d %9.0f %11s %11s | %8s %8s | %7.3f | %.2e / %.2e" % (
            c, cell, float(d["n_ref"]), qi, qii, len(t), t[-1],
            "-" if i60 is None else "%d" % (i60 + 1), "-" if i60 is None else "%.0f s" % t[i60],
            "-" if i60 is None else "%.2e" % (q[i60] / qi), "-" if i60 is None else "%.2e" % (qbar60 / qi),
            slope, float(np.median(rel)) if rel.size else float("nan"),
            float(np.percentile(rel, 99)) if rel.size else float("nan")))
    lines.append("")

    # --- replay every rule
    rules = [("beta", b) for b in BETAS] + [("fixed", w) for w in FIXED]
    results = {}
    for ref in ("i", "ii"):
        for kind, par in rules:
            win = (lambda tt, b=par: b * tt) if kind == "beta" else (lambda tt, w=par: w)
            for eps in EPS:
                for (c, cell), d in data.items():
                    if d is None:
                        continue
                    h = d["hist"]
                    qr = facts[(c, cell)]["qi" if ref == "i" else "qii"]
                    i = stop_index(h[:, 0], h[:, 3], qr, win, eps)
                    if i is None:
                        results[(ref, kind, par, eps, c, cell)] = None
                        continue
                    fm = field_metrics(d, h[i, 0])
                    kind_, n, depth = CASES[c]
                    lm = l_max(n) if n is not None else l_max(float(d["n_ref"]))
                    results[(ref, kind, par, eps, c, cell)] = {
                        "t": float(h[i, 0]), "step": i + 1, "v": float(h[i, 3]), "cells": int(h[i, 5]),
                        "cover": fm["cover"], "updown": fm["updown"], "x_f": fm["x_f"],
                        "x_f_L": fm["x_f"] / lm, "t_snap": fm["t_snap"]}

    # --- grid consistency and acceptance
    lines.append("All combinations: per case, 2.5 m | 1.25 m (t_s, V_in, cover, up/down or x_f) and the checks")
    accepted = []
    for ref in ("i", "ii"):
        for kind, par in rules:
            for eps in EPS:
                ok_all, parts = True, []
                for c in sorted(CASES):
                    a = results.get((ref, kind, par, eps, c, 2.5))
                    b = results.get((ref, kind, par, eps, c, 1.25))
                    if a is None or b is None:
                        ok_all = False
                        parts.append("%s: no stop" % c)
                        continue
                    dv = abs(b["v"] - a["v"]) / max(a["v"], 1e-12)
                    ok = dv <= 0.03
                    if c in ("D1", "D2"):
                        dq = abs(b["updown"] - a["updown"])
                        ok = ok and dq <= 0.05
                        extra = "ud %.3f|%.3f" % (a["updown"], b["updown"])
                    else:
                        dq = abs(b["x_f"] - a["x_f"])
                        ok = ok and dq <= 2.5
                        extra = "xf %.1f|%.1f" % (a["x_f"], b["x_f"])
                    ok_all = ok_all and ok
                    parts.append("%s t %.0f|%.0f V %.1f|%.1f (%+.1f%%) %s %s" % (
                        c, a["t"], b["t"], a["v"], b["v"], 100 * (b["v"] - a["v"]) / max(a["v"], 1e-12),
                        extra, "ok" if ok else "X"))
                lab = "Qref(%s) %s" % (ref, rule_label(kind, par, eps))
                lines.append("%-32s %s | %s" % (lab, "ACCEPT" if ok_all else "      ", "; ".join(parts)))
                if ok_all:
                    accepted.append((ref, kind, par, eps))
    lines.append("")
    lines.append("accepted: " + (", ".join("Qref(%s) %s" % (r, rule_label(k, p, e)) for r, k, p, e in accepted)
                                 if accepted else "none"))
    lines.append("")

    # --- x_f / L_max vs eps (D3-D5), Qref(i), beta 0.1
    lines.append("x_f / L_max against eps (Qref(i), T_w = 0.1 t):")
    for c in ("D3", "D4", "D5"):
        for cell in CELLS:
            row = []
            for eps in EPS:
                r = results.get(("i", "beta", 0.1, eps, c, cell))
                row.append("-" if r is None else "%.2f" % r["x_f_L"])
            lines.append("  %s %5.2f m: %s" % (c, cell, "  ".join("eps %g: %s" % (e, x) for e, x in zip(EPS, row))))
    lines.append("")
    text = "\n".join(lines)
    open(os.path.join(OUT, "evaluation.txt"), "w").write(text + "\n")
    with open(os.path.join(OUT, "evaluation.json"), "w") as fh:
        json.dump([dict(zip(("ref", "kind", "par", "eps", "case", "cell"), k), **v) for k, v in results.items()
                   if v is not None], fh, indent=0)

    # --- figures
    cols = {2.5: "#2a78d6", 1.25: "#eb6834"}
    for what in ("Q", "V"):
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for ax, c in zip(axes, sorted(CASES)):
            for cell in CELLS:
                d = data.get((c, cell))
                if d is None:
                    continue
                h = d["hist"]
                if what == "Q":
                    ax.loglog(h[:, 0], np.maximum(h[:, 2], 1e-30) / float(d["q_ref"]), color=cols[cell], lw=0.8,
                              label=u"%.2f m" % cell)
                    i60 = facts[(c, cell)]["i60"]
                    if i60 is not None:
                        ax.plot([h[i60, 0]], [h[i60, 2] / float(d["q_ref"])], "o", color=cols[cell], mfc="none", ms=7)
                else:
                    ax.semilogx(h[:, 0], h[:, 3], color=cols[cell], lw=1.0, label=u"%.2f m" % cell)
            if what == "Q":
                for e in (0.01, 0.001):
                    ax.axhline(e, color="#999999", lw=0.6, ls="--")
                ax.set_ylabel(u"Q / Q_ref", fontsize=8)
            else:
                ax.set_ylabel(u"累计注入量 (m$^3$/m)", fontsize=8)
            ax.set_title(c, fontsize=9)
            ax.set_xlabel(u"模型时间 (s)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7, loc="best", numpoints=1)
        fig.suptitle(u"注入速率（圆圈：原 60 步判停时刻）" if what == "Q" else u"累计注入量", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(OUT, "fig_%s.png" % what), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ref in zip(axes, ("i", "ii")):
        m = np.zeros((len(rules), len(EPS)))
        for a, (kind, par) in enumerate(rules):
            for b, eps in enumerate(EPS):
                m[a, b] = 1.0 if (ref, kind, par, eps) in accepted else 0.0
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1.3, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels([("T_w = %.2f t" % p) if k == "beta" else ("T_w = %g s" % p) for k, p in rules], fontsize=7)
        ax.set_xticks(range(len(EPS)))
        ax.set_xticklabels(["%g" % e for e in EPS], fontsize=7)
        ax.set_xlabel(u"ε", fontsize=8)
        ax.set_title(u"Q_ref (%s)：满足网格一致性验收的组合（深色）" % ref, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_accept.png"), dpi=150)
    plt.close(fig)
    print(text)
    return 0


def main(argv):
    if argv and argv[0] == "run":
        return run(argv[1], float(argv[2]))
    if argv and argv[0] == "evaluate":
        return evaluate(argv[1:])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
