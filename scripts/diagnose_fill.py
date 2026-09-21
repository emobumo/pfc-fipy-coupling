# -*- coding: utf-8 -*-
"""
Run an inclined-hole case and read it with the fill diagnostics.

    powershell -File scripts\run_local.ps1 scripts\diagnose_fill.py [--case gradient|grouting] [--kind gradient|uniform|both] [--plot]

Marches the case exactly as the case file does (same build, same stall rule)
while recording the injection rate, then reports, for each run:

  - reachable domain (Dijkstra path integral with gravity) vs what filled
  - unfilled cells classified: unreachable / front shortfall / bypass void
  - fill ratios over the reachable domain at S_c = 0.3 / 0.5 / 0.7
  - Q(t) decay: is this a stalled front or a plateau?
  - volume-distance consistency: did the grout travel further than uniform
    filling of the injected volume would allow?

Outputs to outputs/fill_diagnostics/<case>_<kind>/: category map CSV, Q(t)
CSV, summary text, and (with --plot) a two-panel figure. Case files are not
modified; their regression anchors are untouched.
"""
import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.fill_diagnostics import (
    CATEGORY_NAMES, BYPASS_VOID, FRONT_SHORTFALL, UNREACHABLE, OVERSHOOT,
    reachable_domain, classify_unfilled, fill_ratio_ladder,
    record_injection_rate, injection_rate_diagnostics,
    volume_distance_consistency, observed_spread,
)


def load_case(name):
    path = os.path.join(REPO, "cases", "inclined_hole_%s.py" % name)
    return imp.load_source("case_" + name, path)


def march_recording(case, state, mx, my, hcells, vols, label):
    """The case's own march loop, plus Q(t) recording."""
    phi = np.clip(np.asarray(state["porosity"].value, dtype=float), 1e-6, 1.0)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True
    s0 = np.array(state["saturation"].value, copy=True)
    history = []
    t, v_in_src, best, stall, count, v_store = 0.0, 0.0, -1, 0, 0, 0.0
    print("  [%s] marching ..." % label)
    sys.stdout.flush()
    for step in range(case.MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=case.DT_CAP)
        t += float(dt)
        div_q = np.asarray(state["last_div_q"], dtype=float)
        v_in_src += float(np.sum(div_q[hmask] * vols[hmask])) * float(dt)
        record_injection_rate(state, hcells, dt, t, history)
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)
        s_arr = np.asarray(state["saturation"].value, dtype=float)
        v_store = float(np.sum(phi * (s_arr - s0) * vols))
        count = int(np.sum((s_arr >= 0.5) & np.logical_not(hmask)))
        if count > best:
            best, stall = count, 0
        else:
            stall += 1
        if stall >= case.STALL_PATIENCE:
            break
    print("    -> stop at step %d, t=%.1f s, filled=%d, Vstore=%.3f, Vin=%.3f"
          % (step, t, count, v_store, v_in_src))
    sys.stdout.flush()
    return t, v_in_src, v_store, history


def diagnose(case, case_name, kind, plot):
    label = "%s / %s" % (case_name, kind)
    if case_name == "grouting":
        out = case.build_case()
        state, mx, my, nx, ny, phi, hcells, k = (
            out[0], out[1], out[2], out[3], out[4], out[5], out[7], out[8])
    else:
        state, mx, my, nx, ny, phi, hcells, k = case.build_case(kind)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    t, v_in, v_store, history = march_recording(case, state, mx, my, hcells, vols, label)

    reach = reachable_domain(state, hcells, p0=case.P0, use_gravity=True)
    ladder = fill_ratio_ladder(state, hcells, reach["reachable"],
                               thresholds=(0.3, 0.5, 0.7))
    cls = dict(ladder)[0.5]
    q = injection_rate_diagnostics(history, tail_fraction=0.2, stall_ratio=0.05)
    r_obs = observed_spread(state, hcells, s_c=0.5)
    vd = volume_distance_consistency(state, hcells, v_in, r_obs=r_obs)

    lines = []
    lines.append("=== %s ===" % label)
    lines.append("stop t=%.1f s  V_in=%.3f  V_store=%.3f m^3/m" % (t, v_in, v_store))
    lines.append("")
    lines.append("reachable domain : %d of %d cells (%.1f m^2), solver=%s, negative edges=%d"
                 % (int(np.sum(reach["reachable"])), reach["reachable"].size,
                    float(np.sum(vols[reach["reachable"]])), reach["solver"],
                    reach["negative_edges"]))
    lines.append("observed spread  : %.2f m from nearest source cell (S>=0.5)" % r_obs)
    lines.append("")
    lines.append("fill over reachable domain:")
    lines.append("  %-5s %8s %12s %16s %12s %8s" % (
        "S_c", "filled", "unreachable", "front_shortfall", "bypass_void", "#voids"))
    for s_c, res in ladder:
        tb = res["over_reachable"]
        lines.append("  %-5.1f %7.1f%% %11.1f%% %15.1f%% %11.1f%% %8d" % (
            s_c, 100 * tb["filled_fraction"], 100 * tb["unreachable_fraction"],
            100 * tb["front_shortfall_fraction"], 100 * tb["bypass_void_fraction"],
            res["n_bypass_voids"]))
    lines.append("")
    lines.append("reach vs fill (S_c=0.5), whole domain:")
    td = cls["over_domain"]
    n_over = int(np.sum(cls["category"] == OVERSHOOT))
    n_short = int(np.sum(cls["category"] == FRONT_SHORTFALL))
    lines.append("  filled inside reach %d cells | overshoot (filled beyond reach) %d cells, %.1f m^2"
                 " | front shortfall %d cells"
                 % (int(np.sum((cls["category"] == 0))), n_over, td["overshoot"], n_short))
    if n_over:
        over = cls["category"] == OVERSHOOT
        lines.append("  overshoot path cost / p0: min %.3f  max %.3f  (1.0 = budget exhausted)"
                     % (float(np.min(reach["cost"][over])) / case.P0,
                        float(np.max(reach["cost"][over])) / case.P0))
    lines.append("")
    lines.append("injection rate Q(t):")
    lines.append("  peak %.4g at t=%.1f s; final %.4g; tail mean/peak = %.3g; "
                 "tail d(lnQ)/dt = %.3g 1/s  ->  %s"
                 % (q["q_peak"], q["t_peak"], q["q_final"], q["decay_ratio"],
                    q["tail_slope"], "STALLED" if q["stalled"] else "NOT STALLED"))
    lines.append("")
    lines.append("volume-distance consistency:")
    lines.append("  V_in %.3f m^3/m would fill uniformly to r_equiv = %.2f m;"
                 % (vd["v_in"], vd["r_equiv"]))
    lines.append("  observed spread %.2f m needed %.3f m^3/m  ->  volume ratio %.2f, reach ratio %.2f"
                 % (vd["r_obs"], vd["v_required_at_r_obs"], vd["volume_ratio"], vd["reach_ratio"]))
    text = "\n".join(lines)
    print(text)
    sys.stdout.flush()

    out_dir = os.path.join(REPO, "outputs", "fill_diagnostics", "%s_%s" % (case_name, kind))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    s_arr = np.asarray(state["saturation"].value, dtype=float)
    np.savetxt(os.path.join(out_dir, "cells.csv"),
               np.column_stack([mx, my, phi, s_arr, reach["cost"],
                                reach["reachable"].astype(int), cls["category"]]),
               delimiter=",", comments="",
               header="x,y,phi,S,path_cost_Pa,reachable,category(0 filled,1 unreachable,2 front_shortfall,3 bypass_void,4 overshoot)")
    np.savetxt(os.path.join(out_dir, "injection_rate.csv"),
               np.array(history), delimiter=",", comments="", header="t_s,Q_m3_per_m_s")

    if plot:
        _figure(case, state, mx, my, nx, ny, hcells, reach, cls, history, q, label, out_dir)
    return {"label": label, "cls": cls, "q": q, "vd": vd, "reach": reach}


def _figure(case, state, mx, my, nx, ny, hcells, reach, cls, history, q, label, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cat = case.to_grid(cls["category"].astype(float), mx, my, nx, ny)
    rch = case.to_grid(reach["reachable"].astype(float), mx, my, nx, ny)
    xc, yc = case.grid_axes(nx, ny)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    cmap = ListedColormap(["#4a90d9", "#d9d9d9", "#f5c56b", "#d0403f", "#7b4fa3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    im = ax1.pcolormesh(np.linspace(case.X_MIN, case.X_MAX, nx + 1),
                        np.linspace(case.Y_MIN, case.Y_MAX, ny + 1),
                        cat, cmap=cmap, norm=norm)
    ax1.contour(xc, yc, rch, levels=[0.5], colors="k", linewidths=1.6)
    case.draw_hole(ax1, hcells, mx, my)
    cb = fig.colorbar(im, ax=ax1, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["filled", "unreachable", "front shortfall", "bypass void", "overshoot"])
    ax1.set_aspect("equal")
    ax1.set_title("%s\ncategories (S_c=0.5); black = reachable boundary" % label)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")

    hist = np.array(history)
    ax2.semilogy(hist[:, 0], np.maximum(hist[:, 1], 1e-30), "k-")
    ax2.axhline(q["q_peak"] * 0.05, color="r", ls="--", lw=1,
                label="5% of peak (stall line)")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("Q [m^3/m/s]")
    ax2.set_title("injection rate: tail/peak=%.3g, %s"
                  % (q["decay_ratio"], "stalled" if q["stalled"] else "NOT stalled"))
    ax2.legend(loc="upper right")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fill_diagnostics.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote %s" % path)


def main(argv):
    case_name = "gradient"
    kind = "both"
    plot = False
    i = 0
    while i < len(argv):
        if argv[i] == "--case":
            case_name = argv[i + 1]; i += 2
        elif argv[i] == "--kind":
            kind = argv[i + 1]; i += 2
        elif argv[i] == "--plot":
            plot = True; i += 1
        else:
            i += 1
    case = load_case(case_name)
    if case_name == "grouting":
        kinds = ["real"]
    else:
        kinds = ["gradient", "uniform"] if kind == "both" else [kind]
    for k in kinds:
        diagnose(case, case_name, k, plot)
        print("")


if __name__ == "__main__":
    main(sys.argv[1:])
