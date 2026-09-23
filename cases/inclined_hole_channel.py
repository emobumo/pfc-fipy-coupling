# -*- coding: utf-8 -*-
"""
APPLICATION case: runaway and bypass voids from through-going structure.

Same hole, parameters, mesh and stall rule as inclined_hole_gradient.py (the
line-source engineering baseline), with the porosity field replaced by
fields that contain a connected high-porosity feature:

    band      a vertical through-going void band at x = BAND_X, one cell
              wide, full height -- the contact between backfill and host
              rock, or a persistent gap in the fill. Swept over the band's
              porosity (planned case 4, structure-strength sweep).
    random    anisotropic correlated random fields, one per seed -- streaks
              without prescribing where (planned case 6, realisation family).

STOP RULE: stall OR volume quota
--------------------------------
Run to full stall under constant pressure and the Bingham model fills every
reachable cell, so a sealed domain can never show a bypass void that way.
In the field the injection ends when the hole stops taking grout OR when
the budgeted volume is spent -- and runaway is precisely the second ending
arriving before the first. So every structured run gets the same volume the
uniform baseline needed to stall, V_QUOTA = V_in(base at stall), and stops
at min(stall, quota). What that volume filled, and what it left, is the
result. Runaway then shows as: Q(t) not decayed at the quota, spread far
beyond what the volume could fill uniformly, and the baseline's own
footprint (the design target) left short.

Diagnostics come from src/analysis/fill_diagnostics.py. The design target
Omega_design is the baseline's stalled footprint: "the region this hole and
this volume were expected to fill".

    powershell -File scripts\run_local.ps1 cases\inclined_hole_channel.py [--set base|band|position|random|all] [--plot]

Outputs under outputs/inclined_hole_channel/<run>/ and a comparison table.
"""
import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.structure.porosity_fields import uniform, where_band, random_correlated
from src.analysis.fill_diagnostics import (
    FILLED, UNREACHABLE, FRONT_SHORTFALL, BYPASS_VOID, OVERSHOOT,
    reachable_domain, classify_unfilled, record_injection_rate,
    injection_rate_diagnostics, volume_distance_consistency, observed_spread,
)

G = imp.load_source("inclined_hole_gradient",
                    os.path.join(REPO, "cases", "inclined_hole_gradient.py"))

# --- structured fields ------------------------------------------------------
PHI_BASE = 0.18            # the uniform control of the baseline case
# Band sweeps. A vertical through-going band one cell wide, full height.
#   porosity sweep : band at BAND_X_TOE, past the toe, over BAND_PHIS
#   position sweep : band at phi=BAND_PHI_POS, at the collar (the hole
#                    enters the fill THROUGH the contact -- the reference
#                    project traced its runaway to that interface), crossing
#                    the hole mid-length, and past the toe
# Cell-centre columns, placed relative to the hole so they follow the domain:
# the collar column is the first cell inside the left wall, "midhole" is 5 m
# in, and "toe" is the second cell-centre column past the toe x.
_TOE_X = G.MOUTH[0] + G.HOLE_LEN * math.cos(math.radians(G.DIP_DEG))
_COL = lambda x: G.X_MIN + (math.floor((x - G.X_MIN) / G.CELL) + 0.5) * G.CELL
BAND_X_COLLAR = _COL(G.MOUTH[0] + 0.5 * G.CELL)
BAND_X_MID = _COL(G.MOUTH[0] + 2.5 * G.CELL)
BAND_X_TOE = _COL(_TOE_X + 2.0 * G.CELL)
BAND_HALF_WIDTH = 1.25     # one 2.5 m cell
BAND_PHIS = (0.30, 0.45, 0.60)
BAND_PHI_POS = 0.45
BAND_POSITIONS = (("collar", BAND_X_COLLAR), ("midhole", BAND_X_MID), ("toe", BAND_X_TOE))
RANDOM_SEEDS = (1, 2, 3)
RANDOM_STD = 0.06
RANDOM_CORR = (7.5, 2.5)   # (x, y) correlation lengths [m]: horizontal streaks

OUT_ROOT = os.path.join(REPO, "outputs", "inclined_hole_channel")


def build_from_phi(phi):
    """The baseline's build_case with its porosity swapped for `phi`.

    Everything else -- mesh, hole, parameters, Picard budget, initial fields
    -- is the baseline's own code, so the comparison is like for like.
    """
    original = G.phi_field
    G.phi_field = lambda kind, my: np.asarray(phi, dtype=float)
    try:
        return G.build_case("custom")
    finally:
        G.phi_field = original


def field(kind, mx, my, seed=None, phi_band=None, band_x=BAND_X_TOE):
    base = uniform(mx, PHI_BASE)
    if kind == "base":
        return base
    if kind == "band":
        mask = where_band(mx, my, (float(band_x), 0.0), (0.0, 1.0), BAND_HALF_WIDTH)
        return np.where(mask, float(phi_band), base)
    if kind == "random":
        return random_correlated(mx, my, PHI_BASE, RANDOM_STD,
                                 RANDOM_CORR[0], RANDOM_CORR[1], seed, cell=G.CELL)
    raise ValueError(kind)


def march(state, hcells, vols, label, v_quota=None):
    """The baseline's march loop plus Q(t) recording and a volume quota."""
    phi = np.clip(np.asarray(state["porosity"].value, dtype=float), 1e-6, 1.0)
    hmask = np.zeros(vols.size, dtype=bool)
    hmask[hcells] = True
    s0 = np.array(state["saturation"].value, copy=True)
    history = []
    t, v_in, best, stall, count, v_store = 0.0, 0.0, -1, 0, 0, 0.0
    reason = "max_steps"
    print("  [%s] marching%s ..." % (label, "" if v_quota is None else " (quota %.2f)" % v_quota))
    sys.stdout.flush()
    for step in range(G.MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=G.DT_CAP)
        t += float(dt)
        div_q = np.asarray(state["last_div_q"], dtype=float)
        v_in += float(np.sum(div_q[hmask] * vols[hmask])) * float(dt)
        record_injection_rate(state, hcells, dt, t, history)
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)
        v_store = float(np.sum(phi * (s - s0) * vols))
        count = int(np.sum((s >= 0.5) & np.logical_not(hmask)))
        if count > best:
            best, stall = count, 0
        else:
            stall += 1
        if stall >= G.STALL_PATIENCE:
            reason = "stall"
            break
        if v_quota is not None and v_in >= v_quota:
            reason = "quota"
            break
    print("    -> %s at step %d, t=%.1f s, filled=%d, Vin=%.3f, Vstore=%.3f"
          % (reason, step, t, count, v_in, v_store))
    sys.stdout.flush()
    return {"t": t, "v_in": v_in, "v_store": v_store, "history": history,
            "reason": reason, "steps": step}


def run(label, phi, v_quota=None, target_mask=None, plot=False):
    state, mx, my, nx, ny, phi_used, hcells, k = build_from_phi(phi)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    m = march(state, hcells, vols, label, v_quota=v_quota)

    reach = reachable_domain(state, hcells, p0=G.P0, use_gravity=True)
    cls = classify_unfilled(state, hcells, reach["reachable"], s_c=0.5,
                            target_mask=target_mask)
    q = injection_rate_diagnostics(m["history"], tail_fraction=0.2, stall_ratio=0.05)
    r_obs = observed_spread(state, hcells, s_c=0.5)
    vd = volume_distance_consistency(state, hcells, m["v_in"], r_obs=r_obs)
    fm = G.front_metrics(state, mx, my, hcells)
    s = np.asarray(state["saturation"].value, dtype=float)
    filled_mask = s >= 0.5

    cat = cls["category"]
    row = {
        "label": label, "reason": m["reason"], "t": m["t"], "v_in": m["v_in"],
        "filled_cells": int(np.sum(filled_mask & ~np.in1d(np.arange(s.size), hcells))),
        "reach_cells": int(np.sum(reach["reachable"])),
        "overshoot": int(np.sum(cat == OVERSHOOT)),
        "front_shortfall": int(np.sum(cat == FRONT_SHORTFALL)),
        "bypass_void": int(np.sum(cat == BYPASS_VOID)),
        "n_voids": cls["n_bypass_voids"],
        "q_decay": q["decay_ratio"], "stalled": q["stalled"],
        "r_obs": r_obs, "r_equiv": vd["r_equiv"], "reach_ratio": vd["reach_ratio"],
        "up": fm["perp_up"] if fm else 0.0, "down": fm["perp_down"] if fm else 0.0,
        "touches_top": bool(np.any(filled_mask & (my > G.Y_MAX - G.CELL))),
        "touches_bottom": bool(np.any(filled_mask & (my < G.Y_MIN + G.CELL))),
    }
    if target_mask is not None:
        tt = cls["over_target"]
        row["target_filled_fraction"] = tt["filled_fraction"]
        row["target_shortfall_fraction"] = tt["front_shortfall_fraction"] + tt["bypass_void_fraction"]
        row["design_shortfall_fraction"] = cls["design_shortfall_fraction"]

    out = os.path.join(OUT_ROOT, label)
    if not os.path.isdir(out):
        os.makedirs(out)
    np.savetxt(os.path.join(out, "cells.csv"),
               np.column_stack([mx, my, phi_used, s, reach["cost"],
                                reach["reachable"].astype(int), cat]),
               delimiter=",", comments="",
               header="x,y,phi,S,path_cost_Pa,reachable,category(0 filled,1 unreachable,2 front_shortfall,3 bypass_void,4 overshoot)")
    np.savetxt(os.path.join(out, "injection_rate.csv"), np.array(m["history"]),
               delimiter=",", comments="", header="t_s,Q_m3_per_m_s")
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        for key in sorted(row):
            fh.write("%s = %s\n" % (key, row[key]))
    if plot:
        _figure(label, state, mx, my, nx, ny, hcells, phi_used, reach, cls, m, q, out)
    row["filled_mask"] = filled_mask
    return row


def _figure(label, state, mx, my, nx, ny, hcells, phi, reach, cls, m, q, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    xe = np.linspace(G.X_MIN, G.X_MAX, nx + 1)
    ye = np.linspace(G.Y_MIN, G.Y_MAX, ny + 1)
    xc, yc = G.grid_axes(nx, ny)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    im = ax.pcolormesh(xe, ye, G.to_grid(phi, mx, my, nx, ny), cmap="YlGnBu", vmin=0.05, vmax=0.60)
    G.draw_hole(ax, hcells, mx, my)
    fig.colorbar(im, ax=ax, label="phi")
    ax.set_aspect("equal"); ax.set_title("%s: porosity" % label)

    ax = axes[1]
    cmap = ListedColormap(["#4a90d9", "#d9d9d9", "#f5c56b", "#d0403f", "#7b4fa3"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    im = ax.pcolormesh(xe, ye, G.to_grid(cls["category"].astype(float), mx, my, nx, ny),
                       cmap=cmap, norm=norm)
    ax.contour(xc, yc, G.to_grid(reach["reachable"].astype(float), mx, my, nx, ny),
               levels=[0.5], colors="k", linewidths=1.6)
    G.draw_hole(ax, hcells, mx, my)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["filled", "unreachable", "front shortfall", "bypass void", "overshoot"])
    ax.set_aspect("equal")
    ax.set_title("stop: %s at t=%.0f s, V=%.1f\nblack = reachable boundary" % (m["reason"], m["t"], m["v_in"]))

    ax = axes[2]
    h = np.array(m["history"])
    ax.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-30), "k-")
    ax.axhline(q["q_peak"] * 0.05, color="r", ls="--", lw=1, label="5% of peak")
    ax.set_xlabel("t [s]"); ax.set_ylabel("Q [m^3/m/s]")
    ax.set_title("Q(t): tail/peak=%.3g -> %s" % (q["decay_ratio"], "stalled" if q["stalled"] else "NOT stalled"))
    ax.legend(loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "channel_diagnostics.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("    wrote %s" % path)


def main(argv):
    which = "all"
    plot = False
    quota = None
    i = 0
    while i < len(argv):
        if argv[i] == "--set":
            which = argv[i + 1]; i += 2
        elif argv[i] == "--plot":
            plot = True; i += 1
        elif argv[i] == "--quota":
            quota = float(argv[i + 1]); i += 2
        else:
            i += 1

    # Cell centres, so fields can be built before any case is.
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    # Baseline first: it defines the quota and the design target.

    rows = []
    base = run("base_uniform_%.2f" % PHI_BASE, field("base", mx, my), v_quota=None, plot=plot)
    rows.append(base)
    quota = base["v_in"] if quota is None else quota
    target = base["filled_mask"]
    print("  quota for structured runs = V_in(base at stall) = %.3f m^3/m" % quota)
    print("  design target = baseline stalled footprint, %d cells" % int(np.sum(target)))
    sys.stdout.flush()

    if which in ("band", "all"):
        for pb in BAND_PHIS:
            rows.append(run("band_phi%.2f" % pb, field("band", mx, my, phi_band=pb),
                            v_quota=quota, target_mask=target, plot=plot))
    if which in ("position", "all"):
        for name, bx in BAND_POSITIONS:
            rows.append(run("band_%s_x%+.2f" % (name, bx),
                            field("band", mx, my, phi_band=BAND_PHI_POS, band_x=bx),
                            v_quota=quota, target_mask=target, plot=plot))
    if which in ("random", "all"):
        for sd in RANDOM_SEEDS:
            rows.append(run("random_seed%d" % sd, field("random", mx, my, seed=sd),
                            v_quota=quota, target_mask=target, plot=plot))

    hdr = "%-20s %6s %7s %7s %6s %5s %5s %5s %5s %6s %6s %6s %6s %6s %6s %6s" % (
        "run", "stop", "t[s]", "V_in", "fill", "reach", "over", "short", "void",
        "Qtail", "r_obs", "r_eq", "ratio", "tgt%", "dsf%", "wall")
    lines = ["", hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("%-20s %6s %7.0f %7.2f %6d %5d %5d %5d %5d %6.3f %6.2f %6.2f %6.2f %6s %6s %6s" % (
            r["label"], r["reason"], r["t"], r["v_in"], r["filled_cells"], r["reach_cells"],
            r["overshoot"], r["front_shortfall"], r["bypass_void"], r["q_decay"],
            r["r_obs"], r["r_equiv"], r["reach_ratio"],
            ("%.0f" % (100 * r["target_filled_fraction"])) if "target_filled_fraction" in r else "-",
            ("%.0f" % (100 * r["design_shortfall_fraction"])) if "design_shortfall_fraction" in r else "-",
            ("T" if r["touches_top"] else "-") + ("B" if r["touches_bottom"] else "-")))
    lines.append("")
    lines.append("fill = filled cells (excl. hole); reach = reachable cells; over/short/void = overshoot /")
    lines.append("front-shortfall / bypass-void cells; Qtail = tail-mean Q / peak Q (stalled if < 0.05);")
    lines.append("r_obs = observed spread, r_eq = uniform-fill equivalent, ratio = r_obs/r_eq;")
    lines.append("tgt% = share of the baseline footprint filled; dsf% = share of it UNREACHABLE in this")
    lines.append("field (design shortfall, not a fill defect); wall = fill touches Top/Bottom edge.")
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT_ROOT):
        os.makedirs(OUT_ROOT)
    with open(os.path.join(OUT_ROOT, "comparison.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
