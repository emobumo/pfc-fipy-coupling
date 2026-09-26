# -*- coding: utf-8 -*-
"""
4.6 supplement, P1: the collar channel (collar_v06), three columns.

    powershell -File scripts\run_local.ps1 scripts\stage_supplement.py scan        (reach only)
    powershell -File scripts\run_local.ps1 scripts\stage_supplement.py p1 --h 2.5,7.5

    (1) ideal full hole, one pass      must reproduce band_collar_x-28.75
    (2) staged 7 -> 10 -> 17 and 8.5 -> 17 (run_sequence, same budget rule as
        the existing staged runs: the uniform baseline's stalled volume)
    (3) pre-plugged, one pass: before grouting, the band cells within h above
        and below where the hole crosses the band are set to cement (the stage
        closure floor, as the bottom seal is treated). The hole cells are left
        open -- the hole is re-drilled through the plug. An idealized
        "seal first, then grout": the plug is geometry, not a simulated
        double-fluid injection.

Stop rule for (1) and (3): stall or quota 166.15 m3/m, design target = the
uniform baseline's stalled footprint -- exactly as the channel case.
Outputs: outputs/stage_supplement/.
"""
from __future__ import print_function

import imp
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
C, G = S.C, S.G
OUT = os.path.join(REPO, "outputs", "stage_supplement")
QUOTA = 166.15
H_SCAN = (1.25, 2.5, 5.0, 7.5, 10.0)


def setup():
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx = int(round((G.X_MAX - G.X_MIN) / G.CELL))
    ny = int(round((G.Y_MAX - G.Y_MIN) / G.CELL))
    hole = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)
    phi = S.make_field("collar_v06", mx, my)
    band = phi > 0.3
    base = np.loadtxt(os.path.join(REPO, "outputs", "inclined_hole_channel", "base_uniform_0.18", "cells.csv"),
                      delimiter=",", skiprows=1)
    key = dict(((round(u, 6), round(v, 6)), i) for i, (u, v) in enumerate(zip(mx, my)))
    target = np.zeros(mx.size, dtype=bool)
    for row in base:
        if row[3] >= 0.5:
            target[key[(round(row[0], 6), round(row[1], 6))]] = True
    return mx, my, nx, ny, hole, phi, band, target


def plugged(phi, band, hole, my, h):
    hmask = np.zeros(phi.size, dtype=bool)
    hmask[hole] = True
    cross = band & hmask
    y0, y1 = my[cross].min(), my[cross].max()
    plug = band & (my >= y0 - h - 1e-9) & (my <= y1 + h + 1e-9) & np.logical_not(hmask)
    out = phi.copy()
    out[plug] = 1.0e-3                     # the stage-closure floor, as the seal
    return out, plug


def scan():
    mx, my, nx, ny, hole, phi, band, target = setup()
    rows = []
    for h in (0.0,) + H_SCAN:
        field = phi if h == 0.0 else plugged(phi, band, hole, my, h)[0]
        st = C.build_from_phi(field)[0]
        r = reachable_domain(st, hole, p0=G.P0, use_gravity=True)
        reach = r["reachable"]
        bre = band & reach
        rows.append({
            "h": h, "reach": int(reach.sum()), "outside_target": int(np.sum(reach & ~target)),
            "band_reach": int(bre.sum()),
            "band_to_top": bool(np.any(bre & (my > G.Y_MAX - G.CELL))),
            "band_to_bottom": bool(np.any(bre & (my < G.Y_MIN + G.CELL))),
            "plug_cells": 0 if h == 0.0 else int(plugged(phi, band, hole, my, h)[1].sum()),
            "min_band_cost": float(np.min(np.asarray(r["cost"])[band & ~np.in1d(np.arange(mx.size), hole)])) / G.P0,
        })
    lines = ["P1 reach scan (collar_v06, 16-neighbour, gravity on; target = baseline footprint %d cells)"
             % int(target.sum()), "",
             "%6s %6s %6s %14s %10s %8s %8s %14s" % ("h [m]", "plug", "reach", "outside target",
                                                  "band reach", "to top", "to bot", "min band cost/p0")]
    for r in rows:
        lines.append("%6.2f %6d %6d %14d %10d %8s %8s %14.3f" % (
            r["h"], r["plug_cells"], r["reach"], r["outside_target"], r["band_reach"],
            r["band_to_top"], r["band_to_bottom"], r["min_band_cost"]))
    text = "\n".join(lines)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "p1_reach_scan.txt"), "w").write(text + "\n")
    print(text)


def p1(hs, pass2=False):
    mx, my, nx, ny, hole, phi, band, target = setup()
    sfx = "_pass2" if pass2 else ""
    C.OUT_ROOT = os.path.join(OUT, "p1" + sfx)
    S.OUT_ROOT = os.path.join(OUT, "p1" + sfx)
    rows = []
    rows.append(C.run("ideal_full_hole" + sfx, phi, v_quota=QUOTA, target_mask=target))
    for h in hs:
        rows.append(C.run("preplug_h%.2f%s" % (h, sfx), plugged(phi, band, hole, my, h)[0],
                          v_quota=QUOTA, target_mask=target))
    st0 = C.build_from_phi(phi)[0]
    reach_virgin = reachable_domain(st0, hole, p0=G.P0, use_gravity=True)["reachable"]
    staged = []
    for name, depths in S.SEQUENCES:
        staged.append(S.run_sequence("collar_v06/%s%s" % (name, sfx), phi, depths, QUOTA,
                                     reach_virgin, hole, False))
    with open(os.path.join(OUT, "p1_results%s.json" % sfx), "w") as fh:
        # run() rows carry arrays (filled_mask); keep the scalars and lists
        scal = lambda r: dict((k, v) for k, v in r.items() if not isinstance(v, np.ndarray))
        json.dump({"single": [scal(r) for r in rows], "staged": [scal(r) for r in staged]},
                  fh, indent=1, default=float)
    return rows, staged


def _load(path):
    return np.loadtxt(os.path.join(path, "cells.csv"), delimiter=",", skiprows=1)


def table_and_plot(hs, sfx=""):
    """Three-column comparison: grout volume inside / outside the design target
    and the figures. Single passes: cells.csv = x,y,phi,S,cost,reachable,cat;
    staged: x,y,phi0,phi_final,occupied,by_stage,reach_virgin,reach_final,shadow,cat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    mx, my, nx, ny, hole, phi, band, target = setup()
    key = dict(((round(u, 6), round(v, 6)), i) for i, (u, v) in enumerate(zip(mx, my)))
    vol = G.CELL * G.CELL
    root = os.path.join(OUT, "p1" + sfx)
    cols = [(u"① 理想整孔", "ideal_full_hole" + sfx, "single"),
            (u"② 分段 7→10→17 m", os.path.join("collar_v06", "staged_7_10_17" + sfx), "staged"),
            (u"② 分段 8.5→17 m", os.path.join("collar_v06", "staged_8.5_17" + sfx), "staged")]
    cols += [(u"③ 预置封堵 h = %g m" % h, "preplug_h%.2f%s" % (h, sfx), "single") for h in hs]
    rows = []
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 4.2))
    for ax, (title, sub, kind) in zip(axes, cols):
        a = _load(os.path.join(root, sub))
        idx = np.array([key[(round(u, 6), round(v, 6))] for u, v in zip(a[:, 0], a[:, 1])])
        f = np.zeros(mx.size)
        grout = np.zeros(mx.size)
        reach = np.zeros(mx.size, dtype=bool)
        if kind == "single":
            f[idx] = a[:, 3]
            grout[idx] = a[:, 2] * a[:, 3] * vol
            reach[idx] = a[:, 5] > 0.5
        else:
            f[idx] = a[:, 4]
            grout[idx] = (a[:, 2] - a[:, 3]) * vol
            reach[idx] = a[:, 6] > 0.5
        notsrc = np.ones(mx.size, dtype=bool)
        notsrc[hole] = False
        filled = (f >= 0.5) & notsrc
        v_in_t, v_out_t = float(np.sum(grout[target & notsrc])), float(np.sum(grout[~target & notsrc]))
        rows.append((title, float(np.sum(filled & target)) / float(np.sum(target & notsrc)),
                     v_in_t, v_out_t, v_out_t / max(v_in_t + v_out_t, 1e-12),
                     bool(np.any(filled & (my > G.Y_MAX - G.CELL))), bool(np.any(filled & (my < G.Y_MIN + G.CELL)))))
        g = lambda v: np.asarray(v, dtype=float).reshape(ny, nx)
        order = np.lexsort((mx, my))
        ext = [G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX]
        ax.imshow(g(f[order]), origin="lower", extent=ext, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.ma.masked_where(g(band[order]) < 0.5, g(band[order])), origin="lower", extent=ext,
                  cmap=ListedColormap(["#86b6ef"]), alpha=0.35, interpolation="nearest")
        xs, ys = np.unique(mx), np.unique(my)
        if sub.startswith("preplug_h"):
            h = float(sub[len("preplug_h"):].replace(sfx, "") if sfx else sub[len("preplug_h"):])
            plug = plugged(phi, band, hole, my, h)[1]
            ax.imshow(np.ma.masked_where(g(plug[order]) < 0.5, g(plug[order])), origin="lower", extent=ext,
                      cmap=ListedColormap(["#404040"]), interpolation="nearest")
        ax.contour(xs, ys, g(target[order].astype(float)), levels=[0.5], colors="k", linewidths=0.8, linestyles="--")
        ax.contour(xs, ys, g(reach[order].astype(float)), levels=[0.5], colors="#2a78d6", linewidths=0.9)
        end, d, nrm = G.hole_geometry()
        ax.plot([G.MOUTH[0], end[0]], [G.MOUTH[1], end[1]], color="k", lw=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    fig.text(0.5, 0.02, u"红：浆体（充填饱和度或胶结占据率）；浅蓝：通道带；深灰：预置封堵；黑虚线：设计目标域；蓝实线：可达域；黑线：钻孔",
             ha="center", fontsize=8)
    fig.suptitle(u"孔口通道：理想整孔、分段与预置封堵的对照", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    p1 = os.path.join(OUT, "fig_p1_maps%s.png" % sfx)
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rows))
    ax1.bar(x - 0.2, [100 * r[1] for r in rows], 0.4, color="#2a78d6", label=u"目标区充填率 (%)")
    ax1.bar(x + 0.2, [r[3] for r in rows], 0.4, color="#eb6834", label=u"目标区外浆量 (m$^3$/m)")
    for i, r in enumerate(rows):
        ax1.text(i - 0.2, 100 * r[1] + 1, "%.0f" % (100 * r[1]), ha="center", fontsize=7)
        ax1.text(i + 0.2, r[3] + 1, "%.1f" % r[3], ha="center", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(u"目标区充填率与跑浆量", fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(OUT, "fig_p1_bars%s.png" % sfx)
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    lines = ["%-22s %8s %10s %10s %8s %6s %6s" % ("column", "tgt fill", "V in tgt", "V outside", "out %", "top", "bottom")]
    for r in rows:
        lines.append(u"%-22s %7.0f%% %10.2f %10.2f %7.0f%% %6s %6s" % (r[0], 100 * r[1], r[2], r[3],
                                                                     100 * r[4], r[5], r[6]))
    text = u"\n".join(lines)
    io.open(os.path.join(OUT, "p1_table%s.txt" % sfx), "w", encoding="utf-8").write(text + u"\n")
    print(text.encode("utf-8"))
    return p1, p2


def main(argv):
    if argv and argv[0] == "plot":
        hs = [2.5, 10.0]
        if "--h" in argv:
            hs = [float(v) for v in argv[argv.index("--h") + 1].split(",")]
        for p in table_and_plot(hs, "_pass2" if "--pass2" in argv else ""):
            print("wrote", p)
        return 0
    if not argv or argv[0] == "scan":
        scan()
        return 0
    if argv[0] == "p1":
        hs = [2.5, 7.5]
        if "--h" in argv:
            hs = [float(v) for v in argv[argv.index("--h") + 1].split(",")]
        p1(hs, pass2="--pass2" in argv)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
