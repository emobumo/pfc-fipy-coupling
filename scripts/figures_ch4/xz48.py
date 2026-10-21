# -*- coding: utf-8 -*-
"""
Shared inputs of the 4.8 draft figures (scripts/figures_ch4/fig_4_8_*.py).

The figures draw the current end state: geometry configuration $XZ_CONFIG, default 30m here (the 4.8 chain's own
default is 40m, kept for its byte-identical regression). Everything is read from existing outputs; nothing is re-run.
  - 0.25 m geometry, zoning, fan holes        outputs/xz_prejudge<suffix>/step2b.npz, prejudge.npz (cav), seq_reach.npz
  - porosity and p_req of the normal holes    outputs/check_4_8<suffix>/prewrite.npz
  - solver cases                              outputs/xz_solver<suffix>/ (summary.npz, grid_*.npz, snap_*, final_*,
                                              stage2.txt)
"Lower end of the rubble" (散体下端) is the solver's lower end of the basal layer: the 7 cells of the 0.5 m K 1.25
grid (grid_K1.25_s1.npz "low", basal cells with t <= low_t), drawn the same way on every 4.8 figure.
"""
from __future__ import print_function

import imp
import os
import re

os.environ.setdefault("XZ_CONFIG", "30m")

import common as K                       # noqa: E402  (rcParams, save to figures/ch4_draft)
from common import plt                   # noqa: E402

import numpy as np                       # noqa: E402

REPO = K.REPO
PL = imp.load_source("xz_plot", os.path.join(REPO, "scripts", "xz_prejudge", "plot.py"))
P48 = imp.load_source("xz_plot48", os.path.join(REPO, "scripts", "xz_prejudge", "plot_48.py"))
XC, CFG = PL.XC, PL.CFG
PRE = XC.out(CFG, "outputs", "xz_prejudge")
CHK = XC.out(CFG, "outputs", "check_4_8")
SOL = XC.out(CFG, "outputs", "xz_solver")
# plot.py sets its own rcParams on import; the chapter's (common.py) apply to the 4.8 figures
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7})

LOWC = "#c2185b"                         # lower end of the rubble
LOW_NAME = u"散体下端"
ANNOT = u"K = 1.25，情景一"
# residual void: pale cream on all four figures (the 4.8 chain's light blue clashes with the saturation scale)
VOIDC = "#f6ecd2"
ROCKC = "#bdbdbd"
COLORS48 = list(PL.COLORS)
COLORS48[4] = VOIDC


def background(ax, D):
    """Rock (footwall, hanging wall, pillar) and residual void of a 0.25 m class map, as PL.background."""
    from matplotlib.colors import ListedColormap
    cls = D["cls"]
    for mask, col in (((cls >= 1) & (cls <= 3), ROCKC), (cls == 4, VOIDC)):
        ax.imshow(np.ma.masked_where(~mask, np.ones(cls.shape)), origin="lower", extent=PL.ext(D),
                  cmap=ListedColormap([col]), interpolation="nearest")


def low_grid():
    """0.5 m K 1.25 grid: X, Z (relative to the footwall corner of the stope floor), lower-end mask."""
    G = np.load(os.path.join(SOL, "grid_K1.25_s1.npz"))
    return G["X"], G["Z"], G["low"]


def draw_low(ax, label=True, dx=-9.0, dz=-2.6, lw=1.4):
    X, Z, low = low_grid()
    ax.contour(X[0], Z[:, 0], low.astype(float), levels=[0.5], colors=[LOWC], linewidths=lw)
    if label:
        cx, cz = float(X[low].mean()), float(Z[low].mean())
        ax.annotate(LOW_NAME, xy=(cx, cz), xytext=(cx + dx, cz + dz), fontsize=6.5, color=LOWC, va="center",
                    ha="center", arrowprops=dict(arrowstyle="-", color=LOWC, lw=0.6))


def corner_note(ax, text=ANNOT):
    ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=K.INK2)


def stage2():
    """outputs/xz_solver<suffix>/stage2.txt (two-pass identical post-processing) -> per case: pore, arrival V_in / pore,
    0.95 snapshot V_in / pore."""
    txt = open(os.path.join(SOL, "stage2.txt")).read()
    R = {}
    for blk in re.split(r"\n(?=[SKO]\d \()", txt):
        m = re.match(r"([SKO]\d) \(", blk)
        if not m:
            continue
        r = {"pore": float(re.search(r"pore ([\d.]+) m3/m", blk).group(1))}
        a = re.search(r"lower end reached: .*?V_in / pore ([\d.]+)", blk)
        r["arrival"] = float(a.group(1)) if a else None
        s = re.search(r"snapshot 0\.95: .*?V_in / pore ([\d.]+)", blk)
        r["s95"] = float(s.group(1)) if s else None
        R[m.group(1)] = r
    return R
