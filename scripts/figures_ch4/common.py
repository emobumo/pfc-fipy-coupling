# -*- coding: utf-8 -*-
"""
Shared settings for the Ch.4 draft figures (scripts/figures_ch4/fig_*.py).

Visual grammar (framework v1.1, section 1.6): design domain = dashed outline;
reachable domain = solid outline; filled domain = saturation colour patches on
one colour scale for the whole chapter; solids and cement = dark grey; the
hole is always drawn the same way; elevations are relative; all text Chinese.
All wording comes from docs/术语对照表_2026-10-08.md (TERMS below) -- change a
term here and every figure follows. Numbers are read from the rate-rule
outputs (outputs/rerun_rate/ ...); nothing is re-run.
"""
from __future__ import print_function

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "figures", "ch4_draft")
RR = os.path.join(REPO, "outputs", "rerun_rate")

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans SC", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "svg.fonttype": "none",
})

# ---- the chapter's terms (docs/术语对照表_2026-10-08.md) --------------------
TERMS = {
    "design": u"设计域",
    "reach": u"可达域",
    "fill": u"充填域",
    "sat": u"饱和度 S",
    "cum_p": u"累积启动压力",
    "T_field": u"最小累积启动压力场 T(x)",
    "T_rel": u"最小累积启动压力 T/$p_0$",
    "opt_path": u"最优路径",
    "entry_p": u"通道入口累积启动压力",
    "p_ratio": u"累积启动压力比",
    "net_grad": u"净启动压力梯度",
    "stage_loss": u"分段覆盖损失",
    "front_trunc": u"锋面截断区",
    "stage_ok": u"后续段有效",
    "stage_dead": u"后续段失效",
    "cement": u"前序段胶结体",
    "given_V": u"给定注浆量",
    "given_t": u"给定注浆量的注入时间",
    "phi_c_min": u"交界面临界孔隙率下限",
    "updown": u"上下扩散距离比",
    "unreach": u"不可达区",
    "bypass": u"绕流空区",
    "dense": u"致密包裹体",
    "shadow": u"段间阴影区",
    "runaway": u"跑浆",
    "x": u"水平距离 (m)",
    "y": u"相对高程 (m)",
    "hole": u"钻孔",
    "design_fill": u"设计域充填率",
    "outside_V": u"设计域外浆量",
    "unreach_lvl": u"不可达区（可达层面缺陷）",
    "front_lvl": u"锋面截断区（过程层面缺陷）",
}

# mathtext for symbols with subscripts (round-2 comments: x_f, L_max, lambda_c ...)
XF = u"$x_f$"
LMAX = u"$L_{\mathrm{max}}$"
PIG = u"$\Pi_g$"
PHIC = u"$n_c$"

# ---- colours -------------------------------------------------------------
SAT_CMAP = LinearSegmentedColormap.from_list("sat", ["#ffffff", "#9ec5f4", "#2a78d6", "#0d366b"])
SOLID = "#4d4d4d"
INK = "#1a1a1a"
INK2 = "#52514e"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3

MOUTH = (-30.0, 21.0)                              # inclined-hole cases: relative coords from the mouth


def grid(x, y, v, cell):
    """Cell-centre arrays -> 2D image (rows = y) and its extent."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ix = np.round((x - x.min()) / cell).astype(int)
    iy = np.round((y - y.min()) / cell).astype(int)
    g = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    g[iy, ix] = np.asarray(v, dtype=float)
    ext = [x.min() - cell / 2, x.max() + cell / 2, y.min() - cell / 2, y.max() + cell / 2]
    return g, ext


def rel(ext, x0, y0):
    return [ext[0] - x0, ext[1] - x0, ext[2] - y0, ext[3] - y0]


def draw_sat(ax, x, y, s, cell, x0=MOUTH[0], y0=MOUTH[1], solid=None):
    g, ext = grid(x, y, np.clip(s, 0, 1), cell)
    im = ax.imshow(g, origin="lower", extent=rel(ext, x0, y0), cmap=SAT_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    if solid is not None and np.any(solid):
        gs, _ = grid(x, y, np.asarray(solid, dtype=float), cell)
        ax.imshow(np.ma.masked_where(gs < 0.5, gs), origin="lower", extent=rel(ext, x0, y0),
                  cmap=LinearSegmentedColormap.from_list("solid", [SOLID, SOLID]), interpolation="nearest")
    ax.set_aspect("equal")
    return im


def outline(ax, x, y, mask, cell, dashed=False, color=INK, x0=MOUTH[0], y0=MOUTH[1], lw=1.0):
    """Outline of a cell mask: solid = reachable domain, dashed = design domain."""
    g, ext = grid(x, y, np.asarray(mask, dtype=float), cell)
    g = np.nan_to_num(g)
    e = rel(ext, x0, y0)
    xs = np.linspace(e[0] + cell / 2, e[1] - cell / 2, g.shape[1])
    ys = np.linspace(e[2] + cell / 2, e[3] - cell / 2, g.shape[0])
    ax.contour(xs, ys, g, levels=[0.5], colors=[color], linewidths=lw,
               linestyles="dashed" if dashed else "solid")


def draw_hole(ax, start, end, x0=MOUTH[0], y0=MOUTH[1]):
    ax.plot([start[0] - x0, end[0] - x0], [start[1] - y0, end[1] - y0], color="k", lw=2.0, solid_capstyle="butt", zorder=5)
    ax.plot([start[0] - x0], [start[1] - y0], marker="s", ms=4, color="k", zorder=6)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(top="off", right="off", labelsize=8)


def save(fig, name):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(OUT, name + ".png/.svg"))
