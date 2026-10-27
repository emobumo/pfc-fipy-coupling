# -*- coding: utf-8 -*-
"""
Shared helper for the chapter-4 and chapter-5 figures drawn in figures/ch4/ and figures/ch5/ (main line 2026-10-26/27).

Uses the chapter settings of scripts/figures_ch4/common.py (fonts, colours, saturation scale) and writes next to the
calling script: <name>.svg (vector) and <name>.png (600 dpi). Width = A4 text block, 15.5 cm.

Drawing convention of both chapters (framework 1.6, fixed by fig. 4.3-3): design domain = dashed outline (the only
dashed line); reach domain = solid outline; fill domain = saturation patches (SAT_CMAP); solids and cement = dark grey;
hole = thin standpipe + thick outflow section (draw_hole); the five unfilled-zone classes = the light fills in CAT;
relative elevations.
"""
from __future__ import print_function

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "figures_ch4"))

import common as K                      # noqa: E402  (rcParams, colours)
from common import plt                  # noqa: E402

WIDTH = 15.5 / 2.54                     # inches
GREY = "#9a9a9a"                        # thin auxiliary lines (no dashes: dashes are for the design domain only)
SOLID = K.SOLID
ROCK = "#c9c9c9"                        # host rock in schematic sections (light grey, distinct from solids)
DESIGN_LS = (0, (4.0, 2.5))             # dashed: design domain only
LW_REACH, LW_DESIGN = 1.5, 1.1
CAT = {                                 # unfilled-zone classes (table 4.3-3), light fills
    "unreach": "#d9d2e9",               # 不可达区
    "dense": "#e3cfa9",                 # 致密包裹体
    "front": "#fbd6b4",                 # 锋面截断区
    "bypass": "#bfe6cf",                # 绕流空区
    "shadow": "#f2bfd0",                # 段间阴影区
}
CAT_NAME = {"unreach": u"不可达区", "dense": u"致密包裹体", "front": u"锋面截断区", "bypass": u"绕流空区",
            "shadow": u"段间阴影区"}


def tag(ax, letter, dx=0.45, dy=0.08):
    """Panel label (a)(b)(c) at the upper left, dx / dy inches left of / above the axes corner."""
    fig = ax.figure
    w, h = fig.get_size_inches()
    b = ax.get_position()
    fig.text(max(b.x0 - dx / w, 0.004), b.y1 + dy / h, u"(%s)" % letter, ha="left", va="bottom", fontsize=10)


def draw_hole(ax, collar, end, pipe, section_from=None, zorder=6, lw_pipe=1.0, lw_section=2.6):
    """Hole: thin standpipe from the collar over length `pipe`, thick outflow section from `section_from` (default:
    end of the standpipe) to `end`. Returns the unit direction."""
    c, e = np.asarray(collar, float), np.asarray(end, float)
    u = (e - c) / np.hypot(*(e - c))
    a = c + pipe * u
    s0 = a if section_from is None else c + section_from * u
    ax.plot([c[0], s0[0]], [c[1], s0[1]], "-", color="k", lw=lw_pipe, solid_capstyle="butt", zorder=zorder)
    ax.plot([s0[0], e[0]], [s0[1], e[1]], "-", color="k", lw=lw_section, solid_capstyle="butt", zorder=zorder)
    return u


def geodesic(lam, dx, src, offset=None):
    """Multi-source least cumulative integral of lam ds on a grid (16-neighbour stencil, edge weight = length x mean
    of the end values; lam = inf blocks). src: boolean mask of source cells; offset: per-cell non-negative weight of
    the source edge (position-potential reweighting, eq. (4-28)). Returns the distance field."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    nz, nx = lam.shape
    n = nz * nx
    idx = np.arange(n).reshape(nz, nx)
    rows, cols, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)):
        i1 = nz - di
        j0, j1 = max(0, -dj), nx - max(0, dj)
        a, b = idx[0:i1, j0:j1], idx[di:i1 + di, j0 + dj:j1 + dj]
        wv = 0.5 * (lam[0:i1, j0:j1] + lam[di:i1 + di, j0 + dj:j1 + dj]) * dx * np.hypot(di, dj)
        ok = np.isfinite(wv)
        rows.append(a[ok]); cols.append(b[ok]); w.append(wv[ok])
        rows.append(b[ok]); cols.append(a[ok]); w.append(wv[ok])
    s = np.flatnonzero(src.ravel())
    off = np.zeros(len(s)) if offset is None else offset.ravel()[s]
    rows.append(np.full(len(s), n)); cols.append(s); w.append(off + 1e-12)
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 1, n + 1)).tocsr()
    d = dijkstra(g, directed=True, indices=n)
    return d[:n].reshape(nz, nx)


def save(fig, name, out=None):
    out = out or HERE
    if not os.path.isdir(out):
        os.makedirs(out)
    for ext, dpi in (("svg", None), ("png", 600)):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)), dpi=dpi or 300)
    print("wrote", os.path.join(out, name + ".svg/.png"))
