# -*- coding: utf-8 -*-
"""
图 4.6-1(c) S3 的段间阴影区（n 0.14，第 1 段孔深 7 m、终孔孔深 17 m，1.25 m，现行停注判据）。
(a)(b) 为示意图，不在此画。
按式（4-27），均质算例不设设计域：Ω_shadow = Ω_reach^w \\ (Ω_reach^(1) ∪ Ω_reach^(2))。
掩码用 scripts/shadow_eq427.py 的函数重算（后处理，不跑求解器）：
  Ω_reach^w   原始场、整孔出浆段（孔口至 17 m）的可达域，须与 cells.csv 的 reach_w 列逐单元相同；
  Ω_reach^(1) 原始场、第 1 段出浆段（孔口至 7 m）；
  Ω_reach^(2) 终态场、第 2 段出浆段（孔口至 17 m），即第 2 段出浆单元本身（第 2 段失效）。
阴影区 180 个单元、281.25 m²（C13，正文写 281.3 m²）。深灰为前序段胶结体（终态孔隙率降至下限 10⁻³ 的单元）。
数据：outputs/rerun_rate/C/C_p2/phi0.14_H17_c1.25/7_17/cells.csv（两遍一致）。
"""
from __future__ import print_function

import imp
import os

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as K
from common import plt, TERMS

SH = imp.load_source("shadow_eq427", os.path.join(K.REPO, "scripts", "shadow_eq427.py"))
S, G = SH.S, SH.G
LABEL = "C_p2/phi0.14_H17_c1.25/7_17"
CELL = 1.25
FLOOR = 1.0e-3


def masks():
    G.CELL, G.HOLE_LEN = CELL, 17.0
    try:
        mx, my, nx, ny = SH.grid(CELL)
        a = np.loadtxt(os.path.join(SH.RR, "C", LABEL, "cells.csv"), delimiter=",", skiprows=1)
        assert np.allclose(a[:, 0], mx) and np.allclose(a[:, 1], my)
        phi0, n_final = a[:, 2], a[:, 3]
        src1 = np.asarray(S.hole_cells_to_depth(mx, my, nx, ny, 7.0), dtype=int)
        src2 = np.asarray(S.hole_cells_to_depth(mx, my, nx, ny, 17.0), dtype=int)
        rw, r1, r2 = SH.reach(phi0, src2), SH.reach(phi0, src1), SH.reach(n_final, src2)
        assert np.array_equal(rw, a[:, 6] > 0.5), "reach^w differs from the run's reach_w column"
        end = G.hole_geometry()[0]
        d1 = np.asarray(G.MOUTH) + (np.asarray(end) - np.asarray(G.MOUTH)) * 7.0 / 17.0
        return dict(mx=mx, my=my, rw=rw, r1=r1, r2=r2, src2=src2, cement=n_final <= FLOOR * (1 + 1e-9),
                    shadow=rw & ~(r1 | r2), end=end, d1=d1)
    finally:
        G.CELL, G.HOLE_LEN = 2.5, 17.0


def main():
    m = masks()
    mx, my = m["mx"], m["my"]
    n_sh = int(m["shadow"].sum())
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    g, ext = K.grid(mx, my, np.where(m["shadow"], 1.0, np.where(m["cement"], 2.0, 0.0)), CELL)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", K.C2, K.SOLID])
    ax.imshow(g, origin="lower", extent=K.rel(ext, K.MOUTH[0], K.MOUTH[1]), cmap=cmap, vmin=0, vmax=2,
              interpolation="nearest", alpha=1.0)
    K.outline(ax, mx, my, m["rw"], CELL, color=K.INK, lw=1.3)
    K.outline(ax, mx, my, m["r1"], CELL, color=K.C1, lw=1.3)
    K.draw_hole(ax, G.MOUTH, m["end"])
    dx, dy = m["d1"][0] - K.MOUTH[0], m["d1"][1] - K.MOUTH[1]
    ax.plot([dx], [dy], marker="o", ms=5, mfc="white", mec="k", mew=1.2, zorder=7)
    ax.annotate(u"第 1 段孔底", (dx, dy), xytext=(8, -12), textcoords="offset points", fontsize=7.5)
    ax.annotate(u"终孔", (m["end"][0] - K.MOUTH[0], m["end"][1] - K.MOUTH[1]), xytext=(6, 4),
                textcoords="offset points", fontsize=7.5)
    sel = m["rw"]
    xr, yr = mx[sel] - K.MOUTH[0], my[sel] - K.MOUTH[1]
    ax.set_xlim(0, xr.max() + 3.0)
    ax.set_ylim(yr.min() - 2.0, yr.max() + 2.0)
    ax.set_xlabel(u"水平距离 $x$/m"); ax.set_ylabel(u"相对高程 $z$/m")
    ax.set_title(u"(c) S3 的%s" % TERMS["shadow"], fontsize=9)
    handles = [Line2D([], [], color=K.INK, lw=1.3, label=u"理想整孔%s $\\Omega_{\\mathrm{reach}}^{w}$" % TERMS["reach"]),
               Line2D([], [], color=K.C1, lw=1.3, label=u"第 1 段%s $\\Omega_{\\mathrm{reach}}^{(1)}$" % TERMS["reach"]),
               Patch(facecolor=K.C2, edgecolor="none", label=u"%s\n（%d 个单元，%.1f m²）" % (TERMS["shadow"], n_sh,
                                                                       np.floor(n_sh * CELL * CELL * 10 + 0.5) / 10)),
               Patch(facecolor=K.SOLID, edgecolor="none", label=TERMS["cement"])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7.5)
    K.style(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.02, 0.015, u"第 2 段失效，其%s即第 2 段出浆单元，图中不另画；深灰为孔隙率降至下限的单元" % TERMS["reach"],
             fontsize=7.5, color=K.INK2)
    K.save(fig, "fig_4_6_1c")
    print("reach^w %d, reach^(1) %d, reach^(2) %d (= stage-2 source cells: %s), shadow %d cells = %.2f m2, cement %d" % (
        int(m["rw"].sum()), int(m["r1"].sum()), int(m["r2"].sum()),
        bool(np.array_equal(np.where(m["r2"])[0], np.sort(m["src2"]))), n_sh, n_sh * CELL * CELL, int(m["cement"].sum())))


if __name__ == "__main__":
    main()
