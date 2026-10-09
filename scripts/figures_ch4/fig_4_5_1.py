# -*- coding: utf-8 -*-
"""
图 4.5-1 均质与梯度场的充填与最小累积启动压力场（1.25 m，现行停注判据）。
上排：求解器饱和度（停注时）叠可达域轮廓；下排：最小累积启动压力场 T(x)/p₀ 等值线与通往上下极值点的最优路径。
数据：outputs/rerun_rate/A/final_{uniform,gradient}_1.25.npz（两遍一致）；T(x) 由同一孔隙率场重算可达域
（后处理，不跑求解器）。最优路径为在 T(x) 上沿 16 邻域逐步下降的近似路径。
"""
from __future__ import print_function

import imp
import os

import numpy as np

import common as K
from common import plt, TERMS
from src.analysis.fill_diagnostics import reachable_domain

G = imp.load_source("inclined_hole_gradient", os.path.join(K.REPO, "cases", "inclined_hole_gradient.py"))
CELL = 1.25


def field(kind):
    G.CELL = CELL
    st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
    cost = np.asarray(reachable_domain(st, hc, p0=G.P0, use_gravity=True)["cost"], dtype=float) / G.P0
    f = np.load(os.path.join(K.RR, "A", "final_%s_1.25.npz" % kind))
    assert np.allclose(f["x"], mx) and np.allclose(f["y"], my)
    return mx, my, np.asarray(f["s"], dtype=float), cost, np.asarray(hc, dtype=int)


def descend(mx, my, cost, start):
    """Greedy descent on T(x) over the 16-neighbour stencil (an approximate optimal path)."""
    pts = np.column_stack([mx, my])
    path, cur = [start], start
    for _ in range(400):
        d = np.hypot(mx - mx[cur], my - my[cur])
        nb = np.where((d > 0) & (d <= np.sqrt(5.0) * CELL + 1e-9))[0]
        nxt = nb[np.argmin(cost[nb])]
        if cost[nxt] >= cost[cur]:
            break
        path.append(nxt)
        cur = nxt
        if cost[cur] <= 0.0:
            break
    return pts[path]


def main():
    end, d, n = G.hole_geometry()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.6))
    for j, (kind, title) in enumerate((("uniform", u"均质场（n = 0.18）"), ("gradient", u"梯度场（n 0.12→0.30，上疏下密）"))):
        mx, my, s, cost, hc = field(kind)
        reach = cost <= 1.0
        ax = axes[0, j]
        im = K.draw_sat(ax, mx, my, s, CELL)
        K.outline(ax, mx, my, reach, CELL)
        K.draw_hole(ax, G.MOUTH, end)
        ax.set_title(title)
        ax = axes[1, j]
        xs, ys = mx - K.MOUTH[0], my - K.MOUTH[1]
        g, ext = K.grid(mx, my, np.minimum(cost, 1.6), CELL)
        e = K.rel(ext, *K.MOUTH)
        gx = np.linspace(e[0] + CELL / 2, e[1] - CELL / 2, g.shape[1])
        gy = np.linspace(e[2] + CELL / 2, e[3] - CELL / 2, g.shape[0])
        cs = ax.contour(gx, gy, g, levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], colors=K.INK2, linewidths=0.6)
        ax.clabel(cs, fmt="%.1f", fontsize=6)
        K.outline(ax, mx, my, reach, CELL, lw=1.4)
        perp = (mx - G.MOUTH[0]) * n[0] + (my - G.MOUTH[1]) * n[1]
        cand = np.where(reach)[0]
        for idx, col, lab in ((cand[np.argmin(perp[cand])], K.C1, u"向上极值点"), (cand[np.argmax(perp[cand])], K.C2, u"向下极值点")):
            p = descend(mx, my, cost, idx)
            ax.plot(p[:, 0] - K.MOUTH[0], p[:, 1] - K.MOUTH[1], color=col, lw=1.6, label=TERMS["opt_path"] + u"（" + lab + u"）")
            ax.plot([mx[idx] - K.MOUTH[0]], [my[idx] - K.MOUTH[1]], "o", color=col, ms=4)
        K.draw_hole(ax, G.MOUTH, end)
        ax.set_aspect("equal")
        ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
        ax.legend(loc="lower right", frameon=False, fontsize=7)
        ax.set_title(TERMS["T_field"] + u"（等值线为 T/$p_0$）", fontsize=9)
    for ax in axes.ravel():
        K.style(ax)
        ax.set_xlabel(TERMS["x"]); ax.set_ylabel(TERMS["y"])
    cax = fig.add_axes([0.92, 0.56, 0.015, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(TERMS["sat"])
    fig.text(0.5, 0.005, u"实线：%s；色斑：%s（停注时）；钻孔孔口位于左边界（坡面）" % (TERMS["reach"], TERMS["fill"]),
             ha="center", fontsize=8, color=K.INK2)
    K.save(fig, "fig_4_5_1")


if __name__ == "__main__":
    main()
