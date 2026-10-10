# -*- coding: utf-8 -*-
"""
图 4.6-4 前序段胶结体的双重作用（孔口通道场，2.5 m，给定注浆量 185.86 m³/m 停止）。
上：① 理想整孔；② 等长分段 7→10→17（与 8.5→17 在 2.5 m 上逐位相同）；③ 预置封堵 h = 2.5 m、h = 10 m 的充填域，
    叠设计域（虚线）、可达域（实线）、通道（橙线），封堵体与块石同为深灰；
下：目标区充填率与跑浆占比（分栏）。
数据：outputs/rerun_rate/B/B_p1_*/cells.csv（两遍一致）；设计域 A/target_mask.npy。
目标区充填率与跑浆占比均按"除孔单元"计（与框架 4.6.4 的数字一致）。
"""
from __future__ import print_function

import imp
import os

import numpy as np

import common as K
from common import plt, TERMS

G = imp.load_source("inclined_hole_gradient", os.path.join(K.REPO, "cases", "inclined_hole_gradient.py"))
CELL = 2.5
B = os.path.join(K.RR, "B")
COLS = [(u"① 理想整孔", "B_p1_full_hole", False),
        (u"② 等长分段 7→10→17", os.path.join("B_p1_collar_v06", "staged_7_10_17"), True),
        (u"③ 预置封堵 h = 2.5 m", "B_p1_preplug_h2.50", False),
        (u"③ 预置封堵 h = 10 m", "B_p1_preplug_h10.00", False)]


def main():
    target = np.load(os.path.join(K.RR, "A", "target_mask.npy"))
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    mx, my = np.asarray(x), np.asarray(y)
    hole = np.asarray(G.hole_cells(mx, my, 24, 24), dtype=int)
    notsrc = np.ones(mx.size, dtype=bool); notsrc[hole] = False
    end = G.hole_geometry()[0]
    fig = plt.figure(figsize=(10.0, 6.6))
    tg, ru = [], []
    for j, (title, sub, staged) in enumerate(COLS):
        a = np.loadtxt(os.path.join(B, sub, "cells.csv"), delimiter=",", skiprows=1)
        if staged:
            phi0, s, reach = a[:, 2], a[:, 4], a[:, 6] > 0.5
            grout = (a[:, 2] - a[:, 3]) * CELL * CELL
            solid = np.zeros(mx.size, dtype=bool)
        else:
            phi0, s, reach = a[:, 2], a[:, 3], a[:, 5] > 0.5
            grout = a[:, 2] * np.clip(a[:, 3], 0, 1) * CELL * CELL
            solid = a[:, 2] <= 0.01
        ax = fig.add_axes([0.05 + j * 0.24, 0.42, 0.22, 0.52])
        K.draw_sat(ax, mx, my, s, CELL, solid=solid)
        K.outline(ax, mx, my, (phi0 > 0.3) | solid, CELL, color=K.C2, lw=0.8)
        K.outline(ax, mx, my, target, CELL, dashed=True)
        K.outline(ax, mx, my, reach, CELL)
        K.draw_hole(ax, G.MOUTH, end)
        ax.set_xlim(0, 40)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(TERMS["x"]); K.style(ax)
        if j == 0:
            ax.set_ylabel(TERMS["y"])
        tg.append(100.0 * np.sum((s >= 0.5) & target & notsrc) / np.sum(target & notsrc))
        vin, vout = grout[target & notsrc].sum(), grout[~target & notsrc].sum()
        ru.append(100.0 * vout / (vin + vout))
    fig.text(0.05, 0.355, u"虚线：%s；实线：%s；橙线：孔口通道；深灰：预置封堵体；②中色斑为各段累计充填（含%s）"
             % (TERMS["design"], TERMS["reach"], TERMS["cement"]), fontsize=8, color=K.INK2)
    fig.text(0.05, 0.325, u"%s与%s占比均不计钻孔单元；占比 = %s ÷ 计算域内浆量（不计钻孔单元），不是注入量"
             % (TERMS["design_fill"], TERMS["outside_V"], TERMS["outside_V"]), fontsize=8, color=K.INK2)
    names = [u"①\n理想整孔", u"②\n等长分段", u"③\n封堵 h = 2.5 m", u"③\n封堵 h = 10 m"]
    for k, (vals, lab, col) in enumerate(((tg, TERMS["design_fill"] + u" (%)", K.C1), (ru, TERMS["outside_V"] + u"占比 (%)", K.C2))):
        ax = fig.add_axes([0.07 + k * 0.48, 0.07, 0.42, 0.22])
        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.6, color=col, edgecolor="white")
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.0, "%.1f" % v, ha="center", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7); ax.set_xlim(-0.6, len(vals) - 0.4)
        ax.set_ylabel(lab); ax.set_ylim(0, max(vals) * 1.25)
        K.style(ax)
    K.save(fig, "fig_4_6_4")
    print("target", ["%.1f" % v for v in tg], "runaway", ["%.1f" % v for v in ru])


if __name__ == "__main__":
    main()
