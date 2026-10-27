# -*- coding: utf-8 -*-
"""
图 4.6-4 前序胶结体的双重作用（孔口通道场，2.5 m，给定注浆量 185.86 m³/m 停止）。
上：① 理想整孔；② 分段（第 1 段孔深 7 m，与第 1 段孔深 8.5 m 的方案在 2.5 m 上结果相同；色斑即第 1 段充填，
    后续段未注浆）；③ 预置封堵 h = 2.5 m、h = 10 m 的充填域，叠设计域（虚线）、可达域（实线）、通道（橙线），
    封堵体与块石同为深灰；
下：设计域充填率与设计域外浆量占比（分栏）。
数据：outputs/rerun_rate/B/B_p1_*/cells.csv（两遍一致）；设计域 A/target_mask.npy。
设计域充填率按不计钻孔单元的 165 个单元计，预置封堵单元计为已充填（与前序段胶结体同口径；PL01、B09）；
设计域外浆量占比的分母为计算域内浆量（不计钻孔单元）。
第十轮（框架 v2.7）：封堵单元计为已充填（77.0% / 92.1%）；② 改名；坐标写“中文名 符号/单位”。
2026-10-28：按版心宽 15.5 cm 原尺寸排版（不再缩放），字号不小于 7 pt，图注文字分四行，图面内容不变。
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
        (u"② 分段（第 1 段孔深 7 m）", os.path.join("B_p1_collar_v06", "staged_7_10_17"), True),
        (u"③ 预置封堵 h = 2.5 m", "B_p1_preplug_h2.50", False),
        (u"③ 预置封堵 h = 10 m", "B_p1_preplug_h10.00", False)]


def main():
    target = np.load(os.path.join(K.RR, "A", "target_mask.npy"))
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    mx, my = np.asarray(x), np.asarray(y)
    hole = np.asarray(G.hole_cells(mx, my, 24, 24), dtype=int)
    notsrc = np.ones(mx.size, dtype=bool); notsrc[hole] = False
    end = G.hole_geometry()[0]
    # laid out at the text-block width (15.5 cm), no scaling
    H = 5.3
    fig = plt.figure(figsize=(K.TEXT_W, H))
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
            solid = a[:, 2] <= 0.01            # the pre-placed plug (n 1e-3); the field has no other solids
        ax = K.axes_in(fig, 0.45 + j * 1.44, H - 2.1, 1.17, 1.8)
        K.draw_sat(ax, mx, my, s, CELL, solid=solid)
        K.outline(ax, mx, my, (phi0 > 0.3) | solid, CELL, color=K.C2, lw=0.8)
        K.outline(ax, mx, my, target, CELL, dashed=True)
        K.outline(ax, mx, my, reach, CELL)
        K.draw_hole(ax, G.MOUTH, end)
        ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(u"水平距离 $x$/m", fontsize=7.5); K.style(ax); ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel(u"相对高程 $z$/m", fontsize=7.5)
        filled = (s >= 0.5) | solid                # plug cells count as filled
        tg.append(100.0 * np.sum(filled & target & notsrc) / np.sum(target & notsrc))
        vin, vout = grout[target & notsrc].sum(), grout[~target & notsrc].sum()
        ru.append(100.0 * vout / (vin + vout))
    fig.text(0.02 / K.TEXT_W, (H - 2.62) / H, u"虚线：%s；实线：%s；橙线：孔口通道；深灰：预置封堵体；\n② 与第 1 段孔深 8.5 m 的方案结果相同，"
             u"色斑即第 1 段充填，后续段未注浆\n" % (TERMS["design"], TERMS["reach"])
             + u"%s按不计钻孔单元的 165 个单元计，封堵单元计为已充填；\n%s占比 = %s ÷ 计算域内浆量（不计钻孔单元），"
             u"不是注入量" % (TERMS["design_fill"], TERMS["outside_V"], TERMS["outside_V"]), fontsize=7, color=K.INK2,
             va="top", linespacing=1.4)
    names = [u"①\n理想整孔", u"②\n分段", u"③\n封堵\nh = 2.5 m", u"③\n封堵\nh = 10 m"]
    for k, (vals, lab, col) in enumerate(((tg, TERMS["design_fill"] + u"/%", K.C1),
                                          (ru, TERMS["outside_V"] + u"占比/%", K.C2))):
        ax = K.axes_in(fig, 0.5 + k * 3.1, 0.55, 2.4, 1.35)
        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.6, color=col, edgecolor="white", align="center")
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.0, "%.1f" % v, ha="center", fontsize=7)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7); ax.set_xlim(-0.6, len(vals) - 0.4)
        ax.set_ylabel(lab, fontsize=7.5); ax.set_ylim(0, max(vals) * 1.25)
        K.style(ax); ax.tick_params(labelsize=7)
    K.save(fig, "fig_4_6_4")
    print("design fill", ["%.6f" % v for v in tg], "outside share", ["%.6f" % v for v in ru])


if __name__ == "__main__":
    main()
