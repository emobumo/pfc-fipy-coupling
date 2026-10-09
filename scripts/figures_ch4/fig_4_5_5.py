# -*- coding: utf-8 -*-
"""
图 4.5-5 缺陷归因堆叠柱图（2.5 m，给定注浆量停止）：基准、通道 5 场、随机 3 种子。
上：设计域（除孔单元）中未充填部分按成因拆为不可达区与锋面截断区（绕流空区另列，若有）；
下：跑浆量（设计域外浆量，m³/m）——分栏并列，不用双纵轴。
数据：outputs/rerun_rate/B/<label>/cells.csv（列 x,y,phi,S,path,reachable,category）、
outputs/rerun_rate/A/base_uniform_0.18_2.5/cells.csv、A/target_mask.npy（两遍一致）。
"""
from __future__ import print_function

import imp
import os

import numpy as np

import common as K
from common import plt, TERMS

G = imp.load_source("inclined_hole_gradient", os.path.join(K.REPO, "cases", "inclined_hole_gradient.py"))
CASES = [(u"基准\n（均质）", os.path.join("A", "base_uniform_0.18_2.5")),
         (u"孔口\nφ0.45", os.path.join("B", "B_band_collar_phi0.45")),
         (u"孔中\nφ0.45", os.path.join("B", "B_band_midhole_phi0.45")),
         (u"孔底外\nφ0.30", os.path.join("B", "B_band_toe_phi0.30")),
         (u"孔底外\nφ0.45", os.path.join("B", "B_band_toe_phi0.45")),
         (u"孔底外\nφ0.60", os.path.join("B", "B_band_toe_phi0.60")),
         (u"随机场\n种子 1", os.path.join("B", "B_random_seed1")),
         (u"随机场\n种子 2", os.path.join("B", "B_random_seed2")),
         (u"随机场\n种子 3", os.path.join("B", "B_random_seed3"))]


def main():
    target = np.load(os.path.join(K.RR, "A", "target_mask.npy"))
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 2.5)
    mx, my = np.asarray(x), np.asarray(y)
    hole = np.asarray(G.hole_cells(mx, my, 24, 24), dtype=int)
    notsrc = np.ones(mx.size, dtype=bool); notsrc[hole] = False
    T = target & notsrc
    un, fr, bv, run = [], [], [], []
    for _, sub in CASES:
        a = np.loadtxt(os.path.join(K.RR, sub, "cells.csv"), delimiter=",", skiprows=1)
        s, reach, cat = a[:, 3], a[:, 5] > 0.5, a[:, 6].astype(int)
        unfilled = T & (s < 0.5)
        un.append(100.0 * np.sum(unfilled & ~reach) / T.sum())
        bv.append(100.0 * np.sum(unfilled & reach & (cat == 3)) / T.sum())
        fr.append(100.0 * np.sum(unfilled & reach & (cat != 3)) / T.sum())
        run.append(float(np.sum(a[:, 2] * np.clip(s, 0, 1) * 6.25 * (~target & notsrc))))
    xs = np.arange(len(CASES))
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax.bar(xs, un, 0.6, color=K.SOLID, edgecolor="white", label=TERMS["unreach_lvl"])
    ax.bar(xs, fr, 0.6, bottom=un, color=K.C1, edgecolor="white", label=TERMS["front_lvl"])
    if max(bv) > 0:
        ax.bar(xs, bv, 0.6, bottom=np.array(un) + np.array(fr), color=K.C3, edgecolor="white", label=TERMS["bypass"])
    for xi, a, b, c in zip(xs, un, fr, bv):
        ax.text(xi, a + b + c + 0.6, "%.1f" % (a + b + c), ha="center", fontsize=7)
    ax.set_ylabel(u"设计域未充填比例 (%)")
    ax.set_ylim(0, 46)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    bx.bar(xs, run, 0.6, color=K.C2, edgecolor="white")
    for xi, v in zip(xs, run):
        bx.text(xi, v + 0.8, "%.1f" % v, ha="center", fontsize=7)
    bx.set_ylabel(TERMS["outside_V"] + u" (m³/m)")
    bx.set_xlim(-0.6, len(CASES) - 0.4)
    bx.set_ylim(0, 48)
    bx.set_xticks(xs)
    bx.set_xticklabels([c[0] for c in CASES], fontsize=8)
    for a_ in (ax, bx):
        K.style(a_)
    K.save(fig, "fig_4_5_5")
    for (n, _), a, b, c, r in zip(CASES, un, fr, bv, run):
        print(n.replace("\n", " "), "unreach %.1f front %.1f bypass %.1f runaway %.1f" % (a, b, c, r))


if __name__ == "__main__":
    main()
