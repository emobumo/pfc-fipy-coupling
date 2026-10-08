# -*- coding: utf-8 -*-
"""
图 4.5-2 翻转判据验证图：横轴翻转判据 F（各组按自身 ρ 计），纵轴相对几何基线的上下扩散距离比因子。
直射线估计（虚线）、二维可达域（标记）、四组参数；求解器梯度强度扫描（φ_top 0.25/0.30/0.35）；现梯度场位置。
数据：outputs/flip_criterion/flip_criterion.txt（组 1、4：1.25 m）与 flip_criterion_refine_0.625.txt（组 2、3：
0.625 m，L_max ≈ 10 m 须用细网格）；求解器扫描取 outputs/rerun_rate/A/results.json（1.25 m），因子 = 比值 / 无重力
均质 0.994，F 按各扫描场孔轴中点的 γ 计算。
"""
from __future__ import print_function

import json
import os
import re

import numpy as np

import common as K
from common import plt, TERMS

FC = os.path.join(K.REPO, "outputs", "flip_criterion")
RHO_G = 1830.0 * 9.81
P0 = 5.0e6


def table(path):
    rows = {}
    for line in open(path):
        m = re.match(r"(\d) (.{9})\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+) \|\s+[\d.]+\s+[\d.]+ \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            g = int(m.group(1))
            rows.setdefault(g, []).append((float(m.group(4)), float(m.group(7)), float(m.group(8))))
    return rows


def sweep_F(top, bottom=0.12, y_mid=25.875, H=60.0):
    dn = (top - bottom) / H
    n = bottom + dn * y_mid
    return dn / (n * (1 - n)) * P0 / (2 * RHO_G)


def main():
    base = table(os.path.join(FC, "flip_criterion.txt"))
    fine = table(os.path.join(FC, "flip_criterion_refine_0.625.txt"))
    names = {1: u"组 1 基准参数（1.25 m）", 2: u"组 2 $\\tau_0$ 加倍（0.625 m）", 3: u"组 3 A × 0.3（0.625 m）", 4: u"组 4 ρ = 1450（1.25 m）"}
    cols = {1: K.C1, 2: K.C2, 3: K.C3, 4: "#4a3aa7"}
    mk = {1: "o", 2: "s", 3: "^", 4: "D"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for g in (1, 2, 3, 4):
        src = fine[g] if g in (2, 3) else base[g]
        F = np.array([r[0] for r in src]); two = np.array([r[1] for r in src]); ray = np.array([r[2] for r in src])
        ax.plot(F, ray, "--", color=cols[g], lw=1.0)
        ax.plot(F, two, mk[g], color=cols[g], ms=5, mec="white", mew=0.8, label=names[g])
    A = dict((r["label"], r) for r in json.load(open(os.path.join(K.RR, "A", "results.json"))))
    sx = [sweep_F(0.25), sweep_F(0.30), sweep_F(0.35)]
    sy = [A["gradient_1.25_top0.25"]["ratio"] / 0.994, A["gradient_1.25"]["ratio"] / 0.994, A["gradient_1.25_top0.35"]["ratio"] / 0.994]
    ax.plot(sx, sy, "*", color="k", ms=9, label=u"求解器（梯度强度扫描，1.25 m）")
    ax.axhline(1.0, color=K.INK2, lw=0.8)
    ax.axvline(1.0, color=K.INK2, lw=0.8, ls=":")
    ax.axvline(2.635, color=K.INK2, lw=0.8, ls="--")
    ax.text(2.66, 0.76, u"现梯度场\nF = 2.635", fontsize=7, color=K.INK2)
    ax.text(1.03, 0.74, u"F = 1", fontsize=7, color=K.INK2)
    ax.plot([], [], "--", color=K.INK2, lw=1.0, label=u"直射线估计（一阶近似）")
    ax.set_xlabel(u"翻转判据 F = γ$p_0$/(2ρg)")
    ax.set_ylabel(TERMS["updown"] + u"（相对几何基线的因子）")
    ax.legend(loc="upper left", frameon=False, fontsize=7, numpoints=1)
    K.style(ax)
    K.save(fig, "fig_4_5_2")
    print("sweep F:", ["%.3f" % v for v in sx], "factors:", ["%.3f" % v for v in sy])


if __name__ == "__main__":
    main()
