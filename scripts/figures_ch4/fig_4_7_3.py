# -*- coding: utf-8 -*-
"""
图 4.7-3 两种停注规则下的浆液分布（案例工程断面，匿名：相对高程以注浆水平为 0，I/II/III 段，无名称与编号）。
上排：按原设计估算浆量 28.6 m³/m 停止（A3–D3）；
下排：A2–C2 不设浆量上限，持续注浆至浆液到达跑浆中段；D2、E2 按式（4-29）停注（ε 0.1）；
右侧：到达倍数与注浆水平以下占比（分栏）。
数据：outputs/zhaojin_runs/final_{A3,B3,C3,D3,A2,B2,C2}.npz（A3–D3、A2–C2 两遍一致）、
outputs/rerun_rate/D/zhaojin/final_{D2,E2}.npz（速率判据，两遍一致）；倍数与占比取 outputs/zhaojin_runs/original_design.txt
与 docs/results_log.md T2 表。
"""
from __future__ import print_function

import imp
import os

import numpy as np

import common as K
from common import plt, TERMS

Z = imp.load_source("zhaojin_section", os.path.join(K.REPO, "cases", "zhaojin_section.py"))
ZR = os.path.join(K.REPO, "outputs", "zhaojin_runs")
ZD = os.path.join(K.RR, "D", "zhaojin")
CELL = 1.0
# panel labels describe the zone (no run codes on the figure); columns in zone order I, II, III
LAB = {"A": u"II 段 有交界面", "B": u"I 段 有交界面", "C": u"III 段 有交界面", "D": u"I 段 无交界面", "E": u"II 段 封底"}
TOP = [("B", os.path.join(ZR, "final_B3.npz")), ("A", os.path.join(ZR, "final_A3.npz")), ("C", os.path.join(ZR, "final_C3.npz")),
       ("D", os.path.join(ZR, "final_D3.npz"))]
BOT = [("B", os.path.join(ZR, "final_B2.npz")), ("A", os.path.join(ZR, "final_A2.npz")), ("C", os.path.join(ZR, "final_C2.npz")),
       ("D", os.path.join(ZD, "final_D2.npz")), ("E", os.path.join(ZD, "final_E2.npz"))]


def panel(ax, path):
    f = np.load(path)
    x, y, s, region = f["x"], f["y"], f["s"], f["region"]
    solid = (region == Z.ROCK) | (region == Z.SEAL)
    K.draw_sat(ax, x, y, np.where(solid, 0, s), CELL, x0=0.0, y0=Z.Y_LEVEL, solid=solid)
    K.outline(ax, x, y, region == Z.OUTLET, CELL, color="#e34948", x0=0.0, y0=Z.Y_LEVEL, lw=1.0)
    ax.axhline(0.0, color=K.INK2, lw=0.6, ls=":")
    ax.set_xticks([]); ax.tick_params(labelsize=6)


def main():
    fig = plt.figure(figsize=(11.0, 6.2))
    tops = []
    for r, row in enumerate((TOP, BOT)):
        for j, (tag, path) in enumerate(row):
            ax = fig.add_axes([0.04 + j * 0.115, 0.53 - r * 0.45, 0.105, 0.37])
            panel(ax, path)
            ax.set_title(LAB[tag], fontsize=7)
            if j == 0:
                ax.set_ylabel(u"相对高程 (m)", fontsize=8)
            else:
                ax.set_yticklabels([])
        ax.apply_aspect()
        tops.append(ax.get_position().y1 + 0.05)
    fig.text(0.04, tops[0], u"按原设计估算浆量停止", fontsize=8.5)
    fig.text(0.04, tops[1], u"不设浆量上限，持续注浆至浆液到达跑浆中段", fontsize=8.5)
    fig.text(0.04 + 3 * 0.115, tops[1], u"按式（4-29）停注", fontsize=8.5)
    fig.text(0.04, 0.005, u"色斑：%s；深灰：围岩、未采矿体与封底；红框：跑浆中段；点线：注浆水平。下排前三列不设浆量上限，持续注浆至浆液到达跑浆中段；"
             u"后两列按式（4-29）停注（ε 0.1）。\n"
             u"到达倍数 = 到达跑浆中段时的累计注入量 ÷ 估算浆量 28.6 m³/m；注浆水平以下占比 = 按估算浆量停止时注浆水平以下浆量 ÷ 储存浆量（不计出浆单元）"
             % TERMS["fill"], fontsize=7, color=K.INK2)
    names = [u"I 段\n有交界面", u"II 段\n有交界面", u"III 段\n有交界面", u"I 段\n无交界面"]
    mult = [4.51, 5.17, 4.36, None]
    below = [23.9, 8.3, 19.5, 62.6]
    ax = fig.add_axes([0.66, 0.57, 0.31, 0.33])
    xs = np.arange(4)
    ax.bar(xs[:3], mult[:3], 0.6, color=K.C2, edgecolor="white", align="center")
    for xi, v in zip(xs[:3], mult[:3]):
        ax.text(xi, v + 0.1, "%.2f" % v, ha="center", fontsize=7)
    ax.text(3, 0.2, u"不到达", ha="center", fontsize=7, color=K.INK2)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7); ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(u"到达倍数", fontsize=8); ax.set_ylim(0, 6.2)
    K.style(ax)
    ax = fig.add_axes([0.66, 0.10, 0.31, 0.33])
    ax.bar(xs, below, 0.6, color=K.C1, edgecolor="white", align="center")
    for xi, v in zip(xs, below):
        ax.text(xi, v + 1.2, "%.1f" % v, ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7); ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(u"注浆水平以下占比/%", fontsize=8); ax.set_ylim(0, 75)
    K.style(ax)
    K.save(fig, "fig_4_7_3")


if __name__ == "__main__":
    main()
