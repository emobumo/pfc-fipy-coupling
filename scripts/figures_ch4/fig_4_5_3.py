# -*- coding: utf-8 -*-
"""
图 4.5-3 效应分解柱图：上下扩散距离比，四种组合（几何、只有重力、只有结构、两者），可达域与求解器并列，
直射线估计以标记叠加。1.25 m，单元中心读数；柱上数值按表 4.4-2 写两位小数。
数据：可达域 outputs/gravity_control/gravity_control.json（1.25 m 行；第二遍一致见 outputs/rerun_reach/）；
求解器 outputs/rerun_rate/A/results.json（两遍一致）；直射线取孔中点、不计左右边界的估计
（outputs/nonmult_check/nonmult_check.txt 的 mid_free，锚点 NM03、R13：1.0000 / 0.8979 / 1.3784 / 1.2152）；
51 点取最大的估计（R12）在只有结构与两者时被左边界截断，不用。
"""
from __future__ import print_function

import json
import os
import re

import numpy as np

import common as K
from common import plt, TERMS

NAMES = {"uniform, gravity": ("uniform", True), "gradient, no gravity": ("gradient", False),
         "uniform, no gravity": ("uniform", False), "gradient, gravity": ("gradient", True)}


def rays():
    out = {}
    for line in open(os.path.join(K.REPO, "outputs", "nonmult_check", "nonmult_check.txt")):
        m = re.match(r"\s+(uniform, gravity|gradient, no gravity|uniform, no gravity|gradient, gravity)\s+max\s+[\d.]+ "
                     r"\(clipped \w+\)\s+mid_wall\s+[\d.]+\s+mid_free\s+([\d.]+)", line)
        if m:
            out[NAMES[m.group(1)]] = float(m.group(2))
    assert len(out) == 4, out
    return out


def main():
    reach = json.load(open(os.path.join(K.REPO, "outputs", "gravity_control", "gravity_control.json")))["reach"]
    rq = dict(((r["kind"], r["gravity"]), r["ratio"]) for r in reach if r["cell"] == 1.25)
    A = dict((r["label"], r) for r in json.load(open(os.path.join(K.RR, "A", "results.json"))))
    ray = rays()
    combos = [(u"几何\n（均质、无重力）", ("uniform", False), "uniform_1.25_nograv"),
              (u"只有重力\n（均质）", ("uniform", True), "uniform_1.25"),
              (u"只有结构\n（梯度、无重力）", ("gradient", False), "gradient_1.25_nograv"),
              (u"两者\n（梯度、有重力）", ("gradient", True), "gradient_1.25")]
    x = np.arange(len(combos))
    w = 0.36
    rv = [rq[c[1]] for c in combos]
    sv = [A[c[2]]["ratio"] for c in combos]
    yv = [ray[c[1]] for c in combos]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar(x - w / 2 - 0.01, rv, w, color=K.C1, edgecolor="white", label=TERMS["reach"] + u"（路径积分）")
    b2 = ax.bar(x + w / 2 + 0.01, sv, w, color=K.C2, edgecolor="white", label=u"求解器（停注时充填域）")
    ax.plot(x, yv, "kD", ms=5, label=u"直射线估计（孔中点，不计左右边界）")
    for xi, a, b in zip(x, rv, sv):
        ax.text(xi - w / 2 - 0.01, a + 0.015, "%.2f" % a, ha="center", fontsize=7)
        ax.text(xi + w / 2 + 0.01, b + 0.015, "%.2f" % b, ha="center", fontsize=7)
    ax.axhline(1.0, color=K.INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in combos], fontsize=8)
    ax.set_ylim(0.7, 1.5)
    ax.set_ylabel(TERMS["updown"])
    ax.legend(loc="upper left", frameon=False, numpoints=1)
    K.style(ax)
    K.save(fig, "fig_4_5_3")
    print("reach", rv, "solver", sv, "ray", yv)


if __name__ == "__main__":
    main()
