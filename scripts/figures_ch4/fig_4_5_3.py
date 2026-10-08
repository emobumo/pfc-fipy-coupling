# -*- coding: utf-8 -*-
"""
图 4.5-3 效应分解柱图：上下扩散距离比，四种组合（几何、只有重力、只有结构、两者），可达域与求解器并列，
直射线估计以标记叠加。1.25 m。
数据：可达域 outputs/gravity_control/gravity_control.json（1.25 m 行；第二遍一致见 outputs/rerun_reach/）；
求解器 outputs/rerun_rate/A/results.json（两遍一致）；直射线 outputs/flip_criterion/flip_criterion.txt
组 A 的"孔中点自由射线"列（忽略左右边界；"各点最大"列在只有结构时被左边界截断，见 results_log V2）。
"""
from __future__ import print_function

import json
import os
import re

import numpy as np

import common as K
from common import plt, TERMS


def main():
    reach = json.load(open(os.path.join(K.REPO, "outputs", "gravity_control", "gravity_control.json")))["reach"]
    rq = dict(((r["kind"], r["gravity"]), r["ratio"]) for r in reach if r["cell"] == 1.25)
    A = dict((r["label"], r) for r in json.load(open(os.path.join(K.RR, "A", "results.json"))))
    ray = {}
    for line in open(os.path.join(K.REPO, "outputs", "flip_criterion", "flip_criterion.txt")):
        m = re.match(r"(uniform|gradient)\s+(on|off)\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
        if m and (m.group(1), m.group(2) == "on") not in ray:
            ray[(m.group(1), m.group(2) == "on")] = float(m.group(3))
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
    ax.plot(x, yv, "kD", ms=5, label=u"直射线估计（一阶近似）")
    for xi, a, b in zip(x, rv, sv):
        ax.text(xi - w / 2 - 0.01, a + 0.015, "%.3f" % a, ha="center", fontsize=7)
        ax.text(xi + w / 2 + 0.01, b + 0.015, "%.3f" % b, ha="center", fontsize=7)
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
