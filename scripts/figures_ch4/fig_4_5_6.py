# -*- coding: utf-8 -*-
"""
图 4.5-6 架空块石的绕流空区（局部模型 16 × 16 m，0.5 m，现行停注判据）。
上排：三变体首次形成封闭绕流空区时的充填域（块石深灰，口袋绿线，可达域实线）；
下：口袋中绕流空区（及致密包裹体）比例随累计注浆量（占可达孔隙体积）的变化。
数据：outputs/rerun_rate/D/boulder/final_{A,B,C}.npz（first_bypass = 首次封闭时的饱和度）与 results.jsonl
（series：每 25 步的口袋状态）；两遍一致。
框架中的左侧小示意（单块凸形块石的敞开缺口）属示意图，未画。
"""
from __future__ import print_function

import json
import os

import numpy as np

import common as K
from common import plt, TERMS

D = os.path.join(K.RR, "D", "boulder")
CELL = 0.5
NAMES = {"A": u"A 无周边疏松环", "B": u"B 周边疏松环 n = 0.45", "C": u"C 疏松环 + 块石下压密细粒 n = 0.03"}
Y0 = 6.0          # pocket floor: relative elevation 0


def main():
    res = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(D, "results.jsonl")))
    fig = plt.figure(figsize=(9.0, 6.6))
    for j, tag in enumerate(("A", "B", "C")):
        f = np.load(os.path.join(D, "final_%s.npz" % tag))
        ax = fig.add_axes([0.06 + j * 0.31, 0.47, 0.27, 0.46])
        x, y = f["x"], f["y"]
        im = K.draw_sat(ax, x, y, f["first_bypass"], CELL, x0=0.0, y0=Y0, solid=f["solid"])
        K.outline(ax, x, y, f["pocket"], CELL, color=K.C3, x0=0.0, y0=Y0, lw=1.2)
        K.outline(ax, x, y, f["reach"], CELL, x0=0.0, y0=Y0)
        src = np.asarray(f["src"], dtype=int)
        ax.plot(x[src], y[src] - Y0, "s", color="k", ms=3)
        r = res[tag]
        ax.set_title(u"%s\n首次封闭：累计注浆量 %.1f m³/m（%.0f%%）" % (NAMES[tag], r["bypass_window_v"][0],
                                                       100 * r["bypass_window_v"][0] / r["cap"]), fontsize=8)
        ax.set_xlabel(u"水平距离 (m)"); K.style(ax)
        if j == 0:
            ax.set_ylabel(TERMS["y"] + u"（以口袋底为 0）")
    fig.text(0.06, 0.385, u"色斑：%s（首次封闭时）；深灰：块石；绿线：块石下口袋；实线：%s；黑方块：注浆源" % (TERMS["fill"], TERMS["reach"]),
             fontsize=8, color=K.INK2)
    ax = fig.add_axes([0.09, 0.07, 0.86, 0.26])
    cols = {"A": K.C1, "B": K.C2, "C": K.C3}
    for tag in ("A", "B", "C"):
        r = res[tag]
        ser = r["series"]
        v = np.array([p["v_in"] for p in ser]) / r["cap"] * 100
        ax.plot(v, [100 * p["pocket_bypass"] for p in ser], "-", color=cols[tag], lw=1.4, label=NAMES[tag] + u"：" + TERMS["bypass"])
        if tag == "C":
            ax.plot(v, [100 * p["pocket_enclosed_unreachable"] for p in ser], "--", color=cols[tag], lw=1.2,
                    label=NAMES[tag] + u"：" + TERMS["dense"])
    ax.set_xlabel(u"累计注浆量（占可达孔隙体积，%）")
    ax.set_ylabel(u"占口袋比例 (%)")
    ax.set_xlim(0, 101); ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    K.style(ax)
    K.save(fig, "fig_4_5_6")


if __name__ == "__main__":
    main()
