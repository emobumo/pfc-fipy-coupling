# -*- coding: utf-8 -*-
"""
图 4.5-6 架空块石的绕流空区（局部模型 16 × 16 m，0.5 m，现行停注判据）。
上排：左为单块凸形块石对照（停注时充填域），右三幅为架空结构 A、B、C 首次封闭时的充填域（块石深灰，凹形空间
绿线，可达域实线）；凸形块石的绿线为块石正下方与凹形空间同大小的 16 个单元。
下左：单块凸形块石——块石正下方 16 个单元中已充填单元数与封闭分量单元数（4 邻接、8 邻接）随累计注浆量占可达
孔隙体积的变化，阴影标出逐步填满的区间（K06）。
下右：A、B、C 凹形空间（16 个单元）中绕流空区单元数随累计注浆量占可达孔隙体积的变化，C 的致密包裹体以点划线表示
（K01、K02、K05）。时刻与程度分开写，子图标题只写累计注浆量占可达孔隙体积的百分数。
2026-10-28：按版心宽 15.5 cm 原尺寸排版（不再缩放），字号不小于 6 pt；子图标题分三至四行，下右图例移到图下，图面内容不变。
数据：outputs/rerun_rate/D/boulder/final_{A,B,C}.npz（first_bypass = 首次封闭时的饱和度）与 results.jsonl
（series：每 25 步的凹形空间状态）；outputs/boulder_convex/final_convex.npz 与 results.jsonl（每 25 步，含 4、8 邻接
封闭分量）；均两遍一致。
"""
from __future__ import print_function

import json
import os

import numpy as np

import common as K
from common import plt, TERMS

D = os.path.join(K.RR, "D", "boulder")
DC = os.path.join(K.REPO, "outputs", "boulder_convex")
CELL = 0.5
NCELL = 16
NAMES = {"A": u"A 无周边疏松环", "B": u"B 周边疏松环 n = 0.45", "C": u"C 疏松环 + 凹形空间压密细粒 n = 0.03"}
Y0 = 6.0          # underside of the blocks: relative elevation 0


def panel_map(ax, f, s, mask, title, first):
    x, y = f["x"], f["y"]
    K.draw_sat(ax, x, y, s, CELL, x0=0.0, y0=Y0, solid=f["solid"])
    K.outline(ax, x, y, mask, CELL, color=K.C3, x0=0.0, y0=Y0, lw=1.2)
    K.outline(ax, x, y, f["reach"], CELL, x0=0.0, y0=Y0)
    src = np.asarray(f["src"], dtype=int)
    ax.plot(x[src], y[src] - Y0, "s", color="k", ms=3)
    ax.set_title(title, fontsize=6.5)
    ax.set_xlabel(u"水平距离 (m)", fontsize=7)
    K.style(ax); ax.tick_params(labelsize=6.5)
    ax.set_xticks([-8, -4, 0, 4, 8])
    if first:
        ax.set_ylabel(TERMS["y"] + u"\n（以块石底面为 0）", fontsize=7)
    else:
        ax.set_yticklabels([])


def main():
    res = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(D, "results.jsonl")))
    cv = json.loads(open(os.path.join(DC, "results.jsonl")).read().splitlines()[0])
    fc = np.load(os.path.join(DC, "final_convex.npz"))
    # laid out at the text-block width (15.5 cm), no scaling: maps in one row with titles over three or four lines
    H = 5.6
    fig = plt.figure(figsize=(K.TEXT_W, H))
    w, gap = 1.3, 0.13
    ax = K.axes_in(fig, 0.5, H - 1.9, w, w)
    panel_map(ax, fc, fc["s"], fc["pocket"], u"单块凸形块石（对照）\n停注时：累计注浆量占\n可达孔隙体积 %.0f%%"
              % (100 * cv["v_in"] / cv["cap"]), True)
    tname = dict(NAMES, C=u"C 疏松环 + 凹形空间\n压密细粒 n = 0.03")
    for j, tag in enumerate(("A", "B", "C")):
        f = np.load(os.path.join(D, "final_%s.npz" % tag))
        r = res[tag]
        ax = K.axes_in(fig, 0.5 + (j + 1) * (w + gap), H - 1.9, w, w)
        panel_map(ax, f, f["first_bypass"], f["pocket"], u"%s\n首次封闭：累计注浆量占\n可达孔隙体积 %.0f%%"
                  % (tname[tag], 100 * r["bypass_window_v"][0] / r["cap"]), False)
    fig.text(0.02 / K.TEXT_W, (H - 2.42) / H, u"色斑：%s；深灰：块石；绿线：%s（凸形块石为块石正下方同大小的 16 个单元）；\n实线：%s；黑方块：注浆源"
             % (TERMS["fill"], TERMS["concave"], TERMS["reach"]), fontsize=7, color=K.INK2, va="top", linespacing=1.4)

    # lower left: the convex block never encloses anything; the 16 cells below it fill from the sides
    ax = K.axes_in(fig, 0.5, 1.05, 2.2, 1.45)
    ser = cv["series"]
    v = np.array([p["v_in"] for p in ser]) / cv["cap"] * 100
    filled = np.array([p["pocket_filled"] for p in ser]) * NCELL
    lo = v[np.argmax(filled > 0)]
    hi = v[np.argmax(filled >= NCELL)]
    ax.axvspan(lo, hi, color=K.C3, alpha=0.12, lw=0)
    ax.plot(v, filled, "-", color=K.C3, lw=1.4, label=u"块石正下方 16 个单元中已充填单元数")
    ax.plot(v, [p["enc4_cells"] for p in ser], "-", color=K.INK, lw=1.2, label=u"封闭分量单元数（4 邻接）")
    ax.plot(v, [p["enc8_cells"] for p in ser], ":", color=K.C2, lw=1.6, label=u"封闭分量单元数（8 邻接）")
    ax.text(lo - 1.5, 7.0, u"%.0f%%～%.0f%%\n逐步填满" % (lo, hi), ha="right", fontsize=6.5, color=K.INK2)
    ax.set_xlabel(u"累计注浆量占可达孔隙体积 (%)", fontsize=7)
    ax.set_ylabel(u"单元数", fontsize=7)
    ax.set_xlim(0, 101); ax.set_ylim(-0.5, 20)
    ax.set_title(u"单块凸形块石：全程无封闭分量", fontsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=6)
    ax.set_ylim(-0.5, 26)
    K.style(ax); ax.tick_params(labelsize=6.5)

    # lower right: bypass-void cells in the concave space of A, B, C
    ax = K.axes_in(fig, 3.25, 1.05, 2.75, 1.45)
    cols = {"A": K.C1, "B": K.C2, "C": K.C3}
    for tag in ("A", "B", "C"):
        r = res[tag]
        ser = r["series"]
        v = np.array([p["v_in"] for p in ser]) / r["cap"] * 100
        ax.plot(v, [round(NCELL * p["pocket_bypass"]) for p in ser], "-", color=cols[tag], lw=1.4,
                label=NAMES[tag] + u"：" + TERMS["bypass"])
        if tag == "C":
            ax.plot(v, [round(NCELL * p["pocket_enclosed_unreachable"]) for p in ser], "-.", color=cols[tag], lw=1.2,
                    label=NAMES[tag] + u"：" + TERMS["dense"])
    ax.set_xlabel(u"累计注浆量占可达孔隙体积 (%)", fontsize=7)
    ax.set_ylabel(TERMS["bypass"] + u"单元数\n（%s共 16 个）" % TERMS["concave"], fontsize=7)
    ax.set_xlim(0, 101); ax.set_ylim(0, 16.5)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.3), frameon=False, fontsize=6.5)
    K.style(ax); ax.tick_params(labelsize=6.5)
    K.save(fig, "fig_4_5_6")
    print("convex: fills %.1f%%..%.1f%% of reachable pore; enclosed cells max 4/8: %d / %d"
          % (lo, hi, max(p["enc4_cells"] for p in cv["series"]), max(p["enc8_cells"] for p in cv["series"])))


if __name__ == "__main__":
    main()
