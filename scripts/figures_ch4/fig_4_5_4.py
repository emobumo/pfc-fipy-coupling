# -*- coding: utf-8 -*-
"""
图 4.5-4 通道位置与孔隙率的交叉对比（2.5 m，给定注浆量 185.86 m³/m 停止）。
(a) 三个位置（孔口、孔中、孔底外，φ 0.45）的充填域，叠设计域（虚线）与可达域（实线）；
(b) 给定注浆量的注入时间、目标区充填率与通道入口累积启动压力，左列随位置（φ 0.45）、右列随孔隙率（孔底外），
    两列共用"孔底外 φ 0.45"一点。分栏而不用双纵轴。
数据：outputs/rerun_rate/B/<label>/cells.csv 与 results.json（两遍一致）；设计域 outputs/rerun_rate/A/target_mask.npy；
入口累积启动压力 outputs/channel_entry_cost/entry_cost_1.25.txt（1.25 m，第二遍一致）。
目标区充填率按"设计域内除孔单元"计（与框架表一致）。
"""
from __future__ import print_function

import imp
import json
import os
import re

import numpy as np

import common as K
from common import plt, TERMS

G = imp.load_source("inclined_hole_gradient", os.path.join(K.REPO, "cases", "inclined_hole_gradient.py"))
CELL = 2.5
B = os.path.join(K.RR, "B")
POS = [(u"孔口", "B_band_collar_phi0.45", "band_collar_x-28.75"),
       (u"孔中", "B_band_midhole_phi0.45", "band_midhole_x-23.75"),
       (u"孔底外", "B_band_toe_phi0.45", "band_toe_x-11.25")]
PHI = [(0.30, "B_band_toe_phi0.30", "band_phi0.30"), (0.45, "B_band_toe_phi0.45", "band_toe_x-11.25"),
       (0.60, "B_band_toe_phi0.60", "band_phi0.60")]


def entry_costs():
    d = {}
    for line in open(os.path.join(K.REPO, "outputs", "channel_entry_cost", "entry_cost_1.25.txt")):
        m = re.match(r"(band_\S+)\s+.*?\s+(\S+)\s+([\d.]+)\s+\d+\s+\d+", line)
        if m:
            d[m.group(1)] = float(m.group(3))
    return d


def main():
    res = dict((r["label"], r) for r in json.load(open(os.path.join(B, "results.json"))))
    target = np.load(os.path.join(K.RR, "A", "target_mask.npy"))
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    mx, my = np.asarray(x), np.asarray(y)
    hole = np.asarray(G.hole_cells(mx, my, 24, 24), dtype=int)
    notsrc = np.ones(mx.size, dtype=bool); notsrc[hole] = False
    end = G.hole_geometry()[0]
    ent = entry_costs()

    def tgt(label):
        a = np.loadtxt(os.path.join(B, label, "cells.csv"), delimiter=",", skiprows=1)
        return 100.0 * np.sum((a[:, 3] >= 0.5) & target & notsrc) / np.sum(target & notsrc), a

    fig = plt.figure(figsize=(9.0, 8.6))
    for j, (name, lab, _) in enumerate(POS):
        ax = fig.add_axes([0.06 + j * 0.31, 0.56, 0.28, 0.38])
        _, a = tgt(lab)
        im = K.draw_sat(ax, a[:, 0], a[:, 1], a[:, 3], CELL)
        band = a[:, 2] > 0.3
        K.outline(ax, a[:, 0], a[:, 1], band, CELL, color=K.C2, lw=0.8)
        K.outline(ax, a[:, 0], a[:, 1], target, CELL, dashed=True)
        K.outline(ax, a[:, 0], a[:, 1], a[:, 5] > 0.5, CELL)
        K.draw_hole(ax, G.MOUTH, end)
        ax.set_title(u"(a%d) 通道位于%s（φ 0.45）" % (j + 1, name), fontsize=9)
        ax.set_xlabel(TERMS["x"]); K.style(ax)
        if j == 0:
            ax.set_ylabel(TERMS["y"])
    fig.text(0.06, 0.505, u"虚线：%s；实线：%s；橙线：通道；色斑：%s" % (TERMS["design"], TERMS["reach"], TERMS["fill"]),
             fontsize=8, color=K.INK2)
    rows = [("t", TERMS["given_t"] + u" (s)"), ("tgt", TERMS["design_fill"] + u" (%)"), ("ent", TERMS["entry_p"] + u" / $p_0$")]
    for col, (series, xlab) in enumerate(((POS, u"通道位置（φ 0.45）"), (PHI, u"通道孔隙率（孔底外）"))):
        xs = range(len(series))
        vals = {"t": [res[s[1]]["t"] for s in series], "tgt": [tgt(s[1])[0] for s in series],
                "ent": [ent[s[2]] for s in series]}
        for r, (key, ylab) in enumerate(rows):
            ax = fig.add_axes([0.10 + col * 0.46, 0.33 - r * 0.135, 0.38, 0.10])
            c = K.C1 if col == 0 else K.C3
            ax.plot(xs, vals[key], "-o", color=c, ms=5)
            shared = 2 if col == 0 else 1
            ax.plot([shared], [vals[key][shared]], "o", ms=9, mfc="none", mec="k")
            for xi, v in zip(xs, vals[key]):
                ax.annotate(("%.0f" if key == "t" else "%.1f" if key == "tgt" else "%.3f") % v, (xi, v),
                            xytext=(6, 2), textcoords="offset points", fontsize=7)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([s[0] if col == 0 else "%.2f" % s[0] for s in series] if r == 2 else [], fontsize=8)
            ax.set_xlim(-0.4, 2.6)
            pad = 0.15 * (max(vals[key]) - min(vals[key]) + 1e-9)
            ax.set_ylim(min(vals[key]) - pad - (0.02 if key == "ent" else 0), max(vals[key]) + 2 * pad)
            ax.set_ylabel(ylab, fontsize=7)
            K.style(ax)
            if r == 2:
                ax.set_xlabel(xlab)
            if r == 0:
                ax.set_title(u"(b%d) 随%s变化（圆圈为两组共用的孔底外 φ 0.45）" % (col + 1, u"位置" if col == 0 else u"孔隙率"), fontsize=8)
    K.save(fig, "fig_4_5_4")


if __name__ == "__main__":
    main()
