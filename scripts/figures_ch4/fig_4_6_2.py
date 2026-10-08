# -*- coding: utf-8 -*-
"""
图 4.6-2 段长准则图（1.25 m，现行停注判据）。
(a) 各段停注时 x_f 对 L_max（过原点拟合斜率 c）；
(b) 后续段按 Δ/x_f 分类：后续段有效 / 后续段失效（纵轴为后续段注浆量）；
(c) x_f/L_max 随停注阈值 ε（ε = 0.05 只有一个单遍敏感性运行）。
数据：outputs/rerun_rate/C/results.json（P2、P3，两遍一致；stages[].x_f、v_in、reason）；
ε = 0.05：outputs/rerun_rate/E/results.json（单遍）。
Δ 从上一有效段孔底起算（本组均为 7 m 第 1 段之后的第 2 段）。
"""
from __future__ import print_function

import json
import os

import numpy as np

import common as K
from common import plt, TERMS


def main():
    C = json.load(open(os.path.join(K.RR, "C", "results.json")))
    C = [r for r in C if r["group"] in ("p2", "p3phi", "p3len")]
    xf, lm = [], []
    for r in C:
        for st in r["stages"]:
            if st["reason"] == "rate" and st["v_in"] > 1.0 and not (r["n_ref"] == 0.12 and st["k"] == 2):
                xf.append(st["x_f"]); lm.append(r["L_max"])
    xf, lm = np.array(xf), np.array(lm)
    c = float(np.sum(xf * lm) / np.sum(lm * lm))
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax = axs[0]
    ax.plot(lm, xf, "o", color=K.C1, ms=5, mec="white", mew=0.6)
    xx = np.array([0, 25.0])
    ax.plot(xx, c * xx, "-", color=K.INK2, lw=1)
    ax.plot(xx, xx, ":", color=K.INK2, lw=0.8)
    ax.text(15, c * 15 - 3.2, u"x_f = %.3f L_max" % c, fontsize=8)
    ax.text(18.5, 19.4, u"x_f = L_max", fontsize=7, color=K.INK2)
    ax.set_xlabel(u"L_max (m)"); ax.set_ylabel(u"停注时 x_f (m)")
    ax.set_title(u"(a) 各段 x_f 与 L_max（%d 段）" % len(xf), fontsize=9)
    ax = axs[1]
    for r in C:
        if len(r["stages"]) < 2:
            continue
        s1, s2 = r["stages"][0], r["stages"][1]
        ratio = (r["depths"][1] - r["depths"][0]) / s1["x_f"]
        ok = s2["v_in"] > 1.0
        marg = s2["buried"] > 0 and ok and s2["buried"] >= s2["src_cells"] - 1
        mk, col = ("D", K.C3) if marg else (("o", K.C1) if ok else ("x", K.C2))
        ax.plot([ratio], [s2["v_in"]], mk, color=col, ms=6, mew=1.4)
        if ok:
            ax.annotate(u"φ%.2f H%g" % (r["n_ref"], r["H"]), (ratio, s2["v_in"]), xytext=(4, 3), textcoords="offset points", fontsize=6)
    ax.axvline(1.0, color=K.INK2, lw=0.8, ls="--")
    ax.plot([], [], "o", color=K.C1, label=TERMS["stage_ok"])
    ax.plot([], [], "x", color=K.C2, mew=1.4, label=TERMS["stage_dead"])
    ax.plot([], [], "D", color=K.C3, label=u"临界（新出浆段仅剩 1 个未胶结单元）")
    ax.legend(loc="upper left", frameon=False, fontsize=7, numpoints=1)
    ax.set_xlabel(u"Δ / x_f"); ax.set_ylabel(u"后续段注浆量 (m³/m)")
    ax.set_title(u"(b) 后续段有效与失效", fontsize=9)
    ax = axs[2]
    r01 = xf / lm
    ax.plot(np.full(r01.size, 0.10), r01, "o", color=K.C1, ms=4, alpha=0.7, label=u"ε = 0.1（%d 段，两遍一致）" % r01.size)
    E = [r for r in json.load(open(os.path.join(K.RR, "E", "results.json"))) if "stages" in r]
    e05 = E[0]["stages"][0]["x_f_over_L"]
    ref = [r for r in C if r["label"].startswith("C_p2/phi0.18_H17_c1.25/7_17")][0]["stages"][0]["x_f_over_L"]
    ax.plot([0.05], [e05], "s", color=K.C2, ms=6, label=u"ε = 0.05（φ 0.18 7→17 第 1 段，单遍）")
    ax.plot([0.05, 0.10], [e05, ref], "-", color=K.C2, lw=0.8)
    ax.set_xlim(0.0, 0.13); ax.set_ylim(0.75, 1.0)
    ax.set_xlabel(u"停注阈值 ε"); ax.set_ylabel(u"x_f / L_max")
    ax.legend(loc="lower left", frameon=False, fontsize=7, numpoints=1)
    ax.set_title(u"(c) x_f/L_max 随 ε", fontsize=9)
    for a in axs:
        K.style(a)
    fig.tight_layout()
    K.save(fig, "fig_4_6_2")
    print("c = %.4f; x_f/L_max %.3f..%.3f; eps0.05 %.3f vs %.3f" % (c, r01.min(), r01.max(), e05, ref))


if __name__ == "__main__":
    main()
