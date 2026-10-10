# -*- coding: utf-8 -*-
"""
图 4.6-2 段长准则图（1.25 m，现行停注判据）。两格：
(a) 各段停注时 x_f 对 L_max（过原点最小二乘斜率 c，按 19 个独立段：n 0.14 的 7 m 第 1 段在 4 个方案中
是同一计算，只计一次）；
(b) 后续段按 Δ/x_f 分为后续段有效 / 后续段失效（纵轴为后续段注入量）。
数据：outputs/rerun_rate/C/results.json（P2、P3，两遍一致；stages[].x_f、v_in、reason）。
Δ 从上一有效段孔底起算（本组均为 7 m 第 1 段之后的第 2 段）。
第二轮意见：删去原 (c) 格（x_f/L_max 随 ε）；分图标签不用算例代号。
第五轮（框架 v1.7）：c 按 19 个独立段拟合（0.871），图注写 x_f ≈ 0.87 L_max，不写三位小数。
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
    xf, lm, seen = [], [], set()
    for r in C:
        for st in r["stages"]:
            if st["reason"] == "rate" and st["v_in"] > 1.0 and not (r["n_ref"] == 0.12 and st["k"] == 2):
                key = (r["n_ref"], r["depths"][st["k"] - 1], st["k"], round(st["x_f"], 9), round(st["v_in"], 9))
                if key in seen:          # same computation repeated in another plan
                    continue
                seen.add(key)
                xf.append(st["x_f"]); lm.append(r["L_max"])
    xf, lm = np.array(xf), np.array(lm)
    c = float(np.sum(xf * lm) / np.sum(lm * lm))
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.9))
    ax = axs[0]
    ax.plot(lm, xf, "o", color=K.C1, ms=5, mec="white", mew=0.6)
    xx = np.array([0, 25.0])
    ax.plot(xx, c * xx, "-", color=K.INK2, lw=1)
    ax.plot(xx, xx, ":", color=K.INK2, lw=0.8)
    ax.text(15, c * 15 - 3.2, K.XF + u" ≈ 0.87 " + K.LMAX, fontsize=8)
    ax.text(17.5, 19.6, K.XF + u" = " + K.LMAX, fontsize=7, color=K.INK2)
    ax.set_xlabel(K.LMAX + u" (m)"); ax.set_ylabel(u"停注时 " + K.XF + u" (m)")
    ax.set_title(u"(a) 各段 " + K.XF + u" 与 " + K.LMAX + u"（%d 段）" % len(xf), fontsize=9)
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
            ax.annotate(u"孔隙率 %.2f，孔深 %g m" % (r["n_ref"], r["H"]), (ratio, s2["v_in"]), xytext=(5, 3),
                        textcoords="offset points", fontsize=6)
    ax.axvline(1.0, color=K.INK2, lw=0.8, ls="--")
    ax.plot([], [], "o", color=K.C1, label=TERMS["stage_ok"])
    ax.plot([], [], "x", color=K.C2, mew=1.4, label=TERMS["stage_dead"])
    ax.plot([], [], "D", color=K.C3, label=u"临界（新出浆段仅剩 1 个未胶结单元）")
    ax.legend(loc="upper left", frameon=False, fontsize=7, numpoints=1)
    ax.set_xlabel(u"Δ / " + K.XF); ax.set_ylabel(u"后续段注入量 (m³/m)")
    ax.set_xlim(0, 2.6)
    ax.set_title(u"(b) 后续段注入量与 Δ/" + K.XF, fontsize=9)
    for a in axs:
        K.style(a)
    fig.tight_layout()
    K.save(fig, "fig_4_6_2")
    print("c = %.4f; x_f/L_max %.3f..%.3f" % (c, (xf / lm).min(), (xf / lm).max()))


if __name__ == "__main__":
    main()
