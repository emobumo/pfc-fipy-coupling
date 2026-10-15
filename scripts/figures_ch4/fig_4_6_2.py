# -*- coding: utf-8 -*-
"""
图 4.6-2 段长判据图（1.25 m，现行停注判据）。两格：
(a) 各段停注时 x_f 对 L_max（过原点最小二乘斜率 c，按 19 个独立段：n 0.14 的 7 m 第 1 段在 4 个方案中
是同一计算，只计一次；SG01）；
(b) 后续段注入量对 Δ/x_f，图例为后续段有效 / 后续段失效 / 临界。横轴按未修约的 x_f 计算（S2 约 0.97）。
数据：outputs/rerun_rate/C/results.json（P2、P3，两遍一致；stages[].x_f、v_in、reason）。
Δ 从上一有效段孔底起算（本组均为第 1 段之后的第 2 段）。
第二轮意见：删去原 (c) 格（x_f/L_max 随 ε）。
第五轮（框架 v1.7）：c 按 19 个独立段拟合（0.871），图注写 x_f ≈ 0.87 L_max，不写三位小数。
第十轮（框架 v2.7）：坐标写“中文名 符号/单位”；点标注用表 4.6-1 的算例代号与“终孔孔深”。
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
    ax.set_xlabel(u"停滞距离 " + K.LMAX + u"/m"); ax.set_ylabel(u"停注锋面距离 " + K.XF + u"/m")
    ax.set_title(u"(a) 各段 " + K.XF + u" 与 " + K.LMAX + u"（%d 段）" % len(xf), fontsize=9)
    ax = axs[1]
    pts = []
    for r in C:
        code = K.s_code(r)
        if code is None:
            continue
        s1, s2 = r["stages"][0], r["stages"][1]
        ratio = (r["depths"][1] - r["depths"][0]) / s1["x_f"]
        ok = s2["v_in"] > 1.0
        marg = s2["buried"] > 0 and ok and s2["buried"] >= s2["src_cells"] - 1
        mk, col = ("D", K.C3) if marg else (("o", K.C1) if ok else ("x", K.C2))
        ax.plot([ratio], [s2["v_in"]], mk, color=col, ms=6, mew=1.4)
        if ok:
            ax.annotate(u"%s（终孔孔深 %g m）" % (code, r["H"]), (ratio, s2["v_in"]), xytext=(6, 2),
                        textcoords="offset points", fontsize=6.5)
        pts.append((code, ratio, s2["v_in"]))
    ax.axvline(1.0, color=K.INK2, lw=0.8, ls="--")
    ax.plot([], [], "o", color=K.C1, label=TERMS["stage_ok"])
    ax.plot([], [], "x", color=K.C2, mew=1.4, label=TERMS["stage_dead"])
    ax.plot([], [], "D", color=K.C3, label=u"临界")
    ax.legend(loc="upper left", frameon=False, fontsize=7, numpoints=1)
    ax.set_xlabel(u"Δ/" + K.XF); ax.set_ylabel(u"后续段注入量/(m$^3$·m$^{-1}$)")
    ax.set_xlim(0, 2.6)
    ax.set_title(u"(b) 后续段注入量与 Δ/" + K.XF, fontsize=9)
    for a in axs:
        K.style(a)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.015, u"(b) 横轴按未修约的 " + K.XF + u" 计算（S2 约 0.97）；Δ 为新出浆段伸出第 1 段孔底的长度；"
             u"算例代号见表 4.6-1；临界指 Δ 与 " + K.XF + u" 相差不足一个单元", fontsize=7.5, color=K.INK2)
    K.save(fig, "fig_4_6_2")
    print("c = %.6f over %d segments; x_f/L_max %.6f..%.6f" % (c, len(xf), (xf / lm).min(), (xf / lm).max()))
    for p in sorted(pts, key=lambda p: int(p[0][1:])):
        print("  %-3s Delta/x_f %.4f  stage-2 V_in %.6f" % p)


if __name__ == "__main__":
    main()
