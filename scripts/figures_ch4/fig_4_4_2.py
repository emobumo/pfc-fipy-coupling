# -*- coding: utf-8 -*-
"""
图 4.4-2 解析对比三联图：(a) 台阶 2b 锋面—时间对 Gustafson–Stille；(b) 台阶 3 径向停滞剖面；(c) 点源可达域对解析椭圆。
(c)：scripts/conic_check.py 的 build/one（只算可达域，100 × 100 m，中心点源，n = 0.30，1.25 m，含重力）——与
outputs/conic_check/conic_check.txt 同一计算（第二遍一致）。
(a)(b)：仓库未保存这两条曲线的数据（只在 tests/test_verification_ladder.py 的 _run_step2b_march、_run_step3_march
运行时存在），按规格"不自行补算"，暂留占位，见 INDEX。
"""
from __future__ import print_function

import hashlib
import imp
import math
import os

import numpy as np

import common as K
from common import plt, TERMS

CC = imp.load_source("conic_check", os.path.join(K.REPO, "scripts", "conic_check.py"))


def main():
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, t in zip(axs[:2], (u"(a) 一维变饱和充填：锋面位置—时间\n（台阶 2b，对 Gustafson–Stille 解析曲线）",
                               u"(b) 径向停滞：沿轴压力剖面\n（台阶 3，对 max(0, p₀ − λr) 解析线）".replace(u"p₀", u"$p_0$"))):
        ax.text(0.5, 0.5, u"数据待补\n（仓库未保存，需运行验证阶梯测试夹具）", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=K.INK2)
        ax.set_title(t, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    r = CC.one(0.30, 1.25, True, hashlib.md5())
    ax = axs[2]
    mx, my, src = r["mx"], r["my"], r["src"]
    x0, y0 = mx[src], my[src]
    K.outline(ax, mx, my, r["reach"], 1.25, x0=x0, y0=y0, lw=1.4)
    th = np.linspace(0, 2 * np.pi, 400)
    rr = r["L"] / (1 + r["Pi_g"] * np.sin(th))
    ax.plot(rr * np.cos(th), rr * np.sin(th), color=K.C2, lw=1.2, ls="--")
    ax.plot(r["L"] * np.cos(th), r["L"] * np.sin(th), color=K.INK2, lw=0.8, ls=":")
    ax.plot([0], [0], "k+", ms=8, mew=1.4)
    ax.plot([], [], "-", color=K.INK, lw=1.4, label=TERMS["reach"] + u"（二维，16 邻域）")
    ax.plot([], [], "--", color=K.C2, lw=1.2, label=u"解析椭圆 r = L_max/(1 + Π_g sinθ)")
    ax.plot([], [], ":", color=K.INK2, lw=0.8, label=u"无重力解析圆")
    lim = 1.15 * r["L"] / (1 - r["Pi_g"])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(TERMS["x"]); ax.set_ylabel(u"相对注浆点高程 (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=6)
    ax.set_title(u"(c) 均质点源可达域与解析椭圆（n = 0.30）\n上 %.2f / 下 %.2f m（解析 %.2f / %.2f）" % (
        r["up"][0], r["down"][0], r["up"][1], r["down"][1]), fontsize=8)
    for a in axs:
        K.style(a)
    fig.tight_layout()
    K.save(fig, "fig_4_4_2")
    print("Pi_g %.4f up %s down %s sectors %.4f..%.4f" % (r["Pi_g"], r["up"], r["down"], r["sector_min"], r["sector_max"]))


if __name__ == "__main__":
    main()
