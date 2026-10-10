# -*- coding: utf-8 -*-
"""
图 4.4-2 解析对比三联图：(a) 一维变饱和充填锋面—时间对本模型流动律一维解析解（台阶 2b；与 Gustafson 等平行板解同类型，
非其式 9）；
(b) 径向停滞沿轴压力剖面对本模型流动律径向停滞解 max(0, p₀ − λ(r − r₀))（台阶 3）；(c) 均质点源可达域对解析椭圆。
数据：(a)(b) outputs/verification_ladder/step2b_front_time.csv、step3_axis_profile.csv（scripts/save_ladder_curves.py
运行测试夹具 _run_step2b_march、_run_step3_march 得到，两遍逐字节一致）；(c) scripts/conic_check.py 的 build/one
（只算可达域，100 × 100 m，中心点源，n = 0.30，1.25 m，含重力；与 outputs/conic_check/conic_check.txt 同一计算）。
"""
from __future__ import print_function

import hashlib
import imp
import os

import numpy as np

import common as K
from common import plt, TERMS

CC = imp.load_source("conic_check", os.path.join(K.REPO, "scripts", "conic_check.py"))
VL = os.path.join(K.REPO, "outputs", "verification_ladder")


def main():
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.9))
    a = np.loadtxt(os.path.join(VL, "step2b_front_time.csv"), delimiter=",", skiprows=1)
    lmax = 0.6
    ax = axs[0]
    ax.plot(a[:, 0], a[:, 2] / lmax, "-", color=K.C2, lw=1.4, label=u"本模型流动律一维解析解")
    ax.plot(a[::12, 0], a[::12, 1] / lmax, "o", color=K.C1, ms=3.5, label=u"数值（充填饱和度 0.5 位置）")
    ax.set_xlabel(u"时间 t (s)"); ax.set_ylabel(u"锋面位置 " + K.XF + u" / " + K.LMAX)
    ax.legend(loc="lower right", frameon=False, fontsize=7, numpoints=1)
    ax.set_title(u"(a) 一维变饱和充填：锋面位置—时间（台阶 2b）", fontsize=8)
    b = np.loadtxt(os.path.join(VL, "step3_axis_profile.csv"), delimiter=",", skiprows=1)
    ax = axs[1]
    ok = ~np.isnan(b[:, 2])
    ax.plot(b[ok, 0], b[ok, 2] / 1.0e5, "-", color=K.C2, lw=1.4, label=u"本模型流动律径向停滞解\nmax(0, $p_0$ − λ(r − $r_0$))")
    ax.plot(b[:, 0], b[:, 1] / 1.0e5, "o", color=K.C1, ms=3, label=u"数值（沿轴，停滞后）")
    ax.axvline(0.7, color=K.INK2, lw=0.8, ls=":")
    ax.text(0.71, 0.55,u"$I_{\\mathrm{max}}$ = $r_0$ + $p_0$/λ", fontsize=7, color=K.INK2)
    ax.set_xlabel(u"径向距离 r (m)"); ax.set_ylabel(u"压力 p / $p_0$")
    ax.legend(loc="upper right", frameon=False, fontsize=7, numpoints=1)
    ax.set_title(u"(b) 径向停滞：沿轴压力剖面（台阶 3）", fontsize=8)
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
    ax.plot([], [], "--", color=K.C2, lw=1.2, label=u"解析椭圆 r = " + K.LMAX + u"/(1 + " + K.PIG + u" sinθ)")
    ax.plot([], [], ":", color=K.INK2, lw=0.8, label=u"无重力解析圆")
    lim = 1.15 * r["L"] / (1 - r["Pi_g"])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(TERMS["x"]); ax.set_ylabel(u"相对注浆点高程 (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=6)
    ax.set_title(u"(c) 均质点源可达域与解析椭圆（n = 0.30）", fontsize=8)
    for x_ in axs:
        K.style(x_)
    fig.tight_layout()
    K.save(fig, "fig_4_4_2")


if __name__ == "__main__":
    main()
