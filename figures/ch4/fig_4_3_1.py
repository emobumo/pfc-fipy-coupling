# -*- coding: utf-8 -*-
"""
Fig. 4.3-1 permeability, threshold pressure gradient and stagnation distance against porosity (analytic).

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_3_1.py

Eqs. (4-7)-(4-9): k = A n^3/(1-n)^2, lambda = 2 tau0 sqrt(n/(8k)), L_max = p0/lambda; A = 1.25e-7 m2, tau0 = 30 Pa,
p0 = 5 MPa, n 0.05-0.50. (a) lambda and k each divided by their value at n = 0.10, log axis, ratios at n = 0.45
annotated; (b) L_max with points at n = 0.10 / 0.18 / 0.30 / 0.45. No solver output.
"""
from __future__ import print_function

import numpy as np

import ch4fig as F
from ch4fig import K, plt

A, TAU0, P0 = 1.25e-7, 30.0, 5.0e6
MARKS = [0.10, 0.18, 0.30, 0.45]


def k(n):
    return A * n ** 3 / (1 - n) ** 2


def lam(n):
    return 2 * TAU0 * np.sqrt(n / (8 * k(n)))


def lmax(n):
    return P0 / lam(n)


def main():
    n = np.linspace(0.05, 0.50, 400)
    rk, rl = k(0.45) / k(0.10), lam(0.10) / lam(0.45)
    fig = plt.figure(figsize=(F.WIDTH, 2.7))
    ax = fig.add_axes([0.085, 0.16, 0.38, 0.70])
    ax.semilogy(n, k(n) / k(0.10), color=K.C1, lw=1.5, label=u"渗透率 $k/k_{0.10}$")
    ax.semilogy(n, lam(n) / lam(0.10), color=K.C2, lw=1.5, label=u"启动压力梯度 $\\lambda/\\lambda_{0.10}$")
    ax.axhline(1.0, color=F.GREY, lw=0.6, zorder=0)
    ax.plot([0.10], [1.0], "o", color=K.INK, ms=3.5, zorder=4)
    ax.plot([0.45], [rk], "o", color=K.C1, ms=4)
    ax.plot([0.45], [1 / rl], "o", color=K.C2, ms=4)
    ax.annotate(u"k 增大至 %.0f 倍" % rk, (0.45, rk), xytext=(-6, 6), textcoords="offset points", ha="right",
                fontsize=7.5, color=K.C1)
    ax.annotate(u"λ 减小至 1/%.1f" % rl, (0.45, 1 / rl), xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=7.5, color=K.C2)
    ax.yaxis.set_ticks_position("left")
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(0.02, 1000)
    ax.set_xlabel(u"孔隙率 n")
    ax.set_ylabel(u"相对 n = 0.10 时的值")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    K.style(ax)
    F.tag(ax, "a")
    ax = fig.add_axes([0.6, 0.16, 0.38, 0.70])
    ax.plot(n, lmax(n), color=K.C3, lw=1.5)
    for m in MARKS:
        ax.plot([m], [lmax(m)], "o", color=K.C3, ms=4)
        ax.annotate(u"%.1f m" % lmax(m), (m, lmax(m)), xytext=(5, -9) if m < 0.4 else (-6, 3), textcoords="offset points", ha="left" if m < 0.4 else "right",
                    fontsize=7.5)
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(0, 90)
    ax.set_xlabel(u"孔隙率 n")
    ax.set_ylabel(u"停滞距离 $L_{\\mathrm{max}}$ / m")
    K.style(ax)
    F.tag(ax, "b")
    F.save(fig, "fig_4_3_1")
    print("k ratio %.2f, lambda ratio %.3f, L_max" % (rk, rl), ["%.2f" % lmax(m) for m in MARKS])


if __name__ == "__main__":
    main()
