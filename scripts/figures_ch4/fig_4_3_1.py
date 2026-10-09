# -*- coding: utf-8 -*-
"""
图 4.3-1 渗透率 k、启动压力梯度 λ 与停滞距离 L_max 随孔隙率 n 的变化（对数纵轴，三栏共用横轴）。
闭式（参数集 v0.6）：k = A·n³/(1−n)²，A = 1.25e-7 m²；λ = 6.00e4(1−n)/n Pa/m；L_max = 83.33·n/(1−n) m（p₀ = 5 MPa）。
标 n = 0.10 / 0.18 / 0.30 / 0.45；n 由 0.10 增至 0.45：k 升 244 倍，λ 只降 7.4 倍。
"""
from __future__ import print_function

import numpy as np

import common as K
from common import plt

A = 1.25e-7


def k(n):
    return A * n ** 3 / (1 - n) ** 2


def lam(n):
    return 6.00e4 * (1 - n) / n


def lmax(n):
    return 83.33 * n / (1 - n)


def main():
    n = np.linspace(0.08, 0.50, 300)
    marks = [0.10, 0.18, 0.30, 0.45]
    fig, axs = plt.subplots(3, 1, figsize=(5.2, 6.4), sharex=True)
    for ax, f, lab, col, fmt in ((axs[0], k, u"渗透率 k (m²)", K.C1, "%.2e"),
                                 (axs[1], lam, u"启动压力梯度 λ (Pa/m)", K.C2, "%.3g"),
                                 (axs[2], lmax, u"停滞距离 " + K.LMAX + u" (m)（$p_0$ = 5 MPa）", K.C3, "%.1f")):
        ax.semilogy(n, f(n), color=col, lw=1.6)
        for m in marks:
            ax.plot([m], [f(m)], "o", color=col, ms=5)
            ax.annotate(fmt % f(m), (m, f(m)), xytext=(4, -10 if f is lam else 4), textcoords="offset points", fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        K.style(ax)
    axs[0].text(0.22, k(0.12), u"n 0.10 → 0.45：k 升 %.0f 倍" % (k(0.45) / k(0.10)), fontsize=8)
    axs[1].text(0.22, lam(0.12), u"n 0.10 → 0.45：λ 只降 %.1f 倍" % (lam(0.10) / lam(0.45)), fontsize=8)
    axs[2].set_xlabel(u"孔隙率 n")
    for ax in axs:
        for m in marks:
            ax.axvline(m, color="#e4e3dd", lw=0.8, zorder=0)
    fig.tight_layout()
    K.save(fig, "fig_4_3_1")
    print("k ratio %.1f, lambda ratio %.2f" % (k(0.45) / k(0.10), lam(0.10) / lam(0.45)))


if __name__ == "__main__":
    main()
