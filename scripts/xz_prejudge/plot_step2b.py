# -*- coding: utf-8 -*-
"""Figures for docs/xz_step2b.md, deterministic part (from outputs/xz_prejudge/step2b.npz written by step2b.py).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\plot_step2b.py

Writes figures/xz_step2b/fig_b1_mapping_v4, fig_b2_zones_base, fig_b3_zones_compare (.png 300 dpi, .svg).
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PL = imp.load_source("xz_plot", os.path.join(REPO, "scripts", "xz_prejudge", "plot.py"))
plt = PL.plt
from matplotlib.colors import ListedColormap

SRC = os.path.join(PL.XC.out(PL.CFG, "outputs", "xz_prejudge"), "step2b.npz")
PL.OUT = PL.XC.out(PL.CFG, "figures", "xz_step2b")
ZCOL = ["#2a78d6", "#f2b84b", "#e34948"]
ZNAME = [u"可靠区（1 MPa 可达）", u"条件区（仅 2 MPa 可达）", u"难灌区（2 MPa 不可达）"]


def view(D, k):
    return {"cls": D["cls"][k], "X": D["X"], "Z": D["Z"], "drifts": D["drifts"], "holes": D["fan"][k]}


def zone_ax(ax, D, ki, zmap, title):
    V = view(D, ki)
    PL.background(ax, V)
    img = np.where(zmap >= 0, zmap.astype(float), np.nan)
    ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=PL.ext(V), cmap=ListedColormap(ZCOL), vmin=-0.5,
              vmax=2.5, interpolation="nearest")
    ax.contour(D["X"][0], D["Z"][:, 0], D["hang"][ki].astype(float), levels=[0.5], colors=["k"], linewidths=0.6,
               linestyles="dashed")
    PL.frame(ax, V, title)


def legend(fig):
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ZCOL] + [plt.Line2D([], [], color="k", ls="--", lw=0.6),
                                                             plt.Line2D([], [], color="k", lw=1.0)]
    fig.legend(hs, ZNAME + [u"上盘侧区域边界", u"钻孔（每巷道 −30°、0°）"], loc="lower center", ncol=5, frameon=False,
               fontsize=7)


def fig1(D):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(D["ns"], D["zs"], "o", color="#52514e", ms=3.0, mfc="none", label=u"PFC，排除距空区 3$d_{50}$ 以内（原 s）")
    ax.plot(D["sm3"], D["zs"], "-", color="#9a9a9a", lw=1.0, label=u"v3：五点平均（不加权）")
    ax.plot(D["sm4"], D["zs4"], "-", color="#2a78d6", lw=1.2, label=u"v4：面积加权五点平均，剔除 < 20 m² 薄层")
    ax.plot(D["c3"] * D["sm3"], D["zst3"], "-", color="#e8743b", lw=0.8, alpha=0.45, label=u"v3 映射 $n_1$（K = 1.25）")
    ax.plot(D["c4"] * D["sm4"], D["zst4"], "s-", color="#e8743b", ms=3.5, lw=1.2,
            label=u"v4 映射 $n_1$（K = 1.25，c = %.4f；拉伸后的 s）" % float(D["c4"]))
    for z, a in zip(D["zs"], D["ar"]):
        if a < 20:
            ax.text(0.405, z, u"%.1f m²（剔除）" % a, fontsize=6, va="center", color="#52514e")
    ax.axhline(3.0, color="#888888", ls=":", lw=0.8)
    ax.axvspan(0.13, 0.29, color="#eeeeee", zorder=0)
    ax.set_xlabel(u"孔隙率（PFC 为二维面积孔隙率；灰底为 0.13～0.29）")
    ax.set_ylabel(u"距下盘接触面的法向距离 s/m")
    ax.set_xlim(0.10, 0.46)
    ax.legend(frameon=False, loc="center right", fontsize=6.5, numpoints=1)
    fig.tight_layout()
    PL.save(fig, "fig_b1_mapping_v4")


def fig2(D):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    z = D["zones"][1]
    rub = z >= 0
    zone_ax(ax, D, 1, z, u"情景一，K = 1.25：可靠区 %.0f%%，条件区 %.0f%%，难灌区 %.0f%%" % tuple(
        100.0 * (z[rub] == q).mean() for q in (0, 1, 2)))
    ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.94)
    legend(fig)
    PL.save(fig, "fig_b2_zones_base")


def fig3(D):
    fig, axs = plt.subplots(3, 2, figsize=(10.0, 9.6))
    for k in range(2):
        for ki, K in enumerate(D["ks"]):
            z = D["zones"][k * 3 + ki]
            rub = z >= 0
            ax = axs[ki, k]
            zone_ax(ax, D, ki, z, u"情景%s，K = %.2f：%.0f / %.0f / %.0f%%" % ((u"一" if k == 0 else u"二", K) + tuple(
                100.0 * (z[rub] == q).mean() for q in (0, 1, 2))))
            if ki == 2:
                ax.set_xlabel(u"X/m")
            if k == 0:
                ax.set_ylabel(u"Z/m")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.97, hspace=0.18, wspace=0.08)
    legend(fig)
    PL.save(fig, "fig_b3_zones_compare")


def main():
    D = np.load(SRC)
    for f in (fig1, fig2, fig3):
        f(D)
    return 0


if __name__ == "__main__":
    sys.exit(main())
