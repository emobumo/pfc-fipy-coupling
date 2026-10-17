# -*- coding: utf-8 -*-
"""Figures for docs/xz_prejudge_step1.md (from outputs/xz_prejudge/prejudge.npz written by prejudge.py).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\plot.py

Writes figures/xz_prejudge/fig_xz_{1..6}_*.png (300 dpi) and .svg. Coordinates X, Z relative to the footwall
corner of the original stope's lower end.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "outputs", "xz_prejudge", "prejudge.npz")
OUT = os.path.join(REPO, "figures", "xz_prejudge")
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans SC", "SimHei"],
                            "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9,
                            "legend.fontsize": 7, "savefig.dpi": 300, "svg.fonttype": "none"})
CELL = 0.25
COLORS = ["#f2a65a", "#8a8a8a", "#c9c9c9", "#d9c58f", "#cfe3fa", "#3d3d3d"]
VIEW = (-32.0, 72.0, -4.0, 60.0)


def save(fig, name):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    print("wrote", name)


def ext(D):
    X, Z = D["X"], D["Z"]
    return [X.min() - CELL / 2, X.max() + CELL / 2, Z.min() - CELL / 2, Z.max() + CELL / 2]


def frame(ax, D, title=None, holes=True):
    for d in D["drifts"]:
        ax.add_patch(Rectangle((d[0] - 1.25, d[1] - 1.25), 2.5, 2.5, fc=COLORS[5], ec="none"))
    if holes:
        for h in D["holes"]:
            ax.plot([h[0], h[2]], [h[1], h[3]], "-", color="k", lw=1.0)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.tick_params(labelsize=7)


def background(ax, D):
    cls = D["cls"]
    rock = np.ma.masked_where(~((cls >= 1) & (cls <= 3)), np.ones(cls.shape))
    void = np.ma.masked_where(cls != 4, np.ones(cls.shape))
    ax.imshow(rock, origin="lower", extent=ext(D), cmap=ListedColormap(["#bdbdbd"]), interpolation="nearest")
    ax.imshow(void, origin="lower", extent=ext(D), cmap=ListedColormap(["#cfe3fa"]), interpolation="nearest")


def fig1(D):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cls = D["cls"].astype(float)
    ax.imshow(cls, origin="lower", extent=ext(D), cmap=ListedColormap(COLORS),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6), interpolation="nearest")
    ax.contour(D["X"][0], D["Z"][:, 0], D["cav"].astype(float), levels=[0.5], colors=["k"], linewidths=0.8)
    cav = D["cav"]
    xs, yt = D["xs"], D["ytop"]
    col = cav.any(axis=0)
    zc = D["Z"][:, 0]
    ztop = np.array([zc[cav[:, i]].max() + CELL / 2 if col[i] else np.nan for i in range(len(xs))])
    ax.plot(xs[col], np.minimum(yt[col], ztop[col]), "-", color="#a0522d", lw=0.9)
    for dk, ls in zip(D["dks"], (":", "-", "--")):
        ax.plot(xs[col], np.minimum(yt[col] + dk, ztop[col]), ls, color="#1f4e9c", lw=1.0)
    ax.contour(D["X"][0], D["Z"][:, 0], D["basal"].astype(float), levels=[0.5], colors=["#7a3b9e"], linewidths=0.7)
    frame(ax, D)
    for i, h in enumerate(D["holes"]):
        ax.text(h[0] + 1.6, h[1] - 0.6, u"孔 %d" % (i + 1), fontsize=7)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in COLORS]
    hs += [plt.Line2D([], [], color="k", lw=0.8), plt.Line2D([], [], color="#a0522d"),
           plt.Line2D([], [], color="#1f4e9c", ls=":"), plt.Line2D([], [], color="#1f4e9c"),
           plt.Line2D([], [], color="#1f4e9c", ls="--"), plt.Line2D([], [], color="#7a3b9e")]
    ax.legend(hs, [u"散体（K = 1.25）", u"下盘", u"上盘", u"矿柱", u"残余空区（无流）", u"巷道（不透浆）", u"腔体轮廓",
                   u"PFC 散体顶面", u"散体顶面 K = 1.15", u"K = 1.25（基准）", u"K = 1.40", u"下盘松散底层（s ≤ 3 m）"],
              loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
    fig.tight_layout()
    save(fig, "fig_xz_1_geometry")


def fig2(D):
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, key, tt in ((axs[0], "n1", u"(a) 情景一：沿倾向无趋势"), (axs[1], "n2", u"(b) 情景二：沿倾向上松下密（截断 0.10～0.35）")):
        background(ax, D)
        n = np.where(D["cls"] == 0, D[key], np.nan)
        im = ax.imshow(np.ma.masked_invalid(n), origin="lower", extent=ext(D), cmap="YlGnBu", vmin=0.10, vmax=0.35,
                       interpolation="nearest")
        frame(ax, D, tt)
        ax.set_xlabel(u"X/m")
    axs[0].set_ylabel(u"Z/m")
    cb = fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.02); cb.set_label(u"映射孔隙率 n", fontsize=8)
    save(fig, "fig_xz_2_porosity")


def fig3(D):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(D["ns"], D["zs"], "o", color="#888888", ms=3.5, label=u"PFC 颗粒尺度（1.5 m 薄层，原 s）")
    ax.plot(D["sm"], D["zs"], "-", color="#2a78d6", lw=1.2, label=u"五点滑动平均（原 s）")
    ax.plot(D["mapped"], D["zst"], "s-", color="#e8743b", ms=3.5, lw=1.2, label=u"映射值 $n_s(s)$（拉伸后的 s）")
    for zt, a in zip(D["zst"], D["ar"]):
        if a < 40:
            ax.text(0.36, zt, u"%.0f m²" % a, fontsize=6, va="center", color="#52514e")
    ax.axhline(3.0, color="#888888", ls=":", lw=0.8)
    ax.text(0.31, 3.3, u"s = 3 m（以下不拉伸）", fontsize=6.5, color="#52514e")
    ax.set_xlabel(u"孔隙率（PFC 为二维面积孔隙率）"); ax.set_ylabel(u"距下盘接触面的法向距离 s/m")
    ax.set_xlim(0.10, 0.40)
    ax.legend(frameon=False, loc="center right", fontsize=7, numpoints=1)
    fig.tight_layout()
    save(fig, "fig_xz_3_mapping")


def fig4(D):
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.0))
    k = 1e6 / (2 * 1830.0 * 9.81)
    for ax, key, tt in ((axs[0], "gam1", u"(a) 情景一"), (axs[1], "gam2", u"(b) 情景二")):
        background(ax, D)
        g = np.where(D["cls"] == 0, D[key] * k, np.nan)
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext(D), cmap="RdBu_r", vmin=-4, vmax=4,
                       interpolation="nearest")
        frame(ax, D, tt + u"：局部 F/$p_0$（每 MPa），3 m 窗口")
        ax.set_xlabel(u"X/m")
    axs[0].set_ylabel(u"Z/m")
    cb = fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.02)
    cb.set_label(u"F/$p_0$（1/MPa）；$p_0$ = 2 MPa 时乘 2", fontsize=7)
    save(fig, "fig_xz_4_F")


def fig5(D):
    C = D["cost"]
    nh = len(D["holes"])
    fig, axs = plt.subplots(nh, 2, figsize=(10.0, 2.55 * nh + 0.6))
    for k in range(2):
        for hi in range(nh):
            ax = axs[hi, k]
            background(ax, D)
            c = np.where(D["cls"] == 0, C[k * nh + hi] / 1e6, np.nan)
            cm = np.where(np.isfinite(c), np.minimum(np.where(np.isfinite(c), c, 0.0), 4.0), np.nan)
            im = ax.imshow(np.ma.masked_invalid(cm), origin="lower", extent=ext(D), cmap="Purples_r", vmin=0, vmax=4,
                           interpolation="nearest")
            cc = np.where(np.isfinite(c), c, 99.0)
            ax.contour(D["X"][0], D["Z"][:, 0], cc, levels=[1.0, 2.0], colors=["#1b9e77", "#e8743b"], linewidths=1.0)
            frame(ax, D, u"情景%s，孔 %d" % (u"一" if k == 0 else u"二", hi + 1), holes=False)
            h = D["holes"][hi]
            ax.plot([h[0], h[2]], [h[1], h[3]], "-", color="k", lw=1.2)
            if hi == nh - 1:
                ax.set_xlabel(u"X/m")
            if k == 0:
                ax.set_ylabel(u"Z/m")
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.07, top=0.97, hspace=0.3, wspace=0.08)
    fig.legend([plt.Line2D([], [], color="#1b9e77"), plt.Line2D([], [], color="#e8743b")],
               [u"$p_\mathrm{req}$ = 1 MPa（$p_0$ = 1 MPa 可达域边界）", u"$p_\mathrm{req}$ = 2 MPa（$p_0$ = 2 MPa 可达域边界）"],
               loc="lower center", ncol=2, frameon=False, fontsize=7)
    cax = fig.add_axes([0.9, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(u"所需源压 $p_\mathrm{req}$/MPa（截至 4）；除以 $p_0$ 即 $p_\mathrm{req}/p_0$", fontsize=7)
    save(fig, "fig_xz_5_preq")


def fig6(D):
    U = D["union"]
    fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.0))
    cmap = ListedColormap(["#e34948", "#2a78d6"])
    for k in range(2):
        for j, p0 in enumerate((1, 2)):
            ax = axs[j, k]
            background(ax, D)
            u = U[k * 2 + j]
            img = np.where(D["cls"] == 0, u.astype(float), np.nan)
            ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=ext(D), cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.contour(D["X"][0], D["Z"][:, 0], D["hang"].astype(float), levels=[0.5], colors=["k"], linewidths=0.6,
                       linestyles="dashed")
            frame(ax, D, u"情景%s，$p_0$ = %d MPa：并集 %.0f%%" % (u"一" if k == 0 else u"二", p0,
                                                             100.0 * u[D["cls"] == 0].mean()))
            if j == 1:
                ax.set_xlabel(u"X/m")
            if k == 0:
                ax.set_ylabel(u"Z/m")
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc="#2a78d6"), plt.Rectangle((0, 0), 1, 1, fc="#e34948"),
                plt.Line2D([], [], color="k", ls="--", lw=0.6)],
               [u"各孔单独作用的可达域并集", u"并集之外的散体", u"上盘侧区域边界（局部厚度 2/3 处）"],
               loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96, hspace=0.2, wspace=0.08)
    save(fig, "fig_xz_6_union")


def main():
    D = np.load(SRC)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(D)
    return 0


if __name__ == "__main__":
    sys.exit(main())
