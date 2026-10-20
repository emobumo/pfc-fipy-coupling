# -*- coding: utf-8 -*-
"""Figures for docs/pfc_collapse_analysis.md (from outputs/pfc_collapse/*.npz written by analyze.py).

    powershell -File scripts\\run_local.ps1 scripts\\pfc_collapse\\plot.py

Writes figures/pfc_collapse/fig_pc_{1..7}.png (300 dpi) and .svg. Porosity is the 2D area porosity of discs.
"""
from __future__ import print_function

import imp
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import ListedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XC = imp.load_source("xz_config", os.path.join(REPO, "scripts", "xz_config", "xz_config.py"))
CFG = XC.get()
SRC = XC.out(CFG, "outputs", "pfc_collapse")
OUT = XC.out(CFG, "figures", "pfc_collapse")
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans SC", "SimHei"],
                            "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10,
                            "legend.fontsize": 8, "savefig.dpi": 300, "svg.fonttype": "none"})
NLAB = u"二维面积孔隙率"
C_P, C_B = "#2a78d6", "#e8743b"
STOPE = XC.stope_polygon(CFG)
VIEW = tuple(CFG["view"]["pfc"])


def save(fig, name):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    print("wrote", name)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(direction="out", labelsize=8)


def stope(ax):
    ax.add_patch(Polygon(STOPE, closed=True, fill=False, ls="dashed", lw=0.9, ec="k"))


def overlay(ax, F, mask, color, alpha=1.0):
    box, h = F["box"], float(F["h"])
    ax.imshow(np.ma.masked_where(~mask, mask.astype(float)), origin="lower", extent=list(box),
              cmap=ListedColormap([color]), vmin=0, vmax=1, alpha=alpha, interpolation="nearest")


def fig1(F, B):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    x, y, r = B["x"], B["y"], B["r"]
    sel = (x > VIEW[0] - 1) & (x < VIEW[1] + 1) & (y > VIEW[2] - 1) & (y < VIEW[3] + 1)
    frag = B["frag"]
    cnt = np.bincount(frag[frag >= 0])
    cols = {1: "#8a8a8a", 2: "#c9c9c9", 3: "#d9c58f"}
    groups = [(sel & ~B["rub"] & (B["side"] == k), cols[k]) for k in (1, 2, 3)]
    single = B["rub"] & (cnt[np.maximum(frag, 0)] == 1)
    groups += [(sel & B["rub"] & ~single, "#a0522d"), (sel & single, "#f2a65a")]
    for m, col in groups:
        pc = PatchCollection([Circle((a, b_), c) for a, b_, c in zip(x[m], y[m], r[m])], facecolor=col, edgecolor="none", rasterized=True)
        ax.add_collection(pc)
    overlay(ax, F, F["void"], "#9ec5f4", 0.85)
    stope(ax)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
    ax.set_xlabel(u"x/m"); ax.set_ylabel(u"y/m（模型坐标）")
    h = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ("#8a8a8a", "#c9c9c9", "#d9c58f", "#a0522d", "#f2a65a", "#9ec5f4")]
    ax.legend(h, [u"下盘（未动）", u"上盘（未动）", u"矿柱（未动）", u"散体：粘结碎块（≥2 颗）", u"散体：单颗粒",
                  u"残余空区（开运算半径 $d_{50}$）"], loc="upper left", frameon=False, fontsize=7)
    ax.text(VIEW[1] - 1, VIEW[2] + 1, u"虚线：原采场轮廓；终态 = " + CFG["pfc"]["final_label"], ha="right", fontsize=7)
    style(ax)
    fig.tight_layout()
    save(fig, "fig_pc_1_final_state")


def fig2(F):
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    d50 = float(F["d50"])
    for rv, col, lab in ((F["rve_p"], C_P, u"颗粒尺度"), (F["rve_b"], C_B, u"块体尺度")):
        axs[0].loglog(rv[:, 1], rv[:, 4], "o-", color=col, ms=4, label=lab)
        axs[1].plot(rv[:, 1], rv[:, 3], "o-", color=col, ms=4, label=lab)
    L = np.array([0.75, 12.0])
    axs[0].loglog(L, 0.11 * (L / 0.75) ** -1.0, ":", color="k", lw=0.8, label=u"斜率 −1（无相关）")
    for ax in axs:
        ax.axvline(8 * d50, color="#888888", lw=0.8, ls="--")
        ax.set_xlabel(u"窗口边长 L/m（$d_{50}$ = %.2f m）" % d50)
        style(ax)
    axs[0].set_xlim(0.5, 20)
    axs[0].text(8 * d50 * 1.05, 0.2, u"选定分箱\n8 $d_{50}$ = 6 m", fontsize=7)
    axs[0].set_ylabel(u"窗口孔隙率标准差（%s）" % NLAB)
    axs[1].set_ylabel(u"窗口孔隙率平均值（%s）" % NLAB)
    axs[0].legend(frameon=False, loc="lower left")
    axs[0].set_title(u"(a) 标准差与窗口尺寸", fontsize=9)
    axs[1].set_title(u"(b) 平均值与窗口尺寸", fontsize=9)
    axs[1].set_ylim(0, 0.25)
    fig.tight_layout()
    save(fig, "fig_pc_2_rve")


def fig3(F):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    box, BIN = F["box"], float(F["bin"])
    nby, nbx = F["field_p"].shape
    ext = [box[0], box[0] + nbx * BIN, box[2], box[2] + nby * BIN]
    for ax, fld, tt in ((axs[0], F["field_p"], u"(a) 颗粒尺度"), (axs[1], F["field_b"], u"(b) 块体尺度（碎块内部计为实体）")):
        im = ax.imshow(np.ma.masked_invalid(fld), origin="lower", extent=ext, cmap="YlGnBu", vmin=0.0, vmax=0.30,
                       interpolation="nearest")
        overlay(ax, F, F["void"], "#9ec5f4", 0.6)
        overlay(ax, F, F["rock"], "#d0d0d0", 0.5)
        stope(ax)
        ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
        ax.set_title(tt, fontsize=9); ax.set_xlabel(u"x/m"); style(ax)
    axs[0].set_ylabel(u"y/m")
    cb = fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.02)
    cb.set_label(NLAB + u"（6 m 分箱）", fontsize=8)
    fig.text(0.01, 0.01, u"只画散体区占箱面积 ≥ 50% 的箱；浅蓝：残余空区；浅灰：围岩与矿柱；虚线：原采场轮廓", fontsize=7, color="#52514e")
    save(fig, "fig_pc_3_field")


def fig4(P):
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, key, xl in ((axs[0], "y", u"高程 y/m"), (axs[1], "s", u"距下盘接触面的法向距离 s/m")):
        for sc, col, lab in (("particle", C_P, u"颗粒尺度"), ("block", C_B, u"块体尺度")):
            z, n = P["%s_%s_all_z" % (sc, key)], P["%s_%s_all_n" % (sc, key)]
            a0, b1, r2 = float(P["%s_%s_all_a0" % (sc, key)]), float(P["%s_%s_all_b1" % (sc, key)]), float(P["%s_%s_all_r2" % (sc, key)])
            ax.plot(n, z, "o", color=col, ms=3.5, label=u"%s（R² %.2f）" % (lab, r2))
            ax.plot(a0 + b1 * z, z, "-", color=col, lw=1.0)
        ax.set_xlabel(NLAB + u"（1.5 m 薄层）"); ax.set_ylabel(xl); ax.set_xlim(0, 0.4)
        ax.legend(frameon=False, loc="upper right" if key == "y" else "center right", fontsize=7, numpoints=1); style(ax)
    axs[1].axhline(14.0, color="#888888", lw=0.8, ls="--")
    axs[1].text(0.25, 14.4, u"原采场顶板 s = 14 m", fontsize=7, color="#52514e")
    axs[0].set_title(u"(a) 沿高度", fontsize=9); axs[1].set_title(u"(b) 垂直于矿层", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_pc_4_profiles")


def fig5(Bd):
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    names = [str(q) for q in Bd["names"]]
    lab = {"footwall": u"下盘", "hanging wall": u"上盘", "ore pillar": u"矿柱", "free surface": u"空区边", "interior": u"内部"}
    xt, xl = [], []
    pos = 0
    for side in ("footwall", "hanging wall", "ore pillar", "free surface", "interior"):
        idx = [i for i, q in enumerate(names) if q == side]
        for i in idx:
            ax.bar(pos - 0.2, Bd["n_p"][i], 0.4, color=C_P, align="center")
            ax.bar(pos + 0.2, Bd["n_b"][i], 0.4, color=C_B, align="center")
            xt.append(pos)
            xl.append(lab[side] + (u"\n%d～%d" % (Bd["k"][i] - 1, Bd["k"][i]) if Bd["k"][i] else u"\n> 3"))
            pos += 1
        pos += 0.6
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7); ax.set_xlim(-0.8, pos - 0.8)
    ax.set_xlabel(u"条带（距接触面，以 $d_{50}$ 计）", fontsize=8)
    ax.set_ylabel(NLAB)
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=C_P), plt.Rectangle((0, 0), 1, 1, fc=C_B)], [u"颗粒尺度", u"块体尺度"],
              frameon=False, loc="upper right")
    ax.set_title(u"距接触面 0～1、1～2、2～3 倍 $d_{50}$ 的条带与内部（$d_{50}$ = 0.75 m）", fontsize=9)
    style(ax)
    fig.tight_layout()
    save(fig, "fig_pc_5_contact_bands")


def fig6(F, B):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    box, BIN = F["box"], float(F["bin"])
    nby, nbx = F["d50f"].shape
    ext = [box[0], box[0] + nbx * BIN, box[2], box[2] + nby * BIN]
    im = axs[0].imshow(np.ma.masked_invalid(F["d50f"]), origin="lower", extent=ext, cmap="PuOr_r", vmin=0.70, vmax=0.82,
                       interpolation="nearest")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.02); cb.set_label(u"颗粒 $d_{50}$/m（6 m 分箱）", fontsize=8)
    im2 = axs[1].imshow(np.ma.masked_invalid(F["fbin"]), origin="lower", extent=ext, cmap="magma_r" if hasattr(plt.cm, "magma_r") else "YlOrBr",
                        vmin=0, vmax=18, interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=axs[1], fraction=0.04, pad=0.02); cb2.set_label(u"所在碎块等效直径中位数/m", fontsize=8)
    for ax, tt in ((axs[0], u"(a) 颗粒直径"), (axs[1], u"(b) 碎块尺寸（粘结颗粒团）")):
        overlay(ax, F, F["void"], "#9ec5f4", 0.6)
        stope(ax)
        ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
        ax.set_title(tt, fontsize=9); ax.set_xlabel(u"x/m"); style(ax)
    axs[0].set_ylabel(u"y/m")
    fig.tight_layout()
    save(fig, "fig_pc_6_size")


def fig7(F):
    E = F["evo"]
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))
    t = E[:, 1]
    axs[0].plot(t, E[:, 4], "o-", color=C_P, ms=4, label=u"颗粒尺度")
    axs[0].plot(t, E[:, 5], "s-", color=C_B, ms=4, label=u"块体尺度")
    axs[0].plot(t, E[:, 6], "^--", color=C_P, ms=4, label=u"颗粒尺度，内部")
    axs[0].set_ylabel(u"散体区平均" + NLAB); axs[0].set_ylim(0, 0.25); axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    axs[1].plot(t, E[:, 7], "o-", color=C_P, ms=4, label=u"颗粒尺度")
    axs[1].plot(t, E[:, 9], "s-", color=C_B, ms=4, label=u"块体尺度")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel(u"竖向梯度 dn/dy（1/m）"); axs[1].legend(frameon=False, fontsize=7)
    axs[2].semilogy(t, E[:, 12], "o-", color="#52514e", ms=4)
    axs[2].set_ylabel(u"速度 > 0.1 m/s 的颗粒数")
    for ax, tt in zip(axs, (u"(a) 平均孔隙率", u"(b) 竖向梯度", u"(c) 运动颗粒")):
        ax.set_xlabel(u"开挖后模型时间/s"); ax.set_title(tt, fontsize=9); style(ax)
    fig.tight_layout()
    save(fig, "fig_pc_7_time")


def main():
    F = np.load(os.path.join(SRC, "fields_final.npz"))
    B = np.load(os.path.join(SRC, "balls_final.npz"))
    P = np.load(os.path.join(SRC, "profiles.npz"))
    Bd = np.load(os.path.join(SRC, "bands.npz"))
    fig1(F, B)
    fig2(F)
    fig3(F)
    fig4(P)
    fig5(Bd)
    fig6(F, B)
    fig7(F)
    return 0


if __name__ == "__main__":
    sys.exit(main())
