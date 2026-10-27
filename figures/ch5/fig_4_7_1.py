# -*- coding: utf-8 -*-
"""
Fig. 4.7-1 idealised section of the case engineering project (section 4.7.1), first version 2026-10-27.

    powershell -File scripts\\run_local.ps1 figures\\ch5\\fig_4_7_1.py

Geometry from docs/check_4_7_prewrite.md section 5.3 (section.txt): x relative to the hole collar, z relative to the
grouting level (m). Vein dip about 70 deg, footwall contact x_fw(z) = -2.784 + 0.36397 z (the same in all three
sections); hanging-wall contact x_hw = x_fw - width, horizontal width 7.5 / 7.0 / 6.5 m (I / II / III); drawn for
II (7.0 m), the range annotated. Contact bands 2.1284 m horizontal (2 m normal) along both walls from +26 to -37 m;
stored ore -30 to 0 m; unmined ore below -30 m; runaway level x -15.887 to -13.887, z -37 to -35 against the footwall;
original-design reinforcement layer 0 to +22 m (design domain, dashed). Holes: collar (0, +2), 35 deg (bold) and
20 deg (thin), 17.5 m with a 2 m standpipe.
"""
from __future__ import print_function

import os
import sys

import numpy as np
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ch4"))
import ch4fig as F                         # noqa: E402
from ch4fig import K, plt                  # noqa: E402

FW0, DXDZ = -2.784, 0.36397
WIDTH_II, BAND = 7.0, 2.1284
ZTOP, ZDES, ZORE, ZRUN0, ZRUN1, ZBOT = 26.0, 22.0, -30.0, -35.0, -37.0, -42.0
RUN_X = (-15.887, -13.887)
FILL_C, ORE_C, BAND_C, RUN_C, UNMINED_C = "#efe2c4", "#e2c08a", "#d98b2b", "#8a4b0f", "#9e9e9e"
XL = (-31.0, 9.0)


def xfw(z):
    return FW0 + DXDZ * z


def xhw(z):
    return xfw(z) - WIDTH_II


def quad(fx0, fx1, z0, z1):
    return [(fx0(z0), z0), (fx1(z0), z0), (fx1(z1), z1), (fx0(z1), z1)]


def main():
    fig = plt.figure(figsize=(F.WIDTH * 0.72, 7.4))
    ax = fig.add_axes([0.14, 0.07, 0.83, 0.9])
    ax.add_patch(Rectangle((XL[0], ZBOT), XL[1] - XL[0], ZTOP - ZBOT, fc=F.ROCK, ec="none", zorder=0))
    ax.add_patch(Polygon(quad(xhw, xfw, 0.0, ZTOP), fc=FILL_C, ec="none", zorder=1))
    ax.add_patch(Polygon(quad(xhw, xfw, ZORE, 0.0), fc=ORE_C, ec="none", zorder=1))
    ax.add_patch(Polygon(quad(xhw, xfw, ZBOT, ZORE), fc=UNMINED_C, ec="none", zorder=1))
    for f0, f1 in ((xhw, lambda z: xhw(z) + BAND), (lambda z: xfw(z) - BAND, xfw)):
        ax.add_patch(Polygon(quad(f0, f1, ZRUN1, ZTOP), fc=BAND_C, ec="none", alpha=0.75, zorder=2))
    ax.add_patch(Rectangle((RUN_X[0], ZRUN1), RUN_X[1] - RUN_X[0], ZRUN0 - ZRUN1, fc=RUN_C, ec="none", zorder=3))
    for f in (xhw, xfw):
        ax.plot([f(ZBOT), f(ZTOP)], [ZBOT, ZTOP], color=K.INK, lw=0.7, zorder=4)
    ax.plot(XL, [ZTOP, ZTOP], color=K.INK, lw=0.9, zorder=4)
    ax.add_patch(Polygon(quad(lambda z: xhw(z) + 0.3, lambda z: xfw(z) - 0.3, 0.15, ZDES - 0.15), fill=False, ec=K.INK, lw=F.LW_DESIGN, ls="dashed", zorder=5))
    F.draw_hole(ax, (0.0, 2.0), (-16.445, 7.985), 2.0, lw_pipe=0.6, lw_section=1.1)
    F.draw_hole(ax, (0.0, 2.0), (-14.335, 12.038), 2.0)
    ax.text(-14.0, 13.0, u"35°", fontsize=7.5, ha="right", va="bottom")
    ax.text(-16.6, 7.6, u"20°", fontsize=7.5, ha="right", va="top")
    kw = dict(fontsize=7.5, zorder=9)
    lk = dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1)
    ax.annotate(u"充填体\n孔隙率 I 0.10 / II 0.45 / III 0.25", (xfw(18) - 3.5, 18), xytext=(-30.0, 21.5),
                arrowprops=lk, va="center", **kw)
    ax.annotate(u"原设计加固层（设计域）", (xhw(11) + 0.2, 11), xytext=(-30.0, 15.0), arrowprops=lk, va="center", **kw)
    ax.annotate(u"存窿矿石", (xfw(-15) - 3.5, -15), xytext=(-30.0, -12.0), arrowprops=lk, va="center", **kw)
    ax.annotate(u"交界面（$n_c$ 基准 0.35）", (xfw(-25) - 1.0, -25), xytext=(-8.0, -21.0), arrowprops=lk,
                va="center", **kw)
    ax.annotate("", (xhw(-25) + 1.0, -25), xytext=(-8.2, -21.0), arrowprops=lk)
    ax.annotate(u"跑浆中段\n$n$ = 0.60", (RUN_X[1], -36.0), xytext=(-8.0, -38.5), arrowprops=lk, va="center", **kw)
    ax.annotate(u"未采矿体", (xfw(-38) - 3.5, -38.5), xytext=(-30.0, -39.5), arrowprops=lk, va="center", **kw)
    ax.text(-30.0, 5.0, u"上盘", fontsize=8)
    ax.text(3.5, -10.0, u"下盘", fontsize=8)
    ax.text(-30.0, ZBOT + 0.6, u"原岩与未采矿体：孔隙率 10$^{-3}$", fontsize=7)
    z = 6.0
    ax.annotate("", (xhw(z), z), xytext=(xfw(z), z), arrowprops=dict(arrowstyle="<->", lw=0.6, color=K.INK,
                                                                      shrinkA=0, shrinkB=0), zorder=8)
    ax.text(xfw(z) + 0.5, z - 0.2, u"水平宽度 6.5～7.5 m\n（I、II、III 段）", fontsize=7, va="center", zorder=9,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.0))
    ax.plot([0.0], [2.0], "s", color="k", ms=3, zorder=7)
    ax.text(0.6, 1.2, u"孔口", fontsize=7.5, va="top")
    for zz in (ZDES, 0.0, ZORE, ZRUN0, ZRUN1):
        ax.plot(XL, [zz, zz], color=F.GREY, lw=0.4, zorder=0.5)
    ax.set_yticks([ZTOP, ZDES, 0.0, ZORE, ZRUN0, ZRUN1])
    ax.set_yticklabels([u"+26", u"+22", u"0", u"−30", u"−35", u"−37"])
    ax.set_xlim(XL); ax.set_ylim(ZBOT, ZTOP + 1.5); ax.set_aspect("equal")
    ax.set_xlabel(u"水平距离 / m（以孔口为 0）", fontsize=8)
    ax.set_ylabel(u"相对高程 / m（注浆水平为 0）", fontsize=8)
    ax.tick_params(labelsize=7)
    K.style(ax)
    F.save(fig, "fig_4_7_1", out=HERE)


if __name__ == "__main__":
    main()
