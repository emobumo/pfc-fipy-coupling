# -*- coding: utf-8 -*-
"""
Fig. 4.6-1 (final fig. 5-7) staged grouting: (a) sequence schematic, (b) inter-stage shadow of S3. 2026-10-28.

    powershell -File scripts\\run_local.ps1 figures\\ch5\\fig_4_6_1.py

(a) made-up dimensionless section, no solver, as fig. 4.3-3: homogeneous rubble lambda = 1, rho g = 0.08, p0 = 5,
p_req from the multi-source least cumulative integral of lambda ds over the outflow section (ch4fig.geodesic), fill =
{p_req <= 0.8 p0}. Four steps (section 5.2.1): stage 1 drilled to d1 and grouted at constant pressure; its fill
hardens into cement (impassable); the hole is reamed and drilled on to d2; stage 2 grouted, its new outflow section
d1..d2 emitting only where it is outside the cement. Delta = d2 - d1 (new outflow section beyond the previous stage
bottom) and x_f = largest projection on the hole axis of the stage-1 fill cells within one cell of the axis, minus d1
(section 5.2.2); drawn with Delta > x_f (later stage effective).
(b) the S3 map of scripts/figures_ch4/fig_4_6_1c.py (data of the rate-rule run, no solver).
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ch4"))
import ch4fig as F                         # noqa: E402
from ch4fig import K, plt                  # noqa: E402

C = imp.load_source("fig_4_6_1c", os.path.join(K.REPO, "scripts", "figures_ch4", "fig_4_6_1c.py"))

DX = 0.05
XL, ZL = (-1.5, 16.0), (-6.0, 12.5)
COLLAR, ANGLE, PIPE, D1, D2 = (0.0, 0.0), 30.0, 1.0, 4.5, 11.0
RHOG, P0, FILL = 0.08, 5.0, 0.8


def segment_mask(X, Z, s0, s1):
    c = np.array(COLLAR)
    u = np.array([np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE))])
    a, b = c + s0 * u, c + s1 * u
    t = np.clip(((X - a[0]) * (b - a)[0] + (Z - a[1]) * (b - a)[1]) / np.dot(b - a, b - a), 0, 1)
    return np.hypot(X - (a[0] + t * (b - a)[0]), Z - (a[1] + t * (b - a)[1])) <= 0.75 * DX


def p_req(lam, src, Z):
    zmax = Z.max()
    d = F.geodesic(lam, DX, src, offset=RHOG * (zmax - Z))
    return d - RHOG * zmax + RHOG * Z


def show(ax, ext, layer, color, zorder):
    ax.imshow(np.ma.masked_where(~layer, np.ones(layer.shape)), origin="lower", extent=ext,
              cmap=ListedColormap([color]), interpolation="nearest", zorder=zorder)


def show_fill(ax, ext, fill, pr):
    s = np.clip(1 - 0.45 * (pr / (FILL * P0)) ** 6, 0, 1)
    ax.imshow(np.ma.masked_where(~fill, s), origin="lower", extent=ext, cmap=K.SAT_CMAP, vmin=0, vmax=1.7,
              interpolation="nearest", zorder=2)   # lightened: schematic patches


def dim(ax, p, q, nrm, off, text):
    """Dimension line parallel to the hole from p to q, offset `off` along the unit normal nrm, with thin extension
    lines back to the axis (grey: readable on the cement and on the fill)."""
    a, b = p + off * nrm, q + off * nrm
    for e0, e1 in ((p, a), (q, b)):
        ax.plot([e0[0], e1[0] + 0.4 * np.sign(off) * nrm[0]], [e0[1], e1[1] + 0.4 * np.sign(off) * nrm[1]], "-",
                color=F.GREY, lw=0.6, zorder=8)
    ax.annotate("", b, xytext=a, zorder=8,
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=K.INK, mutation_scale=6, shrinkA=0, shrinkB=0))
    m = 0.5 * (a + b) + 0.9 * np.sign(off) * nrm
    ax.text(m[0], m[1], text, fontsize=8, ha="center", va="center", zorder=9,
            bbox=dict(fc="w", ec="none", alpha=0.85, pad=0.6))


def note(ax, text, xy, xytext, ha="center"):
    ax.annotate(text, xy, xytext=xytext, fontsize=7, ha=ha, va="center", zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1))


def frame(ax):
    ax.set_xlim(XL); ax.set_ylim(ZL); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def sequence(axs):
    x = np.arange(XL[0], XL[1] + 1e-9, DX)
    z = np.arange(ZL[0], ZL[1] + 1e-9, DX)
    X, Z = np.meshgrid(x, z)
    ext = [x[0] - DX / 2, x[-1] + DX / 2, z[0] - DX / 2, z[-1] + DX / 2]
    c = np.array(COLLAR)
    u = np.array([np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE))])
    nrm = np.array([-u[1], u[0]])
    wall = X < 0
    lam = np.where(wall, np.inf, 1.0)
    pr1 = p_req(lam, segment_mask(X, Z, PIPE, D1), Z)
    cement = (pr1 <= FILL * P0) & ~wall
    lam2 = lam.copy()
    lam2[cement] = np.inf
    src2 = segment_mask(X, Z, D1, D2) & ~cement
    pr2 = p_req(lam2, src2, Z)
    fill2 = (pr2 <= FILL * P0) & ~wall & ~cement
    s_ax = (X - c[0]) * u[0] + (Z - c[1]) * u[1]
    near = np.abs((X - c[0]) * nrm[0] + (Z - c[1]) * nrm[1]) < DX
    xf = s_ax[cement & near].max() - D1
    if not D2 - D1 > xf:
        raise SystemExit("drawn case must have Delta > x_f")
    p1, p2 = c + D1 * u, c + D2 * u
    titles = (u"① 钻第 1 段并定压出浆", u"② 浆液凝固为胶结体", u"③ 扫孔、钻进第 2 段", u"④ 第 2 段定压出浆")
    for i, ax in enumerate(axs):
        show(ax, ext, wall, F.ROCK, 1)
        if i == 0:
            show_fill(ax, ext, cement, pr1)
            F.draw_hole(ax, c, p1, PIPE)
            note(ax, u"充填域", tuple(c + 2.5 * u - 2.6 * nrm), (8.5, -4.6))
            note(ax, u"孔口管", tuple(c + 0.5 * PIPE * u), (2.6, 7.5))
            note(ax, u"第 1 段出浆段", tuple(c + 0.5 * (PIPE + D1) * u), (8.5, 7.5))
        else:
            show(ax, ext, cement, F.SOLID, 3)
        if i == 1:
            F.draw_hole(ax, c, p1, D1, halo=True)     # cemented hole: no outflow
            note(ax, K.TERMS["cement"], tuple(c + 2.5 * u - 3.0 * nrm), (8.5, -4.6))
        if i >= 2:
            F.draw_hole(ax, c, p2, PIPE, section_from=D1, halo=True)
            ax.plot([p1[0]], [p1[1]], marker="o", ms=3.5, mfc="white", mec="k", mew=0.9, zorder=7)
        if i == 2:
            note(ax, u"第 1 段孔底", tuple(p1), (6.0, -4.6))
            note(ax, u"第 2 段出浆段", tuple(c + 0.5 * (D1 + D2) * u), (4.0, 9.6))
        if i == 3:
            show_fill(ax, ext, fill2, pr2)
            dim(ax, p1, p1 + xf * u, nrm, 5.0, u"$x_f$")
            dim(ax, p1, p2, nrm, -5.0, u"$\\Delta$")
        frame(ax)
        ax.set_title(titles[i], fontsize=7.5)
    return xf


def main():
    fig = plt.figure(figsize=(F.WIDTH, 5.9))
    w, gap = 0.235, 0.02
    axs = [fig.add_axes([0.01 + i * (w + gap), 0.645, w, 0.30]) for i in range(4)]
    xf = sequence(axs)
    for i in range(3):
        b0, b1 = axs[i].get_position(), axs[i + 1].get_position()
        y = b0.y0 + 0.5 * b0.height
        fig.text(0.5 * (b0.x1 + b1.x0), y, u"→", ha="center", va="center", fontsize=9)
    fig.text(0.004, 0.975, u"(a) 分段工序示意", ha="left", va="center", fontsize=9)
    m = C.masks()
    ax = fig.add_axes([0.10, 0.10, 0.40, 0.48])
    hs = C.panel(ax, m, letter="b", fs=7)
    hs.insert(3, Patch(facecolor=K.SAT_CMAP(0.9 / 1.7), edgecolor="none", label=u"充填域（饱和度，(a)）"))
    hs.append(Patch(facecolor=F.ROCK, edgecolor="none", label=u"围岩（(a)）"))
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=7.5)
    fig.text(0.004, 0.012, C.NOTE, fontsize=7, color=K.INK2)
    F.save(fig, "fig_4_6_1", out=HERE)
    print("(a) x_f = %.2f, Delta = %.2f (dimensionless); (b) shadow %d cells" % (xf, D2 - D1, int(m["shadow"].sum())))


if __name__ == "__main__":
    main()
