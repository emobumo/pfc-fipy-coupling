# -*- coding: utf-8 -*-
"""
Fig. 4.1-1 engineering problem and physical idealisation (schematic, no data, no real dimensions).

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_1_1.py

(a) vertical section across strike: hanging-wall and footwall rock, waste rubble (block polygons), a fan of grout holes
    from a footwall drift, the design reinforcement range (dashed), grout running down along the footwall contact, an
    unfilled pocket. (b) the same section as a continuum: equivalent porosity field n(x) (illustrative), holes as line
    sources (standpipe without outflow, outflow section at p0), gravity, no-flow boundary, front at p = 0; inset: scale
    separation d50 << REV << spreading scale (d50 0.15 m as in 4.1.2).
"""
from __future__ import print_function

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon, Rectangle

import ch4fig as F
from ch4fig import K, plt

PHI_CMAP = LinearSegmentedColormap.from_list("phi", ["#fff7e0", "#f2c879", "#d98b2b", "#8a4b0f"])   # as fig. 4.1-2
RUBBLE_BG, BLOCK_FC, BLOCK_EC = "#efe6d4", "#d6c09a", "#8a7660"
X0, X1, Z0, Z1 = 0.0, 10.0, 0.0, 10.0
HW0, FW0, SLOPE = 1.2, 4.6, 0.5            # hanging-wall / footwall contacts x = c + SLOPE z
TOP = 8.6                                  # rubble top
DRIFT = (7.4, 8.1, 3.7, 4.3)               # x0, x1, z0, z1 (footwall drift)
COLLAR = (7.4, 4.0)
HOLES = (160.0, 145.0, 130.0)              # directions (deg from +x)


def hole_geom(a):
    """Standpipe through the footwall rock up to the footwall contact; outflow section to 0.3 short of the
    hanging-wall contact. Returns (unit direction, start of the outflow section, end)."""
    c = np.array(COLLAR)
    u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    t_fw = (FW0 + SLOPE * c[1] - c[0]) / (u[0] - SLOPE * u[1])
    t_hw = (HW0 + SLOPE * c[1] - c[0]) / (u[0] - SLOPE * u[1])
    return u, t_fw, c + t_fw * u, c + (t_hw - 0.3) * u


def hw(z):
    return HW0 + SLOPE * z


def fw(z):
    return FW0 + SLOPE * z


def in_band(x, z, m=0.0):
    return (x > hw(z) + m) & (x < fw(z) - m) & (z < TOP - m) & (z > Z0 + m)


def rock(ax):
    ax.add_patch(Rectangle((X0, Z0), X1 - X0, Z1 - Z0, fc=F.ROCK, ec="none", zorder=0))
    band = [(hw(Z0), Z0), (fw(Z0), Z0), (fw(TOP), TOP), (hw(TOP), TOP)]
    ax.add_patch(Polygon([(hw(TOP), TOP), (fw(TOP), TOP), (fw(Z1), Z1), (hw(Z1), Z1)], fc="w", ec="none", zorder=0.5))
    return band


def drift(ax):
    x0, x1, z0, z1 = DRIFT
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fc="w", ec=K.INK, lw=0.7, zorder=6))


def holes(ax):
    ends = []
    for a in HOLES:
        u, t_fw, p, e = hole_geom(a)
        F.draw_hole(ax, COLLAR, e, t_fw)
        ends.append(e)
    return ends


def frame(ax):
    ax.set_xlim(X0, X1); ax.set_ylim(Z0, Z1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def blocks(ax, rng):
    pts = []
    while len(pts) < 260:
        x, z = rng.uniform(X0, X1), rng.uniform(Z0, TOP)
        if in_band(x, z, 0.15):
            pts.append((x, z))
    for x, z in pts:
        r = rng.uniform(0.10, 0.30)
        k = rng.randint(5, 8)
        th = np.sort(rng.uniform(0, 2 * np.pi, k))
        rr = r * rng.uniform(0.7, 1.0, k)
        poly = np.c_[x + rr * np.cos(th), z + rr * np.sin(th)]
        if np.all(in_band(poly[:, 0], poly[:, 1], 0.02)):
            ax.add_patch(Polygon(poly, fc=BLOCK_FC, ec=BLOCK_EC, lw=0.3, zorder=2))


def label(ax, text, xy, xytext, ha="left", color=K.INK):
    ax.annotate(text, xy, xytext=xytext, fontsize=7.5, ha=ha, va="center", color=color, zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1))


def panel_a(ax, rng):
    band = rock(ax)
    ax.add_patch(Polygon(band, fc=RUBBLE_BG, ec=K.INK, lw=0.7, zorder=1))
    blocks(ax, rng)
    # grout: patch around the holes and a tongue down the footwall contact
    xs, zs = np.meshgrid(np.linspace(X0, X1, 501), np.linspace(Z0, Z1, 501))
    d = np.full(xs.shape, np.inf)
    for a in HOLES:
        u, t_fw, p, e = hole_geom(a)
        t = np.clip(((xs - p[0]) * (e - p)[0] + (zs - p[1]) * (e - p)[1]) / np.dot(e - p, e - p), 0, 1)
        d = np.minimum(d, np.hypot(xs - p[0] - t * (e - p)[0], zs - p[1] - t * (e - p)[1]))
    grout = (d < 0.9) & in_band(xs, zs)
    tongue = in_band(xs, zs) & (xs > fw(zs) - 0.45) & (zs < 4.4) & (zs > 0.4)
    pocket = ((xs - 5.75) / 0.42) ** 2 + ((zs - 7.0) / 0.28) ** 2 < 1
    m = (grout | tongue) & ~pocket
    ax.imshow(np.ma.masked_where(~m, np.ones(m.shape)), origin="lower", extent=[X0, X1, Z0, Z1],
              cmap=ListedColormap([K.SAT_CMAP(0.55)]), alpha=0.85, interpolation="nearest", zorder=3)
    ax.imshow(np.ma.masked_where(~pocket, np.ones(m.shape)), origin="lower", extent=[X0, X1, Z0, Z1],
              cmap=ListedColormap(["w"]), interpolation="nearest", zorder=3)
    ax.annotate("", xy=(fw(0.7) - 0.2, 0.7), xytext=(fw(3.6) - 0.2, 3.6), zorder=8,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=K.C1, mutation_scale=10))
    dz = [(hw(z), z) for z in np.linspace(3.3, 8.2, 2)]
    ax.add_patch(Polygon([(hw(3.4) + 0.2, 3.4), (fw(3.4) - 0.05, 3.4), (fw(8.2) - 0.05, 8.2), (hw(8.2) + 0.2, 8.2)],
                         fill=False, ec=K.INK, lw=F.LW_DESIGN, ls="dashed", zorder=7))
    drift(ax)
    holes(ax)
    label(ax, u"上盘", (1.0, 7.0), (0.25, 9.3))
    label(ax, u"下盘", (8.6, 1.5), (8.6, 0.6))
    label(ax, u"废石散体", (3.2, 5.0), (0.25, 2.2))
    label(ax, u"注浆孔", (5.6, 5.6), (6.3, 9.3))
    label(ax, u"巷道", (7.75, 4.3), (8.9, 5.6))
    label(ax, u"设计加固范围", (hw(8.2) + 1.2, 8.2), (2.4, 9.6))
    label(ax, u"跑浆", (fw(1.8) - 0.3, 1.8), (6.7, 2.3))
    label(ax, u"未充填区", (5.75, 7.0), (0.25, 4.0))
    frame(ax)


def field_n(xs, zs):
    """Illustrative equivalent porosity: slightly looser upwards and along the footwall contact."""
    n = 0.17 + 0.04 * (zs - 4.0) / 4.0 + 0.06 * np.exp(-((fw(zs) - xs) / 0.6) ** 2)
    n += 0.015 * np.sin(1.7 * xs + 0.9 * zs) * np.cos(1.3 * zs - 0.6 * xs)
    return n


def panel_b(fig, ax):
    band = rock(ax)
    xs, zs = np.meshgrid(np.linspace(X0, X1, 501), np.linspace(Z0, Z1, 501))
    inside = in_band(xs, zs)
    im = ax.imshow(np.ma.masked_where(~inside, field_n(xs, zs)), origin="lower", extent=[X0, X1, Z0, Z1],
                   cmap=PHI_CMAP, vmin=0.0, vmax=0.45, interpolation="bilinear", zorder=1)
    ax.add_patch(Polygon(band, fill=False, ec=K.INK, lw=0.7, zorder=2))
    # one illustrative front around the holes
    d = np.full(xs.shape, np.inf)
    for a in HOLES:
        u, t_fw, p, e = hole_geom(a)
        t = np.clip(((xs - p[0]) * (e - p)[0] + (zs - p[1]) * (e - p)[1]) / np.dot(e - p, e - p), 0, 1)
        d = np.minimum(d, np.hypot(xs - p[0] - t * (e - p)[0], zs - p[1] - t * (e - p)[1]))
    fill = (d < 0.85) & inside
    ax.imshow(np.ma.masked_where(~fill, np.clip(1 - 0.5 * (d / 0.85) ** 4, 0, 1)), origin="lower",
              extent=[X0, X1, Z0, Z1], cmap=K.SAT_CMAP, vmin=0, vmax=1.7, interpolation="bilinear", zorder=3)
    ax.contour(xs[0], zs[:, 0], np.where(inside, d, 9.0), levels=[0.85], colors=[K.C1], linewidths=0.8, zorder=4)
    ax.add_patch(Rectangle((X0 + 0.05, Z0 + 0.05), X1 - X0 - 0.1, Z1 - Z0 - 0.1, fill=False, ec=K.INK, lw=1.2,
                           zorder=5))
    drift(ax)
    ends = holes(ax)
    ax.annotate("", xy=(9.3, 7.4), xytext=(9.3, 9.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=K.INK, mutation_scale=9))
    ax.text(9.45, 8.2, u"$g$", fontsize=9, va="center")
    u, t_fw, p, e = hole_geom(HOLES[0])
    mid = 0.5 * (p + e)
    label(ax, u"定压出浆段 $p_0$", tuple(mid), (5.0, 2.6))
    u, t_fw, p, e = hole_geom(HOLES[2])
    label(ax, u"孔口管（不出浆）", tuple(np.array(COLLAR) + 0.5 * t_fw * u), (8.45, 5.6), ha="center")
    ii, jj = np.nonzero((np.abs(d - 0.85) < 0.02) & inside)
    k = np.argmax(zs[ii, jj])
    label(ax, u"锋面 $p = 0$\n（气体自由排出）", (xs[ii[k], jj[k]], zs[ii[k], jj[k]]), (2.4, 9.3))
    label(ax, u"计算域边界：无流", (X1 - 0.05, 2.3), (6.3, 1.2))
    ax.text(0.3, 1.25, u"浆液：宾汉流体\n介质：$\\lambda(n)$、$k(n)$", fontsize=7.5, va="center",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5), zorder=8)
    frame(ax)
    cax = fig.add_axes([0.905, 0.42, 0.012, 0.36])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.15, 0.30, 0.45])
    cb.set_label(u"等效孔隙率 $n$（示意）", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    return ax


def inset(fig):
    ax = fig.add_axes([0.62, 0.075, 0.28, 0.1])
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100); ax.set_ylim(0, 1)
    ax.axvspan(10, 40, color=K.C1, alpha=0.18, lw=0)
    for x, t in ((0.15, u"中值粒径 $d_{50}$\n约 0.15 m"), (2.5, u"REV\n约 2.5 m"), (20, u"扩散尺度\n10～40 m")):
        ax.plot([x], [0.25], "o", color=K.INK, ms=3.5)
        ax.text(x, 0.55, t, fontsize=6.5, ha="center", va="bottom")
    ax.text(0.5, 0.2, u"≪", fontsize=9, ha="center", va="center")
    ax.text(7.5, 0.2, u"≪", fontsize=9, ha="center", va="center")
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel(u"长度 / m", fontsize=6.5, labelpad=1)
    ax.text(-0.04, 0.25, u"尺度分离", transform=ax.transAxes, fontsize=7.5, ha="right", va="center")


def main():
    rng = np.random.RandomState(7)
    fig = plt.figure(figsize=(F.WIDTH, 3.9))
    ax = fig.add_axes([0.0, 0.2, 0.44, 0.76])
    panel_a(ax, rng)
    F.tag(ax, "a", dx=0.02, dy=0.0)
    ax = fig.add_axes([0.455, 0.2, 0.44, 0.76])
    panel_b(fig, ax)
    F.tag(ax, "b", dx=0.02, dy=0.0)
    inset(fig)
    F.save(fig, "fig_4_1_1")


if __name__ == "__main__":
    main()
