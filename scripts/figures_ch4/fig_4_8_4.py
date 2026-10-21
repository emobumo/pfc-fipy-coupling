# -*- coding: utf-8 -*-
"""
Fig. 4.8-4 (draft): scheme prediction, seven panels, current end state; existing solver outputs only (each case was
computed once; the post-processing stage2.txt is two-pass identical).

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_8_4.py

(a)-(e) the five closed cases of K 1.25 scenario 1: saturation when the reach-domain fill first reaches 0.95
        (snap_<case>_0.95.npz), reach-domain outline (summary.npz, same hole and p0, 0.5 m), outflow section (source
        cells projected on the hole axis) and standpipe (collar to the first source cell), lower end of the rubble,
        boundary of the lower-end corner (rubble down-dip of the lowest drift's collar, t < fan_collar_t[1]), cells
        filled outside the reach domain in neutral grey (not counted).
(f)     drawpoint open, both holes: outflow / V_in against V_in / reach pore volume of the closed case, every 20-step
        record of final_O1 / O2.npz "series" (the outflow is recorded every 20 steps only); vertical lines at the
        closed case's arrival at the lower end and at its 0.95 snapshot (stage2.txt).
(g)     V_in / reach pore volume at the arrival at the lower end against K (stage2.txt), normal and -30 deg hole of the
        lowest drift; scenario 2 (K 1.25) with another marker; K 1.15 normal hole: reach domain without the lower end.
No case codes on the figure.
"""
from __future__ import print_function

import os

import numpy as np

import xz48 as X
from common import plt, SAT_CMAP, INK, C1, C2
from matplotlib.colors import ListedColormap

SOL = X.SOL
CELL = 0.5
VIEW = (-30.0, 53.0, -4.5, 33.0)
ROCK, VOIDC, DRIFT, OUTSIDE = "#cdcdcd", X.VOIDC, "#3d3d3d", "#8c8c8c"
# closed cases of K 1.25 scenario 1, panel order (a)-(e): case, panel title, fan hole index of the base geometry
PANELS = [("S1", u"最低巷道法向孔", 1), ("S2", u"孔口管加长", 1), ("S3", u"降压至 1 MPa", 1),
          ("S4", u"最低巷道 −30° 孔", 0), ("S5", u"第二巷道 −30° 孔", 2)]


def hole_parts(line, xs, zs):
    """Standpipe (collar -> first source cell) and outflow section (first -> last source cell) along the hole axis."""
    c = np.array(line[:2]); e = np.array(line[2:])
    u = (e - c) / np.hypot(*(e - c))
    d = (xs - c[0]) * u[0] + (zs - c[1]) * u[1]
    a, b = d.min() - CELL / 2, d.max() + CELL / 2
    return (c, c + a * u), (c + a * u, c + b * u)


def map_panel(ax, Sm, G, B, ci, tag, title, fan):
    Xg, Zg, rub, cls = Sm["X"], Sm["Z"], Sm["rub"], Sm["cls"]
    ext = [Xg.min() - CELL / 2, Xg.max() + CELL / 2, Zg.min() - CELL / 2, Zg.max() + CELL / 2]
    one = np.ones(cls.shape)
    for mask, col in ((~rub & (cls != 4) & (cls != 5), ROCK), (cls == 4, VOIDC), (cls == 5, DRIFT)):
        ax.imshow(np.ma.masked_where(~mask, one), origin="lower", extent=ext, cmap=ListedColormap([col]),
                  interpolation="nearest")
    s = np.clip(np.load(os.path.join(SOL, "snap_%s_0.95.npz" % tag))["s"], 0, 1)
    reach = Sm["reach"][ci]
    im = ax.imshow(np.ma.masked_where(~rub, s), origin="lower", extent=ext, cmap=SAT_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    out = rub & (s >= 0.5) & ~reach
    if out.any():
        ax.imshow(np.ma.masked_where(~out, one), origin="lower", extent=ext, cmap=ListedColormap([OUTSIDE]),
                  interpolation="nearest")
    ax.contour(Xg[0], Zg[:, 0], reach.astype(float), levels=[0.5], colors=[INK], linewidths=1.0)
    # lower-end corner: rubble down-dip of the lowest drift's collar
    j0, j1, i0, i1 = [int(v) for v in G["box"]]
    T = G["T"][j0:j1, i0:i1]
    tc = float(B["fan_collar_t"][1])
    cs = ax.contour(Xg[0], Zg[:, 0], np.ma.masked_where(~rub, T), levels=[tc], colors=[INK], linewidths=0.6,
                    linestyles="dashed")
    for c_ in cs.collections:
        c_.set_zorder(7)
    corner = rub & (T < tc)
    ax.text(float(Xg[corner].mean()) - 6.0, float(Zg[corner].mean()) + 4.0, u"下端角部", fontsize=6.5, color=INK,
            ha="center", va="center", zorder=8, bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.8))
    src = np.load(os.path.join(SOL, "final_%s.npz" % tag))["src"]
    pipe, sec = hole_parts(B["fan_line"][fan], Xg.ravel()[src], Zg.ravel()[src])
    ax.plot([pipe[0][0], pipe[1][0]], [pipe[0][1], pipe[1][1]], "-", color="k", lw=3.0, solid_capstyle="butt", zorder=5)
    ax.plot([pipe[0][0], pipe[1][0]], [pipe[0][1], pipe[1][1]], "-", color="w", lw=1.4, solid_capstyle="butt", zorder=6)
    ax.plot([sec[0][0], sec[1][0]], [sec[0][1], sec[1][1]], "-", color="k", lw=1.8, solid_capstyle="butt", zorder=5)
    X.draw_low(ax, dx=-8.0, dz=-2.6, lw=1.3)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
    ax.set_title(title)
    X.corner_note(ax)
    ax.tick_params(labelsize=7)
    return im


def panel_f(ax, R):
    for tag, ref, name, col in (("O1", "S1", u"最低巷道法向孔", C1), ("O2", "S4", u"最低巷道 −30° 孔", C2)):
        ser = np.load(os.path.join(SOL, "final_%s.npz" % tag))["series"]
        ser = ser[ser[:, 3] > 0]
        pore = R[ref]["pore"]
        if abs(R[tag]["pore"] - pore) > 1e-9:
            raise SystemExit("%s: reach pore volume differs from %s" % (tag, ref))
        ax.plot(ser[:, 3] / pore, ser[:, 7] / ser[:, 3], "-", color=col, lw=1.3, label=name)
        ax.axvline(R[ref]["arrival"], color=col, ls=":", lw=1.0, ymax=0.72)
        ax.axvline(R[ref]["s95"], color=col, ls="--", lw=1.0, ymax=0.72)
    hs, ls = ax.get_legend_handles_labels()
    hs += [plt.Line2D([], [], color=INK, ls=":", lw=1.0), plt.Line2D([], [], color=INK, ls="--", lw=1.0)]
    ls += [u"封闭算例到达散体下端", u"封闭算例可达域充填 0.95"]
    ax.legend(hs, ls, loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    ax.set_xlabel(u"注入量 ÷ 可达域孔隙体积")
    ax.set_ylabel(u"流失占注入量的比例")
    ax.set_xlim(0, None); ax.set_ylim(0, 0.62)
    ax.set_title(u"(f) 出矿口开放")
    X.K.style(ax)


def panel_g(ax, R):
    ks = [1.15, 1.25, 1.40]
    normal = [None, R["S1"]["arrival"], R["K2"]["arrival"]]       # K 1.15 normal hole: lower end not in its reach
    minus30 = [R["K1"]["arrival"], R["S4"]["arrival"], R["K3"]["arrival"]]
    ax.plot(ks[1:], normal[1:], "o-", color=C1, lw=1.2, ms=4.5, label=u"法向孔（情景一）")
    ax.plot(ks, minus30, "s-", color=C2, lw=1.2, ms=4.5, label=u"−30° 孔（情景一）")
    ax.plot([1.25], [R["K4"]["arrival"]], "D", mec=C1, mfc="none", mew=1.2, ms=5, label=u"法向孔（情景二）")
    ax.plot([1.25], [R["K5"]["arrival"]], "D", mec=C2, mfc="none", mew=1.2, ms=5, label=u"−30° 孔（情景二）")
    ax.annotate(u"法向孔：可达域不含散体下端", xy=(1.15, 0.80), fontsize=6.5, color=C1, ha="left", va="center")
    ax.axvline(1.15, color=C1, ls=":", lw=0.6, ymin=0.0, ymax=1.0)
    ax.set_xticks(ks)
    ax.set_xlim(1.11, 1.44); ax.set_ylim(0, 0.9)
    ax.set_xlabel(u"碎胀系数 K")
    ax.set_ylabel(u"注入量 ÷ 可达域孔隙体积\n（到达散体下端时）", fontsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, numpoints=1)
    ax.set_title(u"(g) 到达散体下端")
    X.K.style(ax)


def main():
    Sm = np.load(os.path.join(SOL, "summary.npz"))
    G = np.load(os.path.join(SOL, "grid_K1.25_s1.npz"))
    B = np.load(os.path.join(X.XC.out(X.CFG, "data", "xz_prejudge"), "base_geometry_K1.25_v4.npz"))
    j0, j1, i0, i1 = [int(v) for v in G["box"]]
    if not (np.array_equal(G["low"][j0:j1, i0:i1], Sm["low"]) and np.allclose(G["X"][j0:j1, i0:i1], Sm["X"])):
        raise SystemExit("summary.npz and grid_K1.25_s1.npz differ")
    R = X.stage2()
    tags = [str(t) for t in Sm["tags"]]
    fig = plt.figure(figsize=(7.6, 9.6))
    w, h = 0.43, 0.20
    pos = [(0.06, 0.77), (0.54, 0.77), (0.06, 0.53), (0.54, 0.53), (0.06, 0.29)]
    for k, (tag, title, fan) in enumerate(PANELS):
        ax = fig.add_axes(list(pos[k]) + [w, h])
        im = map_panel(ax, Sm, G, B, tags.index(tag), tag, u"(%s) %s" % ("abcde"[k], title), fan)
        if k % 2 == 0:
            ax.set_ylabel(u"Z/m", fontsize=8)
        ax.set_xlabel(u"X/m", fontsize=8)
    cb = fig.colorbar(im, cax=fig.add_axes([0.56, 0.355, 0.012, 0.13]))
    cb.set_label(u"饱和度 S", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.legend([plt.Line2D([], [], color=INK, lw=1.0), plt.Rectangle((0, 0), 1, 1, fc=OUTSIDE),
                plt.Line2D([], [], color="k", lw=1.8), plt.Line2D([], [], color="k", lw=3.0),
                plt.Line2D([], [], color=INK, lw=0.6, ls="--"), plt.Line2D([], [], color=X.LOWC, lw=1.3),
                plt.Rectangle((0, 0), 1, 1, fc=ROCK), plt.Rectangle((0, 0), 1, 1, fc=VOIDC),
                plt.Rectangle((0, 0), 1, 1, fc=DRIFT)],
               [u"可达域", u"可达域外（不计入）", u"出浆段", u"孔口管", u"下端角部边界", X.LOW_NAME, u"围岩与矿柱",
                u"残余空区", u"巷道"], loc="center left", bbox_to_anchor=(0.64, 0.42), frameon=False, fontsize=7)
    panel_f(fig.add_axes([0.08, 0.05, 0.38, 0.18]), R)
    panel_g(fig.add_axes([0.58, 0.05, 0.38, 0.18]), R)
    X.K.save(fig, "fig_4_8_4")
    for tag in ("S1", "S4", "K1", "K2", "K3", "K4", "K5"):
        print(tag, "arrival V_in / pore", R[tag]["arrival"], "0.95:", R[tag]["s95"], "pore", R[tag]["pore"])


if __name__ == "__main__":
    main()
