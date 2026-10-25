# -*- coding: utf-8 -*-
"""
Fig. 4.3-4 reach criterion and property 3 (schematic, two panels; no solver).

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_3_4.py

A made-up dimensionless lambda field (background 1, a high-lambda block 8 between source and target, a low-lambda
arc-shaped band 0.15 beside it; smooth edges). T(x) = min over paths of the integral of lambda ds is computed on the
grid with Dijkstra (16-neighbour stencil, edge weight = length x mean of the end values); rho g = 0.1 (Pi_g = 0.1 in
the background); p0 = 0.6.
(a) left: field, straight path 1 through the block and the optimal path 2 (Dijkstra predecessor chain, through the
    band); right: cumulative integral of lambda ds along both paths (interpolated field, same sampling for both) and
    the available driving pressure p0 - rho g (z_x - z_s).
(b) left: three source-pressure histories, all <= p0 (constant, stepped, fluctuating); right: the reach domain
    {p_req <= p0}, p_req = T + rho g (z - z_s), and two fill outlines per history at t1 < t2, constructed as
    {p_req + kappa_h |x - x_s| <= a_h(t) p0} with a_h <= 1, kappa_h >= 0 -- inside the reach domain and nested in time
    by construction (illustration, not a solution).
"""
from __future__ import print_function

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib.colors import LinearSegmentedColormap

import ch4fig as F
from ch4fig import K, plt

DX = 0.0125
XLIM, ZLIM = (-0.95, 2.15), (-0.95, 1.55)
SRC, TGT = (0.0, -0.05), (1.25, 0.05)
BLOCK = dict(c=(0.6, 0.0), h=(0.12, 0.45), lam=8.0)
BAND = dict(c=(0.6, 0.10), R=0.64, half=0.05, lam=0.15)
EDGE = 0.025
RHOG, P0 = 0.1, 0.6
LAM_CMAP = LinearSegmentedColormap.from_list("lam", ["#fbf4e8", "#e7c79a", "#b88446", "#6e4517"])
HIST = (  # name, colour, kappa, a(t1), a(t2)
    (u"定压 $p_0$", K.C1, 0.0, 0.55, 0.90),
    (u"逐级升压至 $p_0$", K.C2, 0.15, 0.30, 0.70),
    (u"$p_0$ 以下波动（限流泵送）", K.C3, 0.30, 0.40, 0.68),
)
T1, T2 = 0.32, 0.80                      # (b) the two moments, fraction of the time axis


def field():
    x = np.arange(XLIM[0], XLIM[1] + 1e-9, DX)
    z = np.arange(ZLIM[0], ZLIM[1] + 1e-9, DX)
    X, Z = np.meshgrid(x, z)
    # rounded box signed distance
    qx, qz = np.abs(X - BLOCK["c"][0]) - BLOCK["h"][0], np.abs(Z - BLOCK["c"][1]) - BLOCK["h"][1]
    d_block = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0)
    # upper half-ring (arc) signed distance
    cx, cz = BAND["c"]
    rr = np.hypot(X - cx, Z - cz)
    d_ring = np.abs(rr - BAND["R"]) - BAND["half"]
    ends = np.minimum(np.hypot(X - (cx - BAND["R"]), Z - cz), np.hypot(X - (cx + BAND["R"]), Z - cz)) - BAND["half"]
    d_band = np.where(Z >= cz, d_ring, ends)
    s_block = 0.5 * (1 - np.tanh(d_block / EDGE))
    s_band = 0.5 * (1 - np.tanh(d_band / EDGE))
    lam = np.exp(np.log(BLOCK["lam"]) * s_block + np.log(BAND["lam"]) * s_band)
    return x, z, X, Z, lam


def index(x, z, p):
    return int(round((p[1] - z[0]) / DX)) * len(x) + int(round((p[0] - x[0]) / DX))


def travel_time(x, z, lam, src):
    nz, nx = lam.shape
    idx = np.arange(nz * nx).reshape(nz, nx)
    rows, cols, w = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)):
        i0, i1 = 0, nz - di
        j0, j1 = max(0, -dj), nx - max(0, dj)
        a = idx[i0:i1, j0:j1]
        b = idx[i0 + di:i1 + di, j0 + dj:j1 + dj]
        la, lb = lam[i0:i1, j0:j1], lam[i0 + di:i1 + di, j0 + dj:j1 + dj]
        rows.append(a.ravel()); cols.append(b.ravel())
        w.append((0.5 * (la + lb) * DX * np.hypot(di, dj)).ravel())
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nz * nx,) * 2).tocsr()
    T, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
    return T.reshape(nz, nx), pred


def chain(pred, src, tgt, X, Z):
    pts, k = [], tgt
    while k != src and k >= 0:
        pts.append((X.flat[k], Z.flat[k]))
        k = pred[k]
    pts.append((X.flat[src], Z.flat[src]))
    return np.array(pts[::-1])


def cumulative(x, z, lam, pts, n=4000):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.linspace(0, s[-1], n)
    px, pz = np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])
    lv = map_coordinates(lam, [(pz - z[0]) / DX, (px - x[0]) / DX], order=1)
    c = np.concatenate([[0], np.cumsum(0.5 * (lv[1:] + lv[:-1]) * np.diff(ss))])
    return ss, c


def show_field(ax, x, z, lam, alpha=1.0):
    ext = [x[0] - DX / 2, x[-1] + DX / 2, z[0] - DX / 2, z[-1] + DX / 2]
    im = ax.imshow(np.log(lam), origin="lower", extent=ext, cmap=LAM_CMAP, interpolation="bilinear", alpha=alpha,
                   vmin=np.log(BAND["lam"]), vmax=np.log(BLOCK["lam"]))
    ax.set_xlim(XLIM); ax.set_ylim(ZLIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def points(ax, labels=True):
    ax.plot([SRC[0]], [SRC[1]], "o", color=K.INK, ms=4.5, zorder=6)
    ax.plot([TGT[0]], [TGT[1]], "s", color=K.INK, ms=4.5, zorder=6)
    if labels:
        ax.text(SRC[0] - 0.06, SRC[1] - 0.04, u"$x_s$", ha="right", va="top", fontsize=8.5, zorder=6)
        ax.text(TGT[0] + 0.07, TGT[1] - 0.04, u"$x$", ha="left", va="top", fontsize=8.5, zorder=6)


def axes_in(fig, l, b, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([l / W, b / H, w / W, h / H])


def main():
    x, z, X, Z, lam = field()
    src, tgt = index(x, z, SRC), index(x, z, TGT)
    T, pred = travel_time(x, z, lam, src)
    preq = T + RHOG * (Z - SRC[1])
    p2 = chain(pred, src, tgt, X, Z)
    p1 = np.array([X.flat[[src, tgt]], Z.flat[[src, tgt]]]).T
    s1, c1 = cumulative(x, z, lam, p1)
    s2, c2 = cumulative(x, z, lam, p2)
    avail = P0 - RHOG * (Z.flat[tgt] - Z.flat[src])
    if not (c2[-1] < avail < c1[-1]):
        raise SystemExit("path 2 %.3f, available %.3f, path 1 %.3f: not in the intended order" % (c2[-1], avail, c1[-1]))
    fig = plt.figure(figsize=(F.WIDTH, 6.0))
    # ---- (a) ----
    ax = axes_in(fig, 0.12, 3.25, 2.95, 2.38)
    im = show_field(ax, x, z, lam)
    ax.plot(p1[:, 0], p1[:, 1], "-", color=K.C2, lw=1.5)
    ax.plot(p2[:, 0], p2[:, 1], "-", color=K.C1, lw=1.5)
    points(ax)
    ax.text(0.6, -0.52, u"高 $\\lambda$区", ha="center", va="top", fontsize=7.5)
    ax.text(0.6, 0.88, u"低 $\\lambda$带", ha="center", va="center", fontsize=7.5)
    ax.text(0.30, -0.16, u"路径 1", color=K.C2, fontsize=7.5, ha="center", va="top")
    ax.annotate(u"路径 2\n最优路径：累积量最小的路径", (p2[len(p2) // 3][0], p2[len(p2) // 3][1]),
                xytext=(-0.88, 1.38), color=K.C1, fontsize=7.5, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K.C1, shrinkA=1, shrinkB=2))
    cax = axes_in(fig, 3.12, 3.5, 0.09, 1.85)
    cb = fig.colorbar(im, cax=cax, ticks=[])
    cb.set_label(u"启动压力梯度 $\\lambda$", fontsize=8)
    cax.text(0.5, 1.02, u"大", transform=cax.transAxes, ha="center", va="bottom", fontsize=7)
    cax.text(0.5, -0.02, u"小", transform=cax.transAxes, ha="center", va="top", fontsize=7)
    F.tag(ax, "a", dx=0.1)
    ax = axes_in(fig, 3.85, 3.25, 2.1, 2.3)
    ax.plot(s1, c1, color=K.C2, lw=1.5, label=u"路径 1（直线，穿过高 $\\lambda$区）")
    ax.plot(s2, c2, color=K.C1, lw=1.5, label=u"路径 2（绕行，经低 $\\lambda$带）")
    ax.axhline(avail, color=F.GREY, lw=0.8)
    ax.text(0.30 * s2[-1], avail + 0.06, u"可用驱动压力 $p_0-\\rho g(z_x-z_s)$", fontsize=7, ha="left", va="bottom")
    ax.plot([s1[-1]], [c1[-1]], "s", color=K.C2, ms=4)
    ax.plot([s2[-1]], [c2[-1]], "s", color=K.C1, ms=4)
    ax.annotate(u"绕行更长，\n但累积量更小", (s2[-1], c2[-1]), xytext=(0.97 * s2[-1], 1.05), fontsize=7, ha="right",
                va="bottom", color=K.C1, arrowprops=dict(arrowstyle="-", lw=0.5, color=K.C1, shrinkA=1, shrinkB=3))
    ax.text(0.02 * s2[-1], 0.98 * c1[-1] * 1.15, u"判据比较整条路径的累积量，\n不看 $x$ 处的局部 $\\lambda$", fontsize=7, va="top")
    ax.set_xlim(0, 1.05 * s2[-1]); ax.set_ylim(0, 1.15 * c1[-1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(u"沿路径的距离 $s$")
    ax.set_ylabel(u"累积启动压力 $\\int\\lambda\\,\\mathrm{d}s$")
    ax.legend(loc="center right", frameon=False, fontsize=6.8, bbox_to_anchor=(1.02, 0.62))
    K.style(ax)
    # ---- (b) ----
    ax = axes_in(fig, 0.5, 0.42, 2.3, 2.2)
    t = np.linspace(0, 1, 801)
    steps = np.minimum(np.floor(t / 0.25) + 1, 3) / 3.0
    fluct = 0.78 + 0.12 * np.sin(2 * np.pi * 5.5 * t) - 0.05 * np.cos(2 * np.pi * 2.3 * t)
    fluct = np.minimum(fluct, 0.97)
    for (name, col, kap, a1, a2), ps in zip(HIST, (np.ones_like(t), steps, fluct)):
        ax.plot(t, np.where(t < 0.02, t / 0.02, 1) * ps, color=col, lw=1.3, label=name)
    ax.axhline(1.0, color=F.GREY, lw=0.8, zorder=0)
    ax.text(0.99, 1.04, u"源压上限 $p_0$（如终压）", ha="right", va="bottom", fontsize=7)
    for tt, nm in ((T1, u"$t_1$"), (T2, u"$t_2$")):
        ax.axvline(tt, color=F.GREY, lw=0.6, zorder=0)
        ax.text(tt, -0.03, nm, ha="center", va="top", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(u"时间 $t$", labelpad=12)
    ax.set_ylabel(u"源压 $p_s(t)$")
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    K.style(ax)
    F.tag(ax, "b", dx=0.48)
    ax = axes_in(fig, 3.05, 0.42, 2.95, 2.38)
    show_field(ax, x, z, lam, alpha=0.45)
    ax.contour(x, z, preq, levels=[P0], colors=[K.INK], linewidths=1.5)
    dist = np.hypot(X - SRC[0], Z - SRC[1])
    for name, col, kap, a1, a2 in HIST:
        g = preq + kap * dist
        for a, lw in ((a1, 0.7), (a2, 1.3)):
            if np.any((g <= a * P0) & ~(preq <= P0)):
                raise SystemExit("fill outline leaves the reach domain")
            ax.contour(x, z, g, levels=[a * P0], colors=[col], linewidths=lw)
    points(ax, labels=False)
    ax.text(SRC[0] - 0.06, SRC[1] - 0.04, u"$x_s$", ha="right", va="top", fontsize=8.5)
    ax.text(TGT[0], TGT[1] + 0.08, u"$x$", ha="center", va="bottom", fontsize=8.5)
    ax.text(XLIM[0] + 0.05, ZLIM[1] - 0.05, u"历程不同，充填先后不同；\n任一时刻 $\\Omega_{\\mathrm{fill}}(t)\\subseteq\\Omega_{\\mathrm{reach}}$",
            fontsize=7, va="top")
    hs = [plt.Line2D([], [], color=K.INK, lw=1.5), plt.Line2D([], [], color=F.GREY, lw=0.7),
          plt.Line2D([], [], color=F.GREY, lw=1.3)]
    ax.legend(hs, [u"可达域（按 $p_0$）", u"充填域 $t_1$（细线）", u"充填域 $t_2$（粗线）"], loc="lower right",
              frameon=False, fontsize=6.8, bbox_to_anchor=(1.0, 0.06))
    ax.text(0.97, 0.03, u"颜色对应左图历程", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    F.save(fig, "fig_4_3_4")
    print("path 1 cumulative %.3f, path 2 %.3f, available %.3f; lengths %.3f / %.3f" %
          (c1[-1], c2[-1], avail, s1[-1], s2[-1]))


if __name__ == "__main__":
    main()
