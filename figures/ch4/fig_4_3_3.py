# -*- coding: utf-8 -*-
"""
Fig. 4.3-3 three-level evaluation domains and the classes of unfilled zones (schematic; fixes the drawing convention of
chapters 4 and 5, see ch4fig).

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_3_3.py

Made-up dimensionless section, no solver. lambda = 1 in the rubble, a low-porosity lens lambda = 40, a boulder bridge
and the wall impassable; rho g = 0.08; p0 = 5. p_req from the multi-source least cumulative integral of lambda ds over
the outflow section (16-neighbour grid, position-potential reweighting); reach = {p_req <= p0}.
(a) single stage: fill = {p_req <= 0.8 p0} minus the space under the boulder bridge (bypass void); classes by eq. (4-26)
    and the topology of section 4.3.5.
(b) staged: stage 1 (outflow section up to d1) filled and cemented (cement = its fill, impassable); stage 2 = deep part
    of the hole outside the cement; whole-hole reach on the pristine field (thin solid) and stage-2 reach (thick solid);
    inter-stage shadow by eq. (4-27): design n whole-hole reach minus the union of the stage reaches.
"""
from __future__ import print_function

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import ch4fig as F
from ch4fig import K, plt

DX = 0.05
XL, ZL = (-1.0, 16.0), (-1.5, 15.5)
COLLAR, ANGLE, LENGTH, PIPE = (0.0, 4.0), 30.0, 12.0, 1.5
D1 = 4.5                                   # (b) stage-1 hole depth along the hole
RHOG, P0, FILL = 0.08, 5.0, 0.8
DESIGN = (0.0, 13.0, 1.0, 12.5)            # x0, x1, z0, z1
LENS = ((4.6, 8.0), (1.0, 0.55), 40.0)     # centre, semi-axes, lambda
CAP, LEGS, VOID = (6.0, 8.2, 4.9, 5.35), ((6.0, 6.45), (7.75, 8.2)), (6.45, 7.75, 4.0, 4.9)


def grid():
    x = np.arange(XL[0], XL[1] + 1e-9, DX)
    z = np.arange(ZL[0], ZL[1] + 1e-9, DX)
    return x, z, np.meshgrid(x, z)


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


def setting(X, Z):
    wall = X < 0
    cap = (X >= CAP[0]) & (X <= CAP[1]) & (Z >= CAP[2]) & (Z <= CAP[3])
    legs = np.zeros_like(wall)
    for a, b in LEGS:
        legs |= (X >= a) & (X <= b) & (Z >= VOID[2]) & (Z < CAP[2])
    boulder = cap | legs
    void = (X > VOID[0]) & (X < VOID[1]) & (Z >= VOID[2]) & (Z < VOID[3])
    (cx, cz), (ax_, az), lam_lens = LENS
    lens = ((X - cx) / ax_) ** 2 + ((Z - cz) / az) ** 2 <= 1
    lam = np.ones(X.shape)
    lam[lens] = lam_lens
    lam[wall | boulder] = np.inf
    return wall, boulder, void, lens, lam


def design_mask(X, Z):
    x0, x1, z0, z1 = DESIGN
    return (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)


def show(ax, x, z, layer, color, zorder):
    ext = [x[0] - DX / 2, x[-1] + DX / 2, z[0] - DX / 2, z[-1] + DX / 2]
    ax.imshow(np.ma.masked_where(~layer, np.ones(layer.shape)), origin="lower", extent=ext,
              cmap=ListedColormap([color]), interpolation="nearest", zorder=zorder)


def show_fill(ax, x, z, fill, s):
    ext = [x[0] - DX / 2, x[-1] + DX / 2, z[0] - DX / 2, z[-1] + DX / 2]
    ax.imshow(np.ma.masked_where(~fill, s), origin="lower", extent=ext, cmap=K.SAT_CMAP, vmin=0, vmax=1.7,
              interpolation="nearest", zorder=2)   # lightened: schematic patches


def frame(ax):
    x0, x1, z0, z1 = DESIGN
    ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec=K.INK, lw=F.LW_DESIGN, ls="dashed", zorder=5))
    ax.set_xlim(XL); ax.set_ylim(ZL); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def note(ax, text, xy, xytext, ha="left"):
    ax.annotate(text, xy, xytext=xytext, fontsize=7.5, ha=ha, va="center", zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1))


def centroid(m):
    i, j = np.nonzero(m)
    return i, j


def main():
    x, z, (X, Z) = grid()
    wall, boulder, void, lens, lam = setting(X, Z)
    solid = wall | boulder
    D = design_mask(X, Z)
    u = np.array([np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE))])
    end = np.array(COLLAR) + LENGTH * u
    fig = plt.figure(figsize=(F.WIDTH, 3.95))
    # ---- (a) single stage ----
    pr = p_req(lam, segment_mask(X, Z, PIPE, LENGTH), Z)
    R = pr <= P0
    Fm = (pr <= FILL * P0) & ~void & ~solid
    s = np.clip(1 - 0.45 * (pr / (FILL * P0)) ** 6, 0, 1)
    if np.any(void & ~R):
        raise SystemExit("part of the space under the boulder is unreachable: not a pure bypass void")
    cats = {"unreach": D & ~R & ~solid & ~lens, "dense": lens & D & ~R, "front": D & R & ~Fm & ~void & ~solid,
            "bypass": void}
    ax = fig.add_axes([0.0, 0.2, 0.5, 0.74])
    for k, m in cats.items():
        show(ax, x, z, m, F.CAT[k], 1)
    show_fill(ax, x, z, Fm, s)
    show(ax, x, z, solid, F.SOLID, 3)
    ax.contour(x, z, np.ma.masked_where(solid, pr), levels=[P0], colors=[K.INK], linewidths=F.LW_REACH, zorder=4)
    F.draw_hole(ax, COLLAR, end, PIPE)
    frame(ax)
    outside = Fm & ~D
    ii, jj = np.nonzero(outside)
    note(ax, u"设计域外充填", (x[jj].mean(), z[ii].max() - 0.4), (13.6, 14.6), ha="center")
    note(ax, F.CAT_NAME["unreach"], (12.2, 1.8), (14.0, 0.0), ha="center")
    note(ax, F.CAT_NAME["unreach"], (0.8, 11.6), (2.2, 14.6), ha="center")
    note(ax, F.CAT_NAME["dense"], (LENS[0][0] - 0.6, LENS[0][1] + 0.2), (1.3, 13.3), ha="left")
    note(ax, F.CAT_NAME["bypass"], (7.1, 4.4), (7.1, -0.9), ha="center")
    ii, jj = np.nonzero(cats["front"] & (Z < 2.6) & (X > 1.0) & (X < 6))
    note(ax, F.CAT_NAME["front"], (x[jj].mean(), z[ii].mean()), (2.6, -0.9), ha="center")
    note(ax, u"块石", (8.1, 5.2), (10.6, 3.0), ha="left")
    F.tag(ax, "a", dx=0.05, dy=0.02)
    # ---- (b) staged ----
    lam = np.where(wall, np.inf, 1.0)          # (b): the same hole in plain rubble (no lens, no boulder)
    solid = wall
    D = design_mask(X, Z)
    prw = p_req(lam, segment_mask(X, Z, PIPE, LENGTH), Z)
    R = prw <= P0
    pr1 = p_req(lam, segment_mask(X, Z, PIPE, D1), Z)
    R1 = pr1 <= P0
    cement = (pr1 <= FILL * P0) & ~solid
    lam2 = lam.copy()
    lam2[cement] = np.inf
    src2 = segment_mask(X, Z, D1, LENGTH) & ~cement
    pr2 = p_req(lam2, src2, Z)
    R2 = pr2 <= P0
    shadow = D & R & ~R1 & ~R2 & ~cement & ~solid
    sd = np.linspace(0, LENGTH, 2401)
    pts = np.array(COLLAR)[None, :] + sd[:, None] * u[None, :]
    inc = cement[np.round((pts[:, 1] - z[0]) / DX).astype(int), np.round((pts[:, 0] - x[0]) / DX).astype(int)]
    s_exit = sd[np.nonzero(inc)[0].max()] + DX
    F2 = (pr2 <= FILL * P0) & ~solid & ~cement
    s2 = np.clip(1 - 0.45 * (pr2 / (FILL * P0)) ** 6, 0, 1)
    ax = fig.add_axes([0.5, 0.2, 0.5, 0.74])
    show(ax, x, z, shadow, F.CAT["shadow"], 1)
    show_fill(ax, x, z, F2, s2)
    show(ax, x, z, solid | cement, F.SOLID, 3)
    ax.contour(x, z, np.ma.masked_where(solid, prw), levels=[P0], colors=[K.INK], linewidths=0.8, zorder=4)
    ax.contour(x, z, np.ma.masked_where(solid | cement, pr2), levels=[P0], colors=[K.INK], linewidths=2.0, zorder=4)
    F.draw_hole(ax, COLLAR, end, PIPE, section_from=s_exit)
    frame(ax)
    c = np.array(COLLAR)
    note(ax, u"前序段胶结体", tuple(c + 2.2 * u + np.array([0.4, -1.2])), (0.6, -0.9), ha="left")
    mid = c + 0.5 * (s_exit + LENGTH) * u
    note(ax, u"后续段出浆段", tuple(mid), (9.0, 14.6), ha="center")
    for side in (1, -1):
        m = shadow & (side * ((X - c[0]) * (-u[1]) + (Z - c[1]) * u[0]) > 0)
        if m.sum() > 0:
            ii, jj = np.nonzero(m)
            k = np.argmin(np.abs(z[ii] - z[ii].mean()) + np.abs(x[jj] - x[jj].mean()))
            note(ax, F.CAT_NAME["shadow"], (x[jj][k], z[ii][k]), (13.2 if side < 0 else 2.5, 0.0 if side < 0 else 14.6),
                 ha="center")
    F.tag(ax, "b", dx=0.05, dy=0.02)
    # ---- legend ----
    hs = [Line2D([], [], color=K.INK, lw=F.LW_DESIGN, ls="--"), Line2D([], [], color=K.INK, lw=F.LW_REACH),
          Patch(fc=K.SAT_CMAP(0.9 / 1.7), ec="none"), Patch(fc=F.SOLID, ec="none"),
          Line2D([], [], color=K.INK, lw=0.8), Line2D([], [], color=K.INK, lw=2.0)]
    ls = [u"设计域", u"可达域", u"充填域（饱和度）", u"固体与胶结体",
          u"理想整孔可达域 $\\Omega_{\\mathrm{reach}}^{w}$", u"本段可达域"]
    hs += [Patch(fc=F.CAT[k], ec="none") for k in ("unreach", "dense", "front", "bypass", "shadow")]
    ls += [F.CAT_NAME[k] for k in ("unreach", "dense", "front", "bypass", "shadow")]
    fig.legend(hs, ls, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=7,
               handlelength=3.0, columnspacing=1.2)
    F.save(fig, "fig_4_3_3")
    print("cells: unreach %d dense %d front %d bypass %d shadow %d; s_exit %.2f" %
          tuple([cats[k].sum() for k in ("unreach", "dense", "front", "bypass")] + [shadow.sum(), s_exit]))


if __name__ == "__main__":
    main()
