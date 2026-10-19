# -*- coding: utf-8 -*-
"""
图 4.7-2 跑浆的可达性检验（案例工程断面，匿名，相对高程以注浆水平为 0）。
全图 0.5 m 网格、交界面宽 1 m，与 144 个筛选配置及 C1 一致。
(a) I 段（充填体 10%）无交界面 / 有交界面（φ_c 0.35、宽 1 m）的所需源压场 p_req/p₀（0.5 m，只算可达域，
    存窿矿石 0.35、充填体底面连通、仰角 35°、未封底），实线为可达域边界（p_req = p₀）；
(b) 三段到跑浆中段的所需源压 p_req/p₀ 与交界面孔隙率的关系曲线（底面封闭、35°、未封底、存窿矿石 0.35）、
    比值 = 1 线、与 1 的交点（二分值，锚点 R27）与一维估算线（曲线及其与 1 的交点 0.309333，锚点 R28）。
数据：(a) cases/zhaojin_section.py + 可达域（与 outputs/zhaojin_screen 同一计算）；(b) outputs/check_4_7/reach.txt
B 节（scripts/check_4_7/reach.py，两遍逐字节一致）的 "0.5 / 1.0" 行与宽 1 m 二分行；一维估算为闭式
42.7·(λ(φ_c) − ρg·sin70°)/p₀（与 c1c2 的一维列相同）。
"""
from __future__ import print_function

import imp
import math
import os
import re

import numpy as np

import common as K
from common import plt, TERMS
from src.analysis.fill_diagnostics import reachable_domain

Z = imp.load_source("zhaojin_section", os.path.join(K.REPO, "cases", "zhaojin_section.py"))
CELL = 0.5
ZONE_NAME = {"z1_cuts1-6": u"I 段（孔隙率 10%）", "z2_cuts6-10": u"II 段（孔隙率 45%）", "z3_cuts10-15": u"III 段（孔隙率 25%）"}


def threshold_table():
    rows, cross = {}, {}
    phis = None
    key = {"I": "z1_cuts1-6", "II": "z2_cuts6-10", "III": "z3_cuts10-15"}
    for line in open(os.path.join(K.REPO, "outputs", "check_4_7", "reach.txt")):
        t = line.split()
        if t[:2] == ["grid/w", "zone"]:
            phis = [float(v) for v in t[3:]]
        elif t[:3] == ["0.5", "/", "1.0"] and len(t) > 5 and t[4] == "|":
            rows[key[t[3]]] = [float(v) for v in t[5:]]
        elif line.strip().startswith("bisection") and "width 1.0 m:" in line:
            cross = dict((key[a], float(b)) for a, b in re.findall(r"(I+) ([0-9.]+)", line.split("width 1.0 m:")[1]))
    return phis, rows, cross


def main():
    fig = plt.figure(figsize=(10.0, 5.6))
    bundle = Z.build_mesh(CELL)
    st, mx, my, _, _, _ = Z.build_state(Z.config(), CELL, mesh_bundle=bundle)
    for j, contacts in enumerate((False, True)):
        cfg = Z.config(zone="z1_cuts1-6", contacts=contacts, contact_width=1.0, contact_phi=0.35,
                       base_connected=True, ore_phi=0.35, hole_angle=35.0, seal=False)
        phi, region = Z.section_fields(mx, my, cfg)
        Z.set_porosity(st, phi)
        src = Z.hole_cells(mx, my, cfg, cell=CELL)
        cost = np.asarray(reachable_domain(st, src, p0=Z.P0, use_gravity=True)["cost"], dtype=float) / Z.P0
        out = Z.targets(mx, my, region, cfg)["outlet"]
        ratio = float(np.min(cost[out]))
        ax = fig.add_axes([0.05 + j * 0.27, 0.10, 0.24, 0.80])
        g, ext = K.grid(mx, my, np.minimum(cost, 2.0), CELL)
        e = K.rel(ext, 0.0, Z.Y_LEVEL)
        im = ax.imshow(g, origin="lower", extent=e, cmap="Purples_r", vmin=0, vmax=2.0, interpolation="nearest")
        rock, _ = K.grid(mx, my, (phi <= 0.01).astype(float), CELL)
        ax.imshow(np.ma.masked_where(rock < 0.5, rock), origin="lower", extent=e, cmap=K.LinearSegmentedColormap.from_list("s", [K.SOLID, K.SOLID]), interpolation="nearest")
        gx = np.linspace(e[0] + CELL / 2, e[1] - CELL / 2, g.shape[1])
        gy = np.linspace(e[2] + CELL / 2, e[3] - CELL / 2, g.shape[0])
        ax.contour(gx, gy, g, levels=[1.0], colors=[K.C1], linewidths=1.4)
        K.outline(ax, mx, my, region == Z.CONTACT, CELL, color=K.C2, x0=0.0, y0=Z.Y_LEVEL, lw=0.8)
        K.outline(ax, mx, my, region == Z.ORE, CELL, color=K.C3, x0=0.0, y0=Z.Y_LEVEL, lw=0.8)
        K.outline(ax, mx, my, out, CELL, color="#e34948", x0=0.0, y0=Z.Y_LEVEL, lw=1.2)
        ax.plot(mx[src], my[src] - Z.Y_LEVEL, ".", color="k", ms=2)
        ax.set_aspect("equal")
        ax.set_title(u"(a%d) I 段（孔隙率 10%%），%s交界面\n到跑浆中段 $p_\mathrm{req}/p_0$ %s" % (j + 1, u"有" if contacts else u"无",
                     (u"= %.2f" % ratio) if ratio < 50 else u"> 50（不可达）"), fontsize=8)
        ax.set_xlabel(TERMS["x"]); K.style(ax)
        if j == 0:
            ax.set_ylabel(u"相对高程 (m)（注浆水平为 0）")
    cax = fig.add_axes([0.585, 0.15, 0.01, 0.6])
    cb = fig.colorbar(im, cax=cax); cb.set_label(TERMS["p_req_rel"] + u"（截至 2）", fontsize=7)
    fig.text(0.05, 0.015, u"0.5 m 网格，交界面 $n_c$ 0.35、宽 1 m。蓝线：%s边界（$p_\mathrm{req}$ = $p_0$）；橙线：交界面；绿线：存窿矿石；红框：跑浆中段；黑点：钻孔出浆段；深灰：围岩与未采矿体（不透浆）"
             % TERMS["reach"], fontsize=7, color=K.INK2)
    ax = fig.add_axes([0.71, 0.14, 0.27, 0.74])
    phis, rows, cross = threshold_table()
    for zone, col, mk in (("z1_cuts1-6", K.C1, "o"), ("z2_cuts6-10", K.C2, "s"), ("z3_cuts10-15", K.C3, "^")):
        ax.plot(phis, rows[zone], "-" + mk, color=col, ms=4, label=ZONE_NAME[zone])
        ax.plot([cross[zone]] * 2, [0.2, 1.0], ":", color=col, lw=0.9)
    pf = np.linspace(0.15, 0.45, 100)
    lam = 6.0e4 * (1 - pf) / pf
    one_d = 42.7 * (lam - 1830 * 9.81 * math.sin(math.radians(70.0))) / Z.P0
    ax.plot(pf, one_d, "--", color=K.INK2, lw=1.0, label=u"一维估算")
    nc_1d = 0.309333                     # R28: closed-form crossing of the 1D estimate with 1
    ax.plot([nc_1d] * 2, [0.2, 1.0], "--", color=K.INK2, lw=0.9)
    ax.text(nc_1d + 0.005, 1.04, u"%.2f" % nc_1d, fontsize=7, color=K.INK2)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.text(0.245, 0.24, u"与 1 的交点（二分）\nI 段 %.4f\nII 段 %.4f\nIII 段 %.4f"
            % (cross["z1_cuts1-6"], cross["z2_cuts6-10"], cross["z3_cuts10-15"]), fontsize=7, color=K.INK2)
    ax.set_xlabel(u"交界面孔隙率 " + K.PHIC); ax.set_ylabel(u"到跑浆中段所需源压比 $p_\mathrm{req}/p_0$")
    ax.set_ylim(0.2, 1.8); ax.set_xlim(0.1, 0.47)
    ax.legend(loc="upper right", frameon=False, fontsize=7, numpoints=1)
    ax.set_title(u"(b) 到跑浆中段所需源压比与交界面孔隙率的关系", fontsize=8)
    K.style(ax)
    K.save(fig, "fig_4_7_2")


if __name__ == "__main__":
    main()
