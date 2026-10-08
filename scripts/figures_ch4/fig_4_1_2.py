# -*- coding: utf-8 -*-
"""
图 4.1-2 代表性结构场（同一孔隙率色标）：(a) 孔隙率梯度 0.12→0.30（上疏下密）；(b) 贯通通道（φ 0.45，三个位置
叠画于均质 0.18 背景，实算时每次只放一条）；(c) 随机非均质（均值 0.18、标准差 0.06、相关长度 7.5 m × 2.5 m，种子 1）；
(d) 架空骨架局部模型（块石深灰，周边疏松环 0.45、块石下压密细粒 0.03，变体 C）。
数据：由算例模块生成（cases/inclined_hole_gradient.py、inclined_hole_channel.py、boulder_bypass 结果文件），与重算所用场相同。
"""
from __future__ import print_function

import imp
import os

import numpy as np

import common as K
from common import plt, TERMS
from matplotlib.colors import LinearSegmentedColormap

C = imp.load_source("inclined_hole_channel", os.path.join(K.REPO, "cases", "inclined_hole_channel.py"))
G = C.G
PHI_CMAP = LinearSegmentedColormap.from_list("phi", ["#fff7e0", "#f2c879", "#d98b2b", "#8a4b0f"])
VMAX = 0.45


def show(ax, x, y, phi, cell, x0, y0, solid=None):
    g, ext = K.grid(x, y, phi, cell)
    im = ax.imshow(g, origin="lower", extent=K.rel(ext, x0, y0), cmap=PHI_CMAP, vmin=0.0, vmax=VMAX, interpolation="nearest")
    if solid is not None:
        gs, _ = K.grid(x, y, solid.astype(float), cell)
        ax.imshow(np.ma.masked_where(gs < 0.5, gs), origin="lower", extent=K.rel(ext, x0, y0),
                  cmap=LinearSegmentedColormap.from_list("s", [K.SOLID, K.SOLID]), interpolation="nearest")
    ax.set_aspect("equal")
    return im


def main():
    G.CELL = 2.5
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 2.5)
    mx, my = np.asarray(x), np.asarray(y)
    end = G.hole_geometry()[0]
    fig, axs = plt.subplots(1, 4, figsize=(12.0, 3.8))
    G.CELL = 1.25
    _, x1, y1, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 1.25)
    gx, gy = np.asarray(x1), np.asarray(y1)
    im = show(axs[0], gx, gy, G.phi_field("gradient", gy), 1.25, *K.MOUTH)
    G.CELL = 2.5
    axs[0].set_title(u"(a) 孔隙率梯度 0.12→0.30（上疏下密）", fontsize=8)
    ch = C.field("base", mx, my)
    for bx in (C.BAND_X_COLLAR, C.BAND_X_MID, C.BAND_X_TOE):
        f = C.field("band", mx, my, phi_band=0.45, band_x=bx)
        ch = np.maximum(ch, f)
    show(axs[1], mx, my, ch, 2.5, *K.MOUTH)
    for name, bx, yy in ((u"孔口", C.BAND_X_COLLAR, 36.5), (u"孔中", C.BAND_X_MID, 31.5), (u"孔底外", C.BAND_X_TOE, 36.5)):
        axs[1].text(bx - K.MOUTH[0] + 1.6, yy, name, ha="left", fontsize=6)
    axs[1].set_title(u"(b) 贯通通道 φ 0.45（三个位置，算例中各取其一）", fontsize=8)
    show(axs[2], mx, my, C.field("random", mx, my, seed=1), 2.5, *K.MOUTH)
    axs[2].set_title(u"(c) 随机非均质（种子 1）", fontsize=8)
    for a in axs[:3]:
        K.draw_hole(a, G.MOUTH, end)
        a.set_xlabel(TERMS["x"])
    axs[0].set_ylabel(TERMS["y"])
    f = np.load(os.path.join(K.RR, "D", "boulder", "final_C.npz"))
    show(axs[3], f["x"], f["y"], np.where(f["solid"], 0, f["phi"]), 0.5, 0.0, 6.0, solid=f["solid"])
    src = np.asarray(f["src"], dtype=int)
    axs[3].plot(f["x"][src], f["y"][src] - 6.0, "s", color="k", ms=3)
    axs[3].set_title(u"(d) 架空骨架局部模型（16 × 16 m）", fontsize=8)
    axs[3].set_xlabel(u"水平距离 (m)"); axs[3].set_ylabel(u"相对高程 (m)（口袋底为 0）")
    for a in axs:
        K.style(a)
    cax = fig.add_axes([0.93, 0.2, 0.01, 0.6])
    cb = fig.colorbar(im, cax=cax); cb.set_label(u"孔隙率 φ（≥ 0.45 同色）", fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.91, wspace=0.35)
    K.save(fig, "fig_4_1_2")


if __name__ == "__main__":
    main()
