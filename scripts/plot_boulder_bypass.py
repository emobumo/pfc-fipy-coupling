# -*- coding: utf-8 -*-
"""
Figure for cases/boulder_bypass.py.

    powershell -File scripts\run_local.ps1 scripts\plot_boulder_bypass.py

Top: grout saturation at the first moment a bypass void is sealed under the
lid, one panel per variant. Bottom: share of the pocket that is a sealed
bypass void (and, for C, a dense inclusion) against the injected volume --
the window in which stopping would leave the void behind.
"""
from __future__ import print_function

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "boulder_bypass")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}      # categorical slots 1-3
TITLES = {"A": u"A 架空块石（无周边环）", "B": u"B 架空块石 + 周边疏松环 0.45",
          "C": u"C 周边环 + 块石下压密细粒 0.03"}
SAT = LinearSegmentedColormap.from_list("sat", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#222222", "#6b6b6b"


def grid(d, v):
    xs, ys = np.unique(d["x"]), np.unique(d["y"])
    a = np.full((ys.size, xs.size), np.nan)
    a[np.searchsorted(ys, d["y"]), np.searchsorted(xs, d["x"])] = v
    h = (xs[1] - xs[0]) / 2
    return a, [xs[0] - h, xs[-1] + h, ys[0] - h, ys[-1] + h]


def main():
    rows = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(OUT, "results.jsonl")))
    tags = [t for t in ("A", "B", "C") if t in rows]
    fig = plt.figure(figsize=(12, 8.2))
    for j, tag in enumerate(tags):
        d = np.load(os.path.join(OUT, "final_%s.npz" % tag))
        s = d["first_bypass"] if d["first_bypass"].size else d["s"]
        ax = fig.add_subplot(2, 3, j + 1)
        img, ext = grid(d, np.clip(s, 0, 1))
        im = ax.imshow(img, origin="lower", extent=ext, cmap=SAT, vmin=0, vmax=1, interpolation="nearest")
        solid, _ = grid(d, d["solid"].astype(float))
        ax.imshow(np.ma.masked_where(solid < 0.5, solid), origin="lower", extent=ext,
                  cmap=LinearSegmentedColormap.from_list("k", ["#4a4a4a", "#4a4a4a"]), interpolation="nearest")
        for name, lw, ls in (("rim", 0.8, "--"), ("pocket", 1.4, "-")):
            m, _ = grid(d, d[name].astype(float))
            if np.nanmax(m) > 0:
                ax.contour(m, levels=[0.5], extent=ext, colors=MUTED if name == "rim" else "#eb6834",
                           linewidths=lw, linestyles=ls)
        ax.set_xlim(-5, 5)
        ax.set_ylim(3, 15)
        ax.set_aspect("equal")
        ax.set_title(TITLES[tag], fontsize=9, color=INK)
        w = rows[tag]["bypass_window_v"]
        ax.text(0.02, 0.02, u"首次封闭时 V_in = %.1f m$^3$/m" % (w[0] if w else float("nan")),
                transform=ax.transAxes, fontsize=7, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85))
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
    cax = fig.add_axes([0.92, 0.56, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(u"浆体饱和度 S", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(2, 1, 2)
    for tag in tags:
        ser = rows[tag]["series"]
        v = np.array([r["v_in"] for r in ser])
        frac_v = v / rows[tag]["cap"]
        by = np.array([r["pocket_bypass"] for r in ser])
        ax.plot(frac_v, 100 * by, color=SERIES[tag], lw=2, label=TITLES[tag] + u"：绕流空区")
        if tag == "C":
            de = np.array([r["pocket_enclosed_unreachable"] for r in ser])
            ax.plot(frac_v, 100 * de, color=SERIES[tag], lw=2, ls=":", label=TITLES[tag] + u"：被包围的不可达区")
    ax.set_xlabel(u"累计注入量 / 可达孔隙体积", fontsize=9, color=INK)
    ax.set_ylabel(u"块石下方空间中未填且被封闭的比例 (%)", fontsize=9, color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 100)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(1.0, -0.16, u"曲线非零的区间 = 此时停注会留下封闭空区；注到停滞后 A、B 最终全部填满，C 的末段由正则化蠕动填入（见正文）",
            transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    fig.suptitle(u"大块石架空结构下的绕流空区：出现时机与持续区间", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.92, bottom=0.1, hspace=0.3, wspace=0.15)
    path = os.path.join(OUT, "fig_boulder_bypass.png")
    fig.savefig(path, dpi=160)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
