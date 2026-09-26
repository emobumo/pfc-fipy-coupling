# -*- coding: utf-8 -*-
"""
Figures for the Zhaojin section runs (scripts/zhaojin_runs.py).

    powershell -File scripts\run_local.ps1 scripts\plot_zhaojin_runs.py [--anon]

Reads outputs/zhaojin_runs/final_*.npz and results.jsonl, writes
    fig_maps.png        grout saturation in the section, quota (top) vs arrival (bottom)
    fig_partition.png   where the injected volume is, against the injected volume
    fig_volumes.png     injected volume at the design volume / -56 m / -63 m
--anon writes the thesis versions fig_*_anon.png instead: anonymized labels,
elevations relative to the grouting level, the original-design runs A3-D3 on
the top row (see the section at the end of this file).
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "zhaojin_runs")

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

COLUMNS = [  # (quota tag, arrival tag, title)
    ("A", "A2", u"A  45% 段，有通道"),
    ("B", "B2", u"B  10% 段，有通道"),
    ("C", "C2", u"C  25% 段，有通道"),
    ("D", "D2", u"D  10% 段，无通道"),
    ("E", "E2", u"E  45% 段，封底"),
]
REASON = {"quota": u"注满设计浆量", "arrived_-63m": u"到达 -63 m",
          "stall": u"注不进（停滞）", "volume_cap": u"达上限", "max_steps": u"步数上限"}
# rock, fill, stored ore, contact, seal, outlet drift
REGION_CMAP = ListedColormap(["#8c8c8c", "#e8dcc0", "#c9a77c", "#a8c8e0", "#404040", "#ffffff"])
GROUT_CMAP = plt.get_cmap("Reds")


def load_results():
    rows = {}
    for line in open(os.path.join(OUT, "results.jsonl")):
        r = json.loads(line)
        rows[r["tag"]] = r
    return rows


def grid(npz):
    """Cell arrays -> 2D arrays on the uniform mesh (row = y)."""
    x, y = npz["x"], npz["y"]
    xs, ys = np.unique(x), np.unique(y)
    ix, iy = np.searchsorted(xs, x), np.searchsorted(ys, y)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2]

    def to2d(v, fill=np.nan):
        a = np.full((ys.size, xs.size), fill, dtype=float)
        a[iy, ix] = v
        return a
    return to2d, ext


def draw_map(ax, npz, r, title):
    to2d, ext = grid(npz)
    region, s = npz["region"], npz["s"].astype(float)
    ax.imshow(to2d(region), origin="lower", extent=ext, cmap=REGION_CMAP,
              vmin=-0.5, vmax=5.5, interpolation="nearest")
    grout = np.where(s > 0.02, s, np.nan)
    grout[npz["hole"]] = np.nan
    im = ax.imshow(np.ma.masked_invalid(to2d(grout)), origin="lower", extent=ext,
                   cmap=GROUT_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    hole = np.zeros(s.size, dtype=bool)
    hole[npz["hole"]] = True
    ax.imshow(np.ma.masked_where(~to2d(hole, 0).astype(bool), to2d(hole, 0)), origin="lower",
              extent=ext, cmap=ListedColormap(["#1030ff"]), interpolation="nearest")
    layer = float(Z.ZONES[r["zone"]]["layer"])
    for yl, name in ((Z.Y_LEVEL + layer, u"加固层顶"), (Z.Y_LEVEL, u"-26 m 注浆水平"),
                     (Z.Y_ORE_BOTTOM, u"-56 m"), (Z.Y_OUTLET, u"-63 m")):
        ax.axhline(yl, color="k", lw=0.5, ls="--")
        ax.text(ext[1] - 0.5, yl + 0.4, name, ha="right", va="bottom", fontsize=6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    ax.text(0.02, 0.02, u"V_in = %.1f m$^3$/m\n%s\n注浆水平以下 %.0f%%"
            % (r["v_in"], REASON.get(r["reason"], r["reason"]), 100 * r["below_share"]),
            transform=ax.transAxes, fontsize=6, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8))
    return im


def fig_maps(rows):
    fig, axes = plt.subplots(2, 5, figsize=(15, 9.5))
    im = None
    for j, (qt, at, title) in enumerate(COLUMNS):
        for i, tag in enumerate((qt, at)):
            ax = axes[i, j]
            path = os.path.join(OUT, "final_%s.npz" % tag)
            if tag not in rows or not os.path.exists(path):
                ax.set_axis_off()
                continue
            im = draw_map(ax, np.load(path), rows[tag],
                          title + (u"\n按设计浆量停" if i == 0 else u"\n按终压停（%s）" % tag))
            if j == 0:
                ax.set_ylabel(u"高程 y (m)", fontsize=7)
            if i == 1:
                ax.set_xlabel(u"x (m)，孔口 x=0", fontsize=7)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(u"浆体饱和度 S", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=REGION_CMAP(k)) for k in range(6)] + \
        [plt.Rectangle((0, 0), 1, 1, fc="#1030ff")]
    names = [u"围岩/实体矿", u"废石充填体", u"存窿矿石", u"交界面通道", u"封底", u"-63 m 巷道", u"注浆孔"]
    fig.legend(handles, names, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(u"招金断面：浆体分布（上：注满设计浆量；下：按 5 MPa 终压规则继续注）", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.08, wspace=0.12, hspace=0.25)
    path = os.path.join(OUT, "fig_maps.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_partition(rows):
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=False)
    for ax, (qt, at, title) in zip(axes, COLUMNS):
        r = rows.get(at)
        if r is None:
            ax.set_axis_off()
            continue
        ser = np.load(os.path.join(OUT, "final_%s.npz" % at))["series"]
        v = ser[:, 1]
        ax.stackplot(v, ser[:, 2], ser[:, 3], ser[:, 4],
                     colors=["#6aa84f", "#f1c232", "#cc4125"], alpha=0.85)
        for col, name in zip(["#6aa84f", "#f1c232", "#cc4125"], [u"加固层", u"加固层以上", u"注浆水平以下"]):
            ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=col, alpha=0.85, label=name))
        ax.plot(v, v, "k:", lw=0.8, label=u"累计注入")
        marks = [(r["quota"], u"设计浆量")]
        for key, name in (("reach_-56m", u"到 -56 m"), ("reach_-63m", u"到 -63 m")):
            if key in r["snaps"]:
                marks.append((r["snaps"][key]["v_in"], name))
        top = max(v[-1], r["v_in"]) * 1.05
        for k, (vm, name) in enumerate(marks):
            ax.axvline(vm, color="k", lw=0.7, ls="--")
            ax.text(vm, top * (0.97 - 0.12 * k), u" " + name + u"\n %.1f" % vm, fontsize=6.5, va="top",
                    ha="left" if k < 2 else "right", bbox=dict(fc="white", ec="none", alpha=0.7))
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(title + u"（%s）" % at, fontsize=9)
        ax.set_xlabel(u"累计注入量 V_in (m$^3$/m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(u"各区浆体体积 (m$^3$/m)", fontsize=8)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(u"注入的浆去了哪里（按终压停的算例，每米走向）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT, "fig_partition.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_volumes(rows):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    w = 0.26
    for j, (qt, at, title) in enumerate(COLUMNS):
        r = rows.get(at)
        if r is None:
            continue
        vals = [r["quota"]]
        for key in ("reach_-56m", "reach_-63m"):
            vals.append(r["snaps"][key]["v_in"] if key in r["snaps"] else None)
        for k, (val, col) in enumerate(zip(vals, ["#6aa84f", "#e69138", "#cc0000"])):
            xk = j + (k - 1) * w
            if val is None:
                ax.text(xk, 3, u"未到达", rotation=90, ha="center", va="bottom", fontsize=7)
                continue
            ax.bar(xk - w / 2, val, w, color=col)
            ax.text(xk, val + 2, "%.0f" % val, ha="center", fontsize=7)
        ax.text(j, -0.14, REASON.get(r["reason"], r["reason"]) + u"\n停于 %.0f" % r["v_in"],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#555555")
    ax.set_xlim(-0.6, len(COLUMNS) - 0.4)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels([c[2] for c in COLUMNS], fontsize=8)
    ax.set_ylabel(u"累计注入量 V_in (m$^3$/m)", fontsize=9)
    ax.bar(0, 0, color="#6aa84f", label=u"设计浆量")
    ax.bar(0, 0, color="#e69138", label=u"浆到达 -56 m 时")
    ax.bar(0, 0, color="#cc0000", label=u"浆到达 -63 m 时")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(u"跑到 -63 m 需要注多少浆（按 5 MPa 终压规则）", fontsize=11)
    fig.subplots_adjust(bottom=0.24)
    path = os.path.join(OUT, "fig_volumes.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --- anonymized versions for the thesis (--anon) ------------------------------
# The case study is anonymous in the thesis: no company or place names, no vein
# or cross-cut numbers, no absolute level elevations. Elevations are shown
# relative to the grouting level (y - Y_LEVEL); only the plotting changes.
# Top row: the ORIGINAL design stop (A3-D3, 28.6 m3/m, 22 m layer) -- the design
# in force when the runaway happened; E stays at the changed design (the seal
# is a post-change measure).

ANON_COLUMNS = [  # (top-row tag, arrival tag, title)
    ("A3", "A2", u"II 段（45%），有通道"),
    ("B3", "B2", u"I 段（10%），有通道"),
    ("C3", "C2", u"III 段（25%），有通道"),
    ("D3", "D2", u"I 段（10%），无通道"),
    ("E", "E2", u"II 段（45%），封底"),
]
ANON_REGIONS = [u"围岩/实体矿", u"废石充填体", u"存窿矿石", u"交界面通道", u"封底", u"跑浆巷道", u"注浆孔"]
REL_ORE = Z.Y_ORE_BOTTOM - Z.Y_LEVEL        # -30 m
REL_OUT = Z.Y_OUTLET - Z.Y_LEVEL            # -37 m


def _anon_reason(r):
    if r["reason"] == "quota":
        return u"注满原设计浆量" if r.get("design_mode") == "original" else u"注满变更设计浆量"
    return {"arrived_-63m": u"到达跑浆中段", "stall": u"注不进（停滞）",
            "volume_cap": u"达上限", "max_steps": u"步数上限"}.get(r["reason"], r["reason"])


def _layer_line(r, tag):
    """The design layer drawn in a panel: the original 22 m for the runaway
    columns, the changed-design layer for the seal column."""
    if tag.startswith("E"):
        top = float(Z.ZONES[r["zone"]]["layer"])
        return top, u"变更设计加固层顶（+%.0f m）" % top
    top = float(Z.ORIGINAL_DESIGN["layer"])
    return top, u"原设计加固层顶（+%.0f m）" % top


def draw_map_anon(ax, npz, r, tag, title):
    to2d, ext = grid(npz)
    ext = [ext[0], ext[1], ext[2] - Z.Y_LEVEL, ext[3] - Z.Y_LEVEL]
    region, s = npz["region"], npz["s"].astype(float)
    ax.imshow(to2d(region), origin="lower", extent=ext, cmap=REGION_CMAP,
              vmin=-0.5, vmax=5.5, interpolation="nearest")
    grout = np.where(s > 0.02, s, np.nan)
    grout[npz["hole"]] = np.nan
    im = ax.imshow(np.ma.masked_invalid(to2d(grout)), origin="lower", extent=ext,
                   cmap=GROUT_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    hole = np.zeros(s.size, dtype=bool)
    hole[npz["hole"]] = True
    ax.imshow(np.ma.masked_where(~to2d(hole, 0).astype(bool), to2d(hole, 0)), origin="lower",
              extent=ext, cmap=ListedColormap(["#1030ff"]), interpolation="nearest")
    top, top_name = _layer_line(r, tag)
    for yl, name in ((top, top_name), (0.0, u"注浆水平（0 m）"),
                     (REL_ORE, u"存窿矿石底界（%.0f m）" % REL_ORE),
                     (REL_OUT, u"跑浆中段（%.0f m）" % REL_OUT)):
        ax.axhline(yl, color="k", lw=0.5, ls="--")
        ax.text(ext[1] - 0.5, yl + 0.4, name, ha="right", va="bottom", fontsize=6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    ax.text(0.02, 0.02, u"注入 %.1f m$^3$/m\n%s\n注浆水平以下 %.0f%%"
            % (r["v_in"], _anon_reason(r), 100 * r["below_share"]),
            transform=ax.transAxes, fontsize=6, va="bottom",
            bbox=dict(fc="white", ec="none", alpha=0.8))
    return im


def fig_maps_anon(rows):
    fig, axes = plt.subplots(2, 5, figsize=(15, 9.5))
    im = None
    for j, (qt, at, title) in enumerate(ANON_COLUMNS):
        for i, tag in enumerate((qt, at)):
            ax = axes[i, j]
            path = os.path.join(OUT, "final_%s.npz" % tag)
            if tag not in rows or not os.path.exists(path):
                ax.set_axis_off()
                continue
            if i == 0:
                sub = u"按变更设计浆量停（封底后）" if tag == "E" else u"按原设计浆量停"
            else:
                sub = u"按终压停"
            im = draw_map_anon(ax, np.load(path), rows[tag], tag, title + u"\n" + sub)
            if j == 0:
                ax.set_ylabel(u"相对注浆水平高程 (m)", fontsize=7)
            if i == 1:
                ax.set_xlabel(u"水平距离 (m)，孔口 x = 0", fontsize=7)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(u"浆体饱和度 S", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=REGION_CMAP(k)) for k in range(6)] + \
        [plt.Rectangle((0, 0), 1, 1, fc="#1030ff")]
    fig.legend(handles, ANON_REGIONS, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(u"案例工程断面：浆体分布（上：按设计浆量停；下：按终压规则继续注）", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.08, wspace=0.12, hspace=0.25)
    path = os.path.join(OUT, "fig_maps_anon.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_partition_anon(rows):
    """Above / below the grouting level only: the changed-design layer split
    stored in the series does not apply to the original design."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=False)
    cols = ["#6aa84f", "#cc4125"]
    for ax, (qt, at, title) in zip(axes, ANON_COLUMNS):
        r = rows.get(at)
        if r is None:
            ax.set_axis_off()
            continue
        ser = np.load(os.path.join(OUT, "final_%s.npz" % at))["series"]
        v = ser[:, 1]
        ax.stackplot(v, ser[:, 2] + ser[:, 3], ser[:, 4], colors=cols, alpha=0.85)
        for col, name in zip(cols, [u"注浆水平以上", u"注浆水平以下"]):
            ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=col, alpha=0.85, label=name))
        ax.plot(v, v, "k:", lw=0.8, label=u"累计注入")
        if at.startswith("E"):
            marks = [(r["quota"], u"变更设计浆量")]
        else:
            marks = [(Z.ORIGINAL_DESIGN["grout_per_m"], u"原设计浆量")]
        for key, name in (("reach_-56m", u"到存窿矿石底界"), ("reach_-63m", u"到跑浆中段")):
            if key in r["snaps"]:
                marks.append((r["snaps"][key]["v_in"], name))
        top = max(v[-1], r["v_in"]) * 1.05
        for k, (vm, name) in enumerate(marks):
            ax.axvline(vm, color="k", lw=0.7, ls="--")
            ax.text(vm, top * (0.97 - 0.12 * k), u" " + name + u"\n %.1f" % vm, fontsize=6.5, va="top",
                    ha="left" if k == 0 else "right", bbox=dict(fc="white", ec="none", alpha=0.7))
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(u"累计注入量 (m$^3$/m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(u"各区浆体体积 (m$^3$/m)", fontsize=8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(u"注入的浆去了哪里（按终压规则继续注，每米走向）", fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    path = os.path.join(OUT, "fig_partition_anon.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_volumes_anon(rows):
    q = Z.ORIGINAL_DESIGN["grout_per_m"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    w = 0.34
    for j, (qt, at, title) in enumerate(ANON_COLUMNS):
        r = rows.get(at)
        if r is None:
            continue
        for k, (key, col) in enumerate((("reach_-56m", "#e69138"), ("reach_-63m", "#cc0000"))):
            xk = j + (k - 0.5) * w
            if key not in r["snaps"]:
                ax.text(xk, 3, u"未到达", rotation=90, ha="center", va="bottom", fontsize=7)
                continue
            val = r["snaps"][key]["v_in"]
            ax.bar(xk - w / 2, val, w, color=col)
            label = "%.0f" % val
            if not at.startswith("E"):
                label += u"\n(%.1f×)" % (val / q)
            ax.text(xk, val + 2, label, ha="center", fontsize=7)
        ax.text(j, -0.14, _anon_reason(r) + u"\n停于 %.0f" % r["v_in"],
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="#555555")
    ax.axhline(q, color="#6aa84f", lw=1.4, ls="--")
    ax.set_xlim(-0.6, len(ANON_COLUMNS) - 0.4)
    ax.set_ylim(0, 175)
    ax.set_xticks(range(len(ANON_COLUMNS)))
    ax.set_xticklabels([c[2] for c in ANON_COLUMNS], fontsize=8)
    ax.set_ylabel(u"累计注入量 (m$^3$/m)", fontsize=9)
    ax.plot([], [], color="#6aa84f", lw=1.4, ls="--", label=u"原设计浆量")
    ax.bar(0, 0, color="#e69138", label=u"浆到达存窿矿石底界时")
    ax.bar(0, 0, color="#cc0000", label=u"浆到达跑浆中段时")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(u"到达跑浆中段所需注浆量（按终压规则；括号内为原设计浆量的倍数）", fontsize=10)
    fig.subplots_adjust(bottom=0.24)
    path = os.path.join(OUT, "fig_volumes_anon.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rows = load_results()
    figs = (fig_maps_anon, fig_partition_anon, fig_volumes_anon) if "--anon" in argv \
        else (fig_maps, fig_partition, fig_volumes)
    for f in figs:
        print("wrote", f(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
