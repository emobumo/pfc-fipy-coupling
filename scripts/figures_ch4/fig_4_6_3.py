# -*- coding: utf-8 -*-
"""
图 4.6-3 分段覆盖损失图：分段覆盖损失（理想整孔覆盖率 − 分段覆盖率，百分点）对 Δ/L_max（Δ = H − d₁），
1.25 m，现行停注判据。算例代号按表 4.6-1：孔隙率组 S1～S6（圆点），S7（第 1 段孔深 5 m，三角），孔深组 S3、S8～S10
（方块）；S3 两组共用，以左右两半着色的圆点单独画出；S2 为临界（菱形）。点划线为 c（19 个独立段过原点最小二乘，
ε = 0.1），灰带为 19 段 x_f/L_max 的范围 0.83～0.91（SG01），即 c 的区间在本轴上的位置。
数据：outputs/rerun_rate/C/results.json（P2 + P3，两遍一致）；覆盖率 = Ω_reach^w 中被充填的单元比例
（over_reach_filled），数值与 docs/check_4_6_prewrite.md 第 11 项一致。
第十轮（框架 v2.7）：补入 S7；横轴名改为 Δ/L_max；分组与标注按清单。
"""
from __future__ import print_function

import json
import os

import common as K
from common import plt, TERMS

PURPLE = "#4a3aa7"
# label offsets (points) and alignment, chosen so that labels do not cover markers
OFFS = {"S1": (0, 9, "center"), "S9": (0, 9, "center"), "S10": (0, 9, "center"), "S8": (0, 9, "center"),
        "S2": (8, -3, "left"), "S3": (8, 1, "left"), "S7": (0, 9, "center"), "S4": (5, -13, "left"),
        "S5": (-8, -2, "right"), "S6": (-8, -2, "right")}


def points():
    rows = [r for r in json.load(open(os.path.join(K.RR, "C", "results.json"))) if r["group"] in ("p2", "p3phi", "p3len")]
    single = dict(((r["n_ref"], r["H"]), r) for r in rows if r["plan"] == "single")
    out = []
    for r in rows:
        code = K.s_code(r)
        if code is None:
            continue
        s = single[(r["n_ref"], r["H"])]
        out.append({"code": code, "n": r["n_ref"], "H": r["H"], "d1": r["depths"][0],
                    "dl": (r["H"] - r["depths"][0]) / r["L_max"],
                    "loss": 100.0 * (s["over_reach_filled"] - r["over_reach_filled"])})
    return out


def main():
    pts = dict((p["code"], p) for p in points())
    assert sorted(pts) == sorted(K.S_CODES.values())
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvspan(0.833, 0.906, color="#e4e3dd", zorder=0)
    ax.axvline(0.871, color=K.INK2, lw=0.8, ls="-.")
    style = dict(linestyle="none", mec="white", mew=1.2)
    groups = [(["S1", "S4", "S5", "S6"], dict(marker="o", color=K.C1, ms=8), u"孔隙率组（S1～S6）"),
              (["S8", "S9", "S10"], dict(marker="s", color=K.C2, ms=8), u"孔深组（S3、S8～S10）"),
              (["S3"], dict(marker="o", ms=10, fillstyle="left", color=K.C1, markerfacecoloralt=K.C2),
               u"S3（两组共用）"),
              (["S7"], dict(marker="^", color=PURPLE, ms=9), u"S7（第 1 段孔深 5 m）"),
              (["S2"], dict(marker="D", color=K.C3, ms=7), u"临界（S2）")]
    for codes, kw, lab in groups:
        kw = dict(style, **kw)
        ax.plot([pts[c]["dl"] for c in codes], [pts[c]["loss"] for c in codes], label=lab, zorder=4, **kw)
    for c, p in pts.items():
        if c in ("S8", "S9", "S10"):
            text = u"%s 终孔孔深 %g m" % (c, p["H"])
        elif c == "S3":
            text = u"S3 n %.2f，终孔孔深 %g m" % (p["n"], p["H"])
        else:
            text = u"%s n %.2f" % (c, p["n"])
        dx, dy, ha = OFFS[c]
        box = dict(boxstyle="square,pad=0.15", fc="white", ec="none") if c == "S3" else None
        ax.annotate(text, (p["dl"], p["loss"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=7, color=K.INK2, bbox=box, zorder=5)
    ax.set_xlabel(u"Δ/" + K.LMAX)
    ax.set_ylabel(TERMS["stage_loss"] + u"/百分点")
    ax.set_xlim(0.05, 2.05); ax.set_ylim(-5, 46)
    ax.legend(loc="upper right", frameon=False, numpoints=1, fontsize=7.5)
    ax.set_title(TERMS["stage_loss"] + u"与 Δ/" + K.LMAX + u" 的关系", fontsize=9)
    K.style(ax)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.01, 0.05, u"孔隙率组 S1～S6：终孔孔深 17 m、第 1 段孔深 7 m；S7：第 1 段孔深 5 m；孔深组 S3、S8～S10：n 0.14。"
             u"Δ 为新出浆段伸出第 1 段孔底的长度。", fontsize=7.5, color=K.INK2)
    fig.text(0.01, 0.015, u"灰带：19 个独立段 " + K.XF + u"/" + K.LMAX + u" 的范围 0.83～0.91，即段长系数 c 的区间在本轴上的位置；"
             u"点划线：c ≈ 0.87", fontsize=7.5, color=K.INK2)
    K.save(fig, "fig_4_6_3")
    for c in sorted(pts, key=lambda c: int(c[1:])):
        p = pts[c]
        print("  %-3s n %.2f H %4.1f d1 %.1f  Delta/L_max %.4f  loss %.6f" % (c, p["n"], p["H"], p["d1"], p["dl"], p["loss"]))


if __name__ == "__main__":
    main()
