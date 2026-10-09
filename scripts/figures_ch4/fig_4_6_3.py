# -*- coding: utf-8 -*-
"""
图 4.6-3 分段覆盖损失图：分段覆盖损失（理想整孔覆盖 − 7→H 分段覆盖，百分点）对 Δ/L_max，孔隙率组与孔长组
不同标记（1.25 m，现行停注判据）。竖线为 x_f/L_max 的中心值 0.87（ε = 0.1；全部段 0.83~0.91，灰带）。
数据：outputs/rerun_rate/C/results.json（P2 + P3，两遍一致）；覆盖 = 可达域中被充填的比例。
"""
from __future__ import print_function

import json
import os

import common as K
from common import plt, TERMS


def points():
    rows = [r for r in json.load(open(os.path.join(K.RR, "C", "results.json"))) if r["group"] in ("p2", "p3phi", "p3len")]
    single = dict(((r["n_ref"], r["H"]), r) for r in rows if r["plan"] == "single")
    out = []
    for r in rows:
        if r["plan"] != "7_%g" % r["H"]:
            continue
        s = single[(r["n_ref"], r["H"])]
        rec = {"phi": r["n_ref"], "H": r["H"], "dl": (r["H"] - 7.0) / r["L_max"],
               "loss": 100.0 * (s["over_reach_filled"] - r["over_reach_filled"])}
        if r["H"] == 17.0:
            out.append(dict(rec, g="phi"))
        if r["n_ref"] == 0.14:
            out.append(dict(rec, g="len"))
    return out


def main():
    pts = points()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axvspan(0.833, 0.906, color="#e4e3dd", zorder=0)
    ax.axvline(0.873, color=K.INK2, lw=0.8, ls="--")
    ax.text(0.915, 22, K.XF + u"/" + K.LMAX + u"（ε = 0.1）\n0.83~0.91", fontsize=7, color=K.INK2)
    for g, col, mk, lab in (("phi", K.C1, "o", u"孔隙率组（H = 17 m，φ 0.10~0.22）"), ("len", K.C2, "s", u"孔长组（φ 0.14，H = 10~32 m）")):
        sel = sorted([p for p in pts if p["g"] == g], key=lambda p: p["dl"])
        ax.plot([p["dl"] for p in sel], [p["loss"] for p in sel], linestyle="none", marker=mk, color=col,
                ms=8 if mk == "o" else 11, mec="white", mew=1.5, zorder=4 if mk == "o" else 3, label=lab)
        for p in sel:
            if g == "len" and p["H"] == 17.0:
                continue
            ax.annotate(u"φ %.2f" % p["phi"] if g == "phi" else u"%g m" % p["H"], (p["dl"], p["loss"]),
                        xytext=(0, 9 if g == "phi" else -14), textcoords="offset points", ha="center", fontsize=7, color=K.INK2)
    ax.set_xlabel(u"Δ / " + K.LMAX + u"（Δ = H − 7 m）")
    ax.set_ylabel(TERMS["stage_loss"] + u"（百分点）")
    ax.set_xlim(0.05, 2.05); ax.set_ylim(-5, 45)
    ax.legend(loc="upper right", frameon=False, numpoints=1)
    ax.set_title(TERMS["stage_loss"] + u"与 Δ/" + K.LMAX + u" 的关系", fontsize=9)
    K.style(ax)
    K.save(fig, "fig_4_6_3")
    for p in sorted(pts, key=lambda p: (p["g"], p["dl"])):
        print(p)


if __name__ == "__main__":
    main()
