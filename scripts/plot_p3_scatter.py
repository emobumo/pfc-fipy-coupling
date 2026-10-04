# -*- coding: utf-8 -*-
"""
4.6 P3: coverage loss of staging (single pass minus 7 -> H) against H / L_max
and against Delta / L_max, from the rate-rule reruns (layer C, pass 1; pass 2
is bit-identical).

    powershell -File scripts\run_local.ps1 scripts\plot_p3_scatter.py

Porosity group: H = 17 m, phi 0.10 .. 0.22 (P2 + P3 porosity runs, Delta 10 m).
Length group: phi 0.14, H = 10 / 17 / 25 / 32 m (Delta = H - 7).
Coverage = share of the virgin-reachable cells cemented. The dashed line in
(b) is the measured x_f / L_max at the eps = 0.1 stop (0.86 .. 0.89).
Outputs: outputs/rerun_rate/figs/p3_loss_scatter.png and p3_loss.csv.
"""
from __future__ import print_function

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(ROOT, "figs")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
C_PHI, C_LEN = "#2a78d6", "#eb6834"          # categorical slots 1, 2 (validated)
XF_OVER_L = 0.88


def points():
    rows = [r for r in json.load(open(os.path.join(ROOT, "C", "results.json")))
            if r["group"] in ("p2", "p3phi", "p3len")]
    single = dict(((r["n_ref"], r["H"]), r) for r in rows if r["plan"] == "single")
    out = []
    for r in rows:
        if r["plan"] != "7_%g" % r["H"]:
            continue
        s = single[(r["n_ref"], r["H"])]
        loss = 100.0 * (s["over_reach_filled"] - r["over_reach_filled"])
        st2 = r["stages"][1]["v_in"]
        rec = {"phi": r["n_ref"], "H": r["H"], "L_max": r["L_max"], "H_over_L": r["H"] / r["L_max"],
               "delta_over_L": (r["H"] - 7.0) / r["L_max"], "single": 100 * s["over_reach_filled"],
               "staged": 100 * r["over_reach_filled"], "loss": loss, "stage2_v": st2,
               "shadow": r["shadowed"], "toe_to_wall": r["toe_to_wall"]}
        if r["H"] == 17.0:
            out.append(dict(rec, group="phi"))
        if r["n_ref"] == 0.14:
            out.append(dict(rec, group="len"))
    return out


def main():
    pts = points()
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    with open(os.path.join(OUT, "p3_loss.csv"), "w") as fh:
        keys = ["group", "phi", "H", "L_max", "H_over_L", "delta_over_L", "single", "staged", "loss",
                "stage2_v", "shadow", "toe_to_wall"]
        fh.write(",".join(keys) + "\n")
        for p in sorted(pts, key=lambda p: (p["group"], p["H_over_L"])):
            fh.write(",".join(str(p[k]) if isinstance(p[k], str) else "%.4f" % p[k] for k in keys) + "\n")

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), facecolor=SURFACE)
    series = [("phi", C_PHI, "o", u"孔隙率组（H = 17 m，φ 0.10~0.22）"),
              ("len", C_LEN, "s", u"孔长组（φ 0.14，H = 10~32 m）")]
    for ax, key, xlabel in ((axes[0], "H_over_L", u"H / L_max"),
                            (axes[1], "delta_over_L", u"Δ / L_max（Δ = H - 7 m）")):
        ax.set_axis_bgcolor(SURFACE)          # matplotlib 1.4
        for g, col, mk, lab in series:
            sel = sorted([p for p in pts if p["group"] == g], key=lambda p: p[key])
            ax.plot([p[key] for p in sel], [p["loss"] for p in sel], linestyle="none", marker=mk,
                    markersize=8 if mk == "o" else 11, markerfacecolor=col, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=lab, zorder=4 if mk == "o" else 3)   # shared point: circle on square
            for p in sel:
                txt = (u"φ %.2f" % p["phi"]) if g == "phi" else (u"%g m" % p["H"])
                if g == "len" and p["H"] == 17.0:
                    continue                      # shares the point with phi 0.14
                if key == "delta_over_L" and g == "phi" and p["phi"] in (0.16, 0.18):
                    continue                      # crowded in (b); named in (a)
                dy = 2.2 if g == "phi" else -4.2
                ax.annotate(txt, (p[key], p["loss"]), xytext=(0, -16 if dy < 0 else 9),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        if key == "delta_over_L":
            ax.axvline(XF_OVER_L, color=INK2, linestyle="--", linewidth=1, zorder=1)
            ax.text(XF_OVER_L + 0.03, 41, u"x_f / L_max ≈ 0.88（ε = 0.1）", fontsize=8, color=INK2, va="top")
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(u"覆盖损失（百分点，单段 - 分段）", color=INK)
        ax.set_ylim(-5, 44)
        ax.set_xlim((0.55, 2.55) if key == "H_over_L" else (0.05, 2.05))
        ax.tick_params(top="off", right="off")
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=9)
    axes[0].set_title(u"(a) 对 H / L_max", fontsize=10, color=INK)
    axes[1].set_title(u"(b) 对 Δ / L_max：第 2 段进浆当且仅当 Δ > x_f", fontsize=10, color=INK)
    leg = axes[0].legend(loc="upper right", fontsize=8, frameon=False, numpoints=1)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    p = os.path.join(OUT, "p3_loss_scatter.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    for q in sorted(pts, key=lambda q: (q["group"], q["H_over_L"])):
        print("%-3s phi %.2f H %4.1f H/L %.2f D/L %.2f single %.1f staged %.1f loss %.1f stage2 V %.2f" % (
            q["group"], q["phi"], q["H"], q["H_over_L"], q["delta_over_L"], q["single"], q["staged"],
            q["loss"], q["stage2_v"]))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
