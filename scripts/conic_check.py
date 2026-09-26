# -*- coding: utf-8 -*-
"""
4.3 corollary 1 (V1): in a uniform field with gravity the reachable domain of a
point source is a conic with the source at its focus,

    r(theta) = L_max / (1 + Pi_g sin theta),   L_max = p0/lambda,  Pi_g = rho g / lambda,

an ellipse of eccentricity Pi_g, area pi L_max^2 / (1 - Pi_g^2)^(3/2); without
gravity the circle r = L_max.

    powershell -File scripts\run_local.ps1 scripts\conic_check.py [--plot]

A 100 x 100 m box, point source in the centre cell, uniform n in {0.10, 0.18,
0.30}, parameter set v0.6, production reachable_domain (16-neighbour), cells
1.25 and 0.625 m. The boundary radius along a ray is found where the cost
field, interpolated bilinearly between cell centres, reaches p0 -- sub-cell,
so the comparison is not swamped by the cell size. The 16-neighbour metric
can only lengthen paths, by at most 2.7%, so the reach may be UNDER-stated by
up to that much and never over-stated (plus interpolation error).
Solver-free; prints a digest of the cost arrays for the two-run comparison.
"""
from __future__ import print_function

import hashlib
import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fipy import CellVariable
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.analysis.fill_diagnostics import reachable_domain

Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "conic_check")
P0 = 5.0e6
HALF = 50.0
NS = (0.10, 0.18, 0.30)
CELLS = (1.25, 0.625)
N_SECTORS = 36
METRIC_BOUND = 0.0275


def lam_of(n, params):
    k = params["calibrated_permeability_coefficient"] * n ** 3 / (1.0 - n) ** 2
    return 2.0 * params["yield_stress"] / math.sqrt(8.0 * k / n)


def build(n, cell):
    mesh, x, y, fx, fy = build_mesh_for_domain(-HALF, HALF, -HALF, HALF, cell)
    params = Z.slurry_params()
    st = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    st.update(initialize_slurry_variables(mesh, params=params))
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Z.set_porosity(st, np.full(mx.size, n))
    src = int(np.argmin(mx ** 2 + my ** 2))
    return st, mx, my, src, params


def boundary_radius(cost, xs, ys, x0, y0, theta, rmax, dr):
    """First radius along the ray where the bilinear cost reaches P0."""
    grid = cost.reshape(len(ys), len(xs))
    prev_r, prev_c = 0.0, 0.0
    r = dr
    while r <= rmax:
        px, py = x0 + r * math.cos(theta), y0 + r * math.sin(theta)
        i = np.searchsorted(xs, px) - 1
        j = np.searchsorted(ys, py) - 1
        if i < 0 or j < 0 or i >= len(xs) - 1 or j >= len(ys) - 1:
            return float("nan")
        tx = (px - xs[i]) / (xs[i + 1] - xs[i])
        ty = (py - ys[j]) / (ys[j + 1] - ys[j])
        c = ((1 - tx) * (1 - ty) * grid[j, i] + tx * (1 - ty) * grid[j, i + 1]
             + (1 - tx) * ty * grid[j + 1, i] + tx * ty * grid[j + 1, i + 1])
        if c >= P0:
            return prev_r + (P0 - prev_c) / (c - prev_c) * (r - prev_r)
        prev_r, prev_c = r, c
        r += dr
    return float("nan")


def one(n, cell, gravity, digest):
    st, mx, my, src, params = build(n, cell)
    lam = lam_of(n, params)
    L = P0 / lam
    pig = params["slurry_density"] * 9.81 / lam if gravity else 0.0
    res = reachable_domain(st, [src], p0=P0, use_gravity=gravity)
    cost = np.asarray(res["cost"], dtype=float)
    digest.update(cost.tobytes())
    xs, ys = np.unique(mx), np.unique(my)
    x0, y0 = mx[src], my[src]
    area = float(np.sum(res["reachable"])) * cell * cell
    area_an = math.pi * L * L / (1.0 - pig * pig) ** 1.5
    r_an = lambda th: L / (1.0 + pig * math.sin(th))
    out = {"n": n, "cell": cell, "gravity": gravity, "L": L, "Pi_g": pig,
           "area_ratio": area / area_an, "cost": cost, "mx": mx, "my": my, "src": src,
           "reach": res["reachable"]}
    for name, th in (("up", math.pi / 2), ("down", -math.pi / 2), ("horiz", 0.0)):
        rm = boundary_radius(cost, xs, ys, x0, y0, th, HALF - 2 * cell, cell / 20.0)
        out[name] = (rm, r_an(th), rm / r_an(th) - 1.0)
    errs = []
    for k in range(N_SECTORS):
        th = 2 * math.pi * (k + 0.5) / N_SECTORS
        rm = boundary_radius(cost, xs, ys, x0, y0, th, HALF - 2 * cell, cell / 20.0)
        errs.append(rm / r_an(th) - 1.0)
    errs = np.array(errs)
    out["sector_min"] = float(errs.min())
    out["sector_max"] = float(errs.max())
    out["sector_rms"] = float(np.sqrt(np.mean(errs ** 2)))
    out["in_bound"] = bool(errs.min() >= -METRIC_BOUND - 0.005 and errs.max() <= 0.005)
    return out


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, n in zip(axes, NS):
        r = [q for q in results if q["n"] == n and q["cell"] == 1.25 and q["gravity"]][0]
        c = [q for q in results if q["n"] == n and q["cell"] == 1.25 and not q["gravity"]][0]
        mx, my = r["mx"], r["my"]
        xs, ys = np.unique(mx), np.unique(my)
        g = r["reach"].reshape(len(ys), len(xs)).astype(float)
        h = (xs[1] - xs[0]) / 2
        ax.imshow(np.ma.masked_where(g < 0.5, g), origin="lower", cmap="Blues", vmin=0, vmax=1.6,
                  extent=[xs[0] - h, xs[-1] + h, ys[0] - h, ys[-1] + h], interpolation="nearest")
        th = np.linspace(0, 2 * np.pi, 400)
        x0, y0 = mx[r["src"]], my[r["src"]]
        rr = r["L"] / (1 + r["Pi_g"] * np.sin(th))
        ax.plot(x0 + rr * np.cos(th), y0 + rr * np.sin(th), color="#eb6834", lw=1.6, label=u"解析椭圆（含重力）")
        ax.plot(x0 + c["L"] * np.cos(th), y0 + c["L"] * np.sin(th), color="#555555", lw=1.0, ls="--",
                label=u"解析圆（无重力）")
        ax.plot([x0], [y0], "k+", ms=9, mew=1.5, label=u"注浆点（焦点）")
        lim = 1.15 * r["L"] / (1 - r["Pi_g"])
        ax.set_xlim(x0 - lim, x0 + lim)
        ax.set_ylim(y0 - lim, y0 + lim)
        ax.set_aspect("equal")
        ax.set_title(u"孔隙率 %.2f" % n, fontsize=10)
        ax.set_xlabel(u"水平距离 (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(u"竖向距离 (m)", fontsize=8)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(u"均质场点源可达域（填色）与解析圆锥曲线", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "fig_conic_check.png")
    fig.savefig(p, dpi=160)
    return p


def main(argv):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    digest = hashlib.sha1()
    results = []
    lines = ["V1: uniform point source vs conic (16-neighbour reach, bilinear boundary; relative errors; "
             "metric bound: reach under-stated by <= 2.7%)", "",
             "%5s %6s %4s %7s %6s | %7s | %-22s %-22s %-22s | %7s %7s %7s %s" % (
                 "n", "cell", "grav", "L_max", "Pi_g", "area", "up (meas/an/err)", "down", "horizontal",
                 "sect.min", "sect.max", "rms", "in bound")]
    for n in NS:
        for cell in CELLS:
            for grav in (True, False):
                q = one(n, cell, grav, digest)
                results.append(q)
                f = lambda t: "%6.2f/%6.2f/%+5.1f%%" % (t[0], t[1], 100 * t[2])
                lines.append("%5.2f %6.3f %4s %7.2f %6.4f | %7.4f | %-22s %-22s %-22s | %+6.2f%% %+6.2f%% %6.2f%% %s" % (
                    n, cell, "on" if grav else "off", q["L"], q["Pi_g"], q["area_ratio"],
                    f(q["up"]), f(q["down"]), f(q["horiz"]),
                    100 * q["sector_min"], 100 * q["sector_max"], 100 * q["sector_rms"], q["in_bound"]))
                print(lines[-1])
                sys.stdout.flush()
    lines.append("")
    lines.append("up/down ratio of the point source, analytic (1-Pi_g)/(1+Pi_g): " + ", ".join(
        "n %.2f: %.4f" % (q["n"], (1 - q["Pi_g"]) / (1 + q["Pi_g"])) for q in results if q["cell"] == 1.25 and q["gravity"]))
    lines.append("digest %s" % digest.hexdigest())
    text = "\n".join(lines)
    open(os.path.join(OUT, "conic_check.txt"), "w").write(text + "\n")
    print(text.split("\n")[-2])
    print(text.split("\n")[-1])
    if "--plot" in argv:
        print("wrote", plot(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
