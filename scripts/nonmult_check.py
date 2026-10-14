# -*- coding: utf-8 -*-
"""
Non-multiplicativity of the up/down ratio R_ud (4.5.2), both read-outs.

    powershell -File scripts\\run_local.ps1 scripts\\nonmult_check.py

Four line-source cases: uniform 0.18 / gradient 0.12 -> 0.30, gravity on /
off. Single-factor product  P = R(uniform, on) x R(gradient, off) / R(uniform, off)
against the combined value R(gradient, on); difference D = P - R(gradient, on).

  reach  -- 16-neighbour reachable domain (p_req <= p0), 1.25 m and 0.625 m
  solver -- outputs/rerun_rate/A/final_*_1.25*.npz (S >= 0.5, hole excluded),
            1.25 m only (no other grid has all four solver runs)
  rays   -- straight-ray estimate of scripts/flip_criterion.py (ray_ratio):
            "max" over 51 hole points (rays stopped by the box), and the
            hole mid-point ray with the side walls ignored ("mid_free")

Each R_ud is read two ways, as in scripts/updown_readout.py: cell centres
(the official anchors) and bilinear sub-grid (dx/10). Error of D from the
read-out errors e of the four ratios: worst case
    |dD| <= P (e_a/R_a + e_b/R_b + e_c/R_c) + e_d
and the root-sum-square of the same four terms; e = the RD01 bound for the
grid (1.25 m: 0.03), and separately the observed |sub - cell| of each ratio.

Output: outputs/nonmult_check/nonmult_check.txt (+ md5 of all p_req fields)
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

from src.analysis.fill_diagnostics import reachable_domain

U = imp.load_source("updown_readout", os.path.join(REPO, "scripts", "updown_readout.py"))
FC = imp.load_source("flip_criterion", os.path.join(REPO, "scripts", "flip_criterion.py"))
G = U.G
OUT = os.path.join(REPO, "outputs", "nonmult_check")
CASES = (("uniform", True), ("gradient", False), ("uniform", False), ("gradient", True))   # a, b, c, d
NAMES = ("uniform, gravity", "gradient, no gravity", "uniform, no gravity", "gradient, gravity")
DIGEST = hashlib.md5()


def reach_rows(cell):
    G.CELL = cell
    try:
        out = []
        for kind, grav in CASES:
            st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
            hc = np.asarray(hc, dtype=int)
            cost = np.asarray(reachable_domain(st, hc, p0=G.P0, use_gravity=grav)["cost"], dtype=float)
            DIGEST.update(cost.tobytes())
            m = cost <= G.P0
            m[hc] = False
            c = U.extents_cells(m, mx, my)
            s = U.extents_subgrid(np.minimum(cost, 10.0 * G.P0), mx, my, lambda v: v <= G.P0)
            out.append((c, s, len(hc)))
        return out
    finally:
        G.CELL = 2.5


def solver_rows():
    G.CELL = 1.25
    try:
        out = []
        for kind, grav in CASES:
            st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
            hc = np.asarray(hc, dtype=int)
            f = np.load(os.path.join(U.RR, "final_%s_1.25%s.npz" % (kind, "" if grav else "_nograv")))
            assert np.allclose(f["x"], mx) and np.allclose(f["y"], my)
            sat = np.asarray(f["s"], dtype=float)
            filled = sat >= 0.5
            filled[hc] = False
            c = U.extents_cells(filled, mx, my)
            sat_sub = sat.copy()
            sat_sub[hc] = 0.0
            s = U.extents_subgrid(sat_sub, mx, my, lambda v: v >= 0.5)
            out.append((c, s, len(hc)))
        return out
    finally:
        G.CELL = 2.5


def ray_rows():
    out = []
    for kind, grav in CASES:
        nfun = (lambda y: G.PHI_UNIFORM) if kind == "uniform" else FC.gradient_n
        rr = FC.ray_ratio(nfun, grav)
        out.append(rr)
    return out


def combine(r, e):
    ra, rb, rc, rd = r
    p = ra * rb / rc
    terms = [p * e[0] / ra, p * e[1] / rb, p * e[2] / rc, e[3]]
    return p, p - rd, sum(terms), math.sqrt(sum(t * t for t in terms))


def block(lines, title, rows, e_bound):
    rc = [c[0] / c[1] for c, s, n in rows]
    rs = [s[0] / s[1] for c, s, n in rows]
    obs = [abs(b - a) for a, b in zip(rc, rs)]
    lines.append(title)
    for name, (c, s, n), a, b in zip(NAMES, rows, rc, rs):
        lines.append("  %-22s cell up %7.3f down %7.3f R %.4f | sub up %7.3f down %7.3f R %.4f | sub-cell %+.4f"
                     % (name, c[0], c[1], a, s[0], s[1], b, b - a))
    for lab, r in (("cell centres", rc), ("sub-grid", rs)):
        p, d, wc, rss = combine(r, [e_bound] * 4)
        p2, d2, wc2, rss2 = combine(r, obs)
        lines.append("  %-12s product %.4f  combined %.4f  D = %+.4f | error (e = %.2f): worst %.4f, rss %.4f | "
                     "error (observed |sub-cell|): worst %.4f, rss %.4f" % (lab, p, r[3], d, e_bound, wc, rss, wc2, rss2))
    lines.append("")
    return rc, rs


def main():
    lines = ["Non-multiplicativity of R_ud: P = R(uni, g) x R(grad, no g) / R(uni, no g) vs R(grad, g); D = P - R(grad, g)",
             "read-outs: cell centres (official) / bilinear sub-grid dx/10; line source 17 m, 35 deg", ""]
    r125 = reach_rows(1.25)
    lines.append("hole (source) cells: 1.25 m %d" % r125[0][2])
    r0625 = reach_rows(0.625)
    lines.append("hole (source) cells: 0.625 m %d" % r0625[0][2])
    G.CELL = 2.5
    st, mx, my, nx, ny, phi, hc, k = G.build_case("uniform")
    lines += ["hole (source) cells: 2.5 m %d" % len(hc), ""]
    block(lines, "reach 1.25 m", r125, 0.03)
    block(lines, "reach 0.625 m (error bound taken as the observed |sub-cell|; RD01 has no 0.625 m value)", r0625, 0.03)
    block(lines, "solver 1.25 m (rate rule, outputs/rerun_rate/A)", solver_rows(), 0.03)
    rays = ray_rows()
    lines.append("straight rays (scripts/flip_criterion.py ray_ratio, analytic: no read-out error)")
    for name, rr in zip(NAMES, rays):
        lines.append("  %-22s max %.4f (clipped %s)  mid_wall %.4f  mid_free %.4f" % (
            name, rr["ratio"], rr["clipped"], rr["mid"], rr["mid_free"]))
    for key in ("ratio", "mid_free"):
        r = [rr[key] for rr in rays]
        p = r[0] * r[1] / r[2]
        lines.append("  %-9s product %.4f  combined %.4f  D = %+.4f" % ("max" if key == "ratio" else key, p, r[3], p - r[3]))
    lines += ["", "md5 of all reach p_req fields: %s" % DIGEST.hexdigest()]
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "nonmult_check.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
