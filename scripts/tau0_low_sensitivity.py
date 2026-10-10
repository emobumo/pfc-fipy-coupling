# -*- coding: utf-8 -*-
"""
Reach-only sensitivity at a low yield stress, tau0 = 7.5 Pa (the order of the
dynamic Bingham yield stress of w/c 0.5 grout), A and rho at v0.6, 1.25 m.
No solver run. Everything is computed at tau0 = 30 (v0.6) and 7.5 side by side
on the same grids, so the comparison is like for like.

    powershell -File scripts\run_local.ps1 scripts\tau0_low_sensitivity.py

lambda scales with tau0, so at 7.5 Pa every L_max is 4x the v0.6 value and
Pi_g = rho g / lambda is 4x larger: Pi_g >= 1 where n >= 0.455 (cell value;
the reach graph prices faces, harmonic k and arithmetic n). Where Pi_g >= 1 a
step down costs less than nothing, the reach graph has negative edges and
reachable_domain switches from Dijkstra to Bellman-Ford; no repair is made
here -- the solver used, the negative-edge count and any non-finite cost are
reported as they come.

Part C does not call reachable_domain's per-source Bellman-Ford, which on the
0.5 m section (the -63 m drift has Pi_g ~ 1.8 at 7.5 Pa) takes minutes per
call. Every edge weight reachable_domain builds is (lambda integral) +
rho g (y_j - y_i), and the second term is a potential difference. With the
potential h = rho g y (Johnson's reweighting, with h known in closed form)
the reweighted graph is the lambda-only graph, all weights >= 0, so

    cost(v) = rho g y_v + min over sources s of [ d_lambda(s, v) - rho g y_s ]

with d_lambda from Dijkstra on _stencil_edges(..., use_gravity=False) -- the
same shortest paths, no Bellman-Ford. (scipy 0.15.1's csgraph.johnson was
tried first and reported a negative cycle that cannot exist: around any
closed loop the gravity terms cancel and the cost is sum lambda ds > 0, and
scipy's own bellman_ford on the same graph finds none.) --check compares
this with reachable_domain on one 1.25 m field and one section configuration.

A  Pi_g census of the structural fields (channel past the toe n 0.30 / 0.45 /
   0.60, channel at the collar and mid-hole n 0.45, gradient 0.12->0.30,
   random seeds 1-3 at 1.25 m, uniform 0.18): share and position of the
   Pi_g >= 1 cells, negative edges, solver, reach size, box contact, up/down.
B  flip criterion: scripts/flip_criterion.py group A (2D reach vs rays) and
   V3 (F family, F*) with a tau0 = 7.5 group next to v0.6.
C  case-study contact threshold C1: scripts/zhaojin_c1c2.c1 for tau0 = 30 and
   7.5 (same set-up: 0.5 m, contacts 1 m, base shut, 35 deg, ore 0.35), plus
   the cost ratio to the -63 m drift without a contact and with phi_c = 0.35.

Writes outputs/tau0_low_sensitivity/tau0_low_sensitivity.txt (and a sha1 of
all cost arrays for the two-run check).
"""
from __future__ import print_function

import hashlib
import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.analysis import fill_diagnostics as FD
from src.analysis.fill_diagnostics import reachable_domain

FC = imp.load_source("flip_criterion", os.path.join(REPO, "scripts", "flip_criterion.py"))
C, G = FC.C, FC.G
ZC = imp.load_source("zhaojin_c1c2", os.path.join(REPO, "scripts", "zhaojin_c1c2.py"))
Z = ZC.Z
OUT = os.path.join(REPO, "outputs", "tau0_low_sensitivity")
CELL = 1.25
TAU0S = (30.0, 7.5)
G_ACC = 9.81
DIGEST = hashlib.sha1()


def reach_potential(state, source_cells, p0=None, use_gravity=True, stencil=FD.DEFAULT_STENCIL):
    """reachable_domain via the gravity potential (see the module notes); same return keys.
    negative_edges counts the edges of the full graph with cost < 0, as reachable_domain does."""
    n = state["mesh"].numberOfCells
    src = FD._as_index_array(source_cells, n)
    if p0 is None:
        p0 = float(state.get("interior_dirichlet_value",
                             state["slurry_parameters"].get("inlet_pressure_core_value", 0.0)))
    rows, cols, lam_w = FD._stencil_edges(state, stencil, False)
    params = state["slurry_parameters"]
    rho = float(params.get("slurry_density", 2000.0))
    g = abs(float(params.get("gravity_y", -9.81))) if use_gravity else 0.0
    y = np.asarray(state["mesh"].cellCenters.value, dtype=float)[1]
    negative = int(np.sum(lam_w + rho * g * (y[cols] - y[rows]) < 0.0))
    d = np.atleast_2d(dijkstra(csr_matrix((lam_w, (rows, cols)), shape=(n, n)), directed=True, indices=src))
    cost = rho * g * y + np.min(d - rho * g * y[src][:, None], axis=0)
    cost[src] = 0.0
    return {"cost": cost, "reachable": cost <= float(p0), "budget": float(p0),
            "negative_edges": negative, "solver": "dijkstra+potential", "stencil": stencil}


def pi_g_cells(phi, tau0):
    lam = 2.0 * tau0 / np.sqrt(8.0 * G.A_CAL) * (1.0 - phi) / phi
    return G.RHO * G_ACC / lam


def part_a(lines):
    G.CELL = CELL
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fields = [("uniform 0.18", np.full(mx.size, G.PHI_UNIFORM)),
              ("gradient 0.12->0.30", FC.gradient_n(my))]
    for pb in C.BAND_PHIS:
        fields.append(("channel toe n %.2f" % pb, C.field("band", mx, my, phi_band=pb)))
    for name, bx in C.BAND_POSITIONS:
        if name != "toe":
            fields.append(("channel %s n %.2f" % (name, C.BAND_PHI_POS),
                           C.field("band", mx, my, phi_band=C.BAND_PHI_POS, band_x=bx)))
    for sd in C.RANDOM_SEEDS:
        fields.append(("random seed %d (1.25 m)" % sd, C.field("random", mx, my, seed=sd)))
    end, d, nrm = G.hole_geometry()
    lines += ["A  Pi_g census and reach, %.2f m, 16-neighbour, gravity on; Pi_g = rho g / lambda(n) per cell; "
              "'neg' = directed reach-graph edges with cost < 0" % CELL, "",
              "%-26s %5s | %6s %8s %-34s | %5s %-12s %6s %5s %9s %7s" % (
                  "field", "tau0", "max Pi", "Pi>=1 %", "Pi>=1 cells: n range, x / y range [m]",
                  "neg", "solver", "reach", "box", "finite", "up/dn")]
    rows = {}
    for name, phi in fields:
        for tau0 in TAU0S:
            G.TAU0 = tau0
            try:
                pg = pi_g_cells(phi, tau0)
                hi = pg >= 1.0
                where = "-"
                if hi.any():
                    where = "n %.2f-%.2f, x %.1f..%.1f / y %.1f..%.1f" % (
                        phi[hi].min(), phi[hi].max(), mx[hi].min(), mx[hi].max(), my[hi].min(), my[hi].max())
                t0 = time.time()
                try:
                    st, cx, cy, _, _, _, hc, _ = C.build_from_phi(phi)
                    r = reachable_domain(st, hc, p0=G.P0, use_gravity=True)
                    cost = np.asarray(r["cost"], dtype=float)
                    DIGEST.update(cost.tobytes())
                    hm = np.zeros(mx.size, dtype=bool)
                    hm[hc] = True
                    m = r["reachable"] & np.logical_not(hm)
                    perp = (cx[m] - G.MOUTH[0]) * nrm[0] + (cy[m] - G.MOUTH[1]) * nrm[1]
                    box = bool((cx[m].max() > G.X_MAX - CELL) or (cy[m].max() > G.Y_MAX - CELL)
                               or (cy[m].min() < G.Y_MIN + CELL))
                    fin = "%d/%d" % (int(np.isfinite(cost).sum()), cost.size)
                    res = (r["negative_edges"], r["solver"], int(m.sum()), box, fin, float(-perp.min() / perp.max()))
                except Exception as e:                  # reported, not repaired
                    res = (-1, "ERROR %s" % type(e).__name__, 0, False, "-", float("nan"))
                    lines.append("    %s tau0 %.1f: %s: %s" % (name, tau0, type(e).__name__, e))
                rows[(name, tau0)] = (float(pg.max()), 100.0 * hi.mean()) + res
                lines.append("%-26s %5.1f | %6.3f %8.2f %-34s | %5d %-12s %6d %5s %9s %7.3f" % (
                    name, tau0, pg.max(), 100.0 * hi.mean(), where, res[0], res[1], res[2],
                    "yes" if res[3] else "no", res[4], res[5]))
                print(lines[-1], "(%.1f s)" % (time.time() - t0))
                sys.stdout.flush()
            finally:
                G.TAU0 = FC.BASE["tau0"]
    lines += ["", "  Pi_g = 1 at n = %.4f for tau0 7.5 and n = %.4f for tau0 30 (cell value, rho %.0f, A %.3g)."
              % (n_at_pi1(7.5), n_at_pi1(30.0), G.RHO, G.A_CAL),
              "  L_max(0.18) = %.1f m at tau0 7.5 vs %.1f m at 30; the box is 60 x 60 m, mouth on its left edge." % (
                  G.P0 / (2 * 7.5 / np.sqrt(8 * G.A_CAL) * 0.82 / 0.18), G.P0 / (2 * 30.0 / np.sqrt(8 * G.A_CAL) * 0.82 / 0.18)),
              ""]
    return rows


def n_at_pi1(tau0):
    c = 2.0 * tau0 / np.sqrt(8.0 * G.A_CAL)
    return float(c / (c + G.RHO * G_ACC))


def part_b(lines):
    lines += ["B  flip criterion (scripts/flip_criterion.py, %.2f m)" % CELL, ""]
    for tau0 in TAU0S:
        FC.set_params(tau0, 1.0, 1830.0)
        try:
            sub = []
            FC.group_a(sub)
            lines += ["  tau0 = %.1f:" % tau0] + ["  " + s for s in sub]
        finally:
            FC.restore()
    FC.GROUPS = [("1 v0.6", 30.0, 1.0, 1830.0), ("5 tau0 7.5", 7.5, 1.0, 1830.0)]
    sub = []
    FC.group_v3(sub)
    lines += ["  " + s for s in sub]
    DIGEST.update(FC.DIGEST.digest())


def part_c(lines):
    lines += ["C  case-study contact threshold (scripts/zhaojin_c1c2.c1; 0.5 m, contacts 1 m, base shut, 35 deg, ore 0.35)", ""]
    mesh = Z.build_mesh(ZC.CELL)
    state, mx, my, phi, region, hc = Z.build_state(Z.config(), ZC.CELL, mesh_bundle=mesh)
    base = dict(state["slurry_parameters"])
    ZC.reachable_domain = reach_potential        # used by ZC.out_ratio inside c1
    for tau0 in TAU0S:
        ZC.groups = lambda t=tau0: [(30.0 / t, t, 1.0, 1830.0)]
        t0 = time.time()
        sub = ZC.c1(state, mx, my, base)
        lines += ["  " + s for s in sub] + [""]
        print("C1 tau0 %.1f done (%.0f s)" % (tau0, time.time() - t0))
        sys.stdout.flush()
    lines += ["  cost ratio to the -63 m drift (min path cost / p0) without a contact and with phi_c = 0.35;",
              "  'neg' = negative edges, solver as used:", ""]
    for tau0 in TAU0S:
        ZC.set_params(state, tau0, 1.0, 1830.0, base)
        for zone in ("z1_cuts1-6", "z2_cuts6-10", "z3_cuts10-15"):
            parts = []
            for contacts in (False, True):
                cfg = Z.config(zone=zone, contacts=contacts, contact_width=1.0, base_connected=False, seal=False,
                               hole_angle=35.0, ore_phi=0.35)
                if contacts:
                    cfg["contact_phi"] = 0.35
                p, region = Z.section_fields(mx, my, cfg)
                Z.set_porosity(state, p)
                src = Z.hole_cells(mx, my, cfg, cell=ZC.CELL)
                r = reach_potential(state, src, p0=Z.P0, use_gravity=True)
                cost = np.asarray(r["cost"], dtype=float)
                DIGEST.update(cost.tobytes())
                tg = Z.targets(mx, my, region, cfg)
                ratio = float(np.min(cost[tg["outlet"]])) / Z.P0
                parts.append("%s contact: %s (neg %d, %s)" % (
                    "with" if contacts else "no", ("%.3f" % ratio) if np.isfinite(ratio) else "inf",
                    r["negative_edges"], r["solver"]))
            lines.append("  tau0 %4.1f  %-14s %s" % (tau0, zone, "; ".join(parts)))
    ZC.set_params(state, base["yield_stress"], 1.0, base["slurry_density"], base)
    lines.append("")


def check():
    """reach_potential vs reachable_domain (Bellman-Ford) where the graph has negative edges."""
    G.CELL = CELL
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    mx = np.asarray(x, dtype=float)
    my = np.asarray(y, dtype=float)
    cases = []
    G.TAU0 = 7.5
    try:
        st, _, _, _, _, _, hc, _ = C.build_from_phi(C.field("band", mx, my, phi_band=0.60))
        cases.append(("1.25 m channel toe n 0.60, tau0 7.5", st, hc, G.P0, None))
    finally:
        G.TAU0 = FC.BASE["tau0"]
    mesh = Z.build_mesh(ZC.CELL)
    zs, zx, zy, _, _, _ = Z.build_state(Z.config(), ZC.CELL, mesh_bundle=mesh)
    base = dict(zs["slurry_parameters"])
    ZC.set_params(zs, 7.5, 1.0, 1830.0, base)
    cfg = Z.config(zone="z2_cuts6-10", contacts=True, contact_width=1.0, base_connected=False, seal=False,
                   hole_angle=35.0, ore_phi=0.35)
    cfg["contact_phi"] = 0.35
    p, region = Z.section_fields(zx, zy, cfg)
    Z.set_porosity(zs, p)
    cases.append(("0.5 m section z2, contact 0.35, tau0 7.5", zs, Z.hole_cells(zx, zy, cfg, cell=ZC.CELL), Z.P0,
                  Z.targets(zx, zy, region, cfg)["outlet"]))
    for name, st, src, p0, outlet in cases:
        t0 = time.time()
        a = reachable_domain(st, src, p0=p0, use_gravity=True)
        t1 = time.time()
        b = reach_potential(st, src, p0=p0, use_gravity=True)
        t2 = time.time()
        ca, cb = np.asarray(a["cost"]), np.asarray(b["cost"])
        fin = np.isfinite(ca) & np.isfinite(cb)
        print("%s: neg %d; %s %.1f s, %s %.1f s; same inf pattern %s; max |diff| %.3e Pa (max rel %.3e); "
              "reachable masks equal %s; bit-identical %s" % (
                  name, a["negative_edges"], a["solver"], t1 - t0, b["solver"], t2 - t1,
                  bool(np.array_equal(np.isfinite(ca), np.isfinite(cb))), float(np.max(np.abs(ca[fin] - cb[fin]))),
                  float(np.max(np.abs(ca[fin] - cb[fin]) / np.maximum(np.abs(ca[fin]), 1.0))),
                  bool(np.array_equal(a["reachable"], b["reachable"])), bool(np.array_equal(ca, cb))))
        if outlet is not None:
            print("   outlet cost ratio: bellman_ford %.12f  potential %.12f" % (
                float(np.min(ca[outlet])) / p0, float(np.min(cb[outlet])) / p0))
        sys.stdout.flush()


def main():
    if "--check" in sys.argv:
        check()
        return 0
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    lines = ["tau0 = 7.5 Pa reach-only sensitivity (A, rho at v0.6; 1.25 m structural fields, 0.5 m case-study section); "
             "tau0 = 30 recomputed alongside", ""]
    t0 = time.time()
    part_a(lines)
    part_b(lines)
    part_c(lines)
    lines.append("sha1 of all cost arrays: %s" % DIGEST.hexdigest())
    print("total %.0f s" % (time.time() - t0))
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "tau0_low_sensitivity.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
