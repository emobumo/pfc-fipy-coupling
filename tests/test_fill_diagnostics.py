# -*- coding: utf-8 -*-
"""
Fill diagnostics (src/analysis/fill_diagnostics.py), Python 2.7 / unittest.

Synthetic-field unit tests (instant) plus one marched 1D integration test
that runs a dry strip to stall and checks that the reachable domain the
Dijkstra predicts is where the solver's front actually stopped.
"""
import math
import unittest

import numpy as np
from fipy import CellVariable

from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import (
    build_placeholder_slurry_parameters,
    initialize_slurry_variables,
)
from src.models.slurry_transport.equations import solve_transport_step
from src.coupling.porosity_to_permeability import porosity_to_permeability
from src.analysis.fill_diagnostics import (
    FILLED, UNREACHABLE, FRONT_SHORTFALL, BYPASS_VOID, OVERSHOOT,
    ENCLOSED_UNREACHABLE,
    face_start_gradient,
    interior_face_pairs,
    reachable_domain,
    classify_unfilled,
    fill_ratio_ladder,
    record_injection_rate,
    injection_rate_diagnostics,
    volume_distance_consistency,
    observed_spread,
)


# --- synthetic state builder ----------------------------------------------

TAU0 = 30.0
MU = 0.15
A_CAL = 1.25e-7
RHO = 1830.0


def _state(nx, ny, dx, phi, gravity_on=False, p0=3.0e5):
    """A solver-shaped state with prescribed fields and no solve."""
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=ny, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    params = build_placeholder_slurry_parameters()
    params["porosity_to_permeability_formula"] = "calibrated_power"
    params["calibrated_permeability_coefficient"] = A_CAL
    params["yield_stress"] = TAU0
    params["plastic_viscosity"] = MU
    params["slurry_density"] = RHO
    params["gravity_y"] = -9.81 if gravity_on else 0.0
    state.update(initialize_slurry_variables(mesh, params=params))
    phi = np.asarray(phi, dtype=float) + np.zeros(mesh.numberOfCells)
    k = porosity_to_permeability(phi, params)
    state["porosity"].setValue(phi)
    state["permeability"].setValue(k)
    state["interior_dirichlet_value"] = p0
    return state


def _lambda(phi):
    return 2.0 * TAU0 / (math.sqrt(8.0 * A_CAL) * phi / (1.0 - phi))


# --- reachable domain ------------------------------------------------------

class TestReachableDomain(unittest.TestCase):

    def test_uniform_strip_no_gravity_reaches_p0_over_lambda(self):
        """Cost grows as lambda*x; reachable iff x <= p0/lambda, exactly."""
        nx, dx, phi, p0 = 40, 0.1, 0.30, 2.0e5
        st = _state(nx, 1, dx, phi, p0=p0)
        r = reachable_domain(st, [0], p0=p0, use_gravity=False)
        lam = _lambda(phi)
        x = np.asarray(st["x"], dtype=float)
        expected_cost = lam * (x - x[0])
        np.testing.assert_allclose(r["cost"], expected_cost, rtol=1e-12)
        self.assertTrue(np.array_equal(r["reachable"], expected_cost <= p0))
        self.assertEqual(r["solver"], "dijkstra")
        self.assertEqual(r["negative_edges"], 0)
        # And the reach is neither trivial nor total.
        n_reach = int(np.sum(r["reachable"]))
        self.assertGreater(n_reach, 1)
        self.assertLess(n_reach, nx)

    def test_face_lambda_matches_closed_form_on_uniform_field(self):
        st = _state(10, 1, 0.1, 0.25)
        lam = face_start_gradient(st)
        np.testing.assert_allclose(lam, _lambda(0.25), rtol=1e-12)

    def test_reach_is_a_path_integral_not_the_endpoint(self):
        """Two layers: near lambda_1, far lambda_2. The reach in the far layer
        is set by the budget LEFT after crossing the near layer -- the
        endpoint's own lambda alone would get it wrong."""
        nx, dx = 60, 0.1
        phi = np.where(np.arange(nx) < 20, 0.30, 0.15)   # loose near, tight far
        p0 = 3.0e5
        st = _state(nx, 1, dx, phi, p0=p0)
        r = reachable_domain(st, [0], p0=p0, use_gravity=False)
        x = np.asarray(st["x"], dtype=float)

        lam1, lam2 = _lambda(0.30), _lambda(0.15)
        # Walk the 1D chain summing lambda_f*dx across each face crossed,
        # using the module's own face/cell pairing rather than assuming
        # FiPy's face numbering. The layer-boundary face carries harmonic k
        # and arithmetic phi of the two sides, and the walk picks that up.
        faces, a, b, dist = interior_face_pairs(st["mesh"])
        lam_f = face_start_gradient(st)[faces]
        step = {}
        for fa, fb, lf, ds in zip(a, b, lam_f, dist):
            step[(int(fa), int(fb))] = lf * ds
            step[(int(fb), int(fa))] = lf * ds
        cost = np.zeros(nx)
        for i in range(1, nx):
            cost[i] = cost[i - 1] + step[(i - 1, i)]
        np.testing.assert_allclose(r["cost"], cost, rtol=1e-12)

        # Endpoint-only reasoning would predict reach p0/lam2 from the source;
        # the path integral gives strictly more (the near layer is cheaper).
        naive_reach = p0 / lam2
        actual_reach = float(np.max(x[r["reachable"]]))
        self.assertGreater(actual_reach, naive_reach)
        # And strictly less than if everything were loose.
        self.assertLess(actual_reach, p0 / lam1)

    def test_gravity_makes_descent_cheaper_than_climb(self):
        """Vertical strip, source mid-height. Down: cost (lambda - rho g) per
        metre; up: (lambda + rho g). Reach ratio down/up = (l+rg)/(l-rg)."""
        ny, dx, phi, p0 = 81, 0.05, 0.30, 1.0e5
        st = _state(1, ny, dx, phi, gravity_on=True, p0=p0)
        mid = ny // 2
        r = reachable_domain(st, [mid], p0=p0, use_gravity=True)
        y = np.asarray(st["y"], dtype=float)
        lam = _lambda(phi)
        rg = RHO * 9.81
        self.assertLess(rg, lam, "fixture must keep Pi_g < 1")
        reach = r["reachable"]
        up = float(np.max(y[reach]) - y[mid])
        down = float(y[mid] - np.min(y[reach]))
        # Analytic reaches, quantized to the grid.
        up_exp = p0 / (lam + rg)
        down_exp = p0 / (lam - rg)
        self.assertLess(abs(up - up_exp), dx + 1e-12)
        self.assertLess(abs(down - down_exp), dx + 1e-12)
        self.assertGreater(down, up)
        self.assertEqual(r["solver"], "dijkstra")

    def test_pi_g_above_one_switches_to_bellman_ford(self):
        """If rho*g exceeds lambda, descending faces cost < 0; Dijkstra is
        invalid and Bellman-Ford must take over. Round trips stay positive,
        so there are no negative cycles and it terminates."""
        ny, dx, p0 = 41, 0.05, 1.0e5
        st = _state(1, ny, dx, 0.45, gravity_on=True, p0=p0)
        # Push lambda below rho*g by lowering tau0 on this state only.
        st["slurry_parameters"]["yield_stress"] = 0.5
        lam = float(face_start_gradient(st)[0])
        self.assertLess(lam, RHO * 9.81)
        mid = ny // 2
        r = reachable_domain(st, [mid], p0=p0, use_gravity=True)
        self.assertEqual(r["solver"], "bellman_ford")
        self.assertGreater(r["negative_edges"], 0)
        y = np.asarray(st["y"], dtype=float)
        # Everything below the source is reachable (net negative cost).
        self.assertTrue(np.all(r["reachable"][y < y[mid]]))
        self.assertTrue(np.all(np.isfinite(r["cost"])))

    def test_heterogeneous_2d_reach_follows_the_loose_channel(self):
        """A loose horizontal channel through tight ground: reach along the
        channel exceeds reach across it. A disc could not represent this."""
        n, dx, p0 = 31, 0.1, 2.0e5
        phi = np.zeros((n, n)) + 0.12
        phi[n // 2, :] = 0.35          # row = y index, channel along x
        st = _state(n, n, dx, phi.ravel(), p0=p0)
        src = [(n // 2) * n + n // 2]  # centre cell
        r = reachable_domain(st, src, p0=p0, use_gravity=False)
        reach = r["reachable"].reshape(n, n)
        along = int(np.sum(reach[n // 2, :]))
        across = int(np.sum(reach[:, n // 2]))
        self.assertGreater(along, across)
        self.assertGreater(along, 3)


# --- the graph metric ------------------------------------------------------

def _xy_index(nx, i):
    """(ix, iy) of FiPy cell i on an nx-wide Grid2D (x varies fastest)."""
    return i % nx, i // nx


class TestReachMetric(unittest.TestCase):
    """
    The stall condition integrates lambda along STRAIGHT rays, so the exact
    reach of a point source in uniform ground is a disc. A 4-neighbour
    shortest path measures |dx|+|dy| instead and returns a diamond -- 60% of
    the disc, at every mesh size -- and that bias was read for days as solver
    'overshoot'. These tests pin the metric, and the one thing a wider
    stencil could break: tunnelling through thin cemented walls.
    """

    PHI = 0.18
    CEMENT = 1.0e-3        # the stage_porosity_floor: lambda ~430x virgin

    def _uniform(self, n, dx):
        return _state(n, n, dx, self.PHI, p0=1.0e9)

    def test_legacy_stencil_is_the_manhattan_metric(self):
        """stencil=4 prices a cell at lambda*(|dx|+|dy|) exactly: the bias,
        stated as a fact so nobody mistakes it for the continuum again."""
        n, dx = 9, 0.5
        st = self._uniform(n, dx)
        c = (n * n) // 2
        r = reachable_domain(st, [c], use_gravity=False, stencil=4)
        x = np.asarray(st["x"], float)
        y = np.asarray(st["y"], float)
        manhattan = _lambda(self.PHI) * (np.abs(x - x[c]) + np.abs(y - y[c]))
        np.testing.assert_allclose(r["cost"], manhattan, rtol=1e-12)

    def test_default_stencil_tracks_the_straight_line(self):
        """Never below the straight-line cost (a lattice path cannot beat the
        line, so reach is only ever under-stated) and at most 2.7% above."""
        n, dx = 15, 0.5
        st = self._uniform(n, dx)
        c = (n * n) // 2
        r = reachable_domain(st, [c], use_gravity=False)
        self.assertEqual(r["stencil"], 16)
        x = np.asarray(st["x"], float)
        y = np.asarray(st["y"], float)
        line = _lambda(self.PHI) * np.hypot(x - x[c], y - y[c])
        away = line > 0
        ratio = r["cost"][away] / line[away]
        self.assertGreaterEqual(ratio.min(), 1.0 - 1e-12)
        self.assertLessEqual(ratio.max(), 1.0275)

    def test_disc_is_recovered_at_every_mesh_size(self):
        """The exact reach is a disc of radius R = p0/lambda. Stated so that
        cell quantization at the rim cannot make it flaky: every cell within
        R/1.0275 is reachable (the stencil's worst metric error) and no cell
        beyond R is. Checked at two mesh sizes -- a metric error does not
        shrink with refinement, so this confirms refinement is not what is
        doing the work. The legacy stencil fails the inner bound badly."""
        R_cells = 10
        n = 2 * R_cells + 5
        for dx in (0.5, 0.25):
            st = self._uniform(n, dx)
            c = (n * n) // 2
            R = R_cells * dx
            p0 = _lambda(self.PHI) * R
            x = np.asarray(st["x"], float)
            y = np.asarray(st["y"], float)
            dist = np.hypot(x - x[c], y - y[c])
            inner = dist <= R / 1.0275
            outer = dist > R * (1.0 + 1e-9)
            new = reachable_domain(st, [c], p0=p0, use_gravity=False)["reachable"]
            old = reachable_domain(st, [c], p0=p0, use_gravity=False,
                                   stencil=4)["reachable"]
            self.assertTrue(np.all(new[inner]),
                            "dx=%g: a cell inside the disc is unreachable" % dx)
            self.assertFalse(np.any(new[outer]),
                             "dx=%g: a cell outside the disc is reachable" % dx)
            disc = float(np.sum(dist <= R + 1e-9))
            self.assertGreater(np.sum(new) / disc, 0.95)
            self.assertLess(np.sum(old[inner]) / float(np.sum(inner)), 0.70,
                            "the legacy diamond should miss much of the disc")

    def test_diagonal_reach_matches_axis_reach(self):
        n, dx = 25, 0.5
        st = self._uniform(n, dx)
        c = (n * n) // 2
        p0 = _lambda(self.PHI) * 10 * dx
        r = reachable_domain(st, [c], p0=p0, use_gravity=False)
        x = np.asarray(st["x"], float)
        y = np.asarray(st["y"], float)
        reach = r["reachable"]
        on_axis = reach & (np.abs(y - y[c]) < 1e-9)
        on_diag = reach & (np.abs((x - x[c]) - (y - y[c])) < 1e-9)
        axis_r = np.max(np.abs(x[on_axis] - x[c]))
        diag_r = np.max(np.hypot(x[on_diag] - x[c], y[on_diag] - y[c]))
        self.assertGreaterEqual(diag_r / axis_r, 0.97)

    def _walled(self, n, dx, wall):
        """wall(ix, iy) -> True where the ground is cemented."""
        phi = np.zeros(n * n) + self.PHI
        for i in range(n * n):
            if wall(*_xy_index(n, i)):
                phi[i] = self.CEMENT
        return _state(n, n, dx, phi, p0=1.0e9)

    def test_long_edges_cannot_tunnel_a_one_cell_wall(self):
        """A single column of cemented cells across the whole domain. A
        knight-move edge priced at its end points would hop straight over
        it; priced along its segment it pays the cement and stops."""
        n, dx, w = 15, 0.5, 7
        p0 = _lambda(self.PHI) * 12 * dx     # ample: would reach far past w
        src = [0 * n + 1]                    # (ix=1, iy=0), left of the wall
        right = np.array([_xy_index(n, i)[0] > w for i in range(n * n)])

        open_ground = self._uniform(n, dx)
        r0 = reachable_domain(open_ground, src, p0=p0, use_gravity=False)
        self.assertTrue(np.any(r0["reachable"] & right),
                        "control: without the wall the far side is reachable")

        st = self._walled(n, dx, lambda ix, iy: ix == w)
        r = reachable_domain(st, src, p0=p0, use_gravity=False)
        self.assertFalse(np.any(r["reachable"] & right),
                         "grout crossed a sealed cemented wall")

    def test_a_diagonal_wall_seals_its_corners(self):
        """Cemented cells on the diagonal ix == iy touch only at corners. A
        diagonal edge through such a corner touches neither side cell; it
        must still be refused, because both of its detours are cemented."""
        n, dx = 15, 0.5
        p0 = _lambda(self.PHI) * 20 * dx
        st = self._walled(n, dx, lambda ix, iy: ix == iy)
        src = [0 * n + (n - 1)]              # (ix=n-1, iy=0): below the wall
        r = reachable_domain(st, src, p0=p0, use_gravity=False)
        above = np.array([_xy_index(n, i)[1] > _xy_index(n, i)[0]
                          for i in range(n * n)])
        self.assertFalse(np.any(r["reachable"] & above),
                         "grout squeezed through a corner between cemented cells")

    def test_one_open_side_keeps_a_corner_passable(self):
        """The corner rule must not over-block: with only ONE side cell
        cemented the grout goes round the open side, so the diagonal step
        costs what it costs in open ground."""
        n, dx = 5, 0.5
        c = 2 * n + 2
        lam = _lambda(self.PHI)
        one = self._walled(n, dx, lambda ix, iy: (ix, iy) == (3, 2))
        r = reachable_domain(one, [c], use_gravity=False)
        d = 3 * n + 3                        # (3, 3): diagonal of (2, 2)
        self.assertAlmostEqual(r["cost"][d] / (lam * math.sqrt(2.0) * dx), 1.0,
                               places=9)

        both = self._walled(n, dx, lambda ix, iy: (ix, iy) in ((3, 2), (2, 3)))
        r2 = reachable_domain(both, [c], use_gravity=False)
        self.assertGreater(r2["cost"][d], 1.5 * lam * math.sqrt(2.0) * dx,
                           "both detours cemented, yet the corner stayed cheap")


# --- classification --------------------------------------------------------

def _ring_state(n=21, dx=0.5, r_in=2.0, r_out=4.0, gap=False):
    """S=1 on an annulus around the centre, 0 inside and outside."""
    st = _state(n, n, dx, 0.30, p0=1.0e6)
    mesh = st["mesh"]
    c = np.asarray(mesh.cellCenters.value, dtype=float)
    cx, cy = c[0, (n * n) // 2], c[1, (n * n) // 2]
    d = np.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    s = np.where((d >= r_in) & (d <= r_out), 1.0, 0.0)
    if gap:
        # Cut the ring open along +x so the hole connects to the outside.
        s[(c[0] > cx) & (np.abs(c[1] - cy) < 0.75 * dx)] = 0.0
    st["saturation"].setValue(s)
    src = np.where((d >= r_in) & (d <= r_in + dx))[0][:1]
    return st, src, d


class TestClassifyUnfilled(unittest.TestCase):

    def test_enclosed_hole_is_a_bypass_void(self):
        st, src, d = _ring_state()
        reach = d <= 5.5                      # generous: covers the hole
        res = classify_unfilled(st, src, reach, s_c=0.5)
        cat = res["category"]
        hole = d < 2.0
        outside_reach = d > 5.5
        self.assertTrue(np.all(cat[hole] == BYPASS_VOID))
        self.assertEqual(res["n_bypass_voids"], 1)
        self.assertTrue(np.all(cat[outside_reach] == UNREACHABLE))
        ring_unfilled = (d > 4.0) & (d <= 5.5)
        self.assertTrue(np.all(cat[ring_unfilled] == FRONT_SHORTFALL))
        # Area bookkeeping over the reachable domain closes.
        t = res["over_reachable"]
        parts = t["filled"] + t["unreachable"] + t["front_shortfall"] + t["bypass_void"]
        self.assertAlmostEqual(parts, t["area"], places=9)
        self.assertEqual(t["unreachable"], 0.0)

    def test_open_hole_is_not_a_bypass_void(self):
        """Cut the ring: the hole now touches the exterior through the gap,
        so it is a front shortfall, not something the grout sealed in."""
        st, src, d = _ring_state(gap=True)
        reach = d <= 5.5
        res = classify_unfilled(st, src, reach, s_c=0.5)
        hole = d < 2.0
        self.assertTrue(np.all(res["category"][hole] == FRONT_SHORTFALL))
        self.assertEqual(res["n_bypass_voids"], 0)

    def test_target_mask_reports_design_shortfall(self):
        st, src, d = _ring_state()
        reach = d <= 5.5
        target = d <= 7.0          # design volume larger than the reach
        res = classify_unfilled(st, src, reach, s_c=0.5, target_mask=target)
        self.assertIn("over_target", res)
        self.assertGreater(res["design_shortfall_fraction"], 0.0)
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        expected = float(np.sum(vols[target & np.logical_not(reach)]))
        self.assertAlmostEqual(res["design_shortfall_area"], expected, places=9)


    def test_filled_outside_reach_is_overshoot(self):
        """Shrink the reach so the outer ring lies beyond it: those filled
        cells are OVERSHOOT, the inner hole stays a bypass void, and the
        categories still partition the domain exactly."""
        st, src, d = _ring_state()
        reach = d <= 3.0                       # cuts through the ring (2..4)
        res = classify_unfilled(st, src, reach, s_c=0.5)
        cat = res["category"]
        outer_ring = (d > 3.0) & (d <= 4.0)
        inner_ring = (d >= 2.0) & (d <= 3.0)
        self.assertTrue(np.all(cat[outer_ring] == OVERSHOOT))
        self.assertTrue(np.all(cat[inner_ring] == FILLED))
        self.assertTrue(np.all(cat[d < 2.0] == BYPASS_VOID))
        t = res["over_domain"]
        parts = sum(t[name] for name in ("filled", "unreachable", "front_shortfall",
                                         "bypass_void", "overshoot"))
        self.assertAlmostEqual(parts, t["area"], places=9)
        self.assertGreater(t["overshoot"], 0.0)

    # --- solids and dense inclusions (solid_phi) ---------------------------

    def test_without_solid_phi_the_new_category_never_appears(self):
        """Default stays the original classification: an enclosed pocket that
        is partly unreachable is NOT a bypass void, and nothing is labelled
        enclosed_unreachable."""
        st, src, d = _ring_state()
        reach = (d >= 1.0) & (d <= 5.5)        # the hole's core is unreachable
        res = classify_unfilled(st, src, reach, s_c=0.5)
        cat = res["category"]
        self.assertFalse(np.any(cat == ENCLOSED_UNREACHABLE))
        self.assertEqual(res["n_enclosed_unreachable"], 0)
        self.assertTrue(np.all(cat[d < 1.0] == UNREACHABLE))
        self.assertTrue(np.all(cat[(d >= 1.0) & (d < 2.0)] == FRONT_SHORTFALL))

    def test_unreachable_pocket_sealed_in_grout_is_a_dense_inclusion(self):
        st, src, d = _ring_state()
        reach = (d >= 2.0) & (d <= 5.5)        # the whole hole is unreachable
        res = classify_unfilled(st, src, reach, s_c=0.5, solid_phi=1.0e-2)
        cat = res["category"]
        self.assertTrue(np.all(cat[d < 2.0] == ENCLOSED_UNREACHABLE))
        self.assertEqual(res["n_enclosed_unreachable"], 1)
        self.assertEqual(res["n_bypass_voids"], 0)
        self.assertTrue(np.all(cat[d > 5.5] == UNREACHABLE))   # open ground stays

    def test_mixed_pocket_splits_by_reach(self):
        st, src, d = _ring_state()
        reach = (d >= 1.0) & (d <= 5.5)
        res = classify_unfilled(st, src, reach, s_c=0.5, solid_phi=1.0e-2)
        cat = res["category"]
        self.assertTrue(np.all(cat[d < 1.0] == ENCLOSED_UNREACHABLE))
        self.assertTrue(np.all(cat[(d >= 1.0) & (d < 2.0)] == BYPASS_VOID))
        t = res["over_domain"]
        parts = sum(t[name] for name in ("filled", "unreachable", "front_shortfall",
                                         "bypass_void", "overshoot", "enclosed_unreachable"))
        self.assertAlmostEqual(parts, t["area"], places=9)

    def test_solid_plug_closes_the_gap_and_is_never_a_void(self):
        """Cut the ring open, then plug the cut with rock: with solid_phi the
        rock walls the hole off (a bypass void again); the rock itself is
        not pore space and is never labelled a void. Without solid_phi the
        plugged hole still counts as open (the original behaviour)."""
        st, src, d = _ring_state(gap=True)
        c = np.asarray(st["mesh"].cellCenters.value, dtype=float)
        n = 21
        cx, cy = c[0, (n * n) // 2], c[1, (n * n) // 2]
        plug = (c[0] > cx) & (np.abs(c[1] - cy) < 0.75 * 0.5) & (d >= 2.0) & (d <= 4.0)
        phi = np.where(plug, 1.0e-3, 0.30)
        st["porosity"].setValue(phi)
        reach = (d <= 5.5) & np.logical_not(plug)
        hole = d < 2.0
        old = classify_unfilled(st, src, reach, s_c=0.5)
        self.assertTrue(np.all(old["category"][hole] == FRONT_SHORTFALL))
        res = classify_unfilled(st, src, reach, s_c=0.5, solid_phi=1.0e-2)
        cat = res["category"]
        self.assertTrue(np.all(cat[hole] == BYPASS_VOID))
        self.assertTrue(np.all(cat[plug] == UNREACHABLE))
        self.assertEqual(res["n_bypass_voids"], 1)

    def test_fill_ratio_ladder_is_monotone_in_threshold(self):
        """Raising S_c can only move cells out of 'filled'."""
        st, src, d = _ring_state()
        s = np.asarray(st["saturation"].value, dtype=float)
        s[(d >= 2.0) & (d <= 3.0)] = 0.6      # inner half of the ring partial
        st["saturation"].setValue(s)
        reach = d <= 5.5
        ladder = fill_ratio_ladder(st, src, reach, thresholds=(0.3, 0.5, 0.7))
        fractions = [res["over_reachable"]["filled_fraction"] for _, res in ladder]
        self.assertGreaterEqual(fractions[0], fractions[1])
        self.assertGreaterEqual(fractions[1], fractions[2])
        self.assertGreater(fractions[0], fractions[2])


# --- injection rate --------------------------------------------------------

class TestInjectionRate(unittest.TestCase):

    def test_exponential_decay_reads_as_stalled(self):
        t = np.linspace(0.0, 10.0, 101)
        hist = list(zip(t, np.exp(-t)))
        d = injection_rate_diagnostics(hist, tail_fraction=0.2, stall_ratio=0.05)
        self.assertTrue(d["stalled"])
        self.assertLess(d["decay_ratio"], 0.05)
        self.assertAlmostEqual(d["tail_slope"], -1.0, places=6)

    def test_plateau_reads_as_not_stalled(self):
        t = np.linspace(0.0, 10.0, 101)
        hist = list(zip(t, np.ones_like(t)))
        d = injection_rate_diagnostics(hist)
        self.assertFalse(d["stalled"])
        self.assertAlmostEqual(d["decay_ratio"], 1.0, places=12)
        self.assertAlmostEqual(d["tail_slope"], 0.0, places=9)

    def test_partial_decay_then_plateau_is_the_runaway_signature(self):
        """Falls to 30% of peak and stays there: not stalled, flat tail."""
        t = np.linspace(0.0, 20.0, 201)
        q = 0.3 + 0.7 * np.exp(-t)
        d = injection_rate_diagnostics(list(zip(t, q)), tail_fraction=0.2)
        self.assertFalse(d["stalled"])
        self.assertGreater(d["decay_ratio"], 0.25)
        self.assertGreater(d["tail_slope"], -1e-3)

    def test_record_uses_last_div_q_over_source(self):
        st = _state(10, 1, 0.1, 0.30)
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        div_q = np.zeros(10)
        div_q[0] = 2.0
        div_q[1] = 1.0
        st["last_div_q"] = div_q
        hist = record_injection_rate(st, [0, 1], dt=0.1, t=0.1, history=[])
        self.assertEqual(len(hist), 1)
        self.assertAlmostEqual(hist[0][1], 2.0 * vols[0] + 1.0 * vols[1], places=12)


# --- volume / distance -----------------------------------------------------

class TestVolumeDistance(unittest.TestCase):

    def test_uniform_strip_equivalent_radius(self):
        """1D strip, uniform phi: pore volume out to R is phi*w*R exactly, so
        r_equiv recovers R to within one cell."""
        nx, dx, phi = 50, 0.1, 0.30
        st = _state(nx, 1, dx, phi)
        w = dx            # strip width = cell height
        R = 2.0
        v_in = phi * w * R
        res = volume_distance_consistency(st, [0], v_in)
        self.assertLess(abs(res["r_equiv"] - R), dx + 1e-12)

    def test_channelling_shows_as_reach_ratio_above_one(self):
        """Observed spread three times what the volume could fill uniformly."""
        nx, dx, phi = 100, 0.1, 0.30
        st = _state(nx, 1, dx, phi)
        R = 1.0
        v_in = phi * dx * R
        res = volume_distance_consistency(st, [0], v_in, r_obs=3.0 * R)
        self.assertGreater(res["reach_ratio"], 2.5)
        self.assertLess(res["reach_ratio"], 3.5)
        self.assertLess(res["volume_ratio"], 0.4)

    def test_observed_spread_is_distance_to_nearest_source(self):
        st = _state(20, 1, 0.5, 0.30)
        s = np.zeros(20)
        s[:8] = 1.0
        st["saturation"].setValue(s)
        # Two source cells at 0 and 1; farthest filled is cell 7 -> 3.0 m from cell 1.
        self.assertAlmostEqual(observed_spread(st, [0, 1]), 3.0, places=12)


# --- integration: does the predicted reach match where the solver stopped? --

I_NX, I_DX, I_PHI, I_P0 = 40, 0.1, 0.30, 1.5e5
_I_CACHE = {}


def _march_to_stall():
    """Dry 1D strip, interior source at cell 0, march until the filled-cell
    count stops growing. Returns (state, history)."""
    if "state" in _I_CACHE:
        return _I_CACHE["state"], _I_CACHE["history"]
    st = _state(I_NX, 1, I_DX, I_PHI, p0=I_P0)
    p = st["slurry_parameters"]
    p.update({
        "rheology_model": "porous_bingham",
        "porous_bingham_activation": "threshold",
        "yield_truncation_mode": "papanastasiou",
        "pressure_coeff_form": "face",
        "enable_saturation_transport": True,
        "reference_storage": 1.0e-10,
        "picard_transient_mode": "backward_euler",
        "enable_front_dt_constraint": False,
        "dt_cfl": 0.5,
        "picard_relaxation_fill": 0.15,
        "picard_max_iters": 12,
        "picard_tol": 1.0e-3,
    })
    k = np.asarray(st["permeability"].value, dtype=float)
    for key in ("mobility_structural", "mobility_effective", "mobility", "intrinsic_mobility"):
        st[key].setValue(k / MU)
    mesh = st["mesh"]
    mask = np.zeros(I_NX)
    mask[0] = 1.0
    st["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    st["interior_dirichlet_value"] = I_P0
    s0 = np.zeros(I_NX)
    s0[0] = 1.0
    st["saturation"].setValue(s0)
    st["pressure"].setValue(np.where(mask > 0.5, I_P0, 0.0))

    history = []
    t = 0.0
    best, stall = -1, 0
    for _ in range(400):
        dt = solve_transport_step(st, dt_cap=1.0)
        t += float(dt)
        record_injection_rate(st, [0], dt, t, history)
        s = np.array(st["saturation"].value, copy=True)
        s[0] = 1.0
        st["saturation"].setValue(s)
        count = int(np.sum(s >= 0.5))
        if count > best:
            best, stall = count, 0
        else:
            stall += 1
        if stall >= 40:
            break
    _I_CACHE["state"] = st
    _I_CACHE["history"] = history
    return st, history


class TestIntegrationReachVsFront(unittest.TestCase):

    def test_predicted_reach_matches_the_stalled_front(self):
        """The Dijkstra reach and the solver's stalled front must agree to
        within the ~1-cell overshoot the ladder already documents for a
        sharp Bingham front."""
        st, _ = _march_to_stall()
        r = reachable_domain(st, [0], p0=I_P0, use_gravity=False)
        x = np.asarray(st["x"], dtype=float)
        reach_x = float(np.max(x[r["reachable"]]))
        front_x = observed_spread(st, [0], s_c=0.5)
        l_max = I_P0 / _lambda(I_PHI)
        self.assertLess(abs(reach_x - l_max), I_DX + 1e-12)
        self.assertLess(abs(front_x - reach_x), 2.0 * I_DX + 1e-12,
                        "front %.3f vs reach %.3f (L_max %.3f)" % (front_x, reach_x, l_max))

    def test_stalled_strip_shows_no_bypass_void_and_decayed_rate(self):
        st, history = _march_to_stall()
        r = reachable_domain(st, [0], p0=I_P0, use_gravity=False)
        res = classify_unfilled(st, [0], r["reachable"], s_c=0.5)
        self.assertEqual(res["n_bypass_voids"], 0)
        # Whatever is reachable-but-unfilled is a thin front band, not a hole.
        self.assertLess(res["over_reachable"]["front_shortfall_fraction"], 0.15)
        d = injection_rate_diagnostics(history, tail_fraction=0.2, stall_ratio=0.05)
        self.assertTrue(d["stalled"], "decay_ratio %.3g" % d["decay_ratio"])
        self.assertLess(d["tail_slope"], 0.0)

    def test_stalled_strip_is_volume_consistent(self):
        """Uniform fill to the front: the volume should match the distance,
        reach_ratio ~ 1."""
        st, _ = _march_to_stall()
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        s = np.asarray(st["saturation"].value, dtype=float)
        v_in = float(np.sum(I_PHI * s * vols)) - I_PHI * vols[0]   # exclude prefilled source
        r_obs = observed_spread(st, [0], s_c=0.5)
        res = volume_distance_consistency(st, [0], v_in, r_obs=r_obs)
        self.assertGreater(res["reach_ratio"], 0.8)
        self.assertLess(res["reach_ratio"], 1.25)


if __name__ == "__main__":
    unittest.main()
