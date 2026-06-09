# -*- coding: utf-8 -*-
"""
Gravity sign self-check.

Physical invariant: in a closed column driven only by gravity, hydrostatic
pressure must INCREASE with depth (deeper cells hold higher pressure).

Checks both DIRECTION (deeper > shallower) and MAGNITUDE (steady vertical
gradient ~= rho*g). The magnitude check is meaningful only after the no-flow
walls were made gravity-consistent (gravity flux zeroed on sealed exterior
faces); before that fix the gradient was ~2x the hydrostatic value.

Runs under PFC 5.0's bundled Python 2.7, no PFC required.
"""
import unittest

import numpy as np

from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_slurry_step


def _gravity_column_profile(nx=6, ny=12, dx=0.2, dy=0.2, steps=40, dt=0.05):
    """
    Build a closed column with a uniform reference pressure on top, so gravity
    is the only vertical driver, run to near steady state, and return the
    mean pressure per y-row sorted from bottom (low y) to top (high y).
    """
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=ny, dx=dx, dy=dy)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh))

    params = state["slurry_parameters"]
    width = nx * dx
    # Uniform top pressure across the full width: no horizontal injection
    # differential, so only gravity drives the vertical profile.
    params["inlet_zone_center_x"] = 0.5 * width
    params["inlet_center_x"] = 0.5 * width
    params["inlet_core_width_x"] = 10.0 * width
    params["inlet_spread_width_x"] = 10.0 * width
    params["inlet_zone_width_x"] = 10.0 * width
    params["inlet_pressure_core_value"] = 1.0e5
    params["inlet_pressure_value"] = 1.0e5
    params["inlet_pressure_spread_factor"] = 1.0
    params["inlet_core_min_fraction_of_spread"] = 0.99

    result = None
    for _ in range(steps):
        result = solve_slurry_step(state, dt=dt)

    p = np.asarray(result["scalar_pressure"], dtype=float)
    yv = np.asarray(np.asarray(state["y"]), dtype=float)
    rows = sorted(set(np.round(yv, 9)))
    profile = []
    for yr in rows:
        mask = np.isclose(yv, yr)
        profile.append((yr, float(np.mean(p[mask]))))
    return profile  # sorted bottom -> top


class TestGravitySign(unittest.TestCase):

    def test_pressure_increases_with_depth(self):
        profile = _gravity_column_profile()
        y_bottom, p_bottom = profile[0]
        y_top, p_top = profile[-1]
        self.assertGreater(
            p_bottom,
            p_top,
            "hydrostatic sign inverted: bottom p=%.2f should exceed top p=%.2f"
            % (p_bottom, p_top),
        )

    def test_profile_is_monotonic_in_depth(self):
        # Going from top down to bottom, pressure should not decrease.
        profile = _gravity_column_profile()
        pressures_top_down = [pm for _, pm in reversed(profile)]
        for shallow, deep in zip(pressures_top_down[:-1], pressures_top_down[1:]):
            self.assertGreaterEqual(
                deep,
                shallow - 1.0e-6,
                "pressure decreased going deeper (non-monotonic hydrostatic)",
            )

    def test_gradient_magnitude_matches_hydrostatic(self):
        # After the sealed-wall gravity fix, the steady vertical gradient
        # |dp/dy| should equal rho*g (not ~2x). Allow a 10% tolerance.
        profile = _gravity_column_profile()
        y_bottom, p_bottom = profile[0]
        y_top, p_top = profile[-1]
        slope = abs((p_top - p_bottom) / (y_top - y_bottom))

        # rho/g come from the same placeholder parameters used in the column.
        from src.models.slurry_transport.variables import (
            build_placeholder_slurry_parameters,
        )

        params = build_placeholder_slurry_parameters()
        rho = float(params["slurry_density"])
        g = abs(float(params["gravity_y"]))
        ideal = rho * g
        self.assertAlmostEqual(
            slope / ideal,
            1.0,
            delta=0.10,
            msg="steady gradient %.1f Pa/m vs hydrostatic %.1f Pa/m" % (slope, ideal),
        )


if __name__ == "__main__":
    unittest.main()
