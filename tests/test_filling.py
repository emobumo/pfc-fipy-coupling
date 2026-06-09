# -*- coding: utf-8 -*-
"""
Characterization tests for the filling-indicator update.

Locks in two invariants stated in the project docs:
  - filling is monotonically non-decreasing across steps,
  - filling never exceeds filling_limit_fraction * initial_porosity.

Runs under PFC 5.0's bundled Python 2.7, no PFC required.
"""
import unittest

import numpy as np

from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_slurry_step


def _make_state(nx=10, ny=6, dx=0.2, dy=0.2):
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=ny, dx=dx, dy=dy)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh))
    return state


class TestFilling(unittest.TestCase):

    def test_filling_is_non_decreasing_over_steps(self):
        state = _make_state()
        previous = np.asarray(state["filling"].value, dtype=float).copy()
        for _ in range(5):
            result = solve_slurry_step(state, dt=0.01)
            current = np.asarray(result["scalar_filling"], dtype=float)
            # Allow a tiny numerical tolerance, but no real decrease.
            self.assertTrue(
                np.all(current >= previous - 1.0e-12),
                "filling decreased between steps",
            )
            previous = current.copy()

    def test_filling_respects_porosity_cap(self):
        state = _make_state()
        params = state["slurry_parameters"]
        # Cap = filling_limit_fraction * initial porosity (no PFC -> default 0.35).
        initial_porosity = np.asarray(state["porosity"].value, dtype=float)
        cap = float(params["filling_limit_fraction"]) * initial_porosity
        for _ in range(10):
            result = solve_slurry_step(state, dt=0.05)
            current = np.asarray(result["scalar_filling"], dtype=float)
            self.assertTrue(
                np.all(current <= cap + 1.0e-9),
                "filling exceeded the porosity-based cap",
            )


if __name__ == "__main__":
    unittest.main()
