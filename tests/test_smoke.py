# -*- coding: utf-8 -*-
"""
Smoke / characterization tests for the FiPy-side slurry solve.

These run under PFC 5.0's bundled Python 2.7 and do NOT require PFC (itasca):
the solve path is pure FiPy. Run with:

    powershell -File scripts\run_local.ps1 -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_slurry_step


def _make_state(nx=10, ny=6, dx=0.2, dy=0.2):
    """Build a small mesh + initialized slurry state for testing."""
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=ny, dx=dx, dy=dy)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh))
    return state


class TestLinearBaseline(unittest.TestCase):
    """The default 'linear' rheology must not alter mobility or blow up."""

    def test_effective_equals_structural_in_linear_mode(self):
        state = _make_state()
        result = solve_slurry_step(state, dt=0.01)
        eff = np.asarray(result["scalar_mobility_effective"])
        struct = np.asarray(result["scalar_mobility_structural"])
        # In the linear baseline, effective mobility should stay equal to the
        # structural mobility (no yield/clogging attenuation).
        self.assertTrue(
            np.allclose(eff, struct),
            "linear mode changed mobility: max abs diff = %g"
            % float(np.max(np.abs(eff - struct))),
        )

    def test_pressure_is_finite(self):
        state = _make_state()
        result = solve_slurry_step(state, dt=0.01)
        pressure = np.asarray(result["scalar_pressure"])
        # No NaN / Inf may appear in the solved pressure field.
        self.assertTrue(
            np.all(np.isfinite(pressure)),
            "pressure field contains non-finite values",
        )

    def test_yield_factor_is_one_in_linear_mode(self):
        state = _make_state()
        result = solve_slurry_step(state, dt=0.01)
        yf = np.asarray(result["scalar_yield_factor"])
        # Linear baseline applies no yield gating: factor is identically 1.
        self.assertTrue(np.allclose(yf, 1.0))


if __name__ == "__main__":
    unittest.main()
