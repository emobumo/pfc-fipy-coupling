# -*- coding: utf-8 -*-
"""
Tests for the porosity -> permeability laws.

- Kozeny-Carman with a per-cell grain diameter: k = d^2/C * phi^3/(1-phi)^2.
- The legacy power-law mapping still works when selected via the formula switch.
"""
import unittest

import numpy as np

from src.models.slurry_transport.variables import build_placeholder_slurry_parameters
from src.coupling.porosity_to_permeability import (
    kozeny_carman_permeability,
    porosity_to_permeability,
)


class TestKozenyCarman(unittest.TestCase):

    def test_kc_matches_closed_form(self):
        params = build_placeholder_slurry_parameters()
        params["kozeny_carman_constant"] = 180.0
        phi = np.array([0.3])
        d = np.array([0.25])
        k = kozeny_carman_permeability(phi, d, params)
        expected = (0.25 ** 2 / 180.0) * (0.3 ** 3) / ((1.0 - 0.3) ** 2)
        self.assertAlmostEqual(float(k[0]) / expected, 1.0, places=6)

    def test_kc_increases_with_porosity(self):
        params = build_placeholder_slurry_parameters()
        d = np.array([0.25, 0.25])
        k = kozeny_carman_permeability(np.array([0.20, 0.40]), d, params)
        self.assertGreater(k[1], k[0])

    def test_kc_increases_with_grain_size(self):
        params = build_placeholder_slurry_parameters()
        phi = np.array([0.3, 0.3])
        k = kozeny_carman_permeability(phi, np.array([0.1, 0.3]), params)
        self.assertGreater(k[1], k[0])

    def test_formula_switch_uses_kozeny_carman_by_default(self):
        params = build_placeholder_slurry_parameters()
        # Default formula is kozeny_carman; with cell_diameter given it must
        # match the closed form rather than the legacy power law.
        phi = np.array([0.25])
        d = np.array([0.20])
        k = porosity_to_permeability(phi, params, cell_diameter=d)
        expected = (0.20 ** 2 / 180.0) * (0.25 ** 3) / ((1.0 - 0.25) ** 2)
        self.assertAlmostEqual(float(k[0]) / expected, 1.0, places=6)

    def test_legacy_power_law_still_available(self):
        params = build_placeholder_slurry_parameters()
        params["porosity_to_permeability_formula"] = "power_normalized_linear_range"
        k = porosity_to_permeability(np.array([0.5]), params)
        p_norm = (0.5 - params["porosity_min"]) / (params["porosity_max"] - params["porosity_min"])
        k_rel = p_norm ** params["porosity_to_permeability_exponent"]
        expected = params["permeability_min"] + (
            params["permeability_max"] - params["permeability_min"]
        ) * k_rel
        self.assertAlmostEqual(float(k[0]) / expected, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
