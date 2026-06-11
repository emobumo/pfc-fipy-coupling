# -*- coding: utf-8 -*-
"""
Regression test for the PFC->FiPy porosity cell binning.

The index map (built from cell centers) must agree with the floor-based cell
index that the ball-binning loop computes, otherwise every cell key is shifted
and the top row / right column are left empty. This previously broke because the
index map used round() while the ball loop used floor() (and round(i+0.5)==i+1
on Python 2.7).
"""
import math
import unittest

import numpy as np

from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.pfc_adapter.porosity_reader import _build_cell_index_map


class TestPorosityBinning(unittest.TestCase):

    def _check_domain(self, x_min, x_max, y_min, y_max, cell):
        mesh, x, y, fx, fy = build_mesh_for_domain(x_min, x_max, y_min, y_max, cell)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cmap, x0, y0, dx, dy, nx, ny = _build_cell_index_map(x, y)

        # Index-map key for each cell must equal the floor index a ball at that
        # cell center would land in.
        key_of_cell = {}
        for key, cell_idx in cmap.items():
            key_of_cell[cell_idx] = key
        for i in range(len(x)):
            ix = int(math.floor((x[i] - x0) / dx))
            iy = int(math.floor((y[i] - y0) / dy))
            self.assertEqual(key_of_cell[i], (ix, iy))

        # Keys must span 0..nx-1 and 0..ny-1 (no off-by-one shift).
        kxs = sorted(set(k[0] for k in cmap.keys()))
        kys = sorted(set(k[1] for k in cmap.keys()))
        self.assertEqual(kxs[0], 0)
        self.assertEqual(kxs[-1], nx - 1)
        self.assertEqual(kys[0], 0)
        self.assertEqual(kys[-1], ny - 1)

    def test_engineering_domain(self):
        self._check_domain(-7.5, 7.5, 0.0, 40.0, 1.5)

    def test_unit_origin_domain(self):
        self._check_domain(0.0, 2.0, 0.0, 1.2, 0.2)


if __name__ == "__main__":
    unittest.main()
