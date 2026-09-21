# -*- coding: utf-8 -*-
"""Porosity field generators (src/structure/porosity_fields.py), py2.7."""
import unittest

import numpy as np

from src.structure.porosity_fields import (
    DEFAULT_BOUNDS, uniform, linear_gradient, where_band, random_correlated,
)


def _grid(nx=12, ny=12, cell=2.5, x0=-15.0, y0=0.0):
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


class TestFields(unittest.TestCase):

    def test_uniform_and_bounds(self):
        x, y = _grid()
        self.assertTrue(np.all(uniform(x, 0.18) == 0.18))
        self.assertTrue(np.all(uniform(x, 0.9) == DEFAULT_BOUNDS[1]))
        self.assertTrue(np.all(uniform(x, 0.0) == DEFAULT_BOUNDS[0]))

    def test_linear_gradient_reproduces_the_baseline_prescription(self):
        """phi(y) = 0.12 + 0.18*y/30 -- the gradient the cases hard-code."""
        x, y = _grid()
        phi = linear_gradient(y, 0.12, 0.30, 0.0, 30.0)
        np.testing.assert_allclose(phi, 0.12 + 0.18 * y / 30.0, rtol=1e-12)
        self.assertAlmostEqual(float(phi[np.argmin(y)]), 0.12 + 0.18 * 1.25 / 30.0)

    def test_where_band_vertical_one_cell_wide(self):
        x, y = _grid()
        mask = where_band(x, y, (3.75, 0.0), (0.0, 1.0), 1.25)
        self.assertTrue(np.all(np.isclose(x[mask], 3.75)))
        self.assertEqual(int(mask.sum()), 12)          # full height, one column

    def test_where_band_along_a_dipping_line_with_extent(self):
        x, y = _grid()
        mask = where_band(x, y, (-15.0, 1.0), (np.cos(np.radians(35)), np.sin(np.radians(35))),
                          1.0, extent=(0.0, 10.0))
        self.assertGreater(int(mask.sum()), 0)
        # Nothing selected beyond 10 m along the line.
        along = (x + 15.0) * np.cos(np.radians(35)) + (y - 1.0) * np.sin(np.radians(35))
        self.assertTrue(np.all(along[mask] <= 10.0 + 1e-9))

    def test_where_band_rejects_zero_direction(self):
        x, y = _grid()
        with self.assertRaises(ValueError):
            where_band(x, y, (0.0, 0.0), (0.0, 0.0), 1.0)

    def test_random_correlated_is_reproducible_and_bounded(self):
        x, y = _grid()
        a = random_correlated(x, y, 0.18, 0.06, 7.5, 2.5, seed=7, cell=2.5)
        b = random_correlated(x, y, 0.18, 0.06, 7.5, 2.5, seed=7, cell=2.5)
        c = random_correlated(x, y, 0.18, 0.06, 7.5, 2.5, seed=8, cell=2.5)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(np.all(a >= DEFAULT_BOUNDS[0]) and np.all(a <= DEFAULT_BOUNDS[1]))
        self.assertLess(abs(float(np.mean(a)) - 0.18), 0.03)

    def test_random_correlated_anisotropy_makes_horizontal_streaks(self):
        """Long x-correlation, short y-correlation: neighbouring cells along
        x agree more than neighbouring cells along y."""
        x, y = _grid(nx=40, ny=40, cell=1.0, x0=0.0, y0=0.0)
        phi = random_correlated(x, y, 0.2, 0.05, 8.0, 1.0, seed=3, cell=1.0).reshape(40, 40)
        dx = np.mean(np.abs(np.diff(phi, axis=1)))   # along x
        dy = np.mean(np.abs(np.diff(phi, axis=0)))   # along y
        self.assertLess(dx, dy)


if __name__ == "__main__":
    unittest.main()
