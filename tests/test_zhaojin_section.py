# -*- coding: utf-8 -*-
"""
The Zhaojin section builder (cases/zhaojin_section.py), Python 2.7 / unittest.

The geometry is read off one design document; these tests pin that reading,
so a later edit that silently moves the hole, the levels or the walls fails
here instead of shifting every screening result.
"""
import imp
import math
import os
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = imp.load_source("zhaojin_section_under_test",
                    os.path.join(_REPO, "cases", "zhaojin_section.py"))

CELL = 1.0
_BUNDLE = {}


def _mesh():
    if "b" not in _BUNDLE:
        b = Z.build_mesh(CELL)
        _BUNDLE["b"] = (b, np.asarray(b[1], float), np.asarray(b[2], float))
    return _BUNDLE["b"]


def _walk(cfg, step=0.01):
    """Metres of hole in rock before the fill, in the fill, in rock after."""
    cx, cy = Z.collar()
    d = Z.hole_direction(cfg)
    x_fw, x_hw = Z.walls(cfg)
    before = inside = after = 0.0
    entered = False
    s = 0.0
    while s < cfg["hole_length"]:
        px, py = cx + s * d[0], cy + s * d[1]
        if x_hw(py) <= px <= x_fw(py):
            entered = True
            inside += step
        elif entered:
            after += step
        else:
            before += step
        s += step
    return before, inside, after


class TestHoleGeometry(unittest.TestCase):

    def test_hole_enters_the_fill_where_the_orifice_pipe_ends(self):
        """Drilled in solid rock, stopped on entering loose ground: the fill
        starts exactly at the end of the 2.0 m orifice."""
        for zone in Z.ZONES:
            before, _, _ = _walk(Z.config(zone=zone))
            self.assertAlmostEqual(before, Z.ORIFICE_DEPTH, delta=0.02)

    def test_every_hole_ends_at_least_3_m_in_host_rock(self):
        """The document's own requirement, for every zone and both angles."""
        for zone in Z.ZONES:
            for angle in (35.0, 20.0):
                _, _, after = _walk(Z.config(zone=zone, hole_angle=angle))
                self.assertGreaterEqual(after, 3.0, "%s %g deg" % (zone, angle))

    def test_fill_crossing_matches_the_closed_form(self):
        for zone in Z.ZONES:
            for climb in ("hanging", "footwall"):
                cfg = Z.config(zone=zone, climb=climb)
                _, inside, _ = _walk(cfg)
                self.assertAlmostEqual(inside, Z.fill_crossing_length(cfg), delta=0.05)

    def test_the_10_m_design_width_rules_out_climbing_toward_the_footwall(self):
        """Why 'hanging' is the default: the layout was sized for a 10 m
        fill, and only the hanging-wall direction then leaves >= 3 m of
        rock at the end of a 17.5 m hole."""
        t = math.tan(math.radians(Z.DIP_DEG))
        a = math.radians(35.0)
        hanging = 10.0 / (math.cos(a) + math.sin(a) / t)
        footwall = 10.0 / (math.cos(a) - math.sin(a) / t)
        self.assertGreaterEqual(17.5 - Z.ORIFICE_DEPTH - hanging, 3.0)
        self.assertLess(17.5 - Z.ORIFICE_DEPTH - footwall, 3.0)


class TestRegions(unittest.TestCase):

    def setUp(self):
        _, self.mx, self.my = _mesh()

    def test_levels_stack_as_the_document_says(self):
        cfg = Z.config(contacts=False)
        _, region = Z.section_fields(self.mx, self.my, cfg)
        fill = region == Z.FILL
        ore = region == Z.ORE
        self.assertTrue(np.all(self.my[fill] > Z.Y_LEVEL))
        self.assertTrue(np.all(self.my[ore] <= Z.Y_LEVEL))
        self.assertTrue(np.all(self.my[ore] > Z.Y_ORE_BOTTOM))
        self.assertGreater(np.sum(fill), 0)
        self.assertGreater(np.sum(ore), 0)

    def test_zone_porosity_is_the_document_value(self):
        for zone, spec in Z.ZONES.items():
            phi, region = Z.section_fields(self.mx, self.my, Z.config(zone=zone, contacts=False))
            np.testing.assert_allclose(phi[region == Z.FILL], spec["phi"])

    def test_the_seal_cements_the_whole_vein_width_including_the_contacts(self):
        """The seal holes run >= 3 m into rock, so the contacts are sealed at
        that level too -- otherwise the seal could be bypassed in-plane."""
        cfg = Z.config(contacts=True, seal=True)
        phi, region = Z.section_fields(self.mx, self.my, cfg)
        band = (self.my > Z.Y_LEVEL) & (self.my <= Z.Y_LEVEL + Z.SEAL_THICKNESS)
        self.assertFalse(np.any((region == Z.CONTACT) & band))
        self.assertFalse(np.any((region == Z.FILL) & band))
        self.assertTrue(np.all(phi[region == Z.SEAL] <= Z.NO_GROUT_PHI))

    def test_rock_is_impassable_at_the_mesh_scale(self):
        """The surrogate's only job: half a cell of rock costs more than the
        whole pressure budget."""
        lam = 2.0 * Z.grout_material_v06()["yield_stress"] / math.sqrt(
            8.0 * Z.grout_material_v06()["calibrated_permeability_coefficient"]
            * Z.NO_GROUT_PHI ** 2 / (1.0 - Z.NO_GROUT_PHI) ** 2)
        self.assertGreater(lam * 0.5 * CELL, 5.0 * Z.P0)

    def test_the_outlet_is_the_63_m_level(self):
        _, region = Z.section_fields(self.mx, self.my, Z.config())
        out = region == Z.OUTLET
        self.assertGreater(np.sum(out), 0)
        self.assertTrue(np.all(self.my[out] >= Z.Y_OUTLET))
        self.assertTrue(np.all(self.my[out] <= Z.Y_OUTLET + 2.0))


if __name__ == "__main__":
    unittest.main()
