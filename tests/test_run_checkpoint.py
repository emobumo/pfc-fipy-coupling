# -*- coding: utf-8 -*-
"""
Checkpoint / resume (src/analysis/run_checkpoint.py), Python 2.7 / unittest.

The whole point of the module is one claim: a run that is interrupted and
resumed gives BIT-IDENTICAL results to a run that was never interrupted. If
that fails, a resumed run is a different run and every regression anchor
computed from one is worthless. So the central test marches a dry 1D strip
straight through, marches the same strip with a save/rebuild/restore in the
middle, and compares the raw arrays with assert_array_equal -- not allclose.

The second concern is crash safety: this machine takes random segfaults and
bugchecks, so a crash *during* a save must not eat the previous checkpoint.
"""
import math
import os
import shutil
import tempfile
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
from src.analysis import run_checkpoint as ck


TAU0 = 30.0
MU = 0.15
A_CAL = 1.25e-7
NX, DX, PHI, P0 = 30, 0.1, 0.30, 1.5e5


def _build():
    """A dry 1D strip with an interior constant-pressure source at cell 0."""
    mesh, x, y, fx, fy = build_mesh(nx=NX, ny=1, dx=DX, dy=DX)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    params = build_placeholder_slurry_parameters()
    params["porosity_to_permeability_formula"] = "calibrated_power"
    params["calibrated_permeability_coefficient"] = A_CAL
    params["yield_stress"] = TAU0
    params["plastic_viscosity"] = MU
    params["gravity_y"] = 0.0
    state.update(initialize_slurry_variables(mesh, params=params))

    phi = PHI + np.zeros(mesh.numberOfCells)
    k = porosity_to_permeability(phi, params)
    state["porosity"].setValue(phi)
    state["permeability"].setValue(k)
    for key in ("mobility_structural", "mobility_effective",
                "mobility", "intrinsic_mobility"):
        state[key].setValue(k / MU)

    p = state["slurry_parameters"]
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

    mask = np.zeros(NX)
    mask[0] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    state["interior_dirichlet_value"] = P0
    s0 = np.zeros(NX)
    s0[0] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, P0, 0.0))
    return state


def _march(state, steps, t=0.0):
    """`steps` accepted steps, source cell held full, as the cases do."""
    for _ in range(steps):
        dt = solve_transport_step(state, dt_cap=1.0)
        t += float(dt)
        s = np.array(state["saturation"].value, copy=True)
        s[0] = 1.0
        state["saturation"].setValue(s)
    return t


def _snapshot(state):
    return {
        "pressure": np.array(state["pressure"].value, copy=True),
        "saturation": np.array(state["saturation"].value, copy=True),
        "mobility_face": np.array(state["mobility_face"].value, copy=True),
        "last_div_q": np.array(state["last_div_q"], copy=True),
        "injected": state["injected_volume_total"],
        "clipped": state["clipped_volume_total"],
        "compressed": state["compressed_volume_total"],
    }


class TestResumeIsBitIdentical(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_")
        self.prefix = os.path.join(self.dir, "strip")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_interrupted_and_resumed_run_equals_the_straight_run(self):
        """30 steps == 15 steps + checkpoint + fresh state + restore + 15."""
        straight = _build()
        t_straight = _march(straight, 30)
        want = _snapshot(straight)

        first = _build()
        t_half = _march(first, 15)
        ck.save(self.prefix, first, {"step": 15, "t": t_half})

        resumed = _build()                       # a fresh process would do this
        extra = ck.load(self.prefix, resumed)
        self.assertIsNotNone(extra)
        self.assertEqual(int(extra["step"]), 15)
        t_resumed = _march(resumed, 15, t=float(extra["t"]))
        got = _snapshot(resumed)

        for key in ("pressure", "saturation", "mobility_face", "last_div_q"):
            np.testing.assert_array_equal(
                got[key], want[key],
                err_msg="%s differs after resume -- the carried state is "
                        "incomplete" % key)
        for key in ("injected", "clipped", "compressed"):
            self.assertEqual(got[key], want[key], "%s ledger differs" % key)
        self.assertEqual(t_resumed, t_straight)

    def test_resume_without_the_lagged_mobility_would_drift(self):
        """
        Guards the reason `mobility_face` is in the carried set: drop it from
        the checkpoint and the resumed run must NOT match. Without this, a
        future edit could quietly remove a carried field and the test above
        would still pass for the wrong reason (both runs equally wrong).
        """
        straight = _build()
        _march(straight, 30)
        want = _snapshot(straight)

        first = _build()
        _march(first, 15)
        ck.save(self.prefix, first, {"step": 15})

        resumed = _build()
        ck.load(self.prefix, resumed)
        resumed["mobility_face"] = None          # what "forgetting" looks like
        _march(resumed, 15)

        self.assertFalse(
            np.array_equal(_snapshot(resumed)["saturation"], want["saturation"]),
            "dropping the lagged Picard mobility changed nothing -- either the "
            "field stopped being carried across steps, or this fixture no "
            "longer exercises the yield edge")


class TestStagedState(unittest.TestCase):
    """
    A staged sequence checkpoints across between-stage solidification, which
    rewrites porosity/permeability/mobilities AND re-installs the next stage's
    source. Forgetting the source mask does not crash a resume: it silently
    injects from the whole hole instead of the fresh section, i.e. it returns
    a wrong run rather than losing one. That is what this guards.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_")
        self.prefix = os.path.join(self.dir, "seq")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_stage_state_keys_round_trip(self):
        state = _build()
        _march(state, 3)

        # What a closure would leave behind: cemented porosity, a shrunken
        # source, a bumped stage index.
        cemented = np.asarray(state["porosity"].value, dtype=float).copy()
        cemented[:3] = 1.0e-3
        state["porosity"].setValue(cemented)
        narrowed = np.zeros(NX)
        narrowed[7] = 1.0
        state["interior_dirichlet_mask"].setValue(narrowed)
        state["interior_dirichlet_value"] = 0.5 * P0
        state["stage_index"] = 2
        state["floor_clipped_volume_total"] = 1.25

        ck.save(self.prefix, state, {"stage": 2}, also=ck.STAGE_STATE_KEYS)

        fresh = _build()
        extra = ck.load(self.prefix, fresh, also=ck.STAGE_STATE_KEYS)
        self.assertIsNotNone(extra)
        np.testing.assert_array_equal(
            np.asarray(fresh["porosity"].value, dtype=float), cemented)
        np.testing.assert_array_equal(
            np.asarray(fresh["interior_dirichlet_mask"].value, dtype=float),
            narrowed)
        self.assertEqual(fresh["interior_dirichlet_value"], 0.5 * P0)
        self.assertEqual(fresh["stage_index"], 2)
        self.assertEqual(fresh["floor_clipped_volume_total"], 1.25)

    def test_the_source_mask_is_actually_in_the_carried_set(self):
        """The regression itself: it was missing, and nothing failed loudly."""
        for key in ("interior_dirichlet_mask", "interior_dirichlet_value",
                    "porosity", "permeability"):
            self.assertIn(key, ck.STAGE_STATE_KEYS)

    def test_without_also_the_staged_fields_are_left_alone(self):
        """A single-stage case must not pay for the staged machinery."""
        state = _build()
        _march(state, 3)
        state["stage_index"] = 2
        ck.save(self.prefix, state, {"step": 3})

        fresh = _build()
        ck.load(self.prefix, fresh)
        self.assertNotIn("stage_index", fresh)


class TestCrashSafety(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_")
        self.prefix = os.path.join(self.dir, "strip")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_to_resume_returns_none(self):
        self.assertIsNone(ck.load(self.prefix, _build()))

    def test_saves_alternate_slots_so_a_save_never_overwrites_the_live_one(self):
        state = _build()
        _march(state, 3)
        slots = [ck.save(self.prefix, state, {"step": i}) for i in range(4)]
        self.assertEqual(slots, [0, 1, 0, 1])

    def test_a_torn_write_falls_back_to_the_previous_checkpoint(self):
        """A crash mid-save costs one interval, not the run."""
        state = _build()
        _march(state, 3)
        ck.save(self.prefix, state, {"step": 3})
        _march(state, 3)
        newer = ck.save(self.prefix, state, {"step": 6})

        path = ck._slot_path(self.prefix, newer)      # truncate the newer slot
        handle = open(path, "r+b")
        try:
            handle.truncate(64)
        finally:
            handle.close()

        extra = ck.load(self.prefix, _build())
        self.assertIsNotNone(extra, "both slots lost to one torn write")
        self.assertEqual(int(extra["step"]), 3)

    def test_a_checkpoint_from_a_different_run_is_refused(self):
        """
        A stale checkpoint resumed under changed parameters would produce a
        run that never existed -- worse than losing the run. The fingerprint
        has to notice, and noticing means returning None, not raising.
        """
        state = _build()
        _march(state, 3)
        ck.save(self.prefix, state, {"step": 3},
                fingerprint=ck.state_fingerprint(state, "case=A"))

        same = _build()
        self.assertIsNotNone(
            ck.load(self.prefix, same,
                    fingerprint=ck.state_fingerprint(same, "case=A")))

        other = _build()
        other["slurry_parameters"]["yield_stress"] = 2.0 * TAU0
        self.assertIsNone(
            ck.load(self.prefix, other,
                    fingerprint=ck.state_fingerprint(other, "case=A")),
            "a checkpoint written at a different yield stress was accepted")

        relabelled = _build()
        self.assertIsNone(
            ck.load(self.prefix, relabelled,
                    fingerprint=ck.state_fingerprint(relabelled, "case=B")),
            "a checkpoint from a different sub-case was accepted")

    def test_finish_clears_the_checkpoint_set(self):
        state = _build()
        _march(state, 3)
        keeper = ck.Checkpointer(self.prefix, every=1)
        keeper.save_now(state, step=3)
        keeper.finish()
        self.assertIsNone(ck.load(self.prefix, _build()))


class TestCheckpointerLoop(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_")
        self.prefix = os.path.join(self.dir, "strip")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_maybe_save_only_fires_on_the_interval(self):
        state = _build()
        _march(state, 2)
        keeper = ck.Checkpointer(self.prefix, every=5)
        for step in range(1, 11):
            keeper.maybe_save(state, step=step)
        self.assertEqual(keeper.saves, 2)

    def test_restore_on_a_fresh_prefix_starts_from_zero(self):
        keeper = ck.Checkpointer(self.prefix, every=5)
        self.assertEqual(keeper.restore(_build()), {})
        self.assertIsNone(keeper.resumed_from)


if __name__ == "__main__":
    unittest.main()
