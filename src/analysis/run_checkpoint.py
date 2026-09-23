# -*- coding: utf-8 -*-
"""
Crash-safe checkpoint / resume for the marching loops.

This machine takes random segmentation faults and Windows bugchecks under
sustained numerical load (see CLAUDE.md "分辨率与求解器实测"), so a two-hour
case run is not a safe unit of work. Nothing here touches the solver, the
time stepping or the physics: it saves the fields the marcher carries from one
step to the next, and puts them back.

**Bit-identical resume.** The carried state is exactly:

  pressure, saturation                the IMPES unknowns
  mobility_face                       the LAGGED Picard mobility -- the one
                                      that is easy to forget, and the one
                                      whose absence makes a resume drift
  last_div_q                          previous-step fluxes, read by
                                      compute_adaptive_dt for the fill CFL
  face_yield_latched                  static-yield hysteresis (off by default)
  *_initial_for_ledger                conservation baselines, set once at the
                                      first step -- saved so a resume does not
                                      re-baseline them on the restored state
  injected/clipped/compressed_volume_total, flow_step_index, dt_last

Everything else in `state` is either rebuilt by the case's own `build_case`
(mesh, porosity, permeability, boundary handles) or recomputed inside each
step before it is read (fill masks, `*_last` diagnostics, Picard counters).
`tests/test_run_checkpoint.py` proves the claim rather than asserting it: N
steps straight must equal N/2 + resume + N/2, bit for bit.

**Crash safety.** A crash *during* a save must not destroy the previous good
checkpoint, so saves alternate between two slots and a small pointer file
names the newer one. A load falls back to the other slot if the newer one is
unreadable (a torn write), so the worst case costs one checkpoint interval,
not the run.

Usage in a marching loop:

    ck = Checkpointer(os.path.join(out_dir, "gradient_A"), every=200)
    start = ck.restore(state)            # {} when there is nothing to resume
    t = start.get("t", 0.0)
    for step in range(start.get("step", 0), MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=DT_CAP)
        t += float(dt)
        ...
        ck.maybe_save(state, step=step + 1, t=t, history=history)
"""

from __future__ import print_function

import os

import numpy as np
from fipy import CellVariable, FaceVariable


# FiPy variables: saved as their .value array, restored with setValue.
# `mobility_face` is built lazily on the first solve, so a freshly built state
# does not have it yet and a restore has to create it.
CARRIED_VARIABLES = ("pressure", "saturation", "mobility_face")
_FACE_VARIABLES = ("mobility_face",)

# Plain arrays carried across steps.
CARRIED_ARRAYS = (
    "last_div_q",
    "face_yield_latched",
    "saturation_initial_for_ledger",
    "pressure_initial_for_ledger",
)

# Scalars carried across steps.
CARRIED_SCALARS = (
    "injected_volume_total",
    "clipped_volume_total",
    "compressed_volume_total",
    "flow_step_index",
    "dt_last",
)

# What a STAGED sequence additionally carries: between-stage solidification
# rewrites the porosity, the permeability and every mobility alias, and the
# fill ledger accumulates across stages. Pass these as `also=` so a crash in
# stage 3 does not force stages 1 and 2 to be run again.
STAGE_STATE_KEYS = (
    "porosity",
    "permeability",
    "mobility_structural",
    "mobility_effective",
    "mobility",
    "intrinsic_mobility",
    "stage_porosity_initial",
    "stage_index",
    "floor_clipped_volume_total",
    # apply_stage_closure re-installs the NEXT stage's source before it
    # returns. Miss these two and a resumed sequence injects from the whole
    # hole instead of the fresh section -- a wrong run, not a lost one.
    "interior_dirichlet_mask",
    "interior_dirichlet_value",
)

_PRESENT = "__present__"
_FINGERPRINT = "__fingerprint__"


def _slot_path(prefix, slot):
    return prefix + ".ckpt%d.npz" % slot


def _pointer_path(prefix):
    return prefix + ".ckpt.idx"


def _read_pointer(prefix):
    """Which slot was written last? None when there is no checkpoint."""
    path = _pointer_path(prefix)
    if not os.path.exists(path):
        return None
    try:
        handle = open(path, "rb")
        try:
            text = handle.read().strip()
        finally:
            handle.close()
        slot = int(text)
    except (ValueError, IOError):
        return None
    if slot not in (0, 1):
        return None
    return slot


def _write_pointer(prefix, slot):
    handle = open(_pointer_path(prefix), "wb")
    try:
        handle.write(str(slot).encode("ascii"))
    finally:
        handle.close()


def state_fingerprint(state, extra_text=""):
    """
    A cheap identity for "the run this checkpoint belongs to": mesh size and
    the frozen fields, plus whatever the caller adds (a case label, the source
    cells). Resuming a checkpoint written under a different geometry or a
    different parameter set would silently produce a run that never existed,
    which is worse than losing the run -- so `load` refuses on a mismatch.
    """
    mesh = state["mesh"]
    parts = [
        "cells=%d" % mesh.numberOfCells,
        "faces=%d" % mesh.numberOfFaces,
    ]
    for name in ("porosity", "permeability"):
        var = state.get(name)
        if var is None:
            continue
        arr = np.asarray(var.value, dtype=float)
        parts.append("%s=%.17g/%.17g/%.17g"
                     % (name, float(arr.min()), float(arr.max()),
                        float(arr.sum())))
    params = state.get("slurry_parameters", {})
    for name in ("yield_stress", "plastic_viscosity", "slurry_density",
                 "calibrated_permeability_coefficient", "gravity_y",
                 "picard_relaxation_fill", "picard_max_iters", "picard_tol",
                 "yield_reg_m", "reference_storage"):
        if name in params:
            parts.append("%s=%.17g" % (name, float(params[name])))
    if "interior_dirichlet_value" in state:
        parts.append("p0=%.17g" % float(state["interior_dirichlet_value"]))
    parts.append(str(extra_text))
    return "|".join(parts)


def save(prefix, state, extra=None, fingerprint=None, also=None):
    """
    Write a checkpoint next to `prefix`. Returns the slot written.

    `extra` holds the marching loop's own variables (step, t, stall counters,
    the injection history); values may be scalars, arrays, or lists of
    number pairs. They come back from `load` in the same dict.

    `also` names further state keys to carry. A single-stage case needs none:
    porosity and permeability are frozen for the whole run. A STAGED case
    does -- between-stage solidification rewrites them -- so it passes
    STAGE_STATE_KEYS and can resume without replaying finished stages.
    """
    payload = {}
    present = []

    for name in (also or ()):
        value = state.get(name)
        if value is None:
            continue
        if hasattr(value, "value"):                       # a FiPy variable
            payload["v_" + name] = np.array(value.value, dtype=float, copy=True)
            present.append("v_" + name)
        elif np.isscalar(value):
            payload["s_" + name] = np.array(value)
            present.append("s_" + name)
        else:
            payload["a_" + name] = np.array(value)
            present.append("a_" + name)

    for name in CARRIED_VARIABLES:
        var = state.get(name)
        if var is None:
            continue
        payload["v_" + name] = np.array(var.value, dtype=float, copy=True)
        present.append("v_" + name)

    for name in CARRIED_ARRAYS:
        arr = state.get(name)
        if arr is None:
            continue
        payload["a_" + name] = np.array(arr, copy=True)
        present.append("a_" + name)

    for name in CARRIED_SCALARS:
        if name not in state:
            continue
        payload["s_" + name] = np.array(state[name])
        present.append("s_" + name)

    for name, value in (extra or {}).items():
        payload["x_" + name] = np.array(value)
        present.append("x_" + name)

    payload[_PRESENT] = np.array(present)
    payload[_FINGERPRINT] = np.array(str(fingerprint or ""))

    last = _read_pointer(prefix)
    slot = 1 if last == 0 else 0          # alternate; never overwrite the live one
    np.savez(_slot_path(prefix, slot), **payload)
    _write_pointer(prefix, slot)
    return slot


def _load_slot(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        # Touch the manifest so a torn write fails here, not half-way through
        # a restore that has already mutated the state.
        list(data[_PRESENT])
        return data
    except Exception:                     # truncated / unreadable npz
        return None


def load(prefix, state, fingerprint=None, also=None):
    """
    Restore a checkpoint into an already-built `state`.

    Returns the `extra` dict, or None when there is nothing to resume from --
    no checkpoint, both slots unreadable, or a checkpoint belonging to a
    different run (see `state_fingerprint`). The state is left untouched when
    None is returned, so a caller can simply start from step 0.
    """
    newer = _read_pointer(prefix)
    order = [0, 1] if newer is None else [newer, 1 - newer]

    data = None
    for slot in order:
        candidate = _load_slot(_slot_path(prefix, slot))
        if candidate is None:
            continue
        if fingerprint is not None:
            stored = str(candidate[_FINGERPRINT]) if _FINGERPRINT in candidate.files else ""
            if stored != str(fingerprint):
                continue                  # a checkpoint for some other run
        data = candidate
        break
    if data is None:
        return None

    keys = set(str(k) for k in data[_PRESENT])
    extra = {}

    for name in (also or ()):
        for prefix_char, restore in (("v_", "var"), ("a_", "arr"), ("s_", "num")):
            key = prefix_char + name
            if key not in keys:
                continue
            value = data[key]
            if restore == "var" and state.get(name) is not None:
                state[name].setValue(np.array(value, copy=True))
            elif restore == "num":
                state[name] = value.item() if value.shape == () else np.array(value)
            else:
                state[name] = np.array(value)

    for name in CARRIED_VARIABLES:
        key = "v_" + name
        if key not in keys:
            continue
        value = np.array(data[key], copy=True)
        if state.get(name) is None:
            builder = FaceVariable if name in _FACE_VARIABLES else CellVariable
            state[name] = builder(mesh=state["mesh"], value=value)
        else:
            state[name].setValue(value)

    for name in CARRIED_ARRAYS:
        key = "a_" + name
        if key in keys:
            state[name] = np.array(data[key], copy=True)

    for name in CARRIED_SCALARS:
        key = "s_" + name
        if key in keys:
            value = data[key]
            state[name] = value.item() if value.shape == () else np.array(value)

    for key in keys:
        if key.startswith("x_"):
            value = data[key]
            extra[key[2:]] = value.item() if value.shape == () else np.array(value)

    return extra


def clear(prefix):
    """Remove a checkpoint set (call once a run has finished cleanly)."""
    for path in (_slot_path(prefix, 0), _slot_path(prefix, 1), _pointer_path(prefix)):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


class Checkpointer(object):
    """Save every `every` steps, and on demand. See the module docstring."""

    def __init__(self, prefix, every=200, enabled=True, fingerprint=None,
                 also=None):
        self.prefix = prefix
        self.every = int(every)
        self.enabled = bool(enabled)
        self.fingerprint = fingerprint
        self.also = tuple(also or ())
        self.saves = 0
        self.resumed_from = None

    def restore(self, state):
        """Returns the extra dict, or {} when starting fresh."""
        if not self.enabled:
            return {}
        extra = load(self.prefix, state, fingerprint=self.fingerprint,
                     also=self.also)
        if extra is None:
            return {}
        self.resumed_from = int(extra.get("step", 0))
        return extra

    def save_now(self, state, **extra):
        if not self.enabled:
            return
        save(self.prefix, state, extra, fingerprint=self.fingerprint,
             also=self.also)
        self.saves += 1

    def maybe_save(self, state, step, **extra):
        if not self.enabled or self.every <= 0:
            return
        if step % self.every:
            return
        extra["step"] = step
        self.save_now(state, **extra)

    def finish(self):
        """Drop the checkpoints of a run that completed."""
        if self.enabled:
            clear(self.prefix)
