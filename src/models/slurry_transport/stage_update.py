# -*- coding: utf-8 -*-
"""
Between-stage closure for staged advancing grouting (分段前进式灌浆).

SCOPE AND THE FIXED-SKELETON RULE
---------------------------------
This module is the ONLY place porosity changes after initialization, and it
runs strictly BETWEEN stages -- never from inside the flow loop. The rock
skeleton stays rigid: what changes is the pore space already occupied by
grout that has since hardened.

The distinction matters and is easy to get wrong:

  within a stage   the filled pore space is the FLOW PATH. Reducing n or k
                   as S rises would seal off the very channel carrying the
                   grout. k_r(S)=S^3 already carries this effect correctly.
                   Do not touch n here.

  between stages   the previous stage has set. Hardened grout now occupies
                   pore space permanently, so the next stage sees a denser,
                   less permeable medium. This is what apply_stage_closure
                   does, once, after the stage has stalled.

Closure rule per stage k:

    n_{k+1} = n_k * (1 - S_k)                     [floored, see below]
    k_{k+1} = A * n_{k+1}^3 / (1 - n_{k+1})^2     via porosity_to_permeability
    lambda  follows from (n, k) automatically -- the solver rebuilds
            lambda_face = 2*tau0/sqrt(8*k_f/n_f) every step, so there is no
            stored lambda to update here.
    S       reset to 0 (fresh dry medium for the next stage)
    p       reset (the previous stage's pressure field is meaningless once
            the source moves)

LEDGER
------
All volumes are per unit thickness [m^3/m], matching the existing
conservation ledger in equations.py. The invariant basis is the stage-0 pore
capacity, so cumulative fill ratios from different stages are comparable:

    pore_capacity_initial  = sum n_0 * V             (fixed denominator)
    occupied_cumulative    = sum (n_0 - n_k) * V     (hardened grout to date)
    remaining_capacity     = sum n_k * V

Per stage, the volume that hardens must equal what the flow solve actually
delivered into the pore space:

    occupied_this_stage + floor_clipped == v_store + v_prefill

where v_store is the conservation ledger's stored volume for that stage and
v_prefill = sum n_k * S_init * V accounts for source cells that start at
S=1 by prescription rather than by injection. Verification ladder step 7
asserts this identity.

POROSITY FLOOR
--------------
A cell that reaches S=1 would close to n=0, hence k=0 and lambda=inf. That
is physically the intent (fully cemented, impermeable) but numerically a
division by zero, so n is floored at stage_porosity_floor. The volume the
floor refuses to remove is booked into floor_clipped_volume_total, which is
why the identity above balances exactly rather than approximately.
"""
import numpy as np

from src.coupling.porosity_to_permeability import porosity_to_permeability


def _to_array(value_or_var):
    """Cell array from either a FiPy CellVariable or a plain array."""
    if hasattr(value_or_var, "value"):
        return np.asarray(value_or_var.value, dtype=float)
    return np.asarray(value_or_var, dtype=float)


def get_stage_porosity_floor(params):
    """Lower bound on porosity after closure. See POROSITY FLOOR above."""
    return max(float(params.get("stage_porosity_floor", 1.0e-3)), 1.0e-12)


def initialize_stage_ledger(state):
    """
    Capture the stage-0 porosity as the invariant accounting basis.

    Idempotent: calling it again leaves an existing basis untouched, so a
    case may call it defensively before every stage.
    """
    if state.get("stage_porosity_initial") is not None:
        return state["stage_ledger"]

    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    n0 = np.clip(_to_array(state["porosity"]), 0.0, 1.0)
    state["stage_porosity_initial"] = np.array(n0, copy=True)
    state["stage_index"] = 0
    state["floor_clipped_volume_total"] = 0.0
    state["stage_history"] = []
    state["stage_ledger"] = {
        "pore_capacity_initial": float(np.sum(n0 * vols)),
        "occupied_cumulative": 0.0,
        "remaining_capacity": float(np.sum(n0 * vols)),
        "floor_clipped_total": 0.0,
    }
    return state["stage_ledger"]


def stage_fill_report(state):
    """
    What this stage put into the pore space, measured BEFORE closure.

    Call it after the stage has stalled and before apply_stage_closure; the
    saturation field it reads is consumed by the closure.

    occupied_this_stage is the grout volume about to harden, v_prefill the
    part of it that was prescribed as an initial condition (source cells at
    S=1) rather than injected, and occupied_fraction_of_stage_capacity the
    share of the pore space still open at the start of this stage that the
    stage managed to fill.
    """
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    n = np.clip(_to_array(state["porosity"]), 0.0, 1.0)
    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)

    s_init = state.get("saturation_initial_for_ledger")
    if s_init is None:
        s_init = np.zeros_like(s)
    s_init = np.clip(np.asarray(s_init, dtype=float), 0.0, 1.0)

    stage_capacity = float(np.sum(n * vols))
    occupied = float(np.sum(n * s * vols))
    return {
        "stage_index": int(state.get("stage_index", 0)),
        "stage_capacity": stage_capacity,
        "occupied_this_stage": occupied,
        "v_prefill": float(np.sum(n * s_init * vols)),
        "occupied_fraction_of_stage_capacity": (
            occupied / stage_capacity if stage_capacity > 0.0 else 0.0
        ),
    }


def apply_stage_closure(state, source_mask=None, source_pressure=None,
                        reset_pressure=True):
    """
    Harden this stage and hand a fresh medium to the next one.

    Runs between stages only. Never call this from inside a marching loop --
    it invalidates the pressure and saturation fields on purpose.

    source_mask      cell mask (or index array) for the NEXT stage's interior
                     Dirichlet source; None leaves the current source in
                     place, which is what a same-hole re-injection wants.
    source_pressure  next stage's source pressure [Pa]; None keeps the
                     current value.
    reset_pressure   reset p to the source pressure inside the source and 0
                     outside. The previous stage's field is meaningless once
                     the source has moved, so the default is True.

    Returns the ledger dict after the closure.
    """
    initialize_stage_ledger(state)
    params = state["slurry_parameters"]
    mesh = state["mesh"]
    vols = np.asarray(mesh.cellVolumes, dtype=float)
    floor = get_stage_porosity_floor(params)

    report = stage_fill_report(state)

    n_k = np.clip(_to_array(state["porosity"]), 0.0, 1.0)
    s_k = np.clip(_to_array(state["saturation"]), 0.0, 1.0)

    # --- the closure rule -------------------------------------------------
    n_raw = n_k * (1.0 - s_k)
    n_next = np.maximum(n_raw, floor)
    # Volume the floor refused to remove, so the ledger identity stays exact.
    floor_clipped_step = float(np.sum((n_next - n_raw) * vols))
    state["floor_clipped_volume_total"] = (
        float(state.get("floor_clipped_volume_total", 0.0)) + floor_clipped_step
    )

    k_next = porosity_to_permeability(n_next, params)
    mu_p = max(float(params.get("plastic_viscosity", 0.05)), 1.0e-30)
    mobility_next = k_next / mu_p

    state["porosity"].setValue(n_next)
    state["permeability"].setValue(k_next)
    # All four names alias two CellVariables (see initialize_slurry_variables);
    # set every one so no stale handle survives the closure.
    for key in ("mobility_structural", "mobility_effective",
                "mobility", "intrinsic_mobility"):
        if state.get(key) is not None:
            state[key].setValue(mobility_next)

    # --- next stage's source ---------------------------------------------
    if source_mask is not None:
        mask = np.zeros(n_next.size, dtype=float)
        idx = np.asarray(source_mask)
        if idx.dtype == np.bool_:
            mask[idx] = 1.0
        else:
            mask[idx.astype(int)] = 1.0
        existing = state.get("interior_dirichlet_mask")
        if existing is not None:
            existing.setValue(mask)
        else:
            from fipy import CellVariable
            state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    if source_pressure is not None:
        state["interior_dirichlet_value"] = float(source_pressure)

    mask_var = state.get("interior_dirichlet_mask")
    if mask_var is not None:
        mask_arr = _to_array(mask_var) > 0.5
    else:
        mask_arr = np.zeros(n_next.size, dtype=bool)
    p_src = float(state.get("interior_dirichlet_value", 0.0))

    # --- fresh fields for the next stage ---------------------------------
    s_next = np.where(mask_arr, 1.0, 0.0)
    state["saturation"].setValue(s_next)
    if reset_pressure:
        state["pressure"].setValue(np.where(mask_arr, p_src, 0.0))

    # The per-stage conservation ledger restarts from these fields.
    state["saturation_initial_for_ledger"] = np.array(s_next, copy=True)
    state["pressure_initial_for_ledger"] = np.array(
        _to_array(state["pressure"]), copy=True
    )
    state["injected_volume_total"] = 0.0
    state["clipped_volume_total"] = 0.0
    state["compressed_volume_total"] = 0.0
    state["last_div_q"] = None

    # --- ledger -----------------------------------------------------------
    n0 = np.asarray(state["stage_porosity_initial"], dtype=float)
    ledger = state["stage_ledger"]
    ledger["occupied_cumulative"] = float(np.sum((n0 - n_next) * vols))
    ledger["remaining_capacity"] = float(np.sum(n_next * vols))
    ledger["floor_clipped_total"] = float(state["floor_clipped_volume_total"])

    report["floor_clipped_step"] = floor_clipped_step
    state["stage_history"].append(report)
    state["stage_index"] = int(state.get("stage_index", 0)) + 1
    return ledger


def stage_ledger_report(state):
    """
    Cumulative accounting on the invariant stage-0 basis.

    fill_ratio_cumulative is the share of the ORIGINAL pore capacity now
    occupied by hardened grout; residual_ratio is what is left open. Both
    use the same denominator across every stage, which is the point of
    keeping the stage-0 porosity.
    """
    ledger = state.get("stage_ledger")
    if ledger is None:
        ledger = initialize_stage_ledger(state)
    capacity = ledger["pore_capacity_initial"]
    occupied = ledger["occupied_cumulative"]
    return {
        "stage_index": int(state.get("stage_index", 0)),
        "pore_capacity_initial": capacity,
        "occupied_cumulative": occupied,
        "remaining_capacity": ledger["remaining_capacity"],
        "floor_clipped_total": ledger["floor_clipped_total"],
        "fill_ratio_cumulative": (
            occupied / capacity if capacity > 0.0 else 0.0
        ),
        "residual_ratio": (
            ledger["remaining_capacity"] / capacity if capacity > 0.0 else 0.0
        ),
        "stage_history": list(state.get("stage_history", [])),
    }
