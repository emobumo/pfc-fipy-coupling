import numpy as np

from src.pfc_adapter.porosity_reader import read_cell_porosity_once


def porosity_to_permeability(porosity, params):
    p_min = float(params["porosity_min"])
    p_max = float(params["porosity_max"])
    k_min = float(params["permeability_min"])
    k_max = float(params["permeability_max"])
    exponent = float(params["porosity_to_permeability_exponent"])

    denom = max(p_max - p_min, 1.0e-12)
    p_norm = np.clip((porosity - p_min) / denom, 0.0, 1.0)
    k_rel = np.power(p_norm, exponent)
    return k_min + (k_max - k_min) * k_rel


def _build_structure_init_report(state, params):
    porosity = np.asarray(state["porosity"].value, dtype=float)
    permeability = np.asarray(state["permeability"].value, dtype=float)
    intrinsic_mobility = np.asarray(state["intrinsic_mobility"].value, dtype=float)
    return {
        "porosity_min": float(np.min(porosity)),
        "porosity_max": float(np.max(porosity)),
        "permeability_min": float(np.min(permeability)),
        "permeability_max": float(np.max(permeability)),
        "intrinsic_mobility_min": float(np.min(intrinsic_mobility)),
        "intrinsic_mobility_max": float(np.max(intrinsic_mobility)),
        "porosity_to_permeability_exponent": float(params["porosity_to_permeability_exponent"]),
        "reference_mobility": float(params["reference_mobility"]),
    }


def initialize_structure_mobility_once(state):
    """
    One-time initialization path:
    PFC local porosity/structure -> FiPy cell permeability/mobility.
    """
    if state.get("structure_initialized_once", False):
        if not state.get("structure_init_report"):
            params = state["slurry_parameters"]
            state["structure_init_report"] = _build_structure_init_report(state, params)
        return

    params = state["slurry_parameters"]
    porosity = read_cell_porosity_once(
        x=state["x"],
        y=state["y"],
        default_porosity=params["porosity_default"],
        min_porosity=params["porosity_min"],
        max_porosity=params["porosity_max"],
        fallback_particle_area_ratio=params["fallback_particle_area_ratio"],
    )

    permeability = porosity_to_permeability(porosity, params)
    intrinsic_mobility = params["reference_mobility"] * permeability

    state["porosity"].setValue(porosity)
    state["permeability"].setValue(permeability)
    state["intrinsic_mobility"].setValue(intrinsic_mobility)
    state["mobility"].setValue(intrinsic_mobility)
    state["structure_init_report"] = _build_structure_init_report(state, params)
    state["structure_initialized_once"] = True
