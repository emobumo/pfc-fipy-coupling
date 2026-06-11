# -*- coding: utf-8 -*-
import numpy as np

from src.pfc_adapter.porosity_reader import read_cell_structure_once


def kozeny_carman_permeability(porosity, cell_diameter, params):
    """
    Kozeny-Carman permeability with a per-cell grain size:

        k = (d^2 / C) * phi^3 / (1 - phi)^2      [m^2]

    where d is the local characteristic particle diameter and C the
    Kozeny-Carman constant (~180). NOTE: this is a 3D soil relation applied with
    a 2D areal porosity and a disk-derived grain size, so it is an engineering
    approximation; coarse rockfill may also be non-Darcy. Calibrate against
    measured permeability when available.
    """
    kc_const = max(float(params.get("kozeny_carman_constant", 180.0)), 1.0e-9)
    k_clip_min = float(params.get("permeability_clip_min", 1.0e-12))
    k_clip_max = float(params.get("permeability_clip_max", 1.0e0))
    if k_clip_max < k_clip_min:
        k_clip_min, k_clip_max = k_clip_max, k_clip_min

    phi = np.clip(np.asarray(porosity, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    d = np.asarray(cell_diameter, dtype=float)
    k = (d * d / kc_const) * (phi ** 3) / ((1.0 - phi) ** 2)
    return np.clip(k, k_clip_min, k_clip_max)


def porosity_to_permeability(porosity, params, cell_diameter=None):
    formula = params.get(
        "porosity_to_permeability_formula", "power_normalized_linear_range"
    )
    if formula == "kozeny_carman":
        if cell_diameter is None:
            cell_diameter = np.ones_like(np.asarray(porosity, dtype=float)) * float(
                params.get("default_particle_diameter", 0.25)
            )
        return kozeny_carman_permeability(porosity, cell_diameter, params)

    # Legacy normalized power-law range mapping (kept as a switchable option).
    p_min = float(params["porosity_min"])
    p_max = float(params["porosity_max"])
    k_min = float(params["permeability_min"])
    k_max = float(params["permeability_max"])
    p_clip_min = float(params.get("porosity_clip_min", p_min))
    p_clip_max = float(params.get("porosity_clip_max", p_max))
    k_clip_min = float(params.get("permeability_clip_min", k_min))
    k_clip_max = float(params.get("permeability_clip_max", k_max))
    exponent = float(params["porosity_to_permeability_exponent"])

    if p_clip_max < p_clip_min:
        p_clip_min, p_clip_max = p_clip_max, p_clip_min
    if k_clip_max < k_clip_min:
        k_clip_min, k_clip_max = k_clip_max, k_clip_min

    porosity = np.clip(porosity, p_clip_min, p_clip_max)
    denom = max(p_max - p_min, 1.0e-12)
    p_norm = np.clip((porosity - p_min) / denom, 0.0, 1.0)
    k_rel = np.power(p_norm, exponent)
    permeability = k_min + (k_max - k_min) * k_rel
    permeability = np.clip(permeability, k_clip_min, k_clip_max)
    return permeability


def _build_structure_init_report(state, params):
    porosity = np.asarray(state["porosity"].value, dtype=float)
    permeability = np.asarray(state["permeability"].value, dtype=float)
    mobility_structural = np.asarray(state["mobility_structural"].value, dtype=float)
    report = {
        "porosity_min": float(np.min(porosity)),
        "porosity_max": float(np.max(porosity)),
        "permeability_min": float(np.min(permeability)),
        "permeability_max": float(np.max(permeability)),
        "intrinsic_mobility_min": float(np.min(mobility_structural)),
        "intrinsic_mobility_max": float(np.max(mobility_structural)),
        "porosity_to_permeability_formula": params.get(
            "porosity_to_permeability_formula", "power_normalized_linear_range"
        ),
        "porosity_range": [float(params["porosity_min"]), float(params["porosity_max"])],
        "porosity_clip_range": [
            float(params.get("porosity_clip_min", params["porosity_min"])),
            float(params.get("porosity_clip_max", params["porosity_max"])),
        ],
        "permeability_range": [float(params["permeability_min"]), float(params["permeability_max"])],
        "permeability_clip_range": [
            float(params.get("permeability_clip_min", params["permeability_min"])),
            float(params.get("permeability_clip_max", params["permeability_max"])),
        ],
        "porosity_to_permeability_exponent": float(params["porosity_to_permeability_exponent"]),
        "reference_mobility": float(params["reference_mobility"]),
    }
    if "cell_diameter" in state:
        d = np.asarray(state["cell_diameter"], dtype=float)
        report["cell_diameter_min"] = float(np.min(d))
        report["cell_diameter_max"] = float(np.max(d))
        report["cell_diameter_mean"] = float(np.mean(d))
    return report


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
    porosity, cell_diameter = read_cell_structure_once(
        x=state["x"],
        y=state["y"],
        default_porosity=params["porosity_default"],
        min_porosity=params["porosity_min"],
        max_porosity=params["porosity_max"],
        fallback_particle_area_ratio=params["fallback_particle_area_ratio"],
        default_diameter=float(params.get("default_particle_diameter", 0.25)),
    )
    state["cell_diameter"] = cell_diameter

    permeability = porosity_to_permeability(porosity, params, cell_diameter=cell_diameter)
    # reference_mobility is interpreted as the structural mobility scale
    # at permeability_max [m^2 / (Pa·s)].
    permeability_max = max(float(params["permeability_max"]), 1.0e-30)
    mobility_structural = params["reference_mobility"] * (permeability / permeability_max)

    state["porosity"].setValue(porosity)
    state["permeability"].setValue(permeability)
    state["mobility_structural"].setValue(mobility_structural)
    state["mobility_effective"].setValue(mobility_structural)
    state["intrinsic_mobility"].setValue(mobility_structural)
    state["mobility"].setValue(mobility_structural)
    state["structure_init_report"] = _build_structure_init_report(state, params)
    state["structure_initialized_once"] = True
