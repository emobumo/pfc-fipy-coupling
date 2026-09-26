# -*- coding: utf-8 -*-
"""
The Zhaojin 2# vein, as a vertical section through one cross-cut.

Everything geometric here comes from the one document there is: the 2023-11
design change for the waste-rock-fill grouting project (see the project
memory 'zhaojin-design-document'). What the document does not say is a
switch in CONFIG, not a hidden constant, so the screening script can sweep it.

THE SECTION
-----------
The holes are drilled from the -26 m level cross-cuts (穿脉), perpendicular to
strike, so the natural 2D model is the vertical plane of one cross-cut. In it:

    0 m      surface; the fill reaches it
             waste-rock FILL in a 70-degree vein, horizontal width w
    -26 m    level: the cross-cut, the hole collars in its roof
             stored broken ORE (存窿矿石) in the old shrinkage stope
    -56 m    level: bottom of the stored ore; unmined vein below
    -63 m    level: where the runaway appeared (cuts 5#, 7#, 9#, 15#)

Walls are monzogranite. Each porosity zone of the document (10 / 45 / 25%)
belongs to a stretch of the vein ALONG STRIKE (cuts 1-6 / 6-10 / 10-15), so
each is a different section, not a region within one.

THE HOLE
--------
Collared in the cross-cut roof (-24 m), elevation 35 degrees (15-25 after the
design change), drilled 7 -> 10 -> 17.5 m and required to end >= 3 m in host
rock. The first 2.0 m is the sealed orifice pipe; the perforated pipe bleeds
beyond it. Which wall it climbs toward is not stated; the design itself
settles it. Crossing a 70-degree vein at 35 degrees the hole spends 0.97 x
the horizontal width in the fill when it climbs toward the hanging wall and
1.64 x when it climbs toward the footwall. The layout was sized for a 10 m
wide fill: 2 + 9.7 + 5.8 m (hanging wall) satisfies ">= 3 m into rock";
2 + 16.4 m (footwall) cannot. Default: hanging wall.

MATERIALS THAT ARE NOT POROUS MEDIA
-----------------------------------
Granite and cement are given phi = 1e-3, the stage-closure floor, well below
the 0.05 the permeability law was anchored on. That is deliberate: their only
job is to be impassable to grout, and at 1e-3 the start-up gradient is
6e7 Pa/m, so half a metre costs 30 MPa against a 5 MPa budget. (At phi 0.05
a metre of "granite" would cost only 1.1 MPa and grout would tunnel through
wall rock -- wrong.) The -63 m drift is a cavity; following the fields
module it is capped at phi 0.60 rather than pushed toward 1.

Py2.7. Run as a module from scripts/zhaojin_screen.py (and later cases).
"""
from __future__ import print_function

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fipy import CellVariable

from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.coupling.porosity_to_permeability import porosity_to_permeability
from src.models.slurry_transport.variables import (
    grout_material_v06,
    build_placeholder_slurry_parameters,
    initialize_slurry_variables,
)

# --- the document -----------------------------------------------------------
DIP_DEG = 70.0                 # vein dip
Y_SURFACE = 0.0
Y_LEVEL = -26.0                # grouting level: cross-cut floor
Y_ORE_BOTTOM = -56.0           # bottom of the stored ore
Y_OUTLET = -63.0               # where the runaway appeared
CROSSCUT_HEIGHT = 2.0          # cross-cut section 2.0 x 2.0 m
ORIFICE_DEPTH = 2.0            # sealed orifice pipe
SEAL_THICKNESS = 2.0           # measured bottom-seal thickness (average)
P0 = 5.0e6                     # end pressure

# Zones along strike: fill porosity, horizontal width, changed-design layer
# thickness, and the design grout per metre of strike [m^3/m] (document,
# section 6.3 -- the first term there uses 13 m against 12 m in the text; the
# text value is used for the layer, the document's own product for the volume).
ZONES = {
    "z1_cuts1-6": {"phi": 0.10, "width": 7.5, "layer": 12.0, "grout_per_m": 7.5 * 13 * 0.10},
    "z2_cuts6-10": {"phi": 0.45, "width": 7.0, "layer": 16.0, "grout_per_m": 7.0 * 16 * 0.45},
    "z3_cuts10-15": {"phi": 0.25, "width": 6.5, "layer": 12.0, "grout_per_m": 6.5 * 12 * 0.25},
}
RUNAWAY_CUTS = {"z1_cuts1-6": ["5#"], "z2_cuts6-10": ["7#", "9#"], "z3_cuts10-15": ["15#"]}
# The ORIGINAL design -- the one in force when the runaway happened (single
# fluid, 35 deg, no seal): one 22 m layer above the grouting level (-26 to
# -4 m), 20% porosity assumed throughout, 86 x 6.5 x 22 x 0.2 = 2460 m3 of
# grout, i.e. 2460 / 86 = 28.6 m3 per metre of strike in every zone. ZONES
# above is the CHANGED design, written after the runaway; its porosities are
# the document's estimates back-calculated from grout takes, and they remain
# the best description of the GROUND in every run.
ORIGINAL_DESIGN = {"layer": 22.0, "phi": 0.20, "width": 6.5, "grout_per_m": 6.5 * 22.0 * 0.20}

# --- material surrogates ----------------------------------------------------
NO_GROUT_PHI = 1.0e-3          # granite, unmined ore, cement: see module notes
CAVITY_PHI = 0.60              # the -63 m drift

# --- what the document does not say: switches with defaults ----------------
DEFAULT_CONFIG = {
    "zone": "z2_cuts6-10",
    "hole_angle": 35.0,        # 35 original, 15-25 after the change
    "hole_length": 17.5,
    "climb": "hanging",        # "hanging" (default, see notes) or "footwall"
    "contacts": True,          # through-going gaps along both walls
    "contact_width": 1.0,      # normal width of each contact band [m]
    "contact_phi": 0.35,
    "contact_bottom": Y_OUTLET,  # how far down the contacts run
    "ore_phi": 0.35,           # stored broken ore
    "base_connected": True,    # fill base open to the ore below?
    "seal": False,             # the 2 m double-liquid bottom seal
}

# Region labels, for diagnostics and figures.
ROCK, FILL, ORE, CONTACT, SEAL, OUTLET = 0, 1, 2, 3, 4, 5
REGION_NAMES = {ROCK: "rock", FILL: "fill", ORE: "stored ore",
                CONTACT: "contact", SEAL: "seal", OUTLET: "outlet drift"}

# A fixed domain large enough for every configuration above, so one mesh
# serves a whole screening sweep. Collar at x = 0.
X_MIN, X_MAX = -30.0, 22.0
Y_MIN, Y_MAX = -66.0, 0.0


def config(**overrides):
    cfg = dict(DEFAULT_CONFIG)
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError("unknown config key %r" % k)
        cfg[k] = v
    return cfg


# --- geometry ---------------------------------------------------------------

def _tan_dip():
    return math.tan(math.radians(DIP_DEG))


def hole_direction(cfg):
    """Unit vector of the hole: up at hole_angle, toward the climbed wall."""
    a = math.radians(float(cfg["hole_angle"]))
    s = -1.0 if cfg["climb"] == "hanging" else 1.0
    return (s * math.cos(a), math.sin(a))


def collar():
    return (0.0, Y_LEVEL + CROSSCUT_HEIGHT)


def walls(cfg):
    """
    x of the footwall and hanging-wall contacts as functions of height.

    The vein leans toward +x going up, so the hanging wall is on the -x side.
    The hole enters the fill through the NEAR wall exactly at the end of the
    orifice pipe (drilled in solid rock, stopped on entering loose ground).
    """
    width = float(ZONES[cfg["zone"]]["width"])
    cx, cy = collar()
    d = hole_direction(cfg)
    ex, ey = cx + ORIFICE_DEPTH * d[0], cy + ORIFICE_DEPTH * d[1]   # fill entry
    t = _tan_dip()
    if cfg["climb"] == "hanging":
        def x_fw(y):
            return ex + (y - ey) / t

        def x_hw(y):
            return x_fw(y) - width
    else:
        def x_hw(y):
            return ex + (y - ey) / t

        def x_fw(y):
            return x_hw(y) + width
    return x_fw, x_hw


def fill_crossing_length(cfg):
    """Length of hole inside the fill (straight-line geometry)."""
    width = float(ZONES[cfg["zone"]]["width"])
    a = math.radians(float(cfg["hole_angle"]))
    t = _tan_dip()
    rate = (math.cos(a) + math.sin(a) / t) if cfg["climb"] == "hanging" \
        else (math.cos(a) - math.sin(a) / t)
    return width / rate


# --- fields ---------------------------------------------------------------

def section_fields(mx, my, cfg):
    """(phi, region) per cell for configuration cfg."""
    zone = ZONES[cfg["zone"]]
    x_fw, x_hw = walls(cfg)
    xf = np.array([x_fw(y) for y in my])
    xh = np.array([x_hw(y) for y in my])
    in_vein = (mx >= xh) & (mx <= xf)
    sin_dip = math.sin(math.radians(DIP_DEG))

    region = np.zeros(mx.size, dtype=int) + ROCK
    phi = np.zeros(mx.size) + NO_GROUT_PHI

    fill = in_vein & (my > Y_LEVEL)
    ore = in_vein & (my <= Y_LEVEL) & (my > Y_ORE_BOTTOM)
    region[fill] = FILL
    phi[fill] = zone["phi"]
    region[ore] = ORE
    phi[ore] = float(cfg["ore_phi"])

    if not cfg["base_connected"]:
        # A tight layer across the fill base: grout can only go down through
        # the contacts, if there are any.
        cut = in_vein & (my <= Y_LEVEL) & (my > Y_LEVEL - 1.0)
        region[cut] = ROCK
        phi[cut] = NO_GROUT_PHI

    if cfg["contacts"]:
        half = float(cfg["contact_width"]) / sin_dip          # horizontal width
        near_fw = (mx <= xf) & (mx > xf - half)
        near_hw = (mx >= xh) & (mx < xh + half)
        band = (near_fw | near_hw) & (my <= Y_SURFACE) & (my >= float(cfg["contact_bottom"]))
        region[band] = CONTACT
        phi[band] = float(cfg["contact_phi"])

    if cfg["seal"]:
        seal = in_vein & (my > Y_LEVEL) & (my <= Y_LEVEL + SEAL_THICKNESS)
        region[seal] = SEAL
        phi[seal] = NO_GROUT_PHI

    # The -63 m drift: 2 x 2 m, on the footwall side of the vein, touching it.
    xo = x_fw(Y_OUTLET + 1.0)
    drift = (mx >= xo) & (mx <= xo + 2.0) & (my >= Y_OUTLET) & (my <= Y_OUTLET + 2.0)
    region[drift] = OUTLET
    phi[drift] = CAVITY_PHI
    return phi, region


def hole_cells(mx, my, cfg, depth=None, cell=None):
    """Cells of the perforated pipe: past the orifice, out to `depth`."""
    depth = float(cfg["hole_length"] if depth is None else depth)
    cx, cy = collar()
    d = hole_direction(cfg)
    if cell is None:
        cell = float(np.min(np.diff(np.unique(np.round(mx, 9)))))
    idx = set()
    n = int(math.ceil((depth - ORIFICE_DEPTH) / (0.2 * cell))) + 1
    for k in range(n):
        s = ORIFICE_DEPTH + (depth - ORIFICE_DEPTH) * k / float(max(n - 1, 1))
        px, py = cx + s * d[0], cy + s * d[1]
        j = int(np.argmin((mx - px) ** 2 + (my - py) ** 2))
        idx.add(j)
    return np.array(sorted(idx), dtype=int)


def targets(mx, my, region, cfg):
    """
    Cell masks for what the screening asks about:
        outlet      the -63 m drift
        ore_bottom  the lowest 2 m of the stored ore (draw points at -56 m;
                    ore passes from there to -63 m would make this runaway too)
        design      the changed-design reinforcement layer in the fill
    """
    layer = float(ZONES[cfg["zone"]]["layer"])
    return {
        "outlet": region == OUTLET,
        "ore_bottom": (region == ORE) & (my <= Y_ORE_BOTTOM + 2.0),
        "design": ((region == FILL) | (region == CONTACT))
        & (my > Y_LEVEL) & (my <= Y_LEVEL + layer),
    }


# --- solver state ------------------------------------------------------------

def slurry_params():
    mat = grout_material_v06()
    params = build_placeholder_slurry_parameters()
    params.update({
        "rheology_model": "porous_bingham",
        "porous_bingham_activation": "threshold",
        "pressure_coeff_form": "face",
        "yield_truncation_mode": "papanastasiou",
        "enable_saturation_transport": True,
        "porosity_to_permeability_formula": "calibrated_power",
        "calibrated_permeability_coefficient": mat["calibrated_permeability_coefficient"],
        "yield_stress": mat["yield_stress"],
        "plastic_viscosity": mat["plastic_viscosity"],
        "slurry_density": mat["slurry_density"],
        "gravity_y": -9.81,
        "inlet_pressure_core_value": P0,
        "reference_storage": 1.0e-10,
        "picard_transient_mode": "backward_euler",
        "enable_front_dt_constraint": False,
        "dt_cfl": 0.5,
        "picard_relaxation_fill": 0.15,
        "picard_max_iters": 12,
        "picard_tol": 1.0e-3,
    })
    return params


def build_mesh(cell):
    mesh, x, y, fx, fy = build_mesh_for_domain(X_MIN, X_MAX, Y_MIN, Y_MAX, cell)
    return mesh, x, y, fx, fy


def build_state(cfg, cell, mesh_bundle=None):
    """
    A solver-ready state for configuration cfg. Pass a mesh_bundle from
    build_mesh() to reuse one mesh across a sweep.
    Returns (state, mx, my, phi, region, hcells).
    """
    mesh, x, y, fx, fy = mesh_bundle if mesh_bundle is not None else build_mesh(cell)
    mx = np.asarray(x, dtype=float)
    my = np.asarray(y, dtype=float)
    params = slurry_params()
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh, params=params))
    phi, region = section_fields(mx, my, cfg)
    set_porosity(state, phi)
    hcells = hole_cells(mx, my, cfg, cell=cell)
    mask = np.zeros(mx.size)
    mask[hcells] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    state["interior_dirichlet_value"] = P0
    s0 = np.zeros(mx.size)
    s0[hcells] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, P0, 0.0))
    return state, mx, my, phi, region, hcells


def set_porosity(state, phi):
    params = state["slurry_parameters"]
    k = porosity_to_permeability(phi, params)
    mu = float(params["plastic_viscosity"])
    state["porosity"].setValue(phi)
    state["permeability"].setValue(k)
    for key in ("mobility_structural", "mobility_effective", "mobility",
                "intrinsic_mobility"):
        state[key].setValue(k / mu)


if __name__ == "__main__":
    cfg = config()
    print("default configuration:", cfg)
    print("fill crossing: hanging %.2f m, footwall %.2f m (zone width %.1f m)"
          % (fill_crossing_length(config(climb="hanging")),
             fill_crossing_length(config(climb="footwall")),
             ZONES[cfg["zone"]]["width"]))
