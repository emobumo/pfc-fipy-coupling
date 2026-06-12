# -*- coding: utf-8 -*-
"""
Verification ladder for the slurry-transport solver (Python 2.7 / unittest).

Ladder (see CLAUDE.md for status tracking):
  Step 1: 1D linear Darcy degeneration (lambda=0)  -- full assertions
  Step 2: 1D Bingham stagnation front L_max = p0/lambda -- placeholder
  Step 3: radial Gustafson-Stille benchmark            -- placeholder
  Step 4: mesh-convergence order                       -- placeholder

Step 1 deliberately bypasses apply_boundary_conditions (which hard-codes the
top-borehole placeholder inlet) and drives the core PDE kernel directly:
update_effective_mobility + _solve_pressure_once. That is the actual equation
assembly used in production (TransientTerm == DiffusionTerm + gravity source),
configured so the generalized model degenerates to linear saturated Darcy:
lambda=0 (rheology_model='linear'), uniform k and n, zero gravity, horizontal
1D strip, Dirichlet pressure on both ends.

Analytic steady state: p(x) = p0 * (1 - x/L), q0 = (k/mu) * p0 / L.

Runs under PFC 5.0's bundled Python 2.7, no PFC required:
    powershell -File scripts\\run_local.ps1 -m unittest discover -s tests -v
"""
import math
import unittest

import numpy as np

from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import (
    build_placeholder_slurry_parameters,
    initialize_slurry_variables,
)
from src.models.slurry_transport.equations import (
    update_effective_mobility,
    solve_pressure_step,
    _solve_pressure_once,
)


# --- Step 1 configuration -------------------------------------------------
P0 = 1000.0          # inlet pressure [Pa]
L = 1.0              # strip length [m]
NX = 50              # cells along x
PERMEABILITY = 1.0e-9   # uniform k [m^2]
VISCOSITY = 0.05        # mu [Pa.s]
MOBILITY = PERMEABILITY / VISCOSITY  # k/mu [m^2/(Pa.s)]


def _build_1d_linear_darcy_state():
    """Horizontal 1D strip, uniform k/n, lambda=0, gravity off."""
    dx = L / float(NX)
    mesh, x, y, fx, fy = build_mesh(nx=NX, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}

    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "linear"
    params["enable_bingham_yield"] = False   # lambda = 0
    params["gravity_y"] = 0.0                # horizontal problem
    state.update(initialize_slurry_variables(mesh, params=params))

    # Uniform saturated structure: k, n, and mobility k/mu set directly so the
    # analytic solution is exact (no porosity->permeability mapping involved).
    state["permeability"].setValue(PERMEABILITY)
    state["porosity"].setValue(0.35)
    state["mobility_structural"].setValue(MOBILITY)
    state["mobility_effective"].setValue(MOBILITY)

    # Two-end fixed pressure: p(0)=p0, p(L)=0. Top/bottom of the strip keep
    # FiPy's natural zero-flux condition, so the problem is truly 1D.
    state["pressure"].constrain(P0, mesh.facesLeft)
    state["pressure"].constrain(0.0, mesh.facesRight)
    return state


def _run_to_steady(state, dt=10.0, steps=8):
    """March the transient solve far past its storage time scale.

    Diffusivity = mobility / storage ~ (2e-8)/(1e-7) = 0.2 m^2/s, so the
    diffusion time over L=1 m is ~5 s; 8 steps of dt=10 s is fully steady.
    """
    for _ in range(steps):
        update_effective_mobility(state)
        _solve_pressure_once(state, dt=dt)
    return np.asarray(state["pressure"].value, dtype=float)


class TestStep1LinearDarcyDegeneration(unittest.TestCase):
    """Step 1: with lambda=0 the model must reduce to linear saturated Darcy."""

    def test_steady_pressure_matches_linear_profile(self):
        state = _build_1d_linear_darcy_state()
        p_num = _run_to_steady(state)
        x = np.asarray(state["x"], dtype=float)
        p_exact = P0 * (1.0 - x / L)
        # Normalize by p0 (p_exact crosses 0 at x=L, pointwise relative error
        # is singular there).
        max_rel_err = float(np.max(np.abs(p_num - p_exact))) / P0
        self.assertLess(
            max_rel_err,
            0.01,
            "steady pressure deviates from analytic line: max rel err = %g"
            % max_rel_err,
        )

    def test_steady_flux_matches_analytic(self):
        state = _build_1d_linear_darcy_state()
        p_num = _run_to_steady(state)
        x = np.asarray(state["x"], dtype=float)
        order = np.argsort(x)
        p_sorted = p_num[order]
        x_sorted = x[order]
        # Interior finite-difference flux q = -(k/mu) dp/dx between adjacent
        # cell centers (the profile is linear, so any interior pair works;
        # average over all pairs for robustness).
        dpdx = np.diff(p_sorted) / np.diff(x_sorted)
        q_num = float(np.mean(-MOBILITY * dpdx))
        q_exact = MOBILITY * P0 / L
        rel_err = abs(q_num - q_exact) / q_exact
        self.assertLess(
            rel_err,
            0.01,
            "steady flux %g deviates from analytic %g: rel err = %g"
            % (q_num, q_exact, rel_err),
        )


# --- Step 2 configuration ---------------------------------------------------
# 1D horizontal Bingham stagnation: left end held at p0, right end zero-flux,
# uniform k/phi, zero gravity, porous_bingham "threshold" truncation. The front
# must stall where the driving gradient falls to the start-up gradient lambda:
#     L_max = p0 / lambda,   lambda = 2*tau0 / r_eff,   r_eff = sqrt(8k/phi).
# tau0 is back-computed so that L_max = 0.6 m; the 1.0 m domain is 1.67*L_max.
S2_P0 = 1.0e5        # inlet pressure [Pa]
S2_LMAX = 0.6        # target stagnation length [m]
S2_L = 1.0           # domain length [m] (> 1.5 * L_max)
S2_NX = 100          # cells (dx = 0.01 m)
S2_K = 1.0e-9        # uniform permeability [m^2]
S2_PHI = 0.32        # uniform porosity
S2_MU = 0.05         # plastic viscosity [Pa.s]
S2_LAMBDA = S2_P0 / S2_LMAX                            # [Pa/m]
S2_TAU0 = 0.5 * S2_LAMBDA * math.sqrt(8.0 * S2_K / S2_PHI)  # [Pa]
# Time step must RESOLVE the filling transient: the front time constant is
# T = c*L_max^2/(2*k/mu) ~ 0.9 s here. A large dt lets one frozen-mobility
# implicit solve overshoot the stall profile, and the overshoot then locks in
# (below threshold the mobility is exactly zero, so a Bingham profile cannot
# relax back -- physically correct hysteresis, numerically demands small dt).
# Empirically the locked-in overshoot is ~2.5 cells at dt=0.01 s and scales
# down with dt (stale-mobility motion in the last Picard iteration of a step).
# The face-built mobility (round 4) makes the front face fully conductive
# (the legacy cell coefficient arithmetic-averaged it to half), roughly
# doubling the per-step front motion, so dt is halved again.
S2_DT = 0.0025       # time step [s]
S2_MAX_STEPS = 4800
S2_STATIC_STEPS = 15  # consecutive near-zero front moves that declare a stall

_STEP2_CACHE = {}


def _build_step2_state():
    dx = S2_L / float(S2_NX)
    mesh, x, y, fx, fy = build_mesh(nx=S2_NX, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}

    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "porous_bingham"
    params["porous_bingham_activation"] = "threshold"
    params["gravity_y"] = 0.0
    params["yield_stress"] = S2_TAU0
    params["plastic_viscosity"] = S2_MU
    # picard_tol keeps its default (1e-4 RELATIVE to the step's initial
    # Picard residual), which is pressure-scale invariant.
    # Feeds the auto epsilon default 1e-6 * p0 / L_domain.
    params["inlet_pressure_core_value"] = S2_P0
    state.update(initialize_slurry_variables(mesh, params=params))

    state["permeability"].setValue(S2_K)
    state["porosity"].setValue(S2_PHI)
    # Seed mobility at the unyielded Darcy scale k/mu so the under-relaxed
    # Picard update does not blend with the meaningless init value 1.0.
    state["mobility_structural"].setValue(S2_K / S2_MU)
    state["mobility_effective"].setValue(S2_K / S2_MU)

    # Constant-pressure injection at x=0; right end keeps FiPy's natural
    # zero-flux condition. Initial pressure is 0 everywhere.
    state["pressure"].constrain(S2_P0, state["mesh"].facesLeft)
    return state


def _step2_sorted_profile(state):
    x = np.asarray(state["x"], dtype=float)
    p = np.asarray(state["pressure"].value, dtype=float)
    order = np.argsort(x)
    return x[order], p[order]


def _step2_front_from_pressure(x_sorted, p_sorted):
    """Continuous front proxy: x where p first falls below 1% p0 (linear
    interpolation between cell centers). Used only as the stagnation-detection
    signal for the marching loop."""
    thresh = 0.01 * S2_P0
    below = np.where(p_sorted < thresh)[0]
    if below.size == 0:
        return float(x_sorted[-1])
    i = int(below[0])
    if i == 0:
        return float(x_sorted[0])
    x_a = float(x_sorted[i - 1])
    x_b = float(x_sorted[i])
    p_a = float(p_sorted[i - 1])
    p_b = float(p_sorted[i])
    if p_a <= p_b:
        return x_b
    return x_a + (p_a - thresh) * (x_b - x_a) / (p_a - p_b)


def _step2_stagnation_from_gradient(x_sorted, p_sorted):
    """Stagnation position: first inter-cell face where |dp/dx| drops below
    lambda. At stall the yielded zone plateaus at |dp/dx| ~= lambda and the
    unyielded zone sits at ~0, so the discriminator is set mid-band (0.5
    lambda) to be robust against the +/- few percent numerical noise around
    lambda itself; any value strictly between the two plateaus moves the
    detected face by less than one cell."""
    dpdx = np.abs(np.diff(p_sorted) / np.diff(x_sorted))
    below = np.where(dpdx < 0.5 * S2_LAMBDA)[0]
    if below.size == 0:
        return float(x_sorted[-1])
    i = int(below[0])
    return 0.5 * (float(x_sorted[i]) + float(x_sorted[i + 1]))


def _run_step2_march():
    """March the transient injection until the front velocity stalls; cache
    the result so both assertions reuse one run."""
    if "state" not in _STEP2_CACHE:
        state = _build_step2_state()
        dx = S2_L / float(S2_NX)
        front_history = []
        static_count = 0
        front_prev = None
        for _ in range(S2_MAX_STEPS):
            solve_pressure_step(state, dt=S2_DT)
            x_sorted, p_sorted = _step2_sorted_profile(state)
            front = _step2_front_from_pressure(x_sorted, p_sorted)
            front_history.append(front)
            if (front_prev is not None) and (abs(front - front_prev) < 0.01 * dx):
                static_count += 1
            else:
                static_count = 0
            front_prev = front
            # Stalled: front moved less than 1% of a cell for S2_STATIC_STEPS
            # straight steps (only once the front is clearly inside the domain).
            if (front > 0.2 * S2_LMAX) and (static_count >= S2_STATIC_STEPS):
                break
        _STEP2_CACHE["state"] = state
        _STEP2_CACHE["front_history"] = front_history
    return _STEP2_CACHE["state"], _STEP2_CACHE["front_history"]


class TestStep2BinghamStagnation(unittest.TestCase):
    """Step 2: 1D Bingham front must stall at L_max = p0/lambda."""

    def test_pressure_profile_approaches_truncated_line(self):
        state, _ = _run_step2_march()
        x_sorted, p_sorted = _step2_sorted_profile(state)
        p_exact = np.maximum(0.0, S2_P0 - S2_LAMBDA * x_sorted)
        max_dev = float(np.max(np.abs(p_sorted - p_exact))) / S2_P0
        self.assertLess(
            max_dev,
            0.05,
            "stalled profile deviates from max(0, p0 - lambda*x): "
            "max |dp|/p0 = %g" % max_dev,
        )

    def test_front_stalls_at_p0_over_lambda(self):
        state, front_history = _run_step2_march()
        x_sorted, p_sorted = _step2_sorted_profile(state)
        stall_pos = _step2_stagnation_from_gradient(x_sorted, p_sorted)
        rel_err = abs(stall_pos - S2_LMAX) / S2_LMAX
        self.assertLess(
            rel_err,
            0.05,
            "stagnation front at %.4f m vs analytic L_max %.4f m "
            "(rel err %.3f; front history tail: %s)"
            % (stall_pos, S2_LMAX, rel_err,
               ["%.4f" % f for f in front_history[-5:]]),
        )


# --- Step 3 configuration ---------------------------------------------------
# 2D planar radial constant-pressure injection (Gustafson-Stille stagnation
# limit). Quarter symmetry on a Grid2D: the borehole of radius r0 sits at the
# origin corner, represented by Dirichlet p0 on the bottom faces with x < r0
# and the left faces with y < r0; the two axes are natural no-flux symmetry
# planes, so this equals a full-plane borehole. Outer boundary (right/top,
# r >= 1.1 m > 1.5*I_max) is held at the far-field p = 0.
#
# Stalled state: |dp/dr| = lambda pointwise along each ray, so along the +x
# axis p(x) = p0 - lambda*(x - r0) and the front stalls at
#     I_max = r0 + p0/lambda.
# All assertions sample the +x axis (bottom cell row, a symmetry plane where
# dp/dy = 0 and the gradient is purely radial).
S3_P0 = 1.0e5        # borehole pressure [Pa]
S3_R0 = 0.1          # borehole radius [m]
S3_L = 1.1           # quarter-domain side [m] (> 1.5 * I_max)
S3_NX = 55           # cells per side (dx = 0.02 m)
S3_K = 1.0e-9        # uniform permeability [m^2]
S3_PHI = 0.32        # uniform porosity
S3_MU = 0.05         # plastic viscosity [Pa.s]
S3_LAMBDA = 1.0e5 / 0.6   # start-up gradient [Pa/m], same lambda as step 2
S3_TAU0 = 0.5 * S3_LAMBDA * math.sqrt(8.0 * S3_K / S3_PHI)  # [Pa]
S3_IMAX = S3_R0 + S3_P0 / S3_LAMBDA                         # = 0.7 m
# dt from the front-time-constant rule of section 3 of the saturation design
# doc, with I_max as the characteristic length:
#     T = c * I_max^2 / (2 * k/mu) = 1e-7 * 0.49 / (2 * 2e-8) = 1.225 s
# Round 2 found dt ~ T/100..T/200 keeps the locked-in overshoot around one
# cell; dx here (0.02 m) is twice the step-2 cell. The round-4 face-built
# mobility doubles the front-face conductance (see step-2 note), so dt is
# halved from 0.01 to 0.005 s ~ T/245.
S3_DT = 0.005        # time step [s]
S3_MAX_STEPS = 1200
S3_STATIC_STEPS = 15

_STEP3_CACHE = {}


def _build_step3_state():
    dx = S3_L / float(S3_NX)
    mesh, x, y, fx, fy = build_mesh(nx=S3_NX, ny=S3_NX, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}

    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "porous_bingham"
    params["porous_bingham_activation"] = "threshold"
    params["gravity_y"] = 0.0
    params["yield_stress"] = S3_TAU0
    params["plastic_viscosity"] = S3_MU
    params["inlet_pressure_core_value"] = S3_P0  # feeds the auto-eps default
    state.update(initialize_slurry_variables(mesh, params=params))

    state["permeability"].setValue(S3_K)
    state["porosity"].setValue(S3_PHI)
    state["mobility_structural"].setValue(S3_K / S3_MU)
    state["mobility_effective"].setValue(S3_K / S3_MU)

    mesh_fx, mesh_fy = mesh.faceCenters()
    borehole = (mesh.facesBottom & (mesh_fx < S3_R0)) | (
        mesh.facesLeft & (mesh_fy < S3_R0)
    )
    state["pressure"].constrain(S3_P0, borehole)
    state["pressure"].constrain(0.0, mesh.facesRight | mesh.facesTop)
    return state


def _step3_axis_profile(state):
    """Pressure along the +x axis: the bottom cell row (y = dy/2)."""
    x = np.asarray(state["x"], dtype=float)
    y = np.asarray(state["y"], dtype=float)
    p = np.asarray(state["pressure"].value, dtype=float)
    row = np.isclose(y, np.min(y))
    order = np.argsort(x[row])
    return x[row][order], p[row][order]


def _step3_front_from_pressure(x_sorted, p_sorted):
    """Continuous front proxy along the axis: x where p crosses 1% p0."""
    thresh = 0.01 * S3_P0
    below = np.where(p_sorted < thresh)[0]
    if below.size == 0:
        return float(x_sorted[-1])
    i = int(below[0])
    if i == 0:
        return float(x_sorted[0])
    x_a = float(x_sorted[i - 1])
    x_b = float(x_sorted[i])
    p_a = float(p_sorted[i - 1])
    p_b = float(p_sorted[i])
    if p_a <= p_b:
        return x_b
    return x_a + (p_a - thresh) * (x_b - x_a) / (p_a - p_b)


def _step3_stagnation_from_gradient(x_sorted, p_sorted):
    """First inter-cell face BEYOND the borehole (x >= r0) where |dp/dx|
    drops below 0.5*lambda (same mid-band discriminator as step 2; inside the
    borehole the profile is flat at ~p0, so the search must start at r0)."""
    x_face = 0.5 * (x_sorted[:-1] + x_sorted[1:])
    dpdx = np.abs(np.diff(p_sorted) / np.diff(x_sorted))
    candidates = np.where((x_face >= S3_R0) & (dpdx < 0.5 * S3_LAMBDA))[0]
    if candidates.size == 0:
        return float(x_sorted[-1])
    return float(x_face[int(candidates[0])])


def _run_step3_march():
    if "state" not in _STEP3_CACHE:
        state = _build_step3_state()
        dx = S3_L / float(S3_NX)
        front_history = []
        static_count = 0
        front_prev = None
        for _ in range(S3_MAX_STEPS):
            solve_pressure_step(state, dt=S3_DT)
            x_sorted, p_sorted = _step3_axis_profile(state)
            front = _step3_front_from_pressure(x_sorted, p_sorted)
            front_history.append(front)
            if (front_prev is not None) and (abs(front - front_prev) < 0.01 * dx):
                static_count += 1
            else:
                static_count = 0
            front_prev = front
            if (front > S3_R0 + 0.2 * (S3_P0 / S3_LAMBDA)) and (
                static_count >= S3_STATIC_STEPS
            ):
                break
        _STEP3_CACHE["state"] = state
        _STEP3_CACHE["front_history"] = front_history
    return _STEP3_CACHE["state"], _STEP3_CACHE["front_history"]


class TestStep3RadialGustafsonStille(unittest.TestCase):
    """Step 3: radial constant-pressure grouting must stall at I_max."""

    def test_stall_radius_matches_imax(self):
        state, front_history = _run_step3_march()
        x_sorted, p_sorted = _step3_axis_profile(state)
        stall = _step3_stagnation_from_gradient(x_sorted, p_sorted)
        rel_err = abs(stall - S3_IMAX) / S3_IMAX
        self.assertLess(
            rel_err,
            0.08,
            "radial stagnation at %.4f m vs analytic I_max %.4f m "
            "(rel err %.3f; front history tail: %s)"
            % (stall, S3_IMAX, rel_err,
               ["%.4f" % f for f in front_history[-5:]]),
        )

    def test_axis_profile_matches_radial_line(self):
        state, _ = _run_step3_march()
        x_sorted, p_sorted = _step3_axis_profile(state)
        outside = x_sorted > S3_R0
        p_exact = np.maximum(0.0, S3_P0 - S3_LAMBDA * (x_sorted[outside] - S3_R0))
        max_dev = float(np.max(np.abs(p_sorted[outside] - p_exact))) / S3_P0
        self.assertLess(
            max_dev,
            0.08,
            "stalled radial profile deviates from p0 - lambda*(r - r0): "
            "max |dp|/p0 = %g" % max_dev,
        )


class TestStep4MeshConvergence(unittest.TestCase):
    """Step 4 placeholder: grid-refinement convergence order."""

    def test_l2_error_convergence_order(self):
        self.skipTest(
            "Step 4 not implemented yet. Plan: re-run a case with an exact "
            "solution (Step 1 setup or a manufactured solution with "
            "x-varying k) on successively refined meshes (e.g. nx = 16, 32, "
            "64, 128), fit log2(L2 error) vs log2(h), and assert the slope "
            "is close to the theoretical order (~2 for FiPy's central "
            "DiffusionTerm)."
        )


if __name__ == "__main__":
    unittest.main()
