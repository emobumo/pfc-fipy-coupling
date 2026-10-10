# -*- coding: utf-8 -*-
"""
Verification ladder for the slurry-transport solver (Python 2.7 / unittest).

Ladder (see CLAUDE.md for status tracking):
  Step 1: 1D linear Darcy degeneration (lambda=0)  -- full assertions
  Step 2: 1D Bingham stagnation front L_max = p0/lambda -- placeholder
  Step 3: radial stall solution of the model's flow law -- placeholder
  Step 4: mesh-convergence order                       -- placeholder
  Step 7: between-stage closure conservation           -- full assertions

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
from src.coupling.porosity_to_permeability import porosity_to_permeability
from src.models.slurry_transport.stage_update import (
    get_stage_porosity_floor,
    initialize_stage_ledger,
    stage_fill_report,
    apply_stage_closure,
    stage_ledger_report,
)
from src.models.slurry_transport.equations import (
    update_effective_mobility,
    solve_pressure_step,
    solve_transport_step,
    conservation_report,
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
    # Inlet faces double as the S=1 supply boundary for the upwind k_r when
    # the saturation transport is enabled (2b reuses this builder).
    state["open_pressure_faces"] = state["mesh"].facesLeft
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
# 2D planar radial constant-pressure injection, checked against the radial
# stall solution of this model's own flow law (derived below, no formula
# taken from the literature). Quarter symmetry on a Grid2D: the borehole of radius r0 sits at the
# origin corner, represented by Dirichlet p0 on the bottom faces with x < r0
# and the left faces with y < r0; the two axes are natural no-flux symmetry
# planes, so this equals a full-plane borehole. Outer boundary (right/top,
# r >= 1.1 m > 1.5*I_max) is held at the far-field p = 0.
#
# Derivation. With gravity off and S = 1 the flux is
#     q = -(k/mu) * max(0, 1 - lambda/|grad p|) * grad p,
# which is zero wherever |grad p| <= lambda. Constant-pressure injection
# keeps pushing the front while the gradient behind it exceeds lambda, so
# at the final stall q = 0 everywhere and the wetted region is as large as
# it can be: |dp/dr| = lambda pointwise along each ray. Because q = 0 there
# is no continuity (1/r spreading) term, so the radial stall profile is the
# 1D one shifted by the borehole radius: integrating dp/dr = -lambda from
# p(r0) = p0 to p = 0 gives, along the +x axis,
#     p(x) = p0 - lambda*(x - r0),   I_max = r0 + p0/lambda.
# The stall extent is independent of geometry and of mu; it has the same
# form as the stall length of the Gustafson et al. plate solutions
# (I_max = dp*b/(2*tau0), lambda = 2*tau0/b), i.e. the same type of
# benchmark, but the formula here comes from this model's flow law.
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
    state["open_pressure_faces"] = borehole
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


class TestStep3RadialStallSolution(unittest.TestCase):
    """Step 3: radial constant-pressure grouting must stall at I_max of
    the model's own radial stall solution."""

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


# --- Step 2b configuration ---------------------------------------------------
# Variable-saturation fill on a dry pile (1D): S starts at 0, the inlet
# supplies S=1 slurry, and the fill closure (penalized air-pressure cells +
# explicit face-inflow fill) advances a sharp S front. The reference is the
# 1D analytic solution of this model's own flow law -- a relative
# penetration / relative time curve, NOT a single terminal stall point.
# It is the same type of benchmark as the Gustafson et al. parallel-plate
# solution (same stall-length form: L_max = p0/lambda vs I_max =
# dp*b/(2*tau0), r_eff in the place of the full aperture b), but it is not
# their time curve: the flow laws differ, so t(x_f) differs.
#
# Analytic 1D fill (quasi-steady volume balance, S=1 behind a sharp front):
#   uniform flux behind front  q = (k/mu)*(p0/x_f - lambda)
#   volume balance             n dx_f/dt = q
#   =>  t(x_f) = (n/(M*lambda)) * [ L_max*ln(L_max/(L_max - x_f)) - x_f ],
#       M = k/mu,  L_max = p0/lambda.
# The log term diverges as x_f -> L_max, so the front reaches L_max only as
# t -> infinity: at any FINITE time the analytic front is strictly below
# L_max. The test therefore checks that the numeric front TRACKS this
# analytic time-distance curve over the injection period (the benchmark),
# instead of asserting a terminal stall position (which is an asymptotic
# limit, not reachable in finite steps).
#
# Known open issue (round-5 solver work, documented in saturation_design.md):
# near the yield margin the truncation max(0,1-lambda/|grad|) has diverging
# sensitivity and the Picard iteration does not converge (hits the cap with a
# non-decaying residual). This leaks a slow unphysical creep that, integrated
# over very long times, pushes the front past L_max (~+13% at t ~ 36*L_max/
# v_char). It is negligible over the injection period sampled here (< 3% to
# x_f ~ 0.67*L_max) but is why this test tracks the curve at moderate time
# rather than asserting a hard terminal stall.
#
# mu only sets the time scale (L_max has no mu). NX = 50 (coarser than step 2)
# keeps the long march affordable; the front stays sharp (1-2 cells) and the
# quasi-steady profile is mesh-converged for this smooth problem.
S2B_MU = 0.005
S2B_NX = 50
S2B_K = S2_K
S2B_PHI = S2_PHI
S2B_M0 = S2B_K / S2B_MU
S2B_LMAX = S2_LMAX
S2B_LAMBDA = S2_LAMBDA
# Analytic fill time coefficient n/(M*lambda) [s].
S2B_TIME_COEF = S2B_PHI / (S2B_M0 * S2B_LAMBDA)
# March until the front passes this fraction of L_max. Kept to the injection
# period (before the slow yield-margin phase) so the unit suite stays fast
# (~550 steps). The round-5 convergence / long-time-creep study (marching to
# ~36 characteristic times) lives in outputs/scan_mreg.py, not the suite:
# with the default picard_relaxation_fill=0.15 the long-time front drift is
# ~+2.4% of L_max vs ~+11% at omega=0.5.
S2B_TARGET_FRAC = 0.65
S2B_MAX_STEPS = 1500

_STEP2B_CACHE = {}


def _fill_time(x_f):
    """Analytic fill time to reach front position x_f (< L_max) [s]."""
    if x_f <= 0.0:
        return 0.0
    x_f = min(x_f, S2B_LMAX * (1.0 - 1.0e-9))
    return S2B_TIME_COEF * (
        S2B_LMAX * math.log(S2B_LMAX / (S2B_LMAX - x_f)) - x_f
    )


def _fill_front(t):
    """Invert _fill_time: analytic front position at time t (bisection)."""
    if t <= 0.0:
        return 0.0
    lo, hi = 0.0, S2B_LMAX * (1.0 - 1.0e-12)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _fill_time(mid) < t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _build_step2b_state():
    dx = S2_L / float(S2B_NX)
    mesh, x, y, fx, fy = build_mesh(nx=S2B_NX, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}

    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "porous_bingham"
    params["porous_bingham_activation"] = "threshold"
    params["gravity_y"] = 0.0
    params["yield_stress"] = S2_TAU0
    params["plastic_viscosity"] = S2B_MU
    params["inlet_pressure_core_value"] = S2_P0
    params["enable_saturation_transport"] = True
    # Validated fill configuration (round-4 experiments): chained Picard +
    # symmetric relaxation + yield latch + dt_p ON. The latch kills the
    # head-refill/tail-drain jitter pumping that otherwise creeps the front,
    # and dt_p keeps steps small enough that the step-end gradients settle so
    # the latch can engage (dt_p OFF overran badly even with the latch).
    params["dt_cfl"] = 0.9
    state.update(initialize_slurry_variables(mesh, params=params))

    state["permeability"].setValue(S2B_K)
    state["porosity"].setValue(S2B_PHI)
    state["mobility_structural"].setValue(S2B_M0)
    state["mobility_effective"].setValue(S2B_M0)
    state["pressure"].constrain(S2_P0, mesh.facesLeft)
    state["open_pressure_faces"] = mesh.facesLeft
    # Dry initial pile: the fill transport drives the front.
    state["saturation"].setValue(0.0)
    return state


def _saturation_front(state):
    """Interpolated x where S crosses 0.5 (sharp fill front)."""
    x = np.asarray(state["x"], dtype=float)
    s = np.asarray(state["saturation"].value, dtype=float)
    order = np.argsort(x)
    xs, ss = x[order], s[order]
    above = np.where(ss >= 0.5)[0]
    if above.size == 0:
        return 0.0
    i = int(above[-1])
    if i == len(xs) - 1:
        return float(xs[-1])
    if ss[i] <= ss[i + 1]:
        return float(xs[i])
    return float(xs[i] + (ss[i] - 0.5) / (ss[i] - ss[i + 1]) * (xs[i + 1] - xs[i]))


def _run_step2b_march():
    if "state" not in _STEP2B_CACHE:
        state = _build_step2b_state()
        t = 0.0
        # (time, numeric_front) trajectory for curve-tracking checks.
        trajectory = []
        v_in_history = []
        for _ in range(S2B_MAX_STEPS):
            dt = solve_transport_step(state, dt_cap=0.5)
            t += float(dt)
            front = _saturation_front(state)
            trajectory.append((t, front))
            v_in_history.append(float(state["injected_volume_total"]))
            if front >= S2B_TARGET_FRAC * S2B_LMAX:
                break
        _STEP2B_CACHE["state"] = state
        _STEP2B_CACHE["trajectory"] = trajectory
        _STEP2B_CACHE["v_in_history"] = v_in_history
    return (
        _STEP2B_CACHE["state"],
        _STEP2B_CACHE["trajectory"],
        _STEP2B_CACHE["v_in_history"],
    )


class TestStep2bSaturationFront(unittest.TestCase):
    """Step 2b: the dry-pile S front must track the analytic
    penetration-vs-time curve of the model's own 1D flow law over the
    injection period."""

    def test_front_tracks_analytic_curve(self):
        _, trajectory, _ = _run_step2b_march()
        # Compare the numeric front to the analytic front at the SAME time, at
        # several samples spanning the tracked range (skip the first 20% of
        # the march: the dt ramp / first-cell fill is a startup transient).
        start = max(int(0.2 * len(trajectory)), 1)
        worst = 0.0
        worst_msg = ""
        for t, front in trajectory[start:]:
            analytic = _fill_front(t)
            err = abs(front - analytic) / S2B_LMAX
            if err > worst:
                worst = err
                worst_msg = ("t=%.3f s: numeric front %.4f vs analytic %.4f "
                             "(err %.3f of L_max)" % (t, front, analytic, err))
        self.assertLess(
            worst,
            0.05,
            "front departs from the analytic curve: %s" % worst_msg,
        )

    def test_injected_volume_monotone(self):
        _, _, v_in = _run_step2b_march()
        v = np.asarray(v_in, dtype=float)
        diffs = np.diff(v)
        self.assertTrue(
            np.all(diffs >= -1.0e-12 * max(abs(v[-1]), 1.0)),
            "V_in(t) decreased during the fill",
        )


class TestConservationLedger(unittest.TestCase):
    """Global mass balance of the fill transport (design doc section 5)."""

    def test_relative_drift_below_tolerance(self):
        state, _, _ = _run_step2b_march()
        rep = conservation_report(state)
        self.assertLess(
            rep["drift_rel"],
            1.0e-5,
            "ledger drift %.3e (V_in=%.6e store=%.6e comp=%.6e clip=%.3e)"
            % (rep["drift_rel"], rep["v_in"], rep["v_store"],
               rep["v_comp"], rep["v_clip"]),
        )

    def test_clipped_volume_is_zero(self):
        state, _, _ = _run_step2b_march()
        rep = conservation_report(state)
        self.assertLessEqual(
            abs(rep["v_clip"]),
            1.0e-10 * max(abs(rep["v_in"]), 1.0e-30),
            "fill clipping engaged: clip=%.3e vs V_in=%.6e"
            % (rep["v_clip"], rep["v_in"]),
        )


class TestSaturationDegeneracy(unittest.TestCase):
    """Design doc section 6: with S = 1 everywhere the saturation transport
    must reproduce the plain fully-saturated solve EXACTLY (same pressure
    field arithmetic: no filling cells -> no penalty, k_r = 1)."""

    def _march(self, enable_saturation):
        state = _build_step2_state()
        params = state["slurry_parameters"]
        if enable_saturation:
            params["enable_saturation_transport"] = True
            # dt_p / lagged-CFL must not alter dt: pin dt to the cap so both
            # marches take identical steps.
            params["enable_front_dt_constraint"] = False
            state["saturation"].setValue(1.0)
        for _ in range(10):
            if enable_saturation:
                solve_transport_step(state, dt_cap=S2_DT)
            else:
                solve_pressure_step(state, dt=S2_DT)
        return np.asarray(state["pressure"].value, dtype=float)

    def test_s_equal_one_is_bit_identical_to_plain_solve(self):
        p_plain = self._march(False)
        p_saturated = self._march(True)
        max_diff = float(np.max(np.abs(p_saturated - p_plain)))
        self.assertEqual(
            max_diff,
            0.0,
            "S=1 transport deviates from the plain solve: max |dp| = %g"
            % max_diff,
        )


# --- Step 4 configuration ---------------------------------------------------
# Two sub-cases, deliberately separated:
#
# 4a (smooth solution -> verifies the base discretization is 2nd order). The
#    step-1 LINEAR exact profile cannot show an order: a 2nd-order FV scheme
#    reproduces a linear field exactly (the leading truncation error ~ p'''' is
#    zero), so its L2 error is round-off, not discretization. We therefore use
#    a CURVED exact solution: heterogeneous mobility M(x) = M0*(1 + a*x/L) with
#    no source gives the steady no-flux-divergence solution
#        p(x) = p0 * [1 - ln(1 + a*x/L) / ln(1 + a)],
#    whose 4th derivative is nonzero, so the L2 error decays at the FV order.
#    This stays on the production linear-mode path (FiPy arithmetic face
#    averaging of the cell mobility).
#
# 4b (Bingham stall -> the engineering accuracy at a non-smooth front). The
#    stalled profile max(0, p0 - lambda*x) has a kink at L_max, and the
#    saturated threshold solve locks in a transient overshoot remnant
#    (round-2). Both effects cap the order well below 2. dt is refined WITH dx
#    (dt prop dx) so the temporal overshoot error vanishes together with the
#    spatial error. The robust convergence metric is the STALL-POSITION error
#    (= the horizontal shift of the kinked front); the pointwise profile L2 is
#    erratic (it is near-zero when L_max aligns with a cell edge and the
#    piecewise-linear profile is reproduced, and jumps when an overshoot
#    remnant survives), so it is reported as a diagnostic but not asserted.
S4A_P0 = 1000.0
S4A_L = 1.0
S4A_M0 = 2.0e-8
S4A_A = 1.0  # mobility varies 2x across the domain


def _s4a_exact(x):
    return S4A_P0 * (1.0 - np.log1p(S4A_A * x / S4A_L) / np.log1p(S4A_A))


def _s4a_l2_error(nx):
    dx = S4A_L / float(nx)
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "linear"
    params["gravity_y"] = 0.0
    state.update(initialize_slurry_variables(mesh, params=params))
    xv = np.asarray(state["x"], dtype=float)
    mob = S4A_M0 * (1.0 + S4A_A * xv / S4A_L)
    state["mobility_structural"].setValue(mob)
    state["mobility_effective"].setValue(mob)
    state["pressure"].constrain(S4A_P0, mesh.facesLeft)
    state["pressure"].constrain(0.0, mesh.facesRight)
    for _ in range(12):
        update_effective_mobility(state)
        _solve_pressure_once(state, dt=1.0e6)  # dt -> steady in one solve
    p = np.asarray(state["pressure"].value, dtype=float)
    return np.sqrt(np.mean((p - _s4a_exact(xv)) ** 2)) / S4A_P0


def _s4b_stall_position(nx, t_end=6.0, dt_ref=0.0025, nx_ref=50.0):
    dx = S2_L / float(nx)
    dt = dt_ref * (nx_ref / float(nx))   # dt proportional to dx
    n_steps = int(round(t_end / dt))
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "porous_bingham"
    params["porous_bingham_activation"] = "threshold"
    params["gravity_y"] = 0.0
    params["yield_stress"] = S2_TAU0
    params["plastic_viscosity"] = S2_MU
    params["inlet_pressure_core_value"] = S2_P0
    state.update(initialize_slurry_variables(mesh, params=params))
    state["permeability"].setValue(S2_K)
    state["porosity"].setValue(S2_PHI)
    state["mobility_structural"].setValue(S2_K / S2_MU)
    state["mobility_effective"].setValue(S2_K / S2_MU)
    state["pressure"].constrain(S2_P0, mesh.facesLeft)
    for _ in range(n_steps):
        solve_pressure_step(state, dt=dt)
    xv = np.asarray(state["x"], dtype=float)
    p = np.asarray(state["pressure"].value, dtype=float)
    order = np.argsort(xv)
    xs, ps = xv[order], p[order]
    dpdx = np.abs(np.diff(ps) / np.diff(xs))
    below = np.where(dpdx < 0.5 * S2_LAMBDA)[0]
    return 0.5 * (xs[below[0]] + xs[below[0] + 1]) if below.size else float(xs[-1])


def _observed_order(errs, refine_ratio=2.0):
    """Mean log-ratio convergence order across successive refinements."""
    orders = [
        math.log(errs[i - 1] / errs[i]) / math.log(refine_ratio)
        for i in range(1, len(errs))
        if errs[i] > 0.0 and errs[i - 1] > 0.0
    ]
    return orders, (sum(orders) / len(orders) if orders else 0.0)


class TestStep4aSmoothConvergence(unittest.TestCase):
    """Step 4a: heterogeneous-k linear Darcy with a curved exact solution
    must converge at the ~2nd-order FV rate."""

    def test_l2_order_near_two(self):
        Ns = [20, 40, 80, 160]
        errs = [_s4a_l2_error(n) for n in Ns]
        orders, mean_order = _observed_order(errs)
        self.assertGreater(
            mean_order,
            1.8,
            "smooth L2 order %.3f below 1.8 (errs=%s, orders=%s)"
            % (mean_order, ["%.2e" % e for e in errs],
               ["%.3f" % o for o in orders]),
        )


class TestStep4bBinghamFrontConvergence(unittest.TestCase):
    """Step 4b: the Bingham stall position must converge under refinement
    (dt prop dx). The order is ~1 (not 2): the stalled profile has a kink at
    L_max and the threshold solve locks a transient overshoot remnant, both
    inherently first-order at the front. This is a property of the sharp
    Bingham front, not an implementation defect (cf. the 2nd order of the
    smooth 4a case on the same solver)."""

    def test_stall_position_converges(self):
        Ns = [25, 50, 100]
        stalls = [_s4b_stall_position(n) for n in Ns]
        errs = [abs(s - S2_LMAX) for s in stalls]
        orders, mean_order = _observed_order(errs)
        self.assertTrue(
            all(errs[i] <= errs[i - 1] + 1.0e-12 for i in range(1, len(errs))),
            "stall-position error not monotone under refinement: %s"
            % ["%.4f" % e for e in errs],
        )
        self.assertGreater(
            mean_order,
            0.8,
            "stall-position order %.3f below 0.8 (stalls=%s, errs=%s)"
            % (mean_order, ["%.4f" % s for s in stalls],
               ["%.4f" % e for e in errs]),
        )



# --- Step 7: between-stage closure conservation ---------------------------
# Staged advancing grouting: each stage injects until it stalls, the grout
# sets, and the next stage starts deeper into a medium the previous stage has
# partly cemented. The closure that hardens a stage is the only place porosity
# changes after initialization (see stage_update.py); the skeleton stays rigid.
#
# Fixture: 1D strip, L=6 m in 24 cells, uniform phi0=0.30, calibrated-power
# permeability, interior Dirichlet source advancing 2.5 m per stage -- which
# must EXCEED the local L_max ~2.1 m, or the next stage starts inside its
# predecessor's cemented zone and cannot inject (see the S7_SOURCE_START note).
# Parameters are the v0.6 set (tau0=30 Pa, mu_p=0.15 Pa.s, A=1.25e-7) with a
# reduced p0 so L_max ~2.1 m fits inside the strip several times over.
S7_L = 7.5
S7_NX = 30
S7_PHI0 = 0.30
S7_A = 1.25e-7
S7_TAU0 = 30.0
S7_MU = 0.15
S7_P0 = 3.0e5
S7_STAGES = 3
S7_STEPS_PER_STAGE = 25
S7_DT_CAP = 1.0
# Hole depth, in cells, after each drilling pass. The grouting pipe is
# perforated along its WHOLE length, which is standard practice, so the
# source at stage k is the entire hole [0, S7_HOLE_END[k]) -- it EXTENDS,
# it does not move. The field sequence is drill -> grout -> set -> ream
# through the set grout and drill deeper -> grout again (the reference
# project logged 493 m of reaming against 500 m of drilling, and three
# passes at 7 / 10 / 17.5 m).
#
# What must exceed the local L_max ~2.1 m is therefore the DRILLING
# INCREMENT, not the source position: if the new hole section stays inside
# the zone the previous stage cemented, the pass has nowhere to deliver.
# 10 cells = 2.5 m per pass clears that. The reference project's own
# increments (3 m and 7.5 m against a 12 m design spread) do not.
S7_HOLE_END = (2, 12, 22)

_STEP7_CACHE = {}


def _s7_lambda(phi, a=S7_A, tau0=S7_TAU0):
    """lambda = 2*tau0/r_eff with r_eff = sqrt(8k/phi), k = A phi^3/(1-phi)^2.

    Substituting k gives r_eff = sqrt(8A)*phi/(1-phi), so lambda is a strictly
    DECREASING function of phi: cementing a cell (phi down) raises its start-up
    gradient, which is the whole point of the staged closure.
    """
    phi = np.asarray(phi, dtype=float)
    r_eff = math.sqrt(8.0 * a) * phi / (1.0 - phi)
    return 2.0 * tau0 / r_eff


def _s7_source_cells(stage):
    """Every cell the hole occupies after pass `stage` -- a perforated pipe
    bleeds along all of it."""
    return np.arange(0, S7_HOLE_END[stage], dtype=int)


def _s7_install_source(state, stage):
    """Stage-0 source install; later stages go through apply_stage_closure."""
    from fipy import CellVariable
    cells = _s7_source_cells(stage)
    mask = np.zeros(S7_NX, dtype=float)
    mask[cells] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=state["mesh"], value=mask)
    state["interior_dirichlet_value"] = S7_P0
    s0 = np.zeros(S7_NX, dtype=float)
    s0[cells] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, S7_P0, 0.0))


def _build_step7_state():
    dx = S7_L / float(S7_NX)
    mesh, x, y, fx, fy = build_mesh(nx=S7_NX, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}

    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "porous_bingham"
    params["porous_bingham_activation"] = "threshold"
    params["yield_truncation_mode"] = "papanastasiou"
    params["pressure_coeff_form"] = "face"
    params["enable_saturation_transport"] = True
    params["porosity_to_permeability_formula"] = "calibrated_power"
    params["calibrated_permeability_coefficient"] = S7_A
    params["yield_stress"] = S7_TAU0
    params["plastic_viscosity"] = S7_MU
    params["gravity_y"] = 0.0
    params["reference_storage"] = 1.0e-10
    params["picard_transient_mode"] = "backward_euler"
    params["enable_front_dt_constraint"] = False
    params["dt_cfl"] = 0.5
    params["picard_relaxation_fill"] = 0.15
    params["picard_max_iters"] = 12
    params["picard_tol"] = 1.0e-3
    state.update(initialize_slurry_variables(mesh, params=params))

    phi = np.zeros(S7_NX, dtype=float) + S7_PHI0
    k = porosity_to_permeability(phi, params)
    state["porosity"].setValue(phi)
    state["permeability"].setValue(k)
    for key in ("mobility_structural", "mobility_effective",
                "mobility", "intrinsic_mobility"):
        state[key].setValue(k / S7_MU)

    _s7_install_source(state, 0)
    return state


def _run_step7_stages():
    """March S7_STAGES stages, closing each one, recording everything."""
    if "records" in _STEP7_CACHE:
        return _STEP7_CACHE["records"]

    state = _build_step7_state()
    initialize_stage_ledger(state)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    records = []

    for stage in range(S7_STAGES):
        src = _s7_source_cells(stage)
        phi_at_stage_start = np.array(state["porosity"].value, copy=True)
        # Split the perforated pipe into the part earlier passes already
        # cemented and the part this pass just drilled. A cemented cell sits
        # at the porosity floor; virgin ground is still at phi0.
        floor = get_stage_porosity_floor(state["slurry_parameters"])
        src_spent = src[phi_at_stage_start[src] <= floor * (1.0 + 1.0e-9)]
        src_fresh = np.setdiff1d(src, src_spent)
        phi_during = []
        v_in_src = 0.0
        v_in_spent = 0.0
        v_in_fresh = 0.0

        for _ in range(S7_STEPS_PER_STAGE):
            dt = solve_transport_step(state, dt_cap=S7_DT_CAP)
            div_q = np.asarray(state["last_div_q"], dtype=float)
            v_in_src += float(np.sum(div_q[src] * vols[src])) * float(dt)
            if src_spent.size:
                v_in_spent += float(
                    np.sum(div_q[src_spent] * vols[src_spent])) * float(dt)
            if src_fresh.size:
                v_in_fresh += float(
                    np.sum(div_q[src_fresh] * vols[src_fresh])) * float(dt)
            # Source cells stay full while injecting, as in the cases.
            s = np.array(state["saturation"].value, copy=True)
            s[src] = 1.0
            state["saturation"].setValue(s)
            phi_during.append(np.array(state["porosity"].value, copy=True))

        phi_before = np.array(state["porosity"].value, copy=True)
        s_before = np.array(state["saturation"].value, copy=True)
        report = stage_fill_report(state)
        flow = conservation_report(state)

        next_src = _s7_source_cells(stage + 1) if stage + 1 < S7_STAGES else None
        ledger = apply_stage_closure(state, source_mask=next_src)
        phi_after = np.array(state["porosity"].value, copy=True)

        records.append({
            "stage": stage,
            "phi_at_stage_start": phi_at_stage_start,
            "phi_during": phi_during,
            "phi_before": phi_before,
            "phi_after": phi_after,
            "s_before": s_before,
            "report": dict(state["stage_history"][-1]),
            "report_pre": dict(report),
            "flow": dict(flow),
            "ledger": dict(ledger),
            "v_in_src": v_in_src,
            "v_in_spent": v_in_spent,
            "v_in_fresh": v_in_fresh,
            "n_src_spent": int(src_spent.size),
            "n_src_fresh": int(src_fresh.size),
            "stage_ledger_report": stage_ledger_report(state),
        })

    _STEP7_CACHE["records"] = records
    _STEP7_CACHE["vols"] = vols
    _STEP7_CACHE["state"] = state
    return records


class TestStep7StageClosureConservation(unittest.TestCase):
    """Step 7: the between-stage closure must conserve grout volume exactly
    and hand the next stage a strictly denser, strictly more resistant
    medium."""

    def test_closure_removes_exactly_the_hardened_pore_volume(self):
        """sum (n_k - n_{k+1})V + floor_clip == sum n_k S_k V, to machine
        precision. This is the closure rule itself: n_{k+1} = n_k(1-S_k)
        means the porosity removed IS the grout that was sitting there."""
        records = _run_step7_stages()
        vols = _STEP7_CACHE["vols"]
        for rec in records:
            removed = float(np.sum((rec["phi_before"] - rec["phi_after"]) * vols))
            occupied = rec["report"]["occupied_this_stage"]
            floor_clip = rec["report"]["floor_clipped_step"]
            self.assertAlmostEqual(
                removed + floor_clip, occupied,
                delta=1.0e-12 * max(abs(occupied), 1.0),
                msg="stage %d: removed %.17g + clip %.17g != occupied %.17g"
                    % (rec["stage"], removed, floor_clip, occupied),
            )

    def test_cumulative_basis_is_exact_across_stages(self):
        """Per-stage hardened volumes must sum to the cumulative figure the
        ledger reports against the invariant stage-0 basis, and that basis
        must never move -- otherwise fill ratios from different stages are
        not comparable."""
        records = _run_step7_stages()
        per_stage_sum = sum(
            r["report"]["occupied_this_stage"] - r["report"]["floor_clipped_step"]
            for r in records
        )
        final = records[-1]["ledger"]["occupied_cumulative"]
        self.assertAlmostEqual(
            per_stage_sum, final,
            delta=1.0e-12 * max(abs(final), 1.0),
            msg="per-stage sum %.17g != cumulative %.17g" % (per_stage_sum, final),
        )
        capacities = [r["stage_ledger_report"]["pore_capacity_initial"]
                      for r in records]
        for cap in capacities[1:]:
            self.assertEqual(cap, capacities[0])

    def test_capacity_accounting_closes(self):
        """occupied_cumulative + remaining_capacity == pore_capacity_initial
        after every stage. Nothing appears, nothing vanishes."""
        records = _run_step7_stages()
        for rec in records:
            rep = rec["stage_ledger_report"]
            total = rep["occupied_cumulative"] + rep["remaining_capacity"]
            self.assertAlmostEqual(
                total, rep["pore_capacity_initial"],
                delta=1.0e-12 * rep["pore_capacity_initial"],
                msg="stage %d: %.17g != %.17g"
                    % (rec["stage"], total, rep["pore_capacity_initial"]),
            )
            self.assertAlmostEqual(
                rep["fill_ratio_cumulative"] + rep["residual_ratio"], 1.0,
                delta=1.0e-12,
            )

    def test_porosity_is_frozen_within_a_stage(self):
        """The error the design explicitly warns against: reducing n as S
        rises WITHIN a stage would seal the very channel carrying the grout.
        Porosity must be bit-identical through every step of a stage and
        change only at the closure."""
        records = _run_step7_stages()
        for rec in records:
            base = rec["phi_at_stage_start"]
            for i, phi in enumerate(rec["phi_during"]):
                self.assertTrue(
                    np.array_equal(phi, base),
                    msg="stage %d step %d: porosity moved mid-stage"
                        % (rec["stage"], i),
                )

    def test_closure_rule_applies_exactly(self):
        """n_{k+1} == max(n_k*(1-S_k), floor) on every cell, bit for bit, and
        cells the stage never wetted are untouched."""
        records = _run_step7_stages()
        params = _STEP7_CACHE["state"]["slurry_parameters"]
        floor = get_stage_porosity_floor(params)
        for rec in records:
            before, after = rec["phi_before"], rec["phi_after"]
            n_raw = before * (1.0 - rec["s_before"])
            self.assertTrue(
                np.array_equal(after, np.maximum(n_raw, floor)),
                msg="stage %d: closure is not n_{k+1}=max(n_k(1-S_k), floor)"
                    % rec["stage"],
            )
            self.assertTrue(np.all(after <= before + 1.0e-15))
            dry = rec["s_before"] <= 1.0e-12
            if np.any(dry):
                self.assertTrue(np.array_equal(after[dry], before[dry]))

    def test_filled_cells_cement_to_the_floor(self):
        """The fill closure is a SHARP front: a cell is either still empty or
        essentially full (S >= saturation_active_threshold), so partially
        filled cells barely exist at a stage end. Every filled cell therefore
        closes to n_k*(1-S_k) ~ 0 and lands on the porosity floor.

        This makes stage_porosity_floor the single parameter that fixes what
        set grout looks like: it sets the residual porosity, and the law
        k = A n^3/(1-n)^2 turns that into the cemented permeability
        (1e-3 -> ~1.25e-16 m^2, the right order for hardened cement).
        permeability_clip_min must stay below that, or it takes the decision
        over -- see test_permeability_clip_never_binds_in_the_operating_range.
        """
        records = _run_step7_stages()
        params = _STEP7_CACHE["state"]["slurry_parameters"]
        floor = get_stage_porosity_floor(params)
        threshold = float(params.get("saturation_active_threshold", 1.0 - 1.0e-3))
        for rec in records:
            filled = rec["s_before"] >= threshold
            self.assertTrue(
                np.any(filled), "stage %d filled nothing" % rec["stage"]
            )
            after = rec["phi_after"]
            self.assertTrue(np.all(after[filled] == floor))

            # Compare against VIRGIN ground only: cells left untouched by a
            # later stage include ground its predecessors already cemented,
            # which sits at the same floor.
            virgin = np.isclose(after, S7_PHI0)
            if np.any(virgin):
                k_after = porosity_to_permeability(after, params)
                self.assertTrue(
                    np.all(k_after[filled] < k_after[virgin].min()),
                    msg="stage %d: cemented ground is not less permeable "
                        "than virgin ground" % rec["stage"],
                )

    def test_partial_fill_densifies_monotonically(self):
        """The closure's monotone content, on a prescribed saturation ladder
        rather than a marched one.

        A sharp front rarely leaves partially filled cells, so the marched
        fixture cannot exercise 0 < S < 1. Here S is set directly, which
        isolates the constitutive chain: more grout in a cell -> lower n ->
        lower k -> higher start-up gradient lambda -> shorter local L_max for
        whatever stage comes next. The rungs stay light enough (S <= 0.4) to
        keep k off its clip; test_permeability_clip_caps_the_cemented_contrast
        covers what happens past that.
        """
        state = _build_step7_state()
        initialize_stage_ledger(state)
        params = state["slurry_parameters"]
        floor = get_stage_porosity_floor(params)
        k_clip_min = float(params.get("permeability_clip_min", 1.0e-18))

        ladder = np.array([0.0, 0.10, 0.20, 0.30, 0.40])
        s = np.zeros(S7_NX, dtype=float)
        s[:ladder.size] = ladder
        state["saturation"].setValue(s)

        before = np.array(state["porosity"].value, copy=True)
        apply_stage_closure(state, reset_pressure=False)
        after = np.array(state["porosity"].value, copy=True)

        rung = after[:ladder.size]
        expected = np.maximum(before[:ladder.size] * (1.0 - ladder), floor)
        self.assertTrue(np.array_equal(rung, expected))
        self.assertTrue(np.all(rung > floor))
        self.assertTrue(np.all(np.diff(rung) < 0.0))

        k_rung = porosity_to_permeability(rung, params)
        self.assertTrue(
            np.all(k_rung > k_clip_min),
            msg="ladder reached the permeability clip; k=%s" % (k_rung,),
        )
        self.assertTrue(np.all(np.diff(k_rung) < 0.0))
        self.assertTrue(np.all(np.diff(_s7_lambda(rung)) > 0.0))

    def test_cemented_ground_resists_start_up_far_more_than_virgin(self):
        """The direction that matters for every barrier case.

        lambda = 2*tau0/sqrt(8k/n) puts k over n, so a closure that collapses
        n must let k collapse with it. If a permeability clip holds k up, the
        ratio 8k/n RISES and the model concludes the cemented pores got
        wider -- cemented ground then yields EASIER than the rock around it,
        which is the opposite of a seal.

        Before 2026-09-21 permeability_clip_min was 1e-10 and did exactly
        that: measured lambda_cemented/lambda_virgin = 0.48. This test pins
        the corrected direction.
        """
        params = _build_step7_state()["slurry_parameters"]
        floor = get_stage_porosity_floor(params)

        k_virgin = float(porosity_to_permeability(np.array([S7_PHI0]), params)[0])
        k_cemented = float(porosity_to_permeability(np.array([floor]), params)[0])

        # Permeability must fall by orders, not by a factor.
        self.assertGreater(k_virgin / k_cemented, 1.0e6)

        # Start-up gradient must RISE, and by a lot. lambda from the STORED
        # permeability, which is what the solver actually rebuilds it from.
        lam_virgin = 2.0 * S7_TAU0 / math.sqrt(8.0 * k_virgin / S7_PHI0)
        lam_cemented = 2.0 * S7_TAU0 / math.sqrt(8.0 * k_cemented / floor)
        self.assertGreater(
            lam_cemented / lam_virgin, 100.0,
            msg="cemented lambda is only %.4g x virgin (%.4g vs %.4g Pa/m); "
                "a permeability clip is probably holding k up"
                % (lam_cemented / lam_virgin, lam_cemented, lam_virgin),
        )

    def test_permeability_clip_never_binds_in_the_operating_range(self):
        """permeability_clip_min must stay a numerical guard, not a model.

        It has to sit below what the law produces anywhere the solver can
        go -- from the loosest virgin ground down to a fully cemented cell
        at stage_porosity_floor. The moment it binds it starts deciding
        physics, silently, as it did until 2026-09-21.
        """
        params = _build_step7_state()["slurry_parameters"]
        floor = get_stage_porosity_floor(params)
        k_clip_min = float(params["permeability_clip_min"])

        # The lowest permeability the closure can ever hand the solver.
        k_law_at_floor = S7_A * floor ** 3 / (1.0 - floor) ** 2
        self.assertGreater(
            k_law_at_floor, k_clip_min,
            msg="the clip (%.3g) binds at the porosity floor, where the law "
                "gives %.3g m^2; it is deciding the cemented permeability"
                % (k_clip_min, k_law_at_floor),
        )
        # And the clipped result equals the raw law there, bit for bit.
        self.assertEqual(
            float(porosity_to_permeability(np.array([floor]), params)[0]),
            k_law_at_floor,
        )


    def test_cemented_pipe_length_stops_taking_grout(self):
        """Self-regulation: the source does not need steering.

        A perforated pipe keeps bleeding along its whole length, so after the
        second pass most of the source sits in ground earlier passes already
        cemented. Nothing has to close those perforations -- the closure has
        dropped that ground's permeability by seven orders and raised its
        start-up gradient by ~430x, so grout simply stops going there and the
        pass delivers through the freshly drilled section instead.

        This is why apply_stage_closure accepts an EXTENDING source mask and
        does not need the source to be moved deeper by hand.
        """
        records = _run_step7_stages()
        for rec in records[1:]:
            self.assertGreater(
                rec["n_src_spent"], 0,
                "stage %d has no already-cemented pipe length to test"
                    % rec["stage"],
            )
            self.assertGreater(rec["n_src_fresh"], 0)
            total = rec["v_in_spent"] + rec["v_in_fresh"]
            self.assertGreater(total, 0.0)
            # Not merely small: measured at 1e-34 and 1e-39 of the fresh
            # section's delivery, i.e. float noise. The cemented length is a
            # complete barrier, not a partially blocked one, so the bound is
            # set to match rather than to a forgiving percentage.
            share = abs(rec["v_in_spent"]) / total
            self.assertLess(
                share, 1.0e-6,
                msg="stage %d: %.3g of delivery still came from the "
                    "cemented pipe length (%d cells spent, %d fresh)"
                    % (rec["stage"], share,
                       rec["n_src_spent"], rec["n_src_fresh"]),
            )

    def test_remaining_capacity_decreases_monotonically(self):
        """Successive stages can only consume capacity, never restore it."""
        records = _run_step7_stages()
        remaining = [r["stage_ledger_report"]["remaining_capacity"] for r in records]
        filled = [r["stage_ledger_report"]["fill_ratio_cumulative"] for r in records]
        for i in range(1, len(remaining)):
            self.assertLess(remaining[i], remaining[i - 1])
            self.assertGreater(filled[i], filled[i - 1])
        self.assertGreater(filled[0], 0.0)
        self.assertLess(filled[-1], 1.0)

    def test_hardened_volume_matches_what_the_flow_delivered(self):
        """The physics tie: the pore volume that hardens in a stage must be
        the grout the flow solve actually pushed out of the source, plus the
        source cells' own prescribed fill. Toleranced, not exact -- it
        inherits the flow ledger's Picard transient slip, the same ~1e-3
        relative drift the application cases report."""
        records = _run_step7_stages()
        for rec in records:
            delivered = rec["v_in_src"] + rec["report"]["v_prefill"]
            occupied = rec["report"]["occupied_this_stage"]
            rel = abs(occupied - delivered) / max(abs(occupied), 1.0e-30)
            self.assertLess(
                rel, 5.0e-2,
                msg="stage %d: occupied %.6g vs delivered %.6g (rel %.3g)"
                    % (rec["stage"], occupied, delivered, rel),
            )


if __name__ == "__main__":
    unittest.main()
