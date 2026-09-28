# -*- coding: utf-8 -*-
"""
APPLICATION benchmark (NOT a verification-ladder test): inclined-hole grouting
into an ARTIFICIAL gradient porosity field ("dense bottom, loose top"), to study
how vertical heterogeneity drives asymmetric (top-long / bottom-flat) spread.

Builds on outputs/inclined_hole_grouting.py (same engine, same hole, same
slurry, same time-stepping guards) but replaces the CSV-binned (near-uniform
phi~0.18) field with an analytic vertical gradient:

    phi(y) = phi_bottom + (phi_top - phi_bottom) * (y / H)

Runs two cases for comparison:
  (A) gradient  : phi_bottom=0.12 (dense) -> phi_top=0.30 (loose)
  (B) uniform   : phi = 0.18 everywhere  (the previous benchmark's mean)

Permeability: calibrated_power k = A phi^3/(1-phi)^2, A from grout_material_v06().
Injection : inclined hole, mouth (-15,1), dip 35 deg, length 17 m, source cells
            pinned at p0=5 MPa, held S=1 (interior Dirichlet penalty).
Slurry    : parameter set v0.6 (see variables.grout_material_v06), gravity ON.

Py2.7 / 3 compatible.
Run: powershell -File scripts\\run_local.ps1 cases\\inclined_hole_gradient.py
Figures are written to outputs/ (untracked artifacts).
"""
from __future__ import print_function
import os
import re
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fipy import CellVariable

from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.coupling.porosity_to_permeability import porosity_to_permeability
from src.models.slurry_transport.variables import (
    grout_material_v06,
    build_placeholder_slurry_parameters,
    initialize_slurry_variables,
)
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport.equations import (
    solve_transport_step,
    conservation_report,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Figures are build artifacts: keep them out of the tracked case directory.
_OUT = os.path.join(os.path.dirname(_HERE), "outputs")
_CKPT_DIR = os.path.join(_OUT, "checkpoints")
if not os.path.isdir(_OUT):
    os.makedirs(_OUT)

# --- domain / mesh ---------------------------------------------------------
# Domain sized for the v0.6 parameter set: L_max ~18-20 m at phi 0.18-0.21
# must fit in every direction from a 17 m hole, so 60 x 60 m at the 2.5 m
# REV floor (576 cells; runs clean under the scipy backend). The hole enters
# from the left wall (the drift is in host rock) at mid-height.
X_MIN, X_MAX = -30.0, 30.0
Y_MIN, Y_MAX = 0.0, 60.0
H = 60.0
CELL = 2.5
PHI_CLIP = (0.05, 0.60)

# --- gradient porosity (tunable, top of script for sweeping) ---------------
PHI_BOTTOM = 0.12     # dense bottom
PHI_TOP = 0.30        # loose top
PHI_UNIFORM = 0.18    # homogeneous control

# --- inclined hole ---------------------------------------------------------
MOUTH = (-30.0, 21.0)
DIP_DEG = 35.0
HOLE_LEN = 17.0
P0 = 5.0e6

# --- slurry (PO 42.5, w/c=0.5): the v0.6 set, see grout_material_v06 -------
_MAT = grout_material_v06()
TAU0 = _MAT["yield_stress"]
MU_P = _MAT["plastic_viscosity"]
RHO = _MAT["slurry_density"]
A_CAL = _MAT["calibrated_permeability_coefficient"]

# --- march control ---------------------------------------------------------
# Stall = the S>=0.5 front (filled-cell count) stops advancing for PATIENCE
# steps. (V_store keeps creeping as partially-filled interior cells densify --
# the GS log-singularity tail -- but the front envelope we measure/plot is set
# once no new cell crosses S=0.5; count-plateau captures that without grinding
# through the asymptotic tail.)
#
# Default stop rule since 2026-09-27: the physical rate rule of
# src/analysis/stop_rule.py (window-mean injection rate below 0.1 Q_ref, window
# 0.1 t). STALL_PATIENCE is the legacy rule, used only with STOP_RULE=legacy.
DT_CAP = 5.0
MAX_STEPS = 20000     # never the binding stop under the rate rule (it needs 1.4k-6.3k steps)
STALL_PATIENCE = 60   # legacy rule only
# This machine crashes at random under sustained load (CLAUDE.md), so a run is
# checkpointed and resumes bit-identically; 0 disables it. See --help.
CKPT_EVERY = 200


# ===========================================================================
def kc_like(phi):
    return A_CAL * phi ** 3 / (1.0 - phi) ** 2


def lam_of(phi):
    k = kc_like(phi)
    return 2.0 * TAU0 / math.sqrt(8.0 * k / phi)


def phi_field(kind, my):
    if kind == "gradient":
        phi = PHI_BOTTOM + (PHI_TOP - PHI_BOTTOM) * (my / H)
    else:
        phi = np.zeros_like(my) + PHI_UNIFORM
    return np.clip(phi, PHI_CLIP[0], PHI_CLIP[1])


def hole_geometry():
    a = math.radians(DIP_DEG)
    end = (MOUTH[0] + HOLE_LEN * math.cos(a), MOUTH[1] + HOLE_LEN * math.sin(a))
    d = (math.cos(a), math.sin(a))            # along-hole unit vector
    n = (math.sin(a), -math.cos(a))           # normal, +n points down-right
    return end, d, n


def _cell_index_map(mesh_x, mesh_y, nx, ny):
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    idx = {}
    for n in range(mesh_x.size):
        ix = int(round((mesh_x[n] - X_MIN) / dx - 0.5))
        iy = int(round((mesh_y[n] - Y_MIN) / dy - 0.5))
        idx[(ix, iy)] = n
    return idx, dx, dy


def hole_cells(mesh_x, mesh_y, nx, ny):
    end, _, _ = hole_geometry()
    idx, dx, dy = _cell_index_map(mesh_x, mesh_y, nx, ny)
    cells = set()
    nsamp = 2000
    for s in range(nsamp + 1):
        t = float(s) / nsamp
        x = MOUTH[0] + t * (end[0] - MOUTH[0])
        y = MOUTH[1] + t * (end[1] - MOUTH[1])
        ix = int(math.floor((x - X_MIN) / dx))
        iy = int(math.floor((y - Y_MIN) / dy))
        if (ix, iy) in idx:
            cells.add(idx[(ix, iy)])
    return sorted(cells)


def to_grid(arr, mesh_x, mesh_y, nx, ny):
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    g = np.zeros((ny, nx), dtype=float)
    for n in range(mesh_x.size):
        ix = int(round((mesh_x[n] - X_MIN) / dx - 0.5))
        iy = int(round((mesh_y[n] - Y_MIN) / dy - 0.5))
        g[iy, ix] = arr[n]
    return g


def build_case(kind):
    mesh, x, y, fx, fy = build_mesh_for_domain(X_MIN, X_MAX, Y_MIN, Y_MAX, CELL)
    mx = np.asarray(x, dtype=float)
    my = np.asarray(y, dtype=float)
    nx = int(round((X_MAX - X_MIN) / CELL))
    ny = int(round((Y_MAX - Y_MIN) / CELL))

    phi = phi_field(kind, my)

    params = build_placeholder_slurry_parameters()
    params.update({
        "rheology_model": "porous_bingham",
        "porous_bingham_activation": "threshold",
        "pressure_coeff_form": "face",
        "yield_truncation_mode": "papanastasiou",
        "enable_saturation_transport": True,
        "porosity_to_permeability_formula": "calibrated_power",
        "calibrated_permeability_coefficient": A_CAL,
        "yield_stress": TAU0,
        "plastic_viscosity": MU_P,
        "slurry_density": RHO,
        "gravity_y": -9.81,
        "inlet_pressure_core_value": P0,
        "reference_storage": 1.0e-10,
        "picard_transient_mode": "backward_euler",
        "enable_front_dt_constraint": False,
        "dt_cfl": 0.5,
        "picard_relaxation_fill": 0.15,
        # Application-case Picard budget (looser than the verification tests'
        # 1e-4/20: this is a spread-envelope study, not a stall-precision one).
        "picard_max_iters": 12,
        "picard_tol": 1.0e-3,
    })

    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh, params=params))

    k = porosity_to_permeability(phi, params)
    state["porosity"].setValue(phi)
    state["permeability"].setValue(k)
    state["mobility_structural"].setValue(k / MU_P)
    state["mobility_effective"].setValue(k / MU_P)
    state["mobility"].setValue(k / MU_P)
    state["intrinsic_mobility"].setValue(k / MU_P)

    hcells = hole_cells(mx, my, nx, ny)
    mask = np.zeros(mx.size, dtype=float)
    mask[hcells] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    state["interior_dirichlet_value"] = P0
    s0 = np.zeros(mx.size, dtype=float)
    s0[hcells] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, P0, 0.0))

    return state, mx, my, nx, ny, phi, hcells, k


def front_metrics(state, mx, my, hcells):
    s = np.asarray(state["saturation"].value, dtype=float)
    end, d, n = hole_geometry()
    filled = (s >= 0.5)
    filled[hcells] = False
    if not np.any(filled):
        return None
    fx = mx[filled]
    fy = my[filled]
    perp = (fx - MOUTH[0]) * n[0] + (fy - MOUTH[1]) * n[1]
    return {
        "count": int(np.sum(filled)),
        "x_min": float(np.min(fx)), "x_max": float(np.max(fx)),
        "y_min": float(np.min(fy)), "y_max": float(np.max(fy)),
        "perp_down": float(np.max(perp)),     # down-right (mostly downward)
        "perp_up": float(-np.min(perp)),      # up-left (mostly upward)
    }


def n_ref_for(kind):
    """The stop rule's reference porosity for this case family: the uniform
    porosity, or the gradient field's porosity at the hole's mid-point."""
    if kind == "gradient":
        end, d, n = hole_geometry()
        return float(phi_field("gradient", np.array([0.5 * (MOUTH[1] + end[1])]))[0])
    return float(PHI_UNIFORM)


LAST_REASON = [None]      # stop reason of the latest march(): rate / legacy_stall / max_steps


def march(state, mx, my, hcells, vols, label, keeper=None, n_ref=None):
    phi = np.clip(np.asarray(state["porosity"].value, dtype=float), 1e-6, 1.0)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True
    if n_ref is None:              # fall back to the porosity of the hole's middle cell
        n_ref = float(phi[hcells[len(hcells) // 2]])
    qref = stop_rule.q_ref(n_ref, A_CAL, P0, MU_P)
    t_hist, v_hist = [0.0], [0.0]
    reason = "max_steps"

    t = 0.0
    v_in_src = 0.0
    best_count = -1
    stall = 0
    count = 0
    v_store = 0.0
    step0 = 0
    # s0 is the stage's initial saturation and must survive a resume, or
    # v_store would be measured from the middle of the run.
    s0 = np.array(state["saturation"].value, copy=True)

    resumed = keeper.restore(state) if keeper is not None else {}
    if resumed:
        step0 = int(resumed["step"])
        t = float(resumed["t"])
        v_in_src = float(resumed["v_in_src"])
        best_count = int(resumed["best_count"])
        stall = int(resumed["stall"])
        s0 = np.array(resumed["s0"], copy=True)
        if "t_hist" in resumed:
            t_hist = list(np.atleast_1d(resumed["t_hist"]))
            v_hist = list(np.atleast_1d(resumed["v_hist"]))
        print("  [%s] resuming at step %d (t=%.1f s)" % (label, step0, t))
    else:
        print("  [%s] marching (stop rule: %s) ..." % (label, "legacy" if stop_rule.legacy() else "rate, n_ref %.4f" % n_ref))
    sys.stdout.flush()
    for step in range(step0, MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=DT_CAP)
        t += float(dt)
        div_q = np.asarray(state["last_div_q"], dtype=float)
        v_in_src += float(np.sum(div_q[hmask] * vols[hmask])) * float(dt)
        t_hist.append(t)
        v_hist.append(v_in_src)
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)

        s_arr = np.asarray(state["saturation"].value, dtype=float)
        v_store = float(np.sum(phi * (s_arr - s0) * vols))
        count = int(np.sum((s_arr >= 0.5) & np.logical_not(hmask)))

        # count-plateau: front advanced this step iff a new cell crossed 0.5.
        if count > best_count:
            best_count = count
            stall = 0
        else:
            stall += 1
        if stop_rule.legacy():
            if stall >= STALL_PATIENCE:
                reason = "legacy_stall"
                break
        elif stop_rule.stop_rule_physical(t_hist, v_hist, qref):
            reason = "rate"
            break
        if keeper is not None:
            keeper.maybe_save(state, step=step + 1, t=t, v_in_src=v_in_src,
                              best_count=best_count, stall=stall, s0=s0,
                              t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    if keeper is not None:
        keeper.finish()
    LAST_REASON[0] = reason
    print("    -> %s at step %d, t=%.1f s, filled=%d, Vstore=%.3f, Vin=%.3f"
          % (reason, step, t, count, v_store, v_in_src))
    sys.stdout.flush()
    return t, v_in_src, v_store


def run_one(kind, label, ckpt_every=CKPT_EVERY):
    state, mx, my, nx, ny, phi, hcells, k = build_case(kind)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    keeper = None
    if ckpt_every > 0:
        if not os.path.isdir(_CKPT_DIR):
            os.makedirs(_CKPT_DIR)
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        keeper = Checkpointer(
            os.path.join(_CKPT_DIR, tag), every=ckpt_every,
            fingerprint=state_fingerprint(state, "gradient|%s|%s%s" % (kind, label, stop_rule.rate_tag(n_ref_for(kind)))))
    t, v_in_src, v_store = march(state, mx, my, hcells, vols, label, keeper=keeper, n_ref=n_ref_for(kind))
    rep = conservation_report(state)
    fm = front_metrics(state, mx, my, hcells)
    s_grid = to_grid(np.asarray(state["saturation"].value, dtype=float), mx, my, nx, ny)
    p_grid = to_grid(np.asarray(state["pressure"].value, dtype=float) / 1e6, mx, my, nx, ny)
    phi_grid = to_grid(phi, mx, my, nx, ny)
    return {
        "kind": kind, "label": label, "mx": mx, "my": my, "nx": nx, "ny": ny,
        "phi": phi, "k": k, "hcells": hcells, "t": t,
        "v_in_src": v_in_src, "v_store": v_store, "v_clip": rep["v_clip"],
        "fm": fm, "s_grid": s_grid, "p_grid": p_grid, "phi_grid": phi_grid,
    }


# ===========================================================================
def grid_axes(nx, ny):
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    xc = X_MIN + (np.arange(nx) + 0.5) * dx
    yc = Y_MIN + (np.arange(ny) + 0.5) * dy
    return np.meshgrid(xc, yc)


def draw_hole(ax, hcells, mx, my):
    end, _, _ = hole_geometry()
    ax.plot([MOUTH[0], end[0]], [MOUTH[1], end[1]], "k-", lw=2.0)
    ax.plot([MOUTH[0]], [MOUTH[1]], "ko", ms=6)


def make_comparison_figure(res_a, res_b):
    Xc, Yc = grid_axes(res_a["nx"], res_a["ny"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, res in enumerate((res_a, res_b)):
        # porosity
        ax = axes[row][0]
        cf = ax.contourf(Xc, Yc, res["phi_grid"],
                         levels=np.linspace(0.05, 0.35, 16), cmap="YlOrRd")
        fig.colorbar(cf, ax=ax, fraction=0.046)
        draw_hole(ax, res["hcells"], res["mx"], res["my"])
        ax.set_title("%s: porosity phi" % res["label"])
        ax.set_aspect("equal"); ax.set_ylabel("y [m]")
        # pressure
        ax = axes[row][1]
        cf = ax.contourf(Xc, Yc, res["p_grid"],
                         levels=np.linspace(0.0, P0 / 1e6, 16), cmap="jet")
        fig.colorbar(cf, ax=ax, fraction=0.046)
        draw_hole(ax, res["hcells"], res["mx"], res["my"])
        ax.set_title("%s: pressure [MPa]" % res["label"])
        ax.set_aspect("equal")
        # saturation
        ax = axes[row][2]
        cf = ax.contourf(Xc, Yc, res["s_grid"],
                         levels=np.linspace(0.0, 1.0, 11), cmap="Blues")
        fig.colorbar(cf, ax=ax, fraction=0.046)
        try:
            ax.contour(Xc, Yc, res["s_grid"], levels=[0.5], colors="red", linewidths=2.0)
        except Exception:
            pass
        draw_hole(ax, res["hcells"], res["mx"], res["my"])
        ax.set_title("%s: fill S (red=S=0.5)" % res["label"])
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("x [m]")
    fig.suptitle("Inclined-hole grouting: gradient (top) vs uniform (bottom) "
                 "porosity  [p0=%.1f MPa, tau0=%.0f Pa, rho=%.0f, gravity ON]"
                 % (P0 / 1e6, TAU0, RHO),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, "grout_gradient_compare.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote grout_gradient_compare.png")


def make_front_overlay(res_a, res_b):
    Xc, Yc = grid_axes(res_a["nx"], res_a["ny"])
    fig, ax = plt.subplots(figsize=(8, 8))
    csa = ax.contour(Xc, Yc, res_a["s_grid"], levels=[0.5],
                     colors="red", linewidths=2.5)
    csb = ax.contour(Xc, Yc, res_b["s_grid"], levels=[0.5],
                     colors="blue", linewidths=2.5, linestyles="--")
    draw_hole(ax, res_a["hcells"], res_a["mx"], res_a["my"])
    # proxy legend handles
    ax.plot([], [], "r-", lw=2.5, label="gradient (0.12 bottom -> 0.30 top)")
    ax.plot([], [], "b--", lw=2.5, label="uniform phi=0.18")
    ax.plot([], [], "k-", lw=2.0, label="inclined hole")
    ax.axhline(MOUTH[1], color="gray", lw=0.5, ls=":")
    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("S=0.5 spread front: gradient vs uniform\n"
                 "(gradient front reaches higher / loose top, flatter / dense bottom)")
    ax.legend(loc="upper right", fontsize=9)
    out = os.path.join(_OUT, "grout_gradient_front_overlay.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote grout_gradient_front_overlay.png")


def report_one(res):
    fm = res["fm"]
    drift = res["v_in_src"] - res["v_store"]
    drift_rel = abs(drift) / max(abs(res["v_in_src"]), 1e-30)
    print("--- %s ---" % res["label"])
    print("  phi range %.3f..%.3f ; k range %.3e..%.3e m^2"
          % (res["phi"].min(), res["phi"].max(), res["k"].min(), res["k"].max()))
    if fm:
        ratio = fm["perp_up"] / max(fm["perp_down"], 1e-9)
        print("  filled %d cells; bbox x[%.1f,%.1f] y[%.1f,%.1f]"
              % (fm["count"], fm["x_min"], fm["x_max"], fm["y_min"], fm["y_max"]))
        print("  perp spread up(loose) %.2f m / down(dense) %.2f m  -> up/down = %.2f"
              % (fm["perp_up"], fm["perp_down"], ratio))
    print("  conservation: Vin=%.3f Vstore=%.3f drift_rel=%.2e clip=%.2e"
          % (res["v_in_src"], res["v_store"], drift_rel, res["v_clip"]))


USAGE = """inclined_hole_gradient.py [--only main|sweep|all] [--ckpt-every N]

  --only main    A gradient + B uniform + the comparison figures (~1 h)
  --only sweep   the phi_top 0.25 / 0.35 sweep points only (~1 h)
  --only all     both (default)
  --ckpt-every N checkpoint every N steps; 0 disables. Default %d.

Runs are checkpointed and resume bit-identically after a crash, so an
interrupted run costs one checkpoint interval and not the whole case; just
launch the same command again. Checkpoints live in outputs/checkpoints/ and
are deleted when a sub-run finishes cleanly.""" % CKPT_EVERY


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    which = "all"
    ckpt_every = CKPT_EVERY
    i = 0
    while i < len(argv):
        if argv[i] == "--only":
            which = argv[i + 1]; i += 2
        elif argv[i] == "--ckpt-every":
            ckpt_every = int(argv[i + 1]); i += 2
        elif argv[i] in ("-h", "--help"):
            print(USAGE)
            return
        else:
            i += 1
    if which not in ("main", "sweep", "all"):
        print(USAGE)
        return

    print("=== gradient vs uniform inclined-hole grouting ===")
    print("local L_max=p0/lambda anchors (calibrated k A=%.3g, tau0=%.0f Pa):"
          % (A_CAL, TAU0))
    for pp in (0.12, 0.18, 0.30):
        k = kc_like(pp)
        lm = lam_of(pp)
        print("  phi=%.2f: k=%.3e m^2  lambda=%.3e Pa/m  L_max=%.2f m"
              % (pp, k, lm, P0 / lm))
    print("  k(0.30)/k(0.12) = %.1f x ; lambda(0.12)/lambda(0.30) = %.2f x ; "
          "L_max(0.30)/L_max(0.12) = %.2f x"
          % (kc_like(0.30) / kc_like(0.12), lam_of(0.12) / lam_of(0.30),
             (P0 / lam_of(0.30)) / (P0 / lam_of(0.12))))
    print("")

    sys.stdout.flush()

    if which in ("main", "all"):
        res_a = run_one("gradient", "A gradient 0.12->0.30", ckpt_every=ckpt_every)
        res_b = run_one("uniform", "B uniform 0.18", ckpt_every=ckpt_every)
        print("")
        report_one(res_a)
        report_one(res_b)
        sys.stdout.flush()

        make_comparison_figure(res_a, res_b)
        make_front_overlay(res_a, res_b)
        sys.stdout.flush()

    if which == "main":
        return

    # optional gradient-strength sweep (guarded: a sweep failure must not lose
    # the main results above)
    print("")
    print("=== gradient-strength sweep (phi_bottom=0.12 fixed) ===")
    print("  phi_top   L_max_top[m]  perp_up[m]  perp_down[m]  up/down")
    sys.stdout.flush()
    global PHI_TOP
    base_top = PHI_TOP
    try:
        for top in (0.25, 0.35):
            PHI_TOP = top
            r = run_one("gradient", "sweep top=%.2f" % top,
                        ckpt_every=ckpt_every)
            fm = r["fm"]
            if fm:
                ratio = fm["perp_up"] / max(fm["perp_down"], 1e-9)
                print("  %.2f      %8.2f      %8.2f     %8.2f      %.2f"
                      % (top, P0 / lam_of(top), fm["perp_up"], fm["perp_down"], ratio))
                sys.stdout.flush()
    except Exception as exc:
        print("  sweep aborted: %s" % exc)
    finally:
        PHI_TOP = base_top


if __name__ == "__main__":
    main()
