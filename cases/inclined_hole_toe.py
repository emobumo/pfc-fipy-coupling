# -*- coding: utf-8 -*-
"""
APPLICATION benchmark (NOT a verification-ladder test): a deliberate
source-geometry CONTROL for the line-source gradient case.

This is not the field process. Standard practice is a perforated grouting
pipe that bleeds along its whole length, and the orifice pipe seals only the
collar (the reference project: Phi108x1500 mm in a 2.0 m mouth) -- not the
twelve metres of hole this case leaves un-injecting. inclined_hole_grouting
and inclined_hole_gradient, which inject along the whole 17 m, are the
engineering baseline.

The point of restricting the source is that the 1.26 upward asymmetry of the
line source could be an artifact of injecting along a dipping line. A compact
toe patch is about as different a source geometry as this layout allows, and
it still gives ~1.2, which leaves the porosity gradient as the cause. The
less this case resembles the field, the better it does that job.

Hole geometry is unchanged (mouth (-15,1), dip 35 deg, length 17 m,
toe (-1.07,10.75)); only the source support differs:

    bleed segment = along-hole [HOLE_LEN - BLEED_LEN, HOLE_LEN]   (default 5 m)
    un-injecting  = along-hole [0, HOLE_LEN - BLEED_LEN]          (ordinary
                    rock that still takes part in the seepage)

Two cases for comparison (both toe-segment bleed):
  (A) gradient  : phi_bottom=0.12 -> phi_top=0.30
  (B) uniform   : phi = 0.18

Permeability calibrated_power k=A phi^3/(1-phi)^2 with the v0.6 parameter set
(variables.grout_material_v06), gravity ON. Same engine / time-stepping as the line-source
version. Py2.7 / 3 compatible.

Run: powershell -File scripts\\run_local.ps1 cases\\inclined_hole_toe.py
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

# --- gradient porosity (tunable) -------------------------------------------
PHI_BOTTOM = 0.12
PHI_TOP = 0.30
PHI_UNIFORM = 0.18

# --- inclined hole + toe-segment bleed -------------------------------------
MOUTH = (-30.0, 21.0)
DIP_DEG = 35.0
HOLE_LEN = 17.0
BLEED_LEN = 5.0       # toe-segment bleed length [m]; cased = HOLE_LEN-BLEED_LEN
P0 = 5.0e6

# --- slurry (PO 42.5, w/c=0.5): the v0.6 set, see grout_material_v06 -------
_MAT = grout_material_v06()
TAU0 = _MAT["yield_stress"]
MU_P = _MAT["plastic_viscosity"]
RHO = _MAT["slurry_density"]
A_CAL = _MAT["calibrated_permeability_coefficient"]

# --- march control ---------------------------------------------------------
DT_CAP = 5.0
MAX_STEPS = 20000     # never binding under the rate rule (default stop since 2026-09-27)
STALL_PATIENCE = 60   # legacy rule only (STOP_RULE=legacy)
# Crash-resilience: see src/analysis/run_checkpoint.py. 0 disables.
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


def bleed_segment():
    """Endpoints of the toe-segment bleed in (x,y)."""
    _, d, _ = hole_geometry()
    s0 = HOLE_LEN - BLEED_LEN
    p_start = (MOUTH[0] + s0 * d[0], MOUTH[1] + s0 * d[1])
    p_end = (MOUTH[0] + HOLE_LEN * d[0], MOUTH[1] + HOLE_LEN * d[1])
    return p_start, p_end


def _cell_index_map(mesh_x, mesh_y, nx, ny):
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    idx = {}
    for n in range(mesh_x.size):
        ix = int(round((mesh_x[n] - X_MIN) / dx - 0.5))
        iy = int(round((mesh_y[n] - Y_MIN) / dy - 0.5))
        idx[(ix, iy)] = n
    return idx, dx, dy


def bleed_cells(mesh_x, mesh_y, nx, ny):
    """Cells the TOE segment [HOLE_LEN-BLEED_LEN, HOLE_LEN] passes through."""
    _, d, _ = hole_geometry()
    idx, dx, dy = _cell_index_map(mesh_x, mesh_y, nx, ny)
    cells = set()
    nsamp = 2000
    s_lo = HOLE_LEN - BLEED_LEN
    for i in range(nsamp + 1):
        along = s_lo + (HOLE_LEN - s_lo) * float(i) / nsamp
        x = MOUTH[0] + along * d[0]
        y = MOUTH[1] + along * d[1]
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

    hcells = bleed_cells(mx, my, nx, ny)
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
    _, d, n = hole_geometry()
    p_start, p_end = bleed_segment()
    toe = p_end
    filled = (s >= 0.5)
    filled[hcells] = False
    if not np.any(filled):
        return None
    fx = mx[filled]
    fy = my[filled]
    perp = (fx - MOUTH[0]) * n[0] + (fy - MOUTH[1]) * n[1]      # +=down-right
    along = (fx - MOUTH[0]) * d[0] + (fy - MOUTH[1]) * d[1]     # along hole
    r_toe = np.sqrt((fx - toe[0]) ** 2 + (fy - toe[1]) ** 2)
    return {
        "count": int(np.sum(filled)),
        "x_min": float(np.min(fx)), "x_max": float(np.max(fx)),
        "y_min": float(np.min(fy)), "y_max": float(np.max(fy)),
        "perp_down": float(np.max(perp)),
        "perp_up": float(-np.min(perp)),
        "along_min": float(np.min(along)),
        "along_max": float(np.max(along)),
        "r_toe_max": float(np.max(r_toe)),
        "cx": float(np.mean(fx)), "cy": float(np.mean(fy)),
    }


def march(state, mx, my, hcells, vols, label, keeper=None):
    phi = np.clip(np.asarray(state["porosity"].value, dtype=float), 1e-6, 1.0)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True

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
    # stop rule (src/analysis/stop_rule.py): n_ref = porosity of the hole's middle cell
    n_ref = float(np.asarray(state["porosity"].value, dtype=float)[hcells[len(hcells) // 2]])
    qref = stop_rule.q_ref(n_ref, A_CAL, P0, MU_P)
    t_hist, v_hist = [0.0], [0.0]

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
        print("  [%s] marching ..." % label)
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

        if count > best_count:
            best_count = count
            stall = 0
        else:
            stall += 1
        if stop_rule.legacy():
            if stall >= STALL_PATIENCE:
                break
        elif stop_rule.stop_rule_physical(t_hist, v_hist, qref):
            break
        if keeper is not None:
            keeper.maybe_save(state, step=step + 1, t=t, v_in_src=v_in_src,
                              best_count=best_count, stall=stall, s0=s0,
                              t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    if keeper is not None:
        keeper.finish()
    print("    -> stop at step %d, t=%.1f s, filled=%d, Vstore=%.3f, Vin=%.3f"
          % (step, t, count, v_store, v_in_src))
    sys.stdout.flush()
    return t, v_in_src, v_store


def run_one(kind, label, ckpt_every=CKPT_EVERY):
    state, mx, my, nx, ny, phi, hcells, k = build_case(kind)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    keeper = None
    if ckpt_every > 0:
        if not os.path.isdir(_CKPT_DIR):
            os.makedirs(_CKPT_DIR)
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", "toe_" + label)
        keeper = Checkpointer(
            os.path.join(_CKPT_DIR, tag), every=ckpt_every,
            fingerprint=state_fingerprint(
                state, "toe|%s|%s|bleed=%.3f" % (kind, label, BLEED_LEN)))
    t, v_in_src, v_store = march(state, mx, my, hcells, vols, label, keeper=keeper)
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
    p_start, p_end = bleed_segment()
    # cased (sealed) collar section: thin gray dashed
    ax.plot([MOUTH[0], p_start[0]], [MOUTH[1], p_start[1]],
            color="gray", lw=1.5, ls="--")
    ax.plot([MOUTH[0]], [MOUTH[1]], "o", color="gray", ms=5)
    # bleed (toe) section: thick black
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], "k-", lw=3.0)
    ax.plot(mx[hcells], my[hcells], "ks", ms=4, mfc="none")


def make_comparison_figure(res_a, res_b):
    Xc, Yc = grid_axes(res_a["nx"], res_a["ny"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, res in enumerate((res_a, res_b)):
        ax = axes[row][0]
        cf = ax.contourf(Xc, Yc, res["phi_grid"],
                         levels=np.linspace(0.05, 0.35, 16), cmap="YlOrRd")
        fig.colorbar(cf, ax=ax, fraction=0.046)
        draw_hole(ax, res["hcells"], res["mx"], res["my"])
        ax.set_title("%s: porosity phi" % res["label"])
        ax.set_aspect("equal"); ax.set_ylabel("y [m]")
        ax = axes[row][1]
        cf = ax.contourf(Xc, Yc, res["p_grid"],
                         levels=np.linspace(0.0, P0 / 1e6, 16), cmap="jet")
        fig.colorbar(cf, ax=ax, fraction=0.046)
        draw_hole(ax, res["hcells"], res["mx"], res["my"])
        ax.set_title("%s: pressure [MPa]" % res["label"])
        ax.set_aspect("equal")
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
    fig.suptitle("Toe-segment bleed (%.0f m) grouting: gradient (top) vs uniform "
                 "(bottom)  [p0=%.1f MPa, tau0=%.0f Pa, rho=%.0f, gravity ON]"
                 % (BLEED_LEN, P0 / 1e6, TAU0, RHO),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_OUT, "grout_toe_compare.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote grout_toe_compare.png")


def make_front_overlay(res_a, res_b):
    Xc, Yc = grid_axes(res_a["nx"], res_a["ny"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(Xc, Yc, res_a["s_grid"], levels=[0.5], colors="red", linewidths=2.5)
    ax.contour(Xc, Yc, res_b["s_grid"], levels=[0.5], colors="blue",
               linewidths=2.5, linestyles="--")
    draw_hole(ax, res_a["hcells"], res_a["mx"], res_a["my"])
    toe = bleed_segment()[1]
    ax.plot([toe[0]], [toe[1]], "g*", ms=15, label="toe (%.1f,%.1f)" % toe)
    ax.plot([], [], "r-", lw=2.5, label="gradient 0.12->0.30")
    ax.plot([], [], "b--", lw=2.5, label="uniform phi=0.18")
    ax.plot([], [], "k-", lw=3.0, label="bleed segment (%.0f m)" % BLEED_LEN)
    ax.plot([], [], color="gray", lw=1.5, ls="--", label="cased collar")
    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Toe-segment bleed: S=0.5 front, gradient vs uniform\n"
                 "(ellipsoid around the toe; gradient skews it upward)")
    ax.legend(loc="upper right", fontsize=8)
    out = os.path.join(_OUT, "grout_toe_front_overlay.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote grout_toe_front_overlay.png")


def report_one(res):
    fm = res["fm"]
    drift = res["v_in_src"] - res["v_store"]
    drift_rel = abs(drift) / max(abs(res["v_in_src"]), 1e-30)
    toe = bleed_segment()[1]
    print("--- %s ---" % res["label"])
    print("  phi range %.3f..%.3f ; k range %.3e..%.3e m^2"
          % (res["phi"].min(), res["phi"].max(), res["k"].min(), res["k"].max()))
    if fm:
        ratio = fm["perp_up"] / max(fm["perp_down"], 1e-9)
        print("  filled %d cells; bbox x[%.1f,%.1f] y[%.1f,%.1f]"
              % (fm["count"], fm["x_min"], fm["x_max"], fm["y_min"], fm["y_max"]))
        print("  centroid (%.2f,%.2f) vs toe (%.2f,%.2f); max radius from toe %.2f m"
              % (fm["cx"], fm["cy"], toe[0], toe[1], fm["r_toe_max"]))
        print("  perp up(loose) %.2f m / down(dense) %.2f m  -> up/down = %.2f"
              % (fm["perp_up"], fm["perp_down"], ratio))
        print("  along-hole filled [%.1f, %.1f] m (bleed seg [%.1f, %.1f])"
              % (fm["along_min"], fm["along_max"], HOLE_LEN - BLEED_LEN, HOLE_LEN))
    print("  conservation: Vin=%.3f Vstore=%.3f drift_rel=%.2e clip=%.2e"
          % (res["v_in_src"], res["v_store"], drift_rel, res["v_clip"]))


def main():
    global BLEED_LEN
    print("=== toe-segment bleed (%.0f m) inclined-hole grouting ===" % BLEED_LEN)
    toe = bleed_segment()[1]
    bs = bleed_segment()[0]
    print("hole mouth (%.2f,%.2f) -> toe (%.2f,%.2f); bleed seg %s -> %s"
          % (MOUTH[0], MOUTH[1], toe[0], toe[1],
             ("(%.2f,%.2f)" % bs), ("(%.2f,%.2f)" % toe)))
    phi_toe = PHI_BOTTOM + (PHI_TOP - PHI_BOTTOM) * (toe[1] / H)
    print("local anchors (calibrated k A=%.3g, tau0=%.0f Pa):" % (A_CAL, TAU0))
    for pp, tag in ((0.12, "bottom"), (phi_toe, "toe(gradient)"),
                    (0.18, "uniform"), (0.30, "top")):
        print("  phi=%.3f (%s): lambda=%.3e Pa/m  L_max=%.2f m"
              % (pp, tag, lam_of(pp), P0 / lam_of(pp)))
    print("  (line-source run on the same field gave up/down=1.16)")
    print("")
    sys.stdout.flush()

    res_a = run_one("gradient", "A gradient 0.12->0.30")
    res_b = run_one("uniform", "B uniform 0.18")
    print("")
    report_one(res_a)
    report_one(res_b)
    sys.stdout.flush()

    make_comparison_figure(res_a, res_b)
    make_front_overlay(res_a, res_b)
    sys.stdout.flush()

    # optional bleed-length sweep on the gradient field
    print("")
    print("=== bleed-length sweep (gradient field, phi 0.12->0.30) ===")
    print("  bleed[m]  filled  perp_up[m]  perp_down[m]  up/down  r_toe_max[m]")
    sys.stdout.flush()
    base_bleed = BLEED_LEN
    try:
        for bl in (3.0, 5.0, 8.0):
            BLEED_LEN = bl
            r = run_one("gradient", "sweep bleed=%.0f m" % bl)
            fm = r["fm"]
            if fm:
                ratio = fm["perp_up"] / max(fm["perp_down"], 1e-9)
                print("  %.0f       %4d     %8.2f    %8.2f      %.2f      %.2f"
                      % (bl, fm["count"], fm["perp_up"], fm["perp_down"],
                         ratio, fm["r_toe_max"]))
                sys.stdout.flush()
    except Exception as exc:
        print("  sweep aborted: %s" % exc)
    finally:
        BLEED_LEN = base_bleed


if __name__ == "__main__":
    main()
