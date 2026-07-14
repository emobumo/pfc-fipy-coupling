# -*- coding: utf-8 -*-
"""
APPLICATION benchmark (NOT a verification-ladder test): inclined-hole
constant-pressure grouting of a Bingham cement slurry into a fixed waste-rock
pile, on the validated FiPy IMPES fill-transport engine.

  - particles  : D:/PFC item/pfc_fipy_model1/particles.csv  (x,y,radius)
  - domain     : x in [-15,15], y in [0,30], 2.5 m cells (12x12 = 144 cells)
  - permeability: calibrated_power k = A phi^3/(1-phi)^2, A=9.4e-8  (NOT KC)
  - injection  : inclined hole, mouth (-15,1), dip 35 deg, length 17 m, the
                 cells the line passes through are pinned at p0=5 MPa (interior
                 Dirichlet penalty) and held saturated (S=1)
  - slurry     : tau0=60 Pa, mu_p=0.1 Pa.s, rho=1820 kg/m^3, gravity ON
  - run        : IMPES fill transport to front stall; outputs 3 PNGs + report

Single-run 17 m case (the real process is staged 7/10/17 m; only the final
single 17 m injection is modelled here). Py2.7 / 3 compatible.

Run: powershell -File scripts\\run_local.ps1 cases\\inclined_hole_grouting.py
Env: PARTICLES_CSV=path to override the CSV location.
Figures are written to outputs/ (untracked artifacts).
"""
from __future__ import print_function
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fipy import CellVariable

from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.coupling.porosity_to_permeability import porosity_to_permeability
from src.models.slurry_transport.variables import (
    build_placeholder_slurry_parameters,
    initialize_slurry_variables,
)
from src.models.slurry_transport.equations import (
    solve_transport_step,
    conservation_report,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# Figures are build artifacts: keep them out of the tracked case directory.
_OUT = os.path.join(os.path.dirname(_HERE), "outputs")
if not os.path.isdir(_OUT):
    os.makedirs(_OUT)
CSV_PATH = os.environ.get(
    "PARTICLES_CSV", "D:/PFC item/pfc_fipy_model1/particles.csv"
)

# --- domain / mesh ---------------------------------------------------------
X_MIN, X_MAX = -15.0, 15.0
Y_MIN, Y_MAX = 0.0, 30.0
CELL = 2.5
PHI_CLIP = (0.05, 0.60)

# --- inclined hole ---------------------------------------------------------
MOUTH = (-15.0, 1.0)
DIP_DEG = 35.0
HOLE_LEN = 17.0
P0 = 5.0e6

# --- slurry (PO 42.5, w/c=0.5) ---------------------------------------------
TAU0 = 60.0
MU_P = 0.1
RHO = 1820.0
A_CAL = 9.4e-8

# --- march control ---------------------------------------------------------
DT_CAP = 5.0
MAX_STEPS = 1500
STALL_TOL_CELLS = 0      # stop when filled-cell count stops growing ...
STALL_PATIENCE = 40      # ... for this many consecutive steps
STALL_VSTORE_REL = 1.0e-4


# ===========================================================================
def load_particles(path):
    xs, ys, rs = [], [], []
    f = open(path, "r")
    try:
        for line in f:
            parts = line.replace(",", " ").split()
            try:
                x, y, r = float(parts[0]), float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                continue
            xs.append(x)
            ys.append(y)
            rs.append(r)
    finally:
        f.close()
    return np.array(xs), np.array(ys), np.array(rs)


def bin_porosity(px, py, pr, mesh_x, mesh_y, nx, ny):
    """Areal porosity per FiPy cell: phi = 1 - sum(pi r^2)/cell_area,
    indexed in FiPy cell order via each cell center's (ix,iy)."""
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    solid = np.zeros((ny, nx), dtype=float)
    for i in range(px.size):
        ix = int(math.floor((px[i] - X_MIN) / dx))
        iy = int(math.floor((py[i] - Y_MIN) / dy))
        if 0 <= ix < nx and 0 <= iy < ny:
            solid[iy, ix] += math.pi * pr[i] * pr[i]
    phi_grid = 1.0 - solid / (dx * dy)
    # map to FiPy cell order from each cell center
    phi = np.zeros(mesh_x.size, dtype=float)
    for n in range(mesh_x.size):
        ix = int(round((mesh_x[n] - X_MIN) / dx - 0.5))
        iy = int(round((mesh_y[n] - Y_MIN) / dy - 0.5))
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        phi[n] = phi_grid[iy, ix]
    return np.clip(phi, PHI_CLIP[0], PHI_CLIP[1]), phi_grid


def hole_geometry():
    a = math.radians(DIP_DEG)
    end = (MOUTH[0] + HOLE_LEN * math.cos(a), MOUTH[1] + HOLE_LEN * math.sin(a))
    d = (math.cos(a), math.sin(a))            # along-hole unit vector
    n = (math.sin(a), -math.cos(a))           # normal, +n points down-right
    return end, d, n


def hole_cells(mesh_x, mesh_y, nx, ny):
    """Cells the inclined segment passes through: sample the segment densely
    and collect the cell each sample falls in (FiPy cell order)."""
    end, _, _ = hole_geometry()
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    idx_of_cell = {}
    for n in range(mesh_x.size):
        ix = int(round((mesh_x[n] - X_MIN) / dx - 0.5))
        iy = int(round((mesh_y[n] - Y_MIN) / dy - 0.5))
        idx_of_cell[(ix, iy)] = n
    cells = set()
    nsamp = 2000
    for s in range(nsamp + 1):
        t = float(s) / nsamp
        x = MOUTH[0] + t * (end[0] - MOUTH[0])
        y = MOUTH[1] + t * (end[1] - MOUTH[1])
        ix = int(math.floor((x - X_MIN) / dx))
        iy = int(math.floor((y - Y_MIN) / dy))
        if (ix, iy) in idx_of_cell:
            cells.add(idx_of_cell[(ix, iy)])
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


# ===========================================================================
def build_case():
    mesh, x, y, fx, fy = build_mesh_for_domain(X_MIN, X_MAX, Y_MIN, Y_MAX, CELL)
    mx = np.asarray(x, dtype=float)
    my = np.asarray(y, dtype=float)
    nx = int(round((X_MAX - X_MIN) / CELL))
    ny = int(round((Y_MAX - Y_MIN) / CELL))

    px, py, pr = load_particles(CSV_PATH)
    n_particles = int(np.asarray(px).size)
    phi, phi_grid = bin_porosity(px, py, pr, mx, my, nx, ny)

    params = build_placeholder_slurry_parameters()
    params.update({
        "rheology_model": "porous_bingham",
        "porous_bingham_activation": "threshold",
        "pressure_coeff_form": "face",
        "yield_truncation_mode": "papanastasiou",
        "enable_saturation_transport": True,
        # calibrated permeability (NOT KC); A anchored to field data.
        "porosity_to_permeability_formula": "calibrated_power",
        "calibrated_permeability_coefficient": A_CAL,
        # slurry
        "yield_stress": TAU0,
        "plastic_viscosity": MU_P,
        "slurry_density": RHO,
        "gravity_y": -9.81,
        "inlet_pressure_core_value": P0,   # used only for the eps auto-scale
        # storage: keep c*p0 << phi (incompressible-like); design-review rule.
        "reference_storage": 1.0e-10,
        # time stepping: backward-Euler caps the chained-iterate front overrun
        # structurally (~1 cell/step), so the front dt_p constraint is not
        # needed; the fill CFL + in-step re-solve guard keep clip ~ 0.
        "picard_transient_mode": "backward_euler",
        "enable_front_dt_constraint": False,
        "dt_cfl": 0.5,
        "picard_relaxation_fill": 0.15,
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

    # inclined hole: interior Dirichlet at p0, held saturated.
    hcells = hole_cells(mx, my, nx, ny)
    mask = np.zeros(mx.size, dtype=float)
    mask[hcells] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    state["interior_dirichlet_value"] = P0
    s0 = np.zeros(mx.size, dtype=float)
    s0[hcells] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, P0, 0.0))
    # all outer faces sealed (no open boundary faces) -> default zero-flux.

    return state, mx, my, nx, ny, phi, phi_grid, hcells, k, n_particles


def front_metrics(state, mx, my, hcells):
    """Spread metrics of the S>=0.5 filled region (excluding hole cells)."""
    s = np.asarray(state["saturation"].value, dtype=float)
    end, d, n = hole_geometry()
    filled = (s >= 0.5)
    filled[hcells] = False
    if not np.any(filled):
        return None
    fx = mx[filled]
    fy = my[filled]
    # signed perpendicular distance from the hole axis (+n = down-right)
    perp = (fx - MOUTH[0]) * n[0] + (fy - MOUTH[1]) * n[1]
    # along-hole coordinate
    along = (fx - MOUTH[0]) * d[0] + (fy - MOUTH[1]) * d[1]
    return {
        "count": int(np.sum(filled)),
        "x_min": float(np.min(fx)), "x_max": float(np.max(fx)),
        "y_min": float(np.min(fy)), "y_max": float(np.max(fy)),
        "perp_down": float(np.max(perp)),       # gravity-favored side
        "perp_up": float(-np.min(perp)),        # against gravity
        "along_min": float(np.min(along)),
        "along_max": float(np.max(along)),
    }


def march(state, mx, my, hcells, vols):
    phi = np.clip(np.asarray(state["porosity"].value, dtype=float), 1e-6, 1.0)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True
    s0 = np.array(state["saturation"].value, copy=True)

    t = 0.0
    v_in_src = 0.0           # cumulative slurry pumped OUT of the source cells
    last_count = -1
    stall = 0
    last_vstore = 0.0
    print("step   t[s]      dt[s]     filled  Vstore[m3/m]  Vin_src   front_perp(dn/up)[m]")
    for step in range(MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=DT_CAP)
        t += float(dt)
        # source outflow this step from the explicit flux divergence
        div_q = np.asarray(state["last_div_q"], dtype=float)  # [1/s], +=outflow
        v_in_src += float(np.sum(div_q[hmask] * vols[hmask])) * float(dt)
        # re-assert the source saturation (guard against clip noise)
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)

        s_arr = np.asarray(state["saturation"].value, dtype=float)
        v_store = float(np.sum(phi * (s_arr - s0) * vols))
        count = int(np.sum((s_arr >= 0.5) & np.logical_not(hmask)))

        if step % 25 == 0 or step == MAX_STEPS - 1:
            fm = front_metrics(state, mx, my, hcells)
            pd = fm["perp_down"] if fm else 0.0
            pu = fm["perp_up"] if fm else 0.0
            print("%4d  %8.3f  %8.4f  %5d   %10.4f  %9.4f  %.2f / %.2f"
                  % (step, t, dt, count, v_store, v_in_src, pd, pu))

        # stall detection: filled count plateaus AND V_store barely grows
        grew = count > last_count or (
            abs(v_store - last_vstore) > STALL_VSTORE_REL * max(v_store, 1e-30))
        if grew:
            stall = 0
        else:
            stall += 1
        last_count = max(last_count, count)
        last_vstore = v_store
        if stall >= STALL_PATIENCE:
            print("  -> stalled at step %d (t=%.3f s)" % (step, t))
            break

    return t, v_in_src, v_store


def make_figures(state, mx, my, nx, ny, phi, hcells, k):
    end, _, _ = hole_geometry()
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    xc = X_MIN + (np.arange(nx) + 0.5) * dx
    yc = Y_MIN + (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc)

    def draw_hole(ax):
        ax.plot([MOUTH[0], end[0]], [MOUTH[1], end[1]], "k-", lw=2.5)
        ax.plot([MOUTH[0]], [MOUTH[1]], "ko", ms=7)
        ax.plot(mx[hcells], my[hcells], "ks", ms=4, mfc="none")

    # 1. porosity
    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(Xc, Yc, to_grid(phi, mx, my, nx, ny),
                     levels=np.linspace(PHI_CLIP[0], PHI_CLIP[1], 16),
                     cmap="YlOrRd")
    fig.colorbar(cf, ax=ax, label="areal porosity phi")
    draw_hole(ax)
    ax.set_title("Porosity field phi(x,y), cell %.1f m  (inclined hole marked)" % CELL)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "grout_porosity.png"), dpi=130)
    plt.close(fig)

    # 2. pressure
    p = np.asarray(state["pressure"].value, dtype=float) / 1e6
    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(Xc, Yc, to_grid(p, mx, my, nx, ny),
                     levels=np.linspace(0.0, P0 / 1e6, 16), cmap="jet")
    fig.colorbar(cf, ax=ax, label="pressure p [MPa]")
    draw_hole(ax)
    ax.set_title("Pressure field p(x,y) [MPa]  (p0=%.0f MPa interior Dirichlet)" % (P0 / 1e6))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "grout_pressure.png"), dpi=130)
    plt.close(fig)

    # 3. saturation with S=0.5 front
    s = to_grid(np.asarray(state["saturation"].value, dtype=float), mx, my, nx, ny)
    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(Xc, Yc, s, levels=np.linspace(0.0, 1.0, 11), cmap="Blues")
    fig.colorbar(cf, ax=ax, label="slurry saturation / fill S")
    try:
        cs = ax.contour(Xc, Yc, s, levels=[0.5], colors="red", linewidths=2.0)
        ax.clabel(cs, fmt="S=0.5", fontsize=8)
    except Exception:
        pass
    draw_hole(ax)
    ax.set_title("Slurry fill S(x,y) at stall  (red = S=0.5 spread front)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "grout_saturation.png"), dpi=130)
    plt.close(fig)
    print("wrote grout_porosity.png / grout_pressure.png / grout_saturation.png")


def main():
    print("=== inclined-hole grouting benchmark ===")
    state, mx, my, nx, ny, phi, phi_grid, hcells, k, n_particles = build_case()
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)

    # --- task 1: porosity field report ---
    interior = phi_grid[:, 1:-1]
    print("particles: %d ; domain x[%.0f,%.0f] y[%.0f,%.0f] cell %.1f m -> %dx%d=%d cells"
          % (n_particles, X_MIN, X_MAX, Y_MIN, Y_MAX,
             CELL, nx, ny, nx * ny))
    print("porosity (clipped to %s):" % (PHI_CLIP,))
    print("  all cells : mean %.4f  min %.4f  max %.4f  std %.4f"
          % (phi.mean(), phi.min(), phi.max(), phi.std()))
    print("  interior  : mean %.4f  min %.4f  max %.4f (edge cols dropped)"
          % (interior.mean(), interior.min(), interior.max()))
    print("permeability (calibrated_power A=%.2e): k mean %.3e min %.3e max %.3e m^2"
          % (A_CAL, k.mean(), k.min(), k.max()))
    end, _, _ = hole_geometry()
    print("hole: mouth (%.2f,%.2f) -> end (%.2f,%.2f), %d source cells"
          % (MOUTH[0], MOUTH[1], end[0], end[1], len(hcells)))

    # verify calibration anchors
    for pp in (0.10, 0.25, 0.45):
        kk = A_CAL * pp ** 3 / (1.0 - pp) ** 2
        print("  check k(phi=%.2f) = %.3e m^2" % (pp, kk))

    # --- task 5: march ---
    t, v_in_src, v_store = march(state, mx, my, hcells, vols)

    # --- conservation ---
    rep = conservation_report(state)
    v_clip = rep["v_clip"]
    drift = v_in_src - v_store
    drift_rel = abs(drift) / max(abs(v_in_src), 1e-30)
    print("")
    print("=== conservation (interior-source balance) ===")
    print("  V_in_source (pump outflow) = %.6e m^3/m" % v_in_src)
    print("  V_store (slurry accumulated) = %.6e m^3/m" % v_store)
    print("  drift = V_in - V_store = %.3e  (rel %.3e)" % (drift, drift_rel))
    print("  clipped overshoot volume   = %.3e m^3/m (should be ~0)" % v_clip)

    # --- spread report ---
    fm = front_metrics(state, mx, my, hcells)
    print("")
    print("=== spread of the S>=0.5 region (excl. source cells) ===")
    if fm:
        print("  filled cells: %d (area %.1f m^2)" % (fm["count"], fm["count"] * CELL * CELL))
        print("  bounding box: x[%.2f, %.2f]  y[%.2f, %.2f]"
              % (fm["x_min"], fm["x_max"], fm["y_min"], fm["y_max"]))
        print("  perpendicular spread from hole axis: down-right %.2f m / up-left %.2f m"
              % (fm["perp_down"], fm["perp_up"]))
        print("  along-hole filled coordinate: [%.2f, %.2f] m (hole 0..%.1f)"
              % (fm["along_min"], fm["along_max"], HOLE_LEN))
    print("  (reference uniform L_max at phi: 0.10->4.0 m, 0.25->12.0 m, 0.45->29.5 m)")

    make_figures(state, mx, my, nx, ny, phi, hcells, k)


if __name__ == "__main__":
    main()
