# -*- coding: utf-8 -*-
"""
Round-6 task 2: Roache GCI for the 4a smooth (heterogeneous-k) case.

Reports, in the ASME V&V20 / Roache grid-convergence-index format, the
numerical uncertainty of a representative scalar functional (the domain-
midpoint pressure p(L/2)) over three grids. Run:

  powershell -File scripts\run_local.ps1 scripts\gci_4a.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tests"))

import numpy as np

import test_verification_ladder as lad


def midpoint_pressure(nx):
    """Solve the 4a heterogeneous-k case, return p at x = L/2 (interpolated)."""
    from src.fipy_adapter.mesh_init import build_mesh
    from src.models.slurry_transport.variables import (
        build_placeholder_slurry_parameters, initialize_slurry_variables,
    )
    from src.models.slurry_transport.equations import (
        update_effective_mobility, _solve_pressure_once,
    )
    dx = lad.S4A_L / float(nx)
    mesh, x, y, fx, fy = build_mesh(nx=nx, ny=1, dx=dx, dy=dx)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    params = build_placeholder_slurry_parameters()
    params["rheology_model"] = "linear"
    params["gravity_y"] = 0.0
    state.update(initialize_slurry_variables(mesh, params=params))
    xv = np.asarray(state["x"], dtype=float)
    mob = lad.S4A_M0 * (1.0 + lad.S4A_A * xv / lad.S4A_L)
    state["mobility_structural"].setValue(mob)
    state["mobility_effective"].setValue(mob)
    state["pressure"].constrain(lad.S4A_P0, mesh.facesLeft)
    state["pressure"].constrain(0.0, mesh.facesRight)
    for _ in range(12):
        update_effective_mobility(state)
        _solve_pressure_once(state, dt=1.0e6)
    p = np.asarray(state["pressure"].value, dtype=float)
    return float(np.interp(0.5 * lad.S4A_L, xv[np.argsort(xv)], p[np.argsort(xv)]))


def gci_triple(phi1, phi2, phi3, r=2.0, Fs=1.25):
    """phi1 finest .. phi3 coarsest, constant refinement ratio r."""
    e21 = phi2 - phi1
    e32 = phi3 - phi2
    p = np.log(abs(e32 / e21)) / np.log(r)
    phi_ext = (r ** p * phi1 - phi2) / (r ** p - 1.0)
    ea21 = abs((phi1 - phi2) / phi1)             # approximate rel error
    eext = abs((phi_ext - phi1) / phi_ext)       # extrapolated rel error
    gci21 = Fs * ea21 / (r ** p - 1.0)
    gci32 = Fs * abs((phi2 - phi3) / phi2) / (r ** p - 1.0)
    return p, phi_ext, ea21, eext, gci21, gci32


def main():
    Ns = [40, 80, 160]   # phi1=finest(160) .. but order phi1..phi3 fine->coarse
    phis = {n: midpoint_pressure(n) for n in [40, 80, 160]}
    phi1, phi2, phi3 = phis[160], phis[80], phis[40]
    p, phi_ext, ea21, eext, gci21, gci32 = gci_triple(phi1, phi2, phi3)
    exact = lad._s4a_exact(np.array([0.5 * lad.S4A_L]))[0]
    print("=== 4a Roache GCI (functional: midpoint pressure p(L/2)) ===")
    print("grids N = 40 / 80 / 160 (r = 2)")
    print("phi(40)=%.6f  phi(80)=%.6f  phi(160)=%.6f  [exact %.6f]"
          % (phi3, phi2, phi1, exact))
    print("apparent order p   = %.3f" % p)
    print("Richardson phi_ext = %.6f  (exact %.6f, diff %.2e)"
          % (phi_ext, exact, phi_ext - exact))
    print("GCI_fine (80->160) = %.4f %%" % (100.0 * gci21))
    print("GCI_med  (40->80)  = %.4f %%" % (100.0 * gci32))
    print("asymptotic check GCI_med/(r^p*GCI_fine) = %.4f (want ~1)"
          % (gci32 / (2.0 ** p * gci21)))


if __name__ == "__main__":
    main()
