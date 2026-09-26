# -*- coding: utf-8 -*-
"""
Read-only engineering-model checkup (no src/ changes, no itasca, no FiPy).
Computes the REV binning table and the Bingham/gravity dimensionless numbers
from the recorded model1 specs + the engineering-case slurry parameters.

Run: powershell -File scripts\\run_local.ps1 scripts\\model1_checkup.py
"""
from __future__ import print_function
import math

# --- recorded model1 specs (from a prior PFC run; see memory) --------------
N_BALLS = 8986
KF_X = 15.0            # pile width [m]   (1generate.p2dat kf_x)
KF_H = 40.0            # pile height [m]  (1generate.p2dat kf_h)
PILE_AREA = KF_X * KF_H
MEAN_RADIUS = 0.127    # [m] (radius range 0.10-0.22 after settling)
D50 = 2.0 * MEAN_RADIUS
PHI_MEAN = 0.215       # areal porosity, container basis (edge-inflated)
PHI_BULK = 0.164       # areal porosity, bulk median (representative REV)

# --- engineering-case slurry params (build_engineering_case_parameters) ----
RHO = 1414.0           # [kg/m^3]
TAU0 = 4.6             # [Pa]
MU_P = 0.183           # [Pa.s]
P0 = 2.0e6             # [Pa] inlet_pressure_core_value (2 MPa)
G = 9.81               # [m/s^2]
KC = 180.0             # Kozeny-Carman constant


def kc_perm(phi, d=D50, C=KC):
    return (d * d / C) * phi ** 3 / (1.0 - phi) ** 2


def lam(k, phi):
    r_eff = math.sqrt(8.0 * k / phi)
    return 2.0 * TAU0 / r_eff, r_eff


def report_rev():
    print("=== TASK 2: REV binning table (N=%d balls, pile %.0fx%.0f=%.0f m^2, "
          "d50=%.3f m) ===" % (N_BALLS, KF_X, KF_H, PILE_AREA, D50))
    print(" cell(xd50)  cell[m]  cell_area  domain_cells  balls/cell  REV?")
    for mult in (4, 5, 10, 20, 40):
        cs = mult * D50
        area = cs * cs
        ncells = PILE_AREA / area
        per = N_BALLS * area / PILE_AREA
        if per < 25:
            verdict = "NO  (grain-scale noise)"
        elif per < 60:
            verdict = "marginal"
        else:
            verdict = "YES"
        print("  %4dx     %6.3f   %7.3f    %8.1f     %8.1f   %s"
              % (mult, cs, area, ncells, per, verdict))
    print("  note: production driver uses 1.0-1.5 m cells -> ~15-37 balls/cell")
    print("        (below/at the REV floor; round-1 diag saw clip-rail pinning).")


def report_dimensionless():
    print("")
    print("=== TASK 5: dimensionless numbers (H=%.0f m, p0=%.2g Pa, "
          "tau0=%.1f Pa, mu_p=%.3f, rho=%.0f) ===" % (KF_H, P0, TAU0, MU_P, RHO))
    Gnum = RHO * G * KF_H / P0
    print("gravity number  G = rho*g*H/p0 = %.4f" % Gnum)
    print("")
    print(" scenario              phi     k[m^2]     r_eff[m]  lambda[Pa/m]  "
          "Bn=lam*H/p0   L_max=p0/lam[m]  L_max/H")
    scen = [
        ("bulk median (KC)",  PHI_BULK, kc_perm(PHI_BULK)),
        ("container mean (KC)", PHI_MEAN, kc_perm(PHI_MEAN)),
        ("recorded bulk k",   PHI_BULK, 2.0e-6),
        ("densest cell k",    PHI_BULK, 1.0e-7),
        ("edge-artifact k",   0.45,     1.8e-4),
    ]
    for name, phi, k in scen:
        lm, reff = lam(k, phi)
        Bn = lm * KF_H / P0
        Lmax = P0 / lm
        print("  %-20s %5.3f  %.3e  %8.4f  %10.1f   %.4f       %10.1f     %6.1f"
              % (name, phi, k, reff, lm, Bn, Lmax, Lmax / KF_H))


if __name__ == "__main__":
    report_rev()
    report_dimensionless()
