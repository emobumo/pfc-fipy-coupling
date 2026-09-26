# -*- coding: utf-8 -*-
"""
OFFLINE real-parameter self-consistency checkup (NO FiPy, NO src import, no
numerics solver). Pure closed-form + matplotlib. Python 2.7 / 3 compatible.
KC permeability re-implemented INDEPENDENTLY here (project constant C=180).

Engineering case (Zhaoyuan gold-mine waste-rock backfill, secondary grouting):
  p0      = 5 MPa  (constant terminal injection pressure)
  L_max   ~ 12 m   (design / empirical horizontal-equivalent spread radius)
  grout   = PO 42.5 OPC single-fluid, w/c = 0.5 (1.22 t cement per m^3)
  d50     ~ 0.1 .. 0.2 m   (waste-rock block size)
  porosity zones: 1-6# ~ 0.10 ; 10-15# ~ 0.25 ; 6-10# ~ 0.45
  g = 9.81 ;  characteristic scale L ~ 12 m

KC permeability (project form, C=180, d = d50):
  k = (d^2 / 180) * phi^3 / (1-phi)^2
Yield (startup) gradient:
  lambda = 2 tau0 / sqrt(8 k / phi)
Stall:  L_max = p0 / lambda  (geometry-independent at stall: v->0, only yield).
"""
from __future__ import print_function
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "outputs", "real_param_selfcheck.png")

# --- field / design data ---------------------------------------------------
P0 = 5.0e6              # Pa, constant terminal pressure
LMAX_FIELD = 12.0       # m, empirical spread radius
G = 9.81
R0 = 0.05               # m, borehole radius (for radial geometry note)
KC_C = 180.0            # project Kozeny-Carman constant
D50_LIST = [0.10, 0.20]
PHI_ZONES = [("1-6#  (low)", 0.10),
             ("10-15# (mid)", 0.25),
             ("6-10# (high)", 0.45)]

# --- slurry density from w/c = 0.5 stoichiometry ---------------------------
# w/c = 0.5 (mass), cement specific gravity ~3.1:
#   per 1 kg cement: 0.5 kg water; V = 1/3100 + 0.5/1000 m^3; m = 1.5 kg
CEMENT_SG = 3.10
WC = 0.5
RHO_GROUT = (1.0 + WC) / (1.0 / (CEMENT_SG * 1000.0) + WC / 1000.0)   # kg/m^3
# cross-check with stated 1.22 t cement/m^3: 1220 + 0.5*1220 = 1830 kg/m^3

MU_P = 0.183            # Pa.s, project plastic viscosity (thick grout)

# --- literature yield-stress band for PO 42.5 OPC, w/c = 0.5 ---------------
TAU0_LIT_LO = 30.0      # Pa
TAU0_LIT_HI = 90.0      # Pa
TAU0_FWD = [50.0, 100.0]   # representative forward values (generous upper end)


def kc_perm(d, phi):
    return (d * d / KC_C) * phi ** 3 / (1.0 - phi) ** 2


def lam(d, phi, tau0):
    r_eff = math.sqrt(8.0 * kc_perm(d, phi) / phi)
    return 2.0 * tau0 / r_eff


def tau0_from_lambda(d, phi, lam_val):
    r_eff = math.sqrt(8.0 * kc_perm(d, phi) / phi)
    return lam_val * r_eff / 2.0


def Lmax(d, phi, tau0):
    return P0 / lam(d, phi, tau0)


# ===========================================================================
def task1():
    print("=" * 74)
    print("TASK 1: back out startup gradient lambda from L_max")
    print("=" * 74)
    lam0 = P0 / LMAX_FIELD
    lam_rad = P0 / (LMAX_FIELD - R0)
    print("  planar/first-order:  lambda = p0/L_max = %.3e Pa/m = %.1f kPa/m"
          % (lam0, lam0 / 1e3))
    print("  radial (GS) form:    I_max = r0 + p0/lambda  ->  "
          "lambda = p0/(L_max - r0)")
    print("                       = %.3e Pa/m = %.1f kPa/m   (r0=%.2f m)"
          % (lam_rad, lam_rad / 1e3, R0))
    print("  geometry correction = %.2f%%  (NEGLIGIBLE)"
          % (100.0 * (lam_rad / lam0 - 1.0)))
    print("  NOTE: at stall v->0, every velocity-dependent term (viscous,")
    print("        inertial/Forchheimer) vanishes; only the yield gradient")
    print("        lambda survives, so integral p0 = lambda*(L_max - r0) holds")
    print("        for planar AND radial AND Forchheimer. The ONLY geometric")
    print("        term is the r0 offset (here 0.05/12 = 0.4%).")
    print("  CAVEAT: 12 m is a finite-rate OPERATING radius (injection stopped")
    print("          on time/volume/pressure criteria), not the t->inf stall;")
    print("          true I_max >= 12 m, so this lambda is an UPPER estimate")
    print("          (=> tau0 back-out below is an upper estimate too).")
    return lam0


def task2(lam0):
    print("")
    print("=" * 74)
    print("TASK 2: back out tau0 per zone (self-consistency: should be uniform)")
    print("=" * 74)
    print("  lambda = %.3e Pa/m (from task 1)" % lam0)
    print("  literature tau0 (PO42.5, w/c=0.5): %.0f .. %.0f Pa" % (TAU0_LIT_LO, TAU0_LIT_HI))
    print("")
    print("  zone           phi    d50[m]   k_KC[m^2]    r_eff[m]   tau0_backout[Pa]")
    results = []
    for name, phi in PHI_ZONES:
        for d in D50_LIST:
            k = kc_perm(d, phi)
            reff = math.sqrt(8.0 * k / phi)
            t0 = tau0_from_lambda(d, phi, lam0)
            results.append((name, phi, d, k, t0))
            print("  %-13s %.2f   %.2f    %.3e   %.3e   %10.1f"
                  % (name, phi, d, k, reff, t0))
    t0vals = [r[4] for r in results]
    print("")
    print("  tau0 back-out range: %.0f .. %.0f Pa  (spread factor %.1fx)"
          % (min(t0vals), max(t0vals), max(t0vals) / min(t0vals)))
    print("  => NOT uniform across zones, and %.0f-%.0fx ABOVE the literature"
          % (min(t0vals) / TAU0_LIT_HI, max(t0vals) / TAU0_LIT_LO))
    print("     band (%g-%g Pa). Systematic trend: tau0_backout grows with phi"
          % (TAU0_LIT_LO, TAU0_LIT_HI))
    print("     and with d (k up -> lambda needs bigger tau0). FAILS self-consistency.")
    return results


def task3():
    print("")
    print("=" * 74)
    print("TASK 3: forward self-consistency -- fix tau0, compare zones")
    print("=" * 74)
    d = 0.15
    print("  fixed d50 = %.2f m (mid); volume scalings vs lowest-phi zone" % d)
    for tau0 in TAU0_FWD:
        print("")
        print("  --- tau0 = %.0f Pa (literature representative) ---" % tau0)
        print("  zone           phi    lambda[kPa/m]   L_max[m]   L_max/12   "
              "V2D~phi*L^2   V3D~phi*L^3")
        base2d = base3d = None
        for name, phi in PHI_ZONES:
            lm = lam(d, phi, tau0)
            L = P0 / lm
            v2d = phi * L ** 2
            v3d = phi * L ** 3
            if base2d is None:
                base2d, base3d = v2d, v3d
            print("  %-13s %.2f   %10.1f   %8.1f   %7.1f   %.3e   %.3e"
                  % (name, phi, lm / 1e3, L, L / LMAX_FIELD, v2d, v3d))
        # ratios high/low
        L_lo = P0 / lam(d, PHI_ZONES[0][1], tau0)
        L_hi = P0 / lam(d, PHI_ZONES[-1][1], tau0)
        v2d_lo = PHI_ZONES[0][1] * L_lo ** 2
        v2d_hi = PHI_ZONES[-1][1] * L_hi ** 2
        v3d_lo = PHI_ZONES[0][1] * L_lo ** 3
        v3d_hi = PHI_ZONES[-1][1] * L_hi ** 3
        print("  high(0.45)/low(0.10):  L_max %.1fx ;  V_2D %.0fx ;  V_3D %.0fx"
              % (L_hi / L_lo, v2d_hi / v2d_lo, v3d_hi / v3d_lo))
    print("")
    print("  L_max scaling: lambda ~ tau0*(1-phi)/(d*phi) -> L_max ~ phi/(1-phi).")
    print("  grout intake V ~ phi*L_max^n combines pore-volume AND reach:")
    print("  V_2D ~ phi^3/(1-phi)^2 (= exactly k_KC/d^2 shape). The high-phi")
    print("  zone takes ~2 orders more grout than the low-phi zone for the SAME")
    print("  pressure -> qualitatively explains why 6-10# (45%) took the most.")


def task4(lam0, t0_backout_mid):
    print("")
    print("=" * 74)
    print("TASK 4: dimensionless numbers (real params)")
    print("=" * 74)
    phi, d, L = 0.25, 0.15, LMAX_FIELD
    k = kc_perm(d, phi)
    print("  representative: phi=%.2f, d50=%.2f m, p0=%.1f MPa, L=%.0f m, "
          "rho_grout=%.0f kg/m^3" % (phi, d, P0 / 1e6, L, RHO_GROUT))
    print("  (w/c=0.5 stoichiometry: rho = %.0f kg/m^3 = %.2f g/cm^3; matches"
          % (RHO_GROUT, RHO_GROUT / 1000.0))
    print("   stated 1.22 t cement/m^3 -> 1220+610 = 1830 kg/m^3)")
    print("")

    # Bn with calibrated tau0 (task 2 mid) and with literature tau0
    lam_cal = lam0                       # by construction L_max=12 -> Bn=L/L_max
    Bn_cal = lam_cal * L / P0
    print("  -- Bingham number Bn = lambda*L/p0 --")
    print("  (A) calibrated tau0=%.0f Pa (task2, phi=.25,d=.15): lambda=%.0f -> "
          "Bn=%.3f" % (t0_backout_mid, lam_cal, Bn_cal))
    print("      (Bn=1 BY CONSTRUCTION: L=L_max forces it; tau0 unphysical)")
    for tau0 in TAU0_FWD:
        lm = lam(d, phi, tau0)
        Bn = lm * L / P0
        Lp = P0 / lm
        print("  (B) literature tau0=%3.0f Pa: lambda=%.0f Pa/m -> Bn=%.4f  "
              "(predicts L_max=%.0f m)" % (tau0, lm, Bn, Lp))
    print("      => with PHYSICAL tau0, Bn<<1: yield NOT the dominant stop.")
    print("")

    # gravity number
    Gn = RHO_GROUT * G * L / P0
    print("  -- gravity number G = rho*g*L/p0 = %.4f --" % Gn)
    print("     hydrostatic head rho*g*L = %.0f kPa = %.1f%% of p0 (gravity"
          % (RHO_GROUT * G * L / 1e3, 100.0 * Gn))
    print("     negligible; pressure dominates ~%.0fx)" % (1.0 / Gn))
    print("")

    # pore Reynolds
    q_darcy = (k / MU_P) * (P0 / L)          # Darcy-implied superficial velocity
    Re_darcy = RHO_GROUT * q_darcy * d / MU_P
    v0 = 0.05
    Re_v0 = RHO_GROUT * v0 * d / MU_P
    print("  -- pore Reynolds Re_p = rho*q*d/mu_p --")
    print("  k_KC(phi=.25,d=.15) = %.3e m^2 (= %.2e Darcy: GRAVEL/ROCKFILL scale)"
          % (k, k / 9.87e-13))
    print("  Darcy-IMPLIED superficial q=(k/mu)(p0/L) = %.2f m/s (!!) -> "
          "Re_p=%.0f" % (q_darcy, Re_darcy))
    print("     q~%.0f m/s is physically impossible: Darcy is SELF-CONTRADICTORY"
          % q_darcy)
    print("     here (if it held, flow would be wildly turbulent).")
    print("  realistic injection q=%.2f m/s -> Re_p=%.0f  (still >>1)"
          % (v0, Re_v0))
    print("  => Darcy INVALID near well; Forchheimer/turbulent mandatory.")
    print("     BUT stall L_max=p0/lambda is unchanged (yield only at v->0).")
    print("")
    print("  -- vs OLD result (tau0=4.6 Pa, p0=2 MPa, H=40 m): Bn~0.02 --")
    print("  new (physical tau0~50-100): Bn~%.2f-%.2f (still <<1, same verdict)"
          % (lam(d, phi, 50.0) * L / P0, lam(d, phi, 100.0) * L / P0))
    print("  new (forced to L=12 m):    Bn=1 only by inflating tau0 to ~%.0f Pa"
          % t0_backout_mid)


def make_figure(lam0):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: back-calculated tau0 per zone vs literature band
    ax = axes[0]
    labels = []
    vals = []
    colors = []
    cmap = {0.10: "#2c7bb6", 0.25: "#fdae61", 0.45: "#d7191c"}
    for name, phi in PHI_ZONES:
        for d in D50_LIST:
            t0 = tau0_from_lambda(d, phi, lam0)
            labels.append("phi=%.2f\nd=%.1f" % (phi, d))
            vals.append(t0)
            colors.append(cmap[phi])
    xpos = np.arange(len(vals))
    ax.bar(xpos, vals, color=colors, alpha=0.85)
    ax.axhspan(TAU0_LIT_LO, TAU0_LIT_HI, color="green", alpha=0.25,
               label="literature tau0 (w/c=0.5): %g-%g Pa" % (TAU0_LIT_LO, TAU0_LIT_HI))
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("back-calculated tau0 [Pa]  (log)")
    ax.set_title("TASK 2: tau0 reverse-solved from L_max=12 m\n"
                 "(should be a single grout value; spans %.0f-%.0f Pa, "
                 "1-2 orders above lit.)"
                 % (min(vals), max(vals)))
    for x, v in zip(xpos, vals):
        ax.text(x, v * 1.05, "%.0f" % v, ha="center", fontsize=7)
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: forward L_max vs phi for fixed tau0; mark 12 m and zones
    ax = axes[1]
    phis = np.linspace(0.05, 0.5, 200)
    d = 0.15
    for tau0, c in zip(TAU0_FWD, ["green", "red"]):
        L = np.array([P0 / lam(d, p, tau0) for p in phis])
        ax.plot(phis, L, color=c, lw=2.0,
                label="tau0=%g Pa (lit.), d=0.15 m" % tau0)
    ax.axhline(LMAX_FIELD, color="k", ls="--", lw=1.5, label="field L_max=12 m")
    for name, phi in PHI_ZONES:
        ax.axvline(phi, color="gray", ls=":", lw=0.8)
        ax.text(phi, 1.3, name.split()[0], rotation=90, fontsize=7,
                color="gray", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("porosity phi")
    ax.set_ylabel("predicted yield-stall L_max [m]  (log)")
    ax.set_title("TASK 3: forward L_max(phi) at literature tau0\n"
                 "(all zones >> 12 m: KC k too high -> yield can't stop at 12 m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(1.0, 5.0e3)

    fig.suptitle("Real-parameter self-consistency: PO42.5 w/c=0.5 grout, "
                 "p0=5 MPa, waste rock d50=0.1-0.2 m", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("")
    print("wrote %s" % os.path.abspath(OUT))


def main():
    print("rho_grout (w/c=0.5, cement SG=%.2f) = %.0f kg/m^3 = %.3f g/cm^3"
          % (CEMENT_SG, RHO_GROUT, RHO_GROUT / 1000.0))
    lam0 = task1()
    results = task2(lam0)
    # mid representative back-out tau0 (phi=0.25, d=0.15) for task 4
    t0_mid = tau0_from_lambda(0.15, 0.25, lam0)
    task3()
    task4(lam0, t0_mid)
    make_figure(lam0)


if __name__ == "__main__":
    main()
