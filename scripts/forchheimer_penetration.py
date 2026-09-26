# -*- coding: utf-8 -*-
"""
OFFLINE analytical Forchheimer-Bingham grout-penetration estimate (NO FiPy,
NO src import, no numerics solver). Pure closed-form + bisection + matplotlib.
Python 2.7 / 3 compatible.

Pressure-gradient law (Ergun viscous + Ergun/Forchheimer inertial + Bingham
yield), radial steady, flux conserved (v(r) = v0*r0/r):
    |dp/dr| = lambda + (mu_p/k)*v + rho*B*v^2
    viscous  (mu_p/k)   with k = k_Ergun = phi^3 d^2 / (150 (1-phi)^2)
    inertial  B = 1.75 (1-phi) / (phi^3 d)      [full Ergun inertial term]
    yield     lambda = 2 tau0 / sqrt(8 k / phi)

KEY PHYSICS (reported, not hidden): under CONSTANT injection pressure the
yield-stall distance is L_max = p0/lambda for BOTH Darcy and Forchheimer,
because at stall v -> 0 and both the viscous and inertial terms vanish,
leaving only the yield term. Forchheimer changes the TRANSIENT (and the
finite-rate / pressure-capped operating penetration), not the asymptotic
stall. We therefore plot:
  - L_max = p0/lambda            (yield stall; Darcy == Forchheimer)
  - R_oper(v0)                   (radial penetration at a representative
                                  borehole Darcy velocity v0 where the pump
                                  pressure p0 is exhausted -- here inertia
                                  DOES shorten the reach)
Fixed: phi=0.157, rho=1414, mu_p=0.183, p0=2 MPa, g=9.81, r0=0.05 m.
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "outputs", "forchheimer_penetration.png")

PHI = 0.157
RHO = 1414.0
MU = 0.183
P0 = 2.0e6
G = 9.81
R0 = 0.05               # borehole radius [m]
KC_C = 180.0            # project Kozeny-Carman constant
ERGUN_C = 150.0         # Ergun viscous constant (Blake-Kozeny)
ERGUN_I = 1.75          # Ergun inertial constant (Burke-Plummer)
TAU0_LIST = [4.6, 50.0, 100.0]
V0_FIG = 0.05           # representative borehole Darcy velocity for the figure
H_LINES = [40.0, 10.0, 3.0]
REAL_D50 = [0.10, 0.20, 0.255]


def k_ergun(d):
    return PHI ** 3 * d ** 2 / (ERGUN_C * (1.0 - PHI) ** 2)


def k_kc(d):
    return PHI ** 3 * d ** 2 / (KC_C * (1.0 - PHI) ** 2)


def lam(d, tau0):
    return 2.0 * tau0 / np.sqrt(8.0 * k_ergun(d) / PHI)


def L_max(d, tau0):
    return P0 / lam(d, tau0)


def beta_inertial(d):
    return ERGUN_I * (1.0 - PHI) / (PHI ** 3 * d)


def budget(R, d, tau0, v0, with_inertial):
    """Cumulative |dp/dr| integral r0->R, radial, v(r)=v0*r0/r."""
    k = k_ergun(d)
    qf = v0 * R0                       # = Q/(2 pi); v(r) = qf / r
    visc = (MU / k) * qf * np.log(R / R0)
    yield_ = tau0 and lam(d, tau0) * (R - R0)
    total = lam(d, tau0) * (R - R0) + visc
    if with_inertial:
        total = total + RHO * beta_inertial(d) * qf ** 2 * (1.0 / R0 - 1.0 / R)
    return total


def operating_R(d, tau0, v0, with_inertial):
    """Radius where the budget integral consumes p0 (front, p=0)."""
    lo = R0
    hi = R0 + P0 / lam(d, tau0)        # yield-only stall is the upper bound
    # budget is monotone increasing in R; bisection.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if budget(mid, d, tau0, v0, with_inertial) < P0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    d = np.logspace(np.log10(0.01), np.log10(0.3), 300)

    print("=== Ergun vs KC viscous term (equivalent k) ===")
    print("k_Ergun/k_KC = C_KC/C_Ergun = %.1f/%.1f = %.3f  (Ergun k is %.0f%% higher)"
          % (KC_C, ERGUN_C, KC_C / ERGUN_C, 100.0 * (KC_C / ERGUN_C - 1.0)))
    for dd in REAL_D50:
        print("  d50=%.3f m: k_KC=%.3e  k_Ergun=%.3e m^2" % (dd, k_kc(dd), k_ergun(dd)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {4.6: "blue", 50.0: "green", 100.0: "red"}
    for tau0 in TAU0_LIST:
        Lm = L_max(d, tau0)
        Rf = np.array([operating_R(dd, tau0, V0_FIG, True) for dd in d])
        c = colors[tau0]
        ax.plot(d, Lm, color=c, lw=2.2, ls="-",
                label="L_max=p0/lambda (yield stall, D=F), tau0=%g Pa" % tau0)
        ax.plot(d, Rf, color=c, lw=1.6, ls="--",
                label="R_oper Forchheimer @v0=%g m/s, tau0=%g Pa" % (V0_FIG, tau0))

    for h in H_LINES:
        ax.axhline(h, color="gray", lw=0.8, ls=":")
        ax.text(d[0], h * 1.05, "H=%g m" % h, color="gray", fontsize=8)

    for dd in REAL_D50:
        ax.axvline(dd, color="k", lw=0.5, alpha=0.25)
    ax.text(0.16, 4000, "real waste rock\n0.1-0.255 m", fontsize=8, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d50 [m]  (gravel -> coarse waste rock)")
    ax.set_ylabel("penetration distance [m]")
    ax.set_title("Forchheimer-Bingham grout penetration vs grain size\n"
                 "(phi=%.3f, p0=%.1f MPa, mu_p=%.3f, rho=%.0f; solid=yield stall "
                 "p0/lambda, dashed=operating @v0=%g m/s)"
                 % (PHI, P0 / 1e6, MU, RHO, V0_FIG))
    ax.set_ylim(1.0, 1.0e4)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print("")
    print("=== L_max = p0/lambda (yield stall; SAME for Darcy & Forchheimer) ===")
    print(" d50[m]  tau0=4.6   tau0=50    tau0=100   [m]   (vs H=40/10/3)")
    for dd in REAL_D50:
        row = [L_max(dd, t) for t in TAU0_LIST]
        print("  %.3f   %8.1f  %8.1f  %8.1f" % (dd, row[0], row[1], row[2]))

    print("")
    print("=== operating penetration R (radial, pressure exhausted at p0) ===")
    print("for each (d50, tau0): R_Darcy / R_Forchheimer [m] at v0 in {0.01,0.05,0.1} m/s")
    for dd in REAL_D50:
        for tau0 in TAU0_LIST:
            parts = []
            for v0 in (0.01, 0.05, 0.1):
                rd = operating_R(dd, tau0, v0, False)
                rf = operating_R(dd, tau0, v0, True)
                parts.append("v0=%.2f: %.1f/%.1f" % (v0, rd, rf))
            print("  d50=%.3f tau0=%6.1f  Lmax=%.1f | %s"
                  % (dd, tau0, L_max(dd, tau0), "  ".join(parts)))

    # term decomposition at a representative operating point
    print("")
    print("=== pressure-budget term split at borehole (d50=0.255, v0=0.05) ===")
    dd, v0 = 0.255, 0.05
    k = k_ergun(dd)
    visc_grad = MU / k * v0
    inert_grad = RHO * beta_inertial(dd) * v0 ** 2
    for tau0 in TAU0_LIST:
        lm = lam(dd, tau0)
        tot = lm + visc_grad + inert_grad
        print("  tau0=%6.1f: lambda=%.0f  viscous=%.0f  inertial=%.0f Pa/m  "
              "(yield %.0f%%, visc %.0f%%, inert %.0f%%)"
              % (tau0, lm, visc_grad, inert_grad,
                 100 * lm / tot, 100 * visc_grad / tot, 100 * inert_grad / tot))
    print("")
    print("wrote %s" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
