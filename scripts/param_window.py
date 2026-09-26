# -*- coding: utf-8 -*-
"""
OFFLINE analytical parameter-window sweep (NO FiPy, NO src import, no numerics).
Pure closed-form evaluation + matplotlib. Python 2.7 / 3 compatible.

Maps, on the (d50, tau0) plane, the Bingham number Bn and three physical
boundaries, to find the window where grouting is simultaneously
  (i) yield-dominated  (Bn >= 1),
  (ii) still Darcy      (pore Reynolds Re_p <= 1),
  (iii) injectable      (groutability ratio d_pore/d85 >= 25).

Fixed: phi=0.157, p0=2 MPa, H=40 m, rho=1414, mu_p=0.183, g=9.81.
KC permeability uses the SAME formula as src (k = d^2/180 * phi^3/(1-phi)^2),
re-implemented here independently.

Run: powershell -File scripts\\run_local.ps1 scripts\\param_window.py
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(_HERE), "outputs", "param_window.png")

# --- fixed parameters ------------------------------------------------------
PHI = 0.157
P0 = 2.0e6
H = 40.0
RHO = 1414.0
MU = 0.183
G = 9.81
KC_C = 180.0          # Kozeny-Carman constant (src: kozeny_carman_constant)
PORE_FRAC = 0.18      # pore-throat size ~ 0.18 * d50
D85_ORD = 50.0e-6     # ordinary Portland cement d85 [m]
D85_MICRO = 10.0e-6   # microfine cement d85 [m]
GROUT_MIN = 25.0      # groutability ratio threshold (injectable if >=)

# model1 point
M1_D50 = 0.255
M1_TAU0 = 4.6


# --- closed-form physics (all per the task spec) ---------------------------
def perm(d50):
    return (d50 ** 2 / KC_C) * PHI ** 3 / (1.0 - PHI) ** 2


def lam(d50, tau0):
    r_eff = np.sqrt(8.0 * perm(d50) / PHI)
    return 2.0 * tau0 / r_eff


def bingham(d50, tau0):
    return lam(d50, tau0) * H / P0


def reynolds_p(d50):
    k = perm(d50)
    q = (k / MU) * (P0 / H)      # representative near-well Darcy flux
    v = q / PHI                  # pore velocity
    return RHO * v * d50 / MU


def groutability(d50, d85):
    return (PORE_FRAC * d50) / d85


def solve_d50_for_Re(target):
    # Re_p propto d50^3 -> invert on a fine log grid.
    dd = np.logspace(-4, 0, 200000)
    rr = reynolds_p(dd)
    return float(np.interp(target, rr, dd))


def tau0_for_Bn1(d50):
    # Bn = lam*H/p0 = 1 -> lam = p0/H -> tau0 = (p0/H)*r_eff/2
    r_eff = np.sqrt(8.0 * perm(d50) / PHI)
    return (P0 / H) * r_eff / 2.0


def main():
    d50 = np.logspace(np.log10(0.002), np.log10(0.3), 400)
    tau0 = np.logspace(np.log10(1.0), np.log10(500.0), 400)
    D, T = np.meshgrid(d50, tau0)
    Bn = bingham(D, T)

    # boundary crossings (Re_p and groutability are tau0-independent: vertical)
    d_Re1 = solve_d50_for_Re(1.0)
    d_Re10 = solve_d50_for_Re(10.0)
    d_grout_ord = GROUT_MIN * D85_ORD / PORE_FRAC
    d_grout_micro = GROUT_MIN * D85_MICRO / PORE_FRAC

    fig, ax = plt.subplots(figsize=(9, 7))

    # Bn filled background (log) + labeled contours
    cf = ax.contourf(D, T, np.log10(Bn), levels=np.linspace(-2, 2, 17),
                     cmap="RdYlGn", alpha=0.65)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("log10(Bn)")
    cs = ax.contour(D, T, Bn, levels=[0.1, 0.5, 1.0, 2.0, 5.0],
                    colors="k", linewidths=[0.6, 0.6, 2.2, 0.6, 0.6])
    ax.clabel(cs, fmt="Bn=%g", fontsize=8)

    # Re_p vertical lines
    ax.axvline(d_Re1, color="purple", lw=2.0, ls="-")
    ax.axvline(d_Re10, color="purple", lw=1.2, ls="--")
    ax.text(d_Re1, 520, "Re_p=1", color="purple", fontsize=8, ha="center")
    ax.text(d_Re10, 520, "Re_p=10", color="purple", fontsize=8, ha="center")

    # groutability vertical lines
    ax.axvline(d_grout_ord, color="brown", lw=2.0, ls="-")
    ax.axvline(d_grout_micro, color="brown", lw=1.2, ls="--")
    ax.text(d_grout_ord, 1.15, "grout=25\n(d85=50um)", color="brown",
            fontsize=7, ha="center", va="bottom")
    ax.text(d_grout_micro, 1.15, "grout=25\n(d85=10um)", color="brown",
            fontsize=7, ha="center", va="bottom")

    # the yield+Darcy+groutable(ordinary) window: shade it
    mask = (Bn >= 1.0) & (D <= d_Re1) & (D >= d_grout_ord)
    ax.contourf(D, T, mask.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["xxx"])
    ax.contour(D, T, mask.astype(float), levels=[0.5], colors="blue", linewidths=2.0)

    # model1 point
    m1_bn = bingham(M1_D50, M1_TAU0)
    ax.plot([M1_D50], [M1_TAU0], marker="*", ms=20, color="black",
            markeredgecolor="white")
    ax.annotate("model1\n(d50=0.255, tau0=4.6)\nBn=%.3f, Re_p=%.0f" %
                (m1_bn, reynolds_p(M1_D50)),
                xy=(M1_D50, M1_TAU0), xytext=(0.05, 2.5),
                fontsize=8, color="black",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d50 [m]  (mid sand 2 mm  ->  coarse waste rock 300 mm)")
    ax.set_ylabel("tau0 [Pa]  (thin slurry 1  ->  thick cement 500)")
    ax.set_title("Bingham grouting parameter window  "
                 "(phi=%.3f, p0=%.1f MPa, H=%.0f m, mu_p=%.3f Pa.s)\n"
                 "blue box = yield(Bn>=1) AND Darcy(Re_p<=1) AND "
                 "groutable(ord. cement)" % (PHI, P0 / 1e6, H, MU))
    ax.set_xlim(d50[0], d50[-1])
    ax.set_ylim(tau0[0], tau0[-1])
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    # ------- numeric summary -------
    print("=== boundary crossings (d50) ===")
    print("Re_p=1   at d50 = %.4f m (%.1f mm)" % (d_Re1, d_Re1 * 1e3))
    print("Re_p=10  at d50 = %.4f m (%.1f mm)" % (d_Re10, d_Re10 * 1e3))
    print("grout=25 ordinary (d85=50um) at d50 = %.4f m (%.1f mm)"
          % (d_grout_ord, d_grout_ord * 1e3))
    print("grout=25 microfine (d85=10um) at d50 = %.4f m (%.1f mm)"
          % (d_grout_micro, d_grout_micro * 1e3))
    print("")
    print("=== window (yield+Darcy+groutable) d50 band & Bn=1 tau0 ===")
    print("ordinary cement: d50 in [%.1f, %.1f] mm  (factor %.2f, %.2f decade)"
          % (d_grout_ord * 1e3, d_Re1 * 1e3, d_Re1 / d_grout_ord,
             np.log10(d_Re1 / d_grout_ord)))
    print("microfine cement: d50 in [%.1f, %.1f] mm (factor %.2f, %.2f decade)"
          % (d_grout_micro * 1e3, d_Re1 * 1e3, d_Re1 / d_grout_micro,
             np.log10(d_Re1 / d_grout_micro)))
    print("tau0 for Bn=1 at band edges: at %.1f mm -> %.1f Pa ; at %.1f mm -> %.1f Pa"
          % (d_grout_ord * 1e3, tau0_for_Bn1(d_grout_ord),
             d_Re1 * 1e3, tau0_for_Bn1(d_Re1)))
    print("")
    print("=== example points inside window ===")
    for (dd, tt, mat) in [
        (0.010, 50.0, "fine gravel + thick cement grout"),
        (0.010, 20.0, "fine gravel + medium grout"),
        (0.004, 30.0, "coarse sand + microfine grout"),
    ]:
        print("  d50=%5.1f mm tau0=%5.1f Pa: Bn=%.2f Re_p=%.2f grout_ord=%.0f grout_micro=%.0f L_max=%.0f m"
              % (dd * 1e3, tt, bingham(dd, tt), reynolds_p(dd),
                 groutability(dd, D85_ORD), groutability(dd, D85_MICRO),
                 P0 / lam(dd, tt)))
    print("")
    print("=== model1 point ===")
    print("d50=%.3f m tau0=%.1f Pa: Bn=%.3f  Re_p=%.0f  grout_ord=%.0f  L_max=%.0f m (%.0fx H)"
          % (M1_D50, M1_TAU0, bingham(M1_D50, M1_TAU0), reynolds_p(M1_D50),
             groutability(M1_D50, D85_ORD), P0 / lam(M1_D50, M1_TAU0),
             (P0 / lam(M1_D50, M1_TAU0)) / H))
    print("  d50 must drop from %.0f mm to <= %.1f mm (Re_p=1): factor %.0f (%.2f decade)"
          % (M1_D50 * 1e3, d_Re1 * 1e3, M1_D50 / d_Re1, np.log10(M1_D50 / d_Re1)))
    print("  tau0 must rise from %.1f to >= %.1f Pa (Bn=1 at that d50)"
          % (M1_TAU0, tau0_for_Bn1(d_Re1)))
    print("")
    print("wrote %s" % os.path.abspath(OUT))


if __name__ == "__main__":
    main()
