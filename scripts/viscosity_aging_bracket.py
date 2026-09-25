# -*- coding: utf-8 -*-
"""
Bracket the effect of time-dependent grout viscosity on every finished case,
without a time-dependent model.

    powershell -File scripts\run_local.ps1 scripts\viscosity_aging_bracket.py

WHY THIS IS EXACT FOR ONE END OF THE BRACKET. Every flux in the model is
proportional to 1/mu_p (q = -(k k_r / mu_p) max(0, 1 - lambda/|grad Phi|) grad Phi,
and lambda depends on tau0, not mu_p). If mu_p varies in time but not in space,
mu_p(t) = mu0 f(t), the whole system is the constant-viscosity system run on a
rescaled clock

    d tau = dt / f(t).

For exponential thickening f = exp(t/T) this gives tau = T (1 - exp(-t/T)) < T:
the time-dependent run is the constant-viscosity run STOPPED at tau = T, and
reaching model time tau takes real time t = -T ln(1 - tau/T).

WHY IT IS A BRACKET, NOT A MODEL. Pumping is continuous, so the grout in the
domain has a spread of ages a(x, t) <= t. A global mu(t) gives every parcel the
oldest possible age: it is the SLOW end. Constant viscosity (every parcel
fresh) is the FAST end. The real state is expected between them -- expected,
not proven: in 2D with non-uniform ages the flow can redistribute, so no
pointwise comparison theorem is claimed. A conclusion that holds at both ends
does not depend on the age field; one that differs is reported as a range.

The front grout was injected first and is the oldest, so the slow end is a
fair description of the front itself; what it overstates is the resistance of
the fresh column behind it.

RATE. Lv et al. (2021, Geofluids 7126013), W/C 0.5, 20 C, apparent viscosity
fit mu(t) = 869.75 exp(0.01350 t) [mPa s, t in min], measured 0-40 min only.
T = 1/0.01350 min = 74 min. Caveats carried into the report: apparent (not
plastic) viscosity; tau0 held constant, as in that paper; the fit is
extrapolated past 40 min; initial set ~4 h ends the meaning of any of this.

Each stage of a staged sequence starts with fresh grout, so its clock is
reset per stage.

Reads outputs/*/injection_rate.csv (+ summary.txt) and the Zhaojin
results.jsonl / final_*.npz; writes outputs/viscosity_aging/bracket.{csv,txt}.
"""
from __future__ import print_function

import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "viscosity_aging")

RATE_PER_MIN = 0.01350          # Lv et al. 2021, W/C 0.5, 20 C
T = 60.0 / RATE_PER_MIN         # s, e-folding time of the viscosity
INITIAL_SET_MIN = 240.0         # ~4 h, the brief's value for w/c 0.5
FRACTIONS = (0.5, 0.9, 0.95)


def real_minutes(tau):
    """Real time (min) at which the slow end reaches model time tau; inf if never."""
    if tau >= T:
        return float("inf")
    return -T * math.log(1.0 - tau / T) / 60.0


def fmt_min(m):
    if m == float("inf"):
        return "never"
    return "%.1f" % m


def read_summary(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path):
        if " = " in line:
            k, v = line.split(" = ", 1)
            d[k.strip()] = v.strip()
    return d


def cumulative(t, q):
    """V(t) by trapezoid from a (t, Q) record that starts at the first step."""
    t = np.concatenate([[0.0], t])
    q = np.concatenate([[q[0]], q])
    v = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))])
    return t, v


def bracket(t, v):
    """Slow-end numbers for one constant-viscosity V(t) record (stage-local clock)."""
    v_end, t_end = float(v[-1]), float(t[-1])
    row = {"t_end_model_s": t_end, "v_end": v_end,
           "t_end_real_min": real_minutes(t_end)}
    for f in FRACTIONS:
        if v_end <= 0:
            row["t%02d_real_min" % int(100 * f)] = float("nan")
            continue
        i = int(np.searchsorted(v, f * v_end))
        row["t%02d_real_min" % int(100 * f)] = real_minutes(float(t[min(i, len(t) - 1)]))
    row["v_at_T"] = float(np.interp(T, t, v)) if t_end > T else v_end
    row["v_frac_at_T"] = row["v_at_T"] / v_end if v_end > 0 else 1.0
    return row


def verdict(row):
    if row["v_end"] <= 1e-6:
        return "nothing injected at either end"
    if row["t_end_model_s"] <= T and row["t_end_real_min"] <= INITIAL_SET_MIN:
        return "same end state (reached at %s min)" % fmt_min(row["t_end_real_min"])
    if row["t_end_model_s"] <= T:
        return "same end state only after initial set (%s min); 95%% of the volume by %s min" % (
            fmt_min(row["t_end_real_min"]), fmt_min(row["t95_real_min"]))
    return "RANGE: slow end stops at %.0f%% of the volume (95%% at %s min)" % (
        100 * row["v_frac_at_T"], fmt_min(row["t95_real_min"]))


def solver_cases():
    rows = []
    paths = sorted(glob.glob(os.path.join(REPO, "outputs", "inclined_hole_channel", "*", "injection_rate.csv")) +
                   glob.glob(os.path.join(REPO, "outputs", "inclined_hole_staged", "*", "*", "injection_rate.csv")) +
                   glob.glob(os.path.join(REPO, "outputs", "fill_diagnostics", "*", "injection_rate.csv")))
    for p in paths:
        d = os.path.dirname(p)
        name = os.path.relpath(d, os.path.join(REPO, "outputs")).replace("\\", "/")
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        stage = a[:, 2].astype(int) if a.shape[1] > 2 else np.zeros(a.shape[0], dtype=int)
        summ = read_summary(os.path.join(d, "summary.txt"))
        t0 = 0.0
        for k in sorted(set(stage.tolist())):
            m = stage == k
            tl, v = cumulative(a[m, 0] - t0, a[m, 1])
            t0 = a[m, 0][-1]
            r = bracket(tl, v)
            r["case"] = name + ("" if stage.max() == 0 else " [stage %d]" % (k + 1))
            r["reason"] = summ.get("reason", "")
            v_ref = summ.get("v_in") if stage.max() == 0 else None
            r["v_check"] = float(v_ref) if v_ref else float("nan")
            rows.append(r)
    return rows


def zhaojin_cases():
    rows = []
    path = os.path.join(REPO, "outputs", "zhaojin_runs", "results.jsonl")
    if not os.path.exists(path):
        return rows
    for line in open(path):
        res = json.loads(line)
        if not res["tag"].endswith("2"):
            continue                       # quota runs are prefixes of these
        ser = np.load(os.path.join(REPO, "outputs", "zhaojin_runs", "final_%s.npz" % res["tag"]))["series"]
        t = np.concatenate([[0.0], ser[:, 0], [res["t"]]])
        v = np.concatenate([[0.0], ser[:, 1], [res["v_in"]]])
        r = bracket(t, v)
        r["case"] = "zhaojin/" + res["tag"]
        r["reason"] = res["reason"]
        r["v_check"] = float("nan")
        for key in ("reach_-56m", "reach_-63m"):
            r[key + "_real_min"] = real_minutes(res["snaps"][key]["t"]) if key in res["snaps"] else float("inf")
        if r["t_end_model_s"] > T:
            i = int(np.searchsorted(ser[:, 0], T))
            s = ser[max(i - 1, 0)]
            r["partition_at_T"] = "design %.1f / above %.1f / below %.1f" % (s[2], s[3], s[4])
        rows.append(r)
    return rows


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    rows = solver_cases() + zhaojin_cases()
    cols = ["case", "reason", "v_end", "v_check", "t_end_model_s", "t_end_real_min",
            "t50_real_min", "t90_real_min", "t95_real_min", "v_at_T", "v_frac_at_T"]
    with open(os.path.join(OUT, "bracket.csv"), "w") as h:
        h.write(",".join(cols + ["verdict"]) + "\n")
        for r in rows:
            h.write(",".join(str(r.get(c, "")) for c in cols) + "," + verdict(r) + "\n")
    lines = ["viscosity e-folding T = %.0f s (%.0f min); initial set ~%.0f min" % (T, T / 60, INITIAL_SET_MIN),
             "fast end = constant viscosity (the runs as they are); slow end = every parcel as old as the stage",
             "times below are REAL minutes at the slow end", ""]
    lines.append("%-46s %9s %8s %7s %7s %7s %8s  %s" % ("case", "V_end", "t_end", "t50", "t90", "t95", "V(T)%", "verdict"))
    for r in rows:
        lines.append("%-46s %9.2f %8s %7s %7s %7s %7.0f%%  %s" % (
            r["case"], r["v_end"], fmt_min(r["t_end_real_min"]), fmt_min(r["t50_real_min"]),
            fmt_min(r["t90_real_min"]), fmt_min(r["t95_real_min"]), 100 * r["v_frac_at_T"], verdict(r)))
        if "reach_-63m_real_min" in r:
            lines.append("%-46s reach -56 m at %s min, -63 m at %s min%s" % (
                "", fmt_min(r["reach_-56m_real_min"]), fmt_min(r["reach_-63m_real_min"]),
                ("; at tau=T: " + r["partition_at_T"]) if "partition_at_T" in r else ""))
        if r["v_check"] == r["v_check"] and abs(r["v_check"] - r["v_end"]) > 0.02 * max(r["v_check"], 1e-9):
            lines.append("%-46s WARNING: integrated V %.2f vs summary %.2f" % ("", r["v_end"], r["v_check"]))
    text = "\n".join(lines)
    open(os.path.join(OUT, "bracket.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
