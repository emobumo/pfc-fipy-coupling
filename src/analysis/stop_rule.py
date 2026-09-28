# -*- coding: utf-8 -*-
"""
Physical stop rule for a grouting pass (replaces the step-count stall windows).

    Q_bar(t) = [V_in(t) - V_in(t - T_w)] / T_w ,  T_w = beta * t
    stop when  Q_bar(t) / Q_ref < eps ,           Q_ref = k(n_ref) p0 / mu_p

t is model time since THIS pass started; V_in the volume injected in this
pass [m^3/m]; Q_ref a flux scale from the parameters [m^3/(m s)], fixed per
case family (n_ref declared by the case). Defaults beta = 0.1, eps = 0.1.

Why (2026-09-27, scripts/stop_rule_diagnostic.py): the old rules counted
STEPS ("60 steps without a new cell crossing S = 0.5"; "V_in growth over 400
steps"), and a step is a different length of model time on every mesh --
the 60-step rule amounted to eps 0.07 .. 11.7 across cases and differed by
2-13x between 2.5 and 1.25 m for the same case. The injection rate is a mesh-
independent physical quantity, like a grouting code's "rate below X for Y
minutes at the end pressure". Q_bar comes from the cumulative volume, so a
single-step spike (a cell switching on) is averaged over the window instead
of resetting a counter.

The rule is selected by the environment variable STOP_RULE: "rate"
(default) or "legacy" (the old step-count rules, to reproduce old anchors).
Cases put rate_tag(n_ref) into their checkpoint fingerprints so a run of
one rule can never resume a checkpoint of the other.
"""
import bisect
import os

RULE = os.environ.get("STOP_RULE", "rate").strip().lower()
EPS = 0.1
BETA = 0.1


def legacy():
    """True when the old step-count stall rules are selected."""
    return RULE == "legacy"


def q_ref(n_ref, a_cal, p0, mu_p):
    """Flux scale k(n_ref) p0 / mu_p with the calibrated law k = A n^3/(1-n)^2."""
    n = float(n_ref)
    return float(a_cal) * n ** 3 / (1.0 - n) ** 2 * float(p0) / float(mu_p)


def window_rate(t, v, beta=BETA):
    """Mean injection rate over [t_end - beta t_end, t_end] from the cumulative
    record (t, v) -- lists or arrays starting at the pass origin (t[0] = 0,
    v[0] = 0). Linear interpolation inside the record; O(log n)."""
    t_end = float(t[-1])
    if t_end <= 0.0 or len(t) < 2:
        return float("inf")
    tw = beta * t_end
    ta = t_end - tw
    i = bisect.bisect_right(t, ta) - 1
    i = max(0, min(i, len(t) - 2))
    t0, t1 = float(t[i]), float(t[i + 1])
    v0, v1 = float(v[i]), float(v[i + 1])
    va = v0 if t1 <= t0 else v0 + (v1 - v0) * (ta - t0) / (t1 - t0)
    return (float(v[-1]) - va) / tw


def stop_rule_physical(t, v, q_reference, eps=EPS, beta=BETA):
    """True when the window-mean rate at the last record is below eps * Q_ref.
    With only the origin recorded it never fires."""
    if len(t) < 2:
        return False
    return window_rate(t, v, beta) < eps * float(q_reference)


def rate_tag(n_ref):
    """Checkpoint-fingerprint suffix for the active rule ('' for legacy, so
    the old fingerprints are unchanged)."""
    if legacy():
        return ""
    return "|stop=rate,eps=%g,beta=%g,nref=%.6f" % (EPS, BETA, float(n_ref))
