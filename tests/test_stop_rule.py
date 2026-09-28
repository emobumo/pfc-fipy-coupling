# -*- coding: utf-8 -*-
"""The physical stop rule (src/analysis/stop_rule.py) on known V_in(t) records."""
from __future__ import print_function

import math
import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.stop_rule import stop_rule_physical, window_rate, q_ref


Q0, TAU = 2.0, 50.0            # Q(t) = Q0 / (1 + t/tau)^2  ->  V(t) = Q0 tau t / (tau + t)


def v_of(t):
    return Q0 * TAU * t / (TAU + t)


def first_trigger(ts, vs, qr, eps=0.1, beta=0.1):
    th, vh = [0.0], [0.0]
    for t, v in zip(ts, vs):
        th.append(t)
        vh.append(v)
        if stop_rule_physical(th, vh, qr, eps, beta):
            return t
    return None


class TestStopRule(unittest.TestCase):

    def test_window_rate_is_exact_for_a_linear_record(self):
        t = [0.0, 10.0, 20.0, 30.0]
        v = [0.0, 5.0, 10.0, 15.0]                 # Q = 0.5 throughout
        self.assertAlmostEqual(window_rate(t, v, 0.1), 0.5, places=12)

    def test_triggers_where_the_analytic_window_mean_crosses(self):
        qr, eps, beta = 1.0, 0.1, 0.1
        # analytic: [V(t) - V(0.9 t)] / (0.1 t) = eps * qr
        f = lambda t: (v_of(t) - v_of((1 - beta) * t)) / (beta * t) - eps * qr
        lo, hi = 1.0, 1.0e5
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        t_exact = hi
        ts = np.arange(0.5, 2000.0, 0.5)
        t_hit = first_trigger(ts, v_of(ts), qr, eps, beta)
        self.assertIsNotNone(t_hit)
        self.assertLessEqual(abs(t_hit - t_exact), 0.5 + 1e-9)       # one sample

    def test_a_single_spike_does_not_reset_the_rule(self):
        """A one-step spike adds its volume once; the trigger moves by a bounded
        amount instead of restarting a window as a step counter would."""
        qr = 1.0
        ts = np.arange(0.5, 3000.0, 0.5)
        vs = v_of(ts)
        base = first_trigger(ts, vs, qr)
        spiked = vs.copy()
        k = int(np.searchsorted(ts, 0.8 * base))
        spiked[k:] += 0.05 * 0.1 * qr * 0.1 * ts[k]    # 5% of one window's budget
        hit = first_trigger(ts, spiked, qr)
        self.assertIsNotNone(hit)
        self.assertLess(hit - base, 0.15 * base)

    def test_no_false_trigger_early(self):
        qr = 1.0
        self.assertFalse(stop_rule_physical([0.0], [0.0], qr))
        ts = np.arange(0.5, 20.0, 0.5)                # Q/qr ~ 2 early on
        self.assertIsNone(first_trigger(ts, v_of(ts), qr))

    def test_a_pass_that_takes_nothing_stops_at_once(self):
        """A buried pass (all source cells cemented) injects ~0 and must stop at
        its first step, not after a step-count window."""
        self.assertEqual(first_trigger([5.0, 10.0, 15.0], [1e-12, 2e-12, 3e-12], 1.0), 5.0)

    def test_q_ref_matches_the_closed_form(self):
        a, p0, mu = 1.25e-7, 5.0e6, 0.15
        self.assertAlmostEqual(q_ref(0.18, a, p0, mu) / (a * 0.18 ** 3 / 0.82 ** 2 * p0 / mu), 1.0, places=12)
        self.assertAlmostEqual(q_ref(0.18, a, p0, mu), 3.6139e-2, places=5)


if __name__ == "__main__":
    unittest.main()
