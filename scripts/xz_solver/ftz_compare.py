# -*- coding: utf-8 -*-
"""Before enabling the subnormal flush in S2 (main line, 2026-10-18): from the same S2 checkpoint, N steps as is and
N steps with ftz.py active; report the largest saturation difference, the V_in difference and the wall time per
step of both. No checkpoint is written. Appends to outputs/xz_solver/ftz_compare.txt.

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\ftz_compare.py 30 S2
"""
from __future__ import print_function

import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
FZ = R.FZ
from src.analysis.run_checkpoint import load as ck_load, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport import equations as EQ


def run(tag, nsteps, flush, D, C, srcs, box):
    p0 = dict((c[0], c[3]) for c in R.CASES)[tag]
    st, src, _ = R.build(C["n"], srcs[tag], box, p0)
    fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % (tag, p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(R.N_REF)
    res = ck_load(os.path.join(R.OUT, "ckpt_" + tag), st, fingerprint=state_fingerprint(st, fp))
    vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
    hm = np.zeros(vols.size, bool)
    hm[src] = True
    v_in, t, walls = float(res["v_in"]), float(res["t"]), []
    FZ.STATE.update({"calls": 0, "diag_subnormal": 0, "diag_steps": [], "flushed": 0})
    FZ.set_active(flush)
    for k in range(nsteps):
        FZ.STATE["step"] = int(res["step"]) + k + 1
        w0 = time.time()
        dt = EQ.solve_transport_step(st, dt_cap=R.DT_CAP)
        walls.append(time.time() - w0)
        t += float(dt)
        dq = np.asarray(st["last_div_q"], dtype=float)
        v_in += float(np.sum(dq[hm] * vols[hm])) * float(dt)
        s = np.array(st["saturation"].value, copy=True)
        s[hm] = 1.0
        st["saturation"].setValue(s)
    FZ.set_active(False)
    return (int(res["step"]), np.asarray(st["saturation"].value, float).copy(), v_in, t, np.array(walls),
            dict(FZ.STATE))


def main():
    nsteps, tag = int(sys.argv[1]), sys.argv[2]
    D, C, srcs, collar_t = R.load()
    box = R.crop_box(C)
    s0, sa, va, ta, wa, _ = run(tag, nsteps, False, D, C, srcs, box)
    _, sb, vb, tb, wb, fz = run(tag, nsteps, True, D, C, srcs, box)
    line = ("%s from step %d, %d steps (%s): max |S_flush - S_asis| %.3e (cells differing %d); V_in as is %.9f, flush"
            " %.9f, difference %.3e m3/m; model time difference %.3e s; wall per step as is %.3f s, flush %.3f s"
            " (x%.2f); flush: CG calls %d, off-diagonal entries zeroed %d, subnormal diagonal entries %d (steps %s)"
            % (tag, s0, nsteps, time.strftime("%Y-%m-%d %H:%M"), np.abs(sb - sa).max(), int((sb != sa).sum()), va, vb,
               vb - va, tb - ta, wa.mean(), wb.mean(), wa.mean() / wb.mean(), fz["calls"], fz["flushed"],
               fz["diag_subnormal"], fz["diag_steps"][:10]))
    open(os.path.join(R.OUT, "ftz_compare.txt"), "a").write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
