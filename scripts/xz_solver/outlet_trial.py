# -*- coding: utf-8 -*-
"""Feasibility of a drawpoint outflow boundary (main line 2026-10-20), scripts only, no change to src/.

    $env:MKL_NUM_THREADS=1; $env:OMP_NUM_THREADS=1
    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\outlet_trial.py [--steps 500] [--case S4]

Idea: the fill closure in src (equations._update_fill_mask / _fill_penalty_beta) already pins every cell with
S < saturation_active_threshold at air pressure ~0 by a strong implicit sink, and books its net inflow into its
saturation (update_saturation_from_flux). So a cell that is kept EMPTY is a pressure-0 outflow cell. The outlet
cells (the lower end of the basal layer on the 0.5 m grid) are reset to S = 0 after every step; what flowed into
them during the step leaves the model and is booked as outflow, from the same explicit flux field the solver used
(-div q * V * dt), and cross-checked against the saturation it had gained (n S V before the reset; equal unless the
explicit update clipped at S = 1, which is booked as clip).

Run: same case, settings and grid as run_cases.py (40 m end state by default: S4, drift 1 -30 deg, 2 MPa), one
thread, subnormal flush from step 1, no stop rule. Mass balance per step and cumulated:
    V_in (source cells) = dStore (n S V of every non-source, non-outlet cell) + V_comp (active non-source cells)
                          + V_out + V_clip
reported with the main line's measure |V_in - (dStore + V_out)| / V_in and the full ledger. Until the grout reaches
the outlet the run must equal run_cases.py's case bit for bit (empty outlet cells are pinned at 0 there too), which
is checked against final_<case>.npz.
Output: outputs/xz_solver/outlet_trial_<case>_<steps>.txt (configuration suffix applies).
"""
from __future__ import print_function

import ctypes
import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RC = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
FZ = RC.FZ
from src.models.slurry_transport.equations import solve_transport_step


def peak_mb():
    """Peak working set and peak pagefile (private) usage of this process, MB (Windows)."""
    try:
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                     "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                     "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2.0 ** 20, c.PeakPagefileUsage / 2.0 ** 20
    except Exception:
        return float("nan"), float("nan")


def main():
    a = sys.argv[1:]
    nsteps = int(a[a.index("--steps") + 1]) if "--steps" in a else 500
    tag = a[a.index("--case") + 1] if "--case" in a else "S4"
    case = [c_ for c_ in RC.CASES if c_[0] == tag][0]
    p0 = case[3]
    D, C, srcs, collar_t = RC.load()
    box = RC.crop_box(C)
    j0, j1, i0, i1 = box
    st, src, _ = RC.build(C["n"], srcs[tag], box, p0)
    vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
    n = np.clip(np.asarray(st["porosity"].value, dtype=float), 1.0e-6, 1.0)
    hm = np.zeros(vols.size, bool)
    hm[src] = True
    om = C["low"][j0:j1, i0:i1].ravel().copy()
    assert not (om & hm).any()
    rest = ~hm & ~om
    ref = os.path.join(RC.OUT, "final_%s.npz" % tag)
    v_ref = np.load(ref)["v_hist"] if os.path.isfile(ref) else None
    L = ["Outlet trial: case %s (%s), configuration %s, %d steps; outlet = lower end of the basal layer on the 0.5 m grid"
         " (%d cells, t <= %.4f m); sources %d; MKL_NUM_THREADS=%s OMP_NUM_THREADS=%s; subnormal flush from step 1"
         % (tag, case[1], RC.CFG["name"], nsteps, om.sum(), RC.LOW_T, hm.sum(), os.environ.get("MKL_NUM_THREADS", "default"),
            os.environ.get("OMP_NUM_THREADS", "default")),
         "  step      t (s)     V_in      dStore    V_comp    V_out(flux) V_out(S)  V_clip    |V_in-(dStore+V_out)|/V_in"
         "  full-ledger rel   outlet max p (Pa)  same as run_cases"]
    t, v_in, v_out, v_out_s, v_comp = 0.0, 0.0, 0.0, 0.0, 0.0
    store0 = float(np.sum((n * np.asarray(st["saturation"].value, dtype=float) * vols)[rest]))
    first_out, diverged = None, None
    w0 = time.time()
    for step in range(nsteps):
        FZ.STATE["step"] = step + 1
        FZ.set_active(True)
        dt = float(solve_transport_step(st, dt_cap=RC.DT_CAP))
        t += dt
        dq = np.asarray(st["last_div_q"], dtype=float)
        empty = np.asarray(st["fill_empty_mask"], dtype=bool)
        v_in += float(np.sum(dq[hm] * vols[hm])) * dt
        v_comp += float(np.sum((-dq * vols)[~empty & rest])) * dt
        out_step = float(np.sum(-dq[om] * vols[om])) * dt
        v_out += out_step
        s = np.array(st["saturation"].value, copy=True)
        v_out_s += float(np.sum((n * s * vols)[om]))
        if first_out is None and out_step > 0.0:
            first_out = (step + 1, t, v_in)
        s[hm] = 1.0
        s[om] = 0.0
        st["saturation"].setValue(s)
        if v_ref is not None and diverged is None and step + 1 < len(v_ref) and v_in != v_ref[step + 1]:
            diverged = step + 1
        if (step + 1) % 100 == 0 or step + 1 == nsteps:
            store = float(np.sum((n * s * vols)[rest])) - store0
            clip = float(st.get("clipped_volume_total", 0.0))
            r1 = abs(v_in - (store + v_out)) / max(v_in, 1e-30)
            r2 = abs(v_in - (store + v_comp + v_out + clip)) / max(v_in, 1e-30)
            pmax = float(np.max(np.asarray(st["pressure"].value, dtype=float)[om]))
            L.append("  %5d  %9.3f  %8.4f  %8.4f  %8.2e  %9.4f   %8.4f  %8.2e  %.3e                   %.3e        %.4g          %s"
                     % (step + 1, t, v_in, store, v_comp, v_out, v_out_s, clip, r1, r2, pmax,
                        "-" if v_ref is None else ("yes" if diverged is None else "no (from step %d)" % diverged)))
            print(L[-1])
            sys.stdout.flush()
    FZ.set_active(False)
    pw, pp = peak_mb()
    L.append("  first outflow (first step with net inflow into the outlet cells): %s"
             % ("step %d, t %.3f s, V_in %.4f m3/m" % first_out if first_out else "none in this run"))
    L.append("  wall %.1f s (%.3f s/step); peak working set %.0f MB, peak private %.0f MB" % (
        time.time() - w0, (time.time() - w0) / nsteps, pw, pp))
    txt = "\n".join(L) + "\n"
    if not os.path.isdir(RC.OUT):
        os.makedirs(RC.OUT)
    open(os.path.join(RC.OUT, "outlet_trial_%s_%d.txt" % (tag, nsteps)), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
