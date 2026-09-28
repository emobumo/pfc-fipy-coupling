# -*- coding: utf-8 -*-
"""
Case 3 (no-gravity control): split the up/down asymmetry into its gravity
part and its structure part.

    powershell -File scripts\run_local.ps1 scripts\gravity_control.py            reach only (minutes)
    powershell -File scripts\run_local.ps1 scripts\gravity_control.py --solver   + solver pair (~1 h)

The headline case (cases/inclined_hole_gradient.py, line source) is run with
gravity on and off, on the uniform 0.18 and the 0.12 -> 0.30 gradient field:

    uniform, no gravity     the geometric baseline (the hole, the face on the
                            left boundary) -- what "no bias" looks like here
    uniform, gravity        gravity alone
    gradient, no gravity    structure alone
    gradient, gravity       both (the headline 1.175 reach / 1.157 solver)

Reach numbers use the production reachable_domain (16-neighbour stencil) at
2.5 and 1.25 m and are solver-free. The solver pair marches exactly as the
case does (same march, same stall rule; gravity_y is read from the state's
parameters at every step, so setting it to 0 is the only change).
"""
from __future__ import print_function

import imp
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
OUT = os.path.join(REPO, "outputs", "gravity_control")


def up_down(mask, mx, my, nrm):
    perp = (mx[mask] - G.MOUTH[0]) * nrm[0] + (my[mask] - G.MOUTH[1]) * nrm[1]
    up, dn = -perp.min(), perp.max()
    return up, dn, up / dn


def reach_table():
    rows = []
    base_cell = G.CELL
    for cell in (2.5, 1.25):
        G.CELL = cell
        end, d, nrm = G.hole_geometry()
        for kind in ("uniform", "gradient"):
            st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
            hm = np.zeros(mx.size, dtype=bool)
            hm[hc] = True
            for grav in (False, True):
                r = reachable_domain(st, hc, p0=G.P0, use_gravity=grav)
                m = r["reachable"] & np.logical_not(hm)
                up, dn, ratio = up_down(m, mx, my, nrm)
                rows.append({"cell": cell, "kind": kind, "gravity": grav, "cells": int(m.sum()),
                             "area": float(m.sum()) * cell * cell, "up": up, "down": dn, "ratio": ratio})
    G.CELL = base_cell
    return rows


def solver_pair():
    rows = []
    end, d, nrm = G.hole_geometry()
    for kind in ("uniform", "gradient"):
        st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
        st["slurry_parameters"]["gravity_y"] = 0.0
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        label = "%s, no gravity" % kind
        from src.analysis import stop_rule
        keeper = Checkpointer(os.path.join(OUT, "ckpt_" + re.sub(r"[^A-Za-z0-9._-]+", "_", label)),
                              every=200, fingerprint=state_fingerprint(
                                  st, "nograv|%s" % kind + stop_rule.rate_tag(G.n_ref_for(kind))))
        t, v_in, v_store = G.march(st, mx, my, hc, vols, label, keeper=keeper, n_ref=G.n_ref_for(kind))
        fm = G.front_metrics(st, mx, my, hc)
        rows.append({"kind": kind, "t": t, "v_in": v_in, "cells": fm["count"],
                     "up": fm["perp_up"], "down": fm["perp_down"],
                     "ratio": fm["perp_up"] / fm["perp_down"]})
    return rows


def main(argv):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    lines = ["reach (16-neighbour, solver-free): up/down of the reachable domain about the hole axis", ""]
    lines.append("%-6s %-9s %-8s %6s %8s %7s %7s %7s" % ("cell", "field", "gravity", "cells", "area", "up", "down", "up/dn"))
    reach = reach_table()
    for r in reach:
        lines.append("%-6.2f %-9s %-8s %6d %8.0f %7.2f %7.2f %7.3f" % (
            r["cell"], r["kind"], "on" if r["gravity"] else "off", r["cells"], r["area"],
            r["up"], r["down"], r["ratio"]))
    lines.append("")
    lines.append("decomposition (ratios of up/down):")
    for cell in (2.5, 1.25):
        q = dict(((r["kind"], r["gravity"]), r["ratio"]) for r in reach if r["cell"] == cell)
        lines.append("  %.2f m: geometry %.3f | gravity alone x%.3f | structure alone x%.3f | both x%.3f"
                     % (cell, q[("uniform", False)], q[("uniform", True)] / q[("uniform", False)],
                        q[("gradient", False)] / q[("uniform", False)],
                        q[("gradient", True)] / q[("uniform", False)]))
    result = {"reach": reach}
    if "--solver" in argv:
        solver = solver_pair()
        result["solver_no_gravity"] = solver
        lines.append("")
        lines.append("solver, no gravity (2.5 m; with-gravity anchors: uniform 0.909, gradient 1.157):")
        for r in solver:
            lines.append("  %-9s stall t=%.0f s  V_in=%.2f  cells=%d  up=%.2f down=%.2f  up/dn=%.3f"
                         % (r["kind"], r["t"], r["v_in"], r["cells"], r["up"], r["down"], r["ratio"]))
    text = "\n".join(lines)
    open(os.path.join(OUT, "gravity_control.txt"), "w").write(text + "\n")
    json.dump(result, open(os.path.join(OUT, "gravity_control.json"), "w"), indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
