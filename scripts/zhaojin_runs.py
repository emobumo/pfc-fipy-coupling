# -*- coding: utf-8 -*-
"""
Step 1 for the Zhaojin section: where does a design volume of grout go?

    powershell -File scripts\run_local.ps1 scripts\zhaojin_runs.py [--only A,B] [--cell 1.0]

Step 0 (scripts/zhaojin_screen.py) answered "can grout reach -63 m" with the
path integral. This runs the solver on a handful of configurations until the
zone's design grout per metre of strike (the document's own quantities) has
been injected, and books where it ended up:

    design     the changed-design reinforcement layer (12 / 16 / 12 m)
    above      fill above that layer
    below      everything under the -26 m level: stored ore, contacts, the
               -63 m drift -- grout that ran away from the target downward

READ THE TIME AXIS WITH CARE. The source is constant 5 MPa; the site pumped
at a limited rate (BW250, ~2 t/h) and read runaway as "pressure does not
build within 10 minutes". Through contacts with k ~ 1e-8 m2 the model's
initial rate is orders of magnitude above the pump, so times here are not
site times. What is comparable is the PARTITION of a given volume.

Boundaries are closed. That is adequate while the quota is smaller than the
stored-ore pore space below (~68 m3/m at phi 0.35), which holds for every
zone; the -63 m drift is a 2x2 m pocket whose first fill time marks arrival.

Contacts are 2 m wide here: at 1 m cells a 1 m band inclined at 70 degrees is
only corner-connected where it steps sideways, which neither the solver
(face fluxes) nor the reach diagnostic can pass. Reachability does not depend
on the width (tested), but the volume a contact can take does -- keep that
in mind reading the 'below' share.

Crash-safe: each run is checkpointed; finished runs are skipped on re-run.
"""
from __future__ import print_function

import imp
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint

Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "zhaojin_runs")

COMMON = {"contact_width": 2.0, "contact_phi": 0.35, "ore_phi": 0.35,
          "base_connected": True, "hole_angle": 35.0}
RUNS = [
    ("A", "z2 45%, contacts, no seal (7#/9#)", dict(zone="z2_cuts6-10", contacts=True, seal=False)),
    ("B", "z1 10%, contacts, no seal (5#)", dict(zone="z1_cuts1-6", contacts=True, seal=False)),
    ("C", "z3 25%, contacts, no seal (15#)", dict(zone="z3_cuts10-15", contacts=True, seal=False)),
    ("D", "z1 10%, NO contacts, no seal (control)", dict(zone="z1_cuts1-6", contacts=False, seal=False)),
    ("E", "z2 45%, contacts, SEAL (remedy)", dict(zone="z2_cuts6-10", contacts=True, seal=True)),
]

# Pressure-terminated runs: the site's own stop rule is the 5 MPa end
# pressure, not a volume ("注浆终压为5MPa（即注浆结束标准）"). A runaway never
# builds pressure, so the site kept pumping well past the design volume. These
# runs keep going until the grout reaches the -63 m drift, stalls (pressure
# built everywhere), or hits a volume cap, and book how much it took.
ARRIVAL_RUNS = [
    ("A2", "z2 45%, contacts, no seal -- to arrival", dict(zone="z2_cuts6-10", contacts=True, seal=False)),
    ("B2", "z1 10%, contacts, no seal -- to arrival", dict(zone="z1_cuts1-6", contacts=True, seal=False)),
    ("C2", "z3 25%, contacts, no seal -- to arrival", dict(zone="z3_cuts10-15", contacts=True, seal=False)),
    ("E2", "z2 45%, contacts, SEAL -- to arrival", dict(zone="z2_cuts6-10", contacts=True, seal=True)),
    ("D2", "z1 10%, NO contacts -- to arrival", dict(zone="z1_cuts1-6", contacts=False, seal=False)),
]
ARRIVAL_CAP = 300.0            # m3/m: ~6x the largest design volume

DT_CAP = 5.0
MAX_STEPS = 60000
STALL_WINDOW = 400            # steps
STALL_REL = 1.0e-5            # V_in growth over the window, relative to the quota
RECORD_EVERY = 50


def groups(mx, my, region, hcells, cfg):
    layer = float(Z.ZONES[cfg["zone"]]["layer"])
    grout = (region == Z.FILL) | (region == Z.CONTACT) | (region == Z.ORE) | (region == Z.OUTLET)
    not_src = np.ones(mx.size, dtype=bool)
    not_src[hcells] = False
    return {
        "design": grout & not_src & (my > Z.Y_LEVEL) & (my <= Z.Y_LEVEL + layer),
        "above": grout & not_src & (my > Z.Y_LEVEL + layer),
        "below": grout & not_src & (my <= Z.Y_LEVEL),
    }


def _snapshot(g, phi, s, vols, t, v_in, step):
    """Where the injected volume is, by group, at one instant."""
    vg = dict((k, float(np.sum(phi[m] * s[m] * vols[m]))) for k, m in g.items())
    vg.update({"t": t, "v_in": v_in, "step": step + 1})
    return vg


def run_one(tag, label, overrides, cell, arrival=False):
    """
    arrival=False: stop at the zone's design volume (the quota).
    arrival=True:  the site's stop rule -- keep going until the grout reaches
                   the -63 m drift, stalls, or passes ARRIVAL_CAP; the state
                   at the design volume is still booked, for comparison.
    """
    cfg = Z.config(**dict(COMMON, **overrides))
    quota = float(Z.ZONES[cfg["zone"]]["grout_per_m"])
    stop_volume = ARRIVAL_CAP if arrival else quota
    state, mx, my, phi, region, hc = Z.build_state(cfg, cell)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    hm = np.zeros(mx.size, dtype=bool)
    hm[hc] = True
    g = groups(mx, my, region, hc, cfg)
    tg = Z.targets(mx, my, region, cfg)
    design_capacity = float(np.sum(phi[g["design"]] * vols[g["design"]]))

    keeper = Checkpointer(os.path.join(OUT, "ckpt_" + tag), every=500,
                          fingerprint=state_fingerprint(state, "zhaojin|%s|%r" % (tag, sorted(cfg.items()))))
    t, v_in, step0 = 0.0, 0.0, 0
    t_ore = t_out = -1.0
    series = []
    snaps = {}                       # partition at the design volume / arrivals
    resumed = keeper.restore(state)
    if resumed:
        step0 = int(resumed["step"])
        t, v_in = float(resumed["t"]), float(resumed["v_in"])
        t_ore, t_out = float(resumed["t_ore"]), float(resumed["t_out"])
        series = json.loads(str(resumed["series_json"]))
        snaps = json.loads(str(resumed["snaps_json"]))
        print("  [%s] resuming at step %d (t=%.2f s)" % (tag, step0, t))
    else:
        print("  [%s] %s -- quota %.2f m3/m, %d cells" % (tag, label, quota, mx.size))
    sys.stdout.flush()

    reason = "max_steps"
    history = []
    w0 = time.time()
    for step in range(step0, MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=DT_CAP)
        t += float(dt)
        dq = np.asarray(state["last_div_q"], dtype=float)
        v_in += float(np.sum(dq[hm] * vols[hm])) * float(dt)
        s = np.array(state["saturation"].value, copy=True)
        s[hm] = 1.0
        state["saturation"].setValue(s)
        if "design_volume" not in snaps and v_in >= quota:
            snaps["design_volume"] = _snapshot(g, phi, s, vols, t, v_in, step)
        if t_ore < 0 and np.any(s[tg["ore_bottom"]] >= 0.5):
            t_ore = t
            snaps["reach_-56m"] = _snapshot(g, phi, s, vols, t, v_in, step)
        if t_out < 0 and np.any(s[tg["outlet"]] >= 0.5):
            t_out = t
            snaps["reach_-63m"] = _snapshot(g, phi, s, vols, t, v_in, step)
        history.append(v_in)
        if step % RECORD_EVERY == 0:
            vg = dict((k, float(np.sum(phi[m] * s[m] * vols[m]))) for k, m in g.items())
            series.append([t, v_in, vg["design"], vg["above"], vg["below"]])
        if step % 1000 == 0:
            print("    step %6d  t=%8.2f s  dt=%.2e  V_in=%7.2f / %.2f   below=%6.2f"
                  % (step, t, dt, v_in, stop_volume, series[-1][4] if series else 0.0))
            sys.stdout.flush()
        if arrival and t_out >= 0:
            reason = "arrived_-63m"
            break
        if v_in >= stop_volume:
            reason = "volume_cap" if arrival else "quota"
            break
        if len(history) > STALL_WINDOW and \
                history[-1] - history[-1 - STALL_WINDOW] < STALL_REL * quota:
            reason = "stall"
            break
        keeper.maybe_save(state, step=step + 1, t=t, v_in=v_in, t_ore=t_ore, t_out=t_out,
                          series_json=json.dumps(series), snaps_json=json.dumps(snaps))
    keeper.finish()

    s = np.asarray(state["saturation"].value, dtype=float)
    vg = dict((k, float(np.sum(phi[m] * s[m] * vols[m]))) for k, m in g.items())
    stored = vg["design"] + vg["above"] + vg["below"]
    np.savez(os.path.join(OUT, "final_%s.npz" % tag), s=s, phi=phi, region=region,
             x=mx, y=my, hole=hc, series=np.asarray(series, dtype=float))
    return {
        "tag": tag, "label": label, "zone": cfg["zone"], "reason": reason,
        "steps": step + 1, "t": t, "wall": time.time() - w0, "quota": quota,
        "v_in": v_in, "design": vg["design"], "above": vg["above"], "below": vg["below"],
        "below_share": vg["below"] / stored if stored > 0 else 0.0,
        "design_fill": vg["design"] / design_capacity if design_capacity > 0 else 0.0,
        "t_ore": t_ore, "t_out": t_out, "snaps": snaps,
    }


def main(argv):
    only = None
    cell = 1.0
    arrival = False
    i = 0
    while i < len(argv):
        if argv[i] == "--only":
            only = set(argv[i + 1].split(",")); i += 2
        elif argv[i] == "--cell":
            cell = float(argv[i + 1]); i += 2
        elif argv[i] == "--arrival":
            arrival = True; i += 1
        else:
            i += 1
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    done_path = os.path.join(OUT, "results.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            try:
                done.add(json.loads(line)["tag"])
            except ValueError:
                pass

    for tag, label, overrides in (ARRIVAL_RUNS if arrival else RUNS):
        if only is not None and tag not in only:
            continue
        if tag in done:
            print("  [%s] done already, skipped" % tag)
            continue
        r = run_one(tag, label, overrides, cell, arrival=arrival)
        handle = open(done_path, "a")
        try:
            handle.write(json.dumps(r) + "\n")
        finally:
            handle.close()
        print("  [%s] %s after %d steps, t=%.2f s (wall %.0f s): V_in %.2f | design %.2f above %.2f "
              "BELOW %.2f (%.0f%% of stored) | design layer %.0f%% full | reached -56 m at %s, -63 m at %s"
              % (tag, r["reason"], r["steps"], r["t"], r["wall"], r["v_in"], r["design"], r["above"],
                 r["below"], 100 * r["below_share"], 100 * r["design_fill"],
                 ("%.2f s" % r["t_ore"]) if r["t_ore"] >= 0 else "never",
                 ("%.2f s" % r["t_out"]) if r["t_out"] >= 0 else "never"))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
