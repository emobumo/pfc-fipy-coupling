# -*- coding: utf-8 -*-
"""Geometry configuration of the 4.8 chain: PFC end state -> PFC structure analysis -> step 1 / 2A / 2B ->
solver cases -> check_4_8 -> figures (main line 2026-10-20: everything that depends on the stope outline or
height comes from one file, so a new PFC end state is a configuration change, not a code change).

    XZ_CONFIG=40m (default) | 30m | <path to a .json>

Each script loads this module with imp.load_source and calls get(). The 40 m file reproduces the existing
results byte for byte (its output-directory suffix is empty); any other configuration writes to directories
with its own suffix (outputs/xz_prejudge_30m, data/xz_prejudge_30m, ...), so the 40 m results are never
overwritten.
"""
from __future__ import print_function

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _plain(o):
    """json gives unicode strings on Python 2: turn ASCII ones into str (paths, labels), keep the rest."""
    if isinstance(o, dict):
        return dict((_plain(k), _plain(v)) for k, v in o.items())
    if isinstance(o, list):
        return [_plain(v) for v in o]
    if str is bytes and type(o) is type(u""):
        try:
            return o.encode("ascii")
        except UnicodeError:
            return o
    return o


def get(name=None):
    name = name or os.environ.get("XZ_CONFIG", "40m")
    path = name if os.path.isfile(name) else os.path.join(HERE, "stope_%s.json" % name)
    c = _plain(json.load(open(path)))
    st = c["stope"]
    cot = 1.0 / math.tan(math.radians(st["dip_deg"]))
    hx = 0.5 * st["true_thick"] / math.sin(math.radians(st["dip_deg"]))
    fw = (st["ore_ref"][0] + (st["y_floor"] - st["ore_ref"][1]) * cot + hx, st["y_floor"])
    if abs(fw[0] - st["origin"][0]) > 1e-6 or abs(fw[1] - st["origin"][1]) > 1e-6:
        raise ValueError("%s: origin %s is not the footwall corner of the stope floor %s" % (path, st["origin"], fw))
    c["path"] = path
    return c


def out(c, *parts):
    """Repository path of an output directory for this configuration (suffix appended to the last part)."""
    return os.path.join(REPO, *parts) + c["suffix"]


def stope_polygon(c):
    """Four corners of the excavated stope (model x, y): floor hanging-wall side, floor footwall side, roof
    footwall side, roof hanging-wall side."""
    st = c["stope"]
    cot = 1.0 / math.tan(math.radians(st["dip_deg"]))
    hx = 0.5 * st["true_thick"] / math.sin(math.radians(st["dip_deg"]))
    xb = st["ore_ref"][0] + (st["y_floor"] - st["ore_ref"][1]) * cot
    xt = st["ore_ref"][0] + (st["y_top"] - st["ore_ref"][1]) * cot
    return [(xb - hx, st["y_floor"]), (xb + hx, st["y_floor"]), (xt + hx, st["y_top"]), (xt - hx, st["y_top"])]


def state(c, key):
    """PFC export base name of a state: key 'initial' / 'final' / 'prev' / 'earlier', or an integer index."""
    idx = c["pfc"][key] if not isinstance(key, int) else key
    return c["pfc"]["state_fmt"] % idx


if __name__ == "__main__":
    import sys
    cc = get(sys.argv[1] if len(sys.argv) > 1 else None)
    print(cc["path"])
    print("stope polygon", ["(%.3f, %.3f)" % p for p in stope_polygon(cc)])
