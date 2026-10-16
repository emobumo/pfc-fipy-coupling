# -*- coding: utf-8 -*-
"""
4.7 check, part 1: geometry and set-up of the case-study section (items
1.1-1.5 and 5.3 of the 4.7 pre-draft check). No solver; reach recomputed only
for the channel-entry pressure.

    powershell -File scripts\\run_local.ps1 scripts\\check_4_7\\section.py

All elevations are printed RELATIVE to the grouting level (0, up positive),
never as mine levels. Zones are I / II / III (fill porosity 0.10 / 0.45 / 0.25).

  - domain, grids, elevations of the surface, collar, layer tops, stored-ore
    bottom, outlet drift; fill width per zone; the 70 deg walls (which side);
  - materials: porosities, how impassable ground is represented, the contact
    bands (side, width, vertical extent), "fill base shut", the bottom seal;
  - the hole: collar, elevation, length, source section, where it crosses the
    contact bands (source cells in each band, distance along the hole) and how
    far it runs into rock beyond the far wall; source cells per grid;
  - contact bands on the 0.5 m and 1 m grids: cells per row, and whether
    successive rows share a face (a band whose rows only touch at corners is
    closed to both the solver's face fluxes and the reach graph's short
    edges);
  - channel-entry pressure as defined in 4.5.3 (minimum p_req over the
    contact-band cells, hole cells included);
  - geometry data for figure 4.7-1 (relative coordinates).

Output: outputs/check_4_7/section.txt
"""
from __future__ import print_function

import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

Z = imp.load_source("case_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "check_4_7")
ZONES = [("z1_cuts1-6", "I (0.10)"), ("z2_cuts6-10", "II (0.45)"), ("z3_cuts10-15", "III (0.25)")]


def rel(y):
    return y - Z.Y_LEVEL


def band_rows(mx, my, cfg, cell, wall):
    """Per row of cells: x-indices of the contact band along one wall."""
    x_fw, x_hw = Z.walls(cfg)
    half = float(cfg["contact_width"]) / math.sin(math.radians(Z.DIP_DEG))
    rows = {}
    for x, y in zip(mx, my):
        if y > Z.Y_SURFACE or y < float(cfg["contact_bottom"]):
            continue
        xf, xh = x_fw(y), x_hw(y)
        inside = (x <= xf and x > xf - half) if wall == "fw" else (x >= xh and x < xh + half)
        if inside:
            rows.setdefault(round(y, 6), set()).add(int(round((x - Z.X_MIN) / cell - 0.5)))
    return rows


def band_stats(rows):
    ys = sorted(rows)
    widths = [len(rows[y]) for y in ys]
    corner_only = sum(1 for a, b in zip(ys[:-1], ys[1:]) if not (rows[a] & rows[b]))
    return min(widths), max(widths), len(ys), corner_only


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    w("4.7 check, part 1: section geometry and set-up (elevations relative to the grouting level, up positive)")
    w("")
    w("1.1 Domain and elevations")
    w("  domain x %.1f .. %.1f m, z %.1f .. %.1f m (relative); collar at x = 0" % (
        Z.X_MIN, Z.X_MAX, rel(Z.Y_MIN), rel(Z.Y_MAX)))
    for cell in (0.5, 1.0):
        nx = int(round((Z.X_MAX - Z.X_MIN) / cell)); ny = int(round((Z.Y_MAX - Z.Y_MIN) / cell))
        w("  grid %.1f m: %d x %d = %d cells" % (cell, nx, ny, nx * ny))
    w("  ground surface (fill reaches it)          z = %+.1f m" % rel(Z.Y_SURFACE))
    w("  original design layer top (22 m)          z = %+.1f m" % rel(Z.Y_LEVEL + Z.ORIGINAL_DESIGN["layer"]))
    w("  changed design layer tops I / II / III    z = %s m" % " / ".join(
        "%+.1f" % rel(Z.Y_LEVEL + Z.ZONES[k]["layer"]) for k, _ in ZONES))
    w("  hole collar (cross-cut roof)              z = %+.1f m" % rel(Z.collar()[1]))
    w("  grouting level (cross-cut floor)          z = %+.1f m" % rel(Z.Y_LEVEL))
    w("  stored-ore bottom                         z = %+.1f m" % rel(Z.Y_ORE_BOTTOM))
    w("  outlet drift (2 x 2 m cavity, n %.2f)     z = %+.1f .. %+.1f m" % (
        Z.CAVITY_PHI, rel(Z.Y_OUTLET), rel(Z.Y_OUTLET + 2.0)))
    w("  vein dip %.0f deg; walls x_fw(z), x_hw(z) are straight lines of slope 1/tan(dip); the vein leans to +x going up," % Z.DIP_DEG)
    w("  so the hanging wall is on the -x side and the footwall on the +x side; the drift touches the footwall side.")
    w("  horizontal fill width per zone (one value per zone, constant with height): %s m" % " / ".join(
        "%s %.1f" % (lab, Z.ZONES[k]["width"]) for k, lab in ZONES))
    w("")
    w("1.2 Materials")
    w("  fill n: %s; stored ore n %.2f (screening 0.25 / 0.35 / 0.45); outlet drift n %.2f" % (
        " / ".join("%s %.2f" % (lab, Z.ZONES[k]["phi"]) for k, lab in ZONES), Z.DEFAULT_CONFIG["ore_phi"], Z.CAVITY_PHI))
    lam = 2.0 * 30.0 / 1.0e-3 * (1 - Z.NO_GROUT_PHI) / Z.NO_GROUT_PHI
    w("  host rock, unmined ore below the stored ore, seal: ordinary cells with n = %.0e (lambda %.3e Pa/m, i.e. %.1f MPa per 0.5 m);" % (
        Z.NO_GROUT_PHI, lam, lam * 0.5 / 1e6))
    w("    no mask; the domain boundary is no-flow everywhere (including the ground surface)")
    w("  contact bands: both walls, inside the vein (the band takes the outermost vein cells), normal width w, horizontal w/sin(dip):")
    for cw in (0.5, 1.0, 2.0):
        w("    w = %.1f m -> horizontal %.4f m" % (cw, cw / math.sin(math.radians(Z.DIP_DEG))))
    w("    vertical extent: from the surface (z %+.1f) down to z %+.1f m (the outlet-drift floor), through the fill, the stored ore and the" % (
        rel(Z.Y_SURFACE), rel(Z.DEFAULT_CONFIG["contact_bottom"])))
    w("    unmined vein below it; n_c %.2f by default" % Z.DEFAULT_CONFIG["contact_phi"])
    w("  'fill base shut' (base_connected False): a 1 m layer of n %.0e across the vein just below the grouting level" % Z.NO_GROUT_PHI)
    w("    (z 0 .. -1 m); the contact bands are written after it and stay open through it")
    w("  'fill base open' (default): fill (z > 0) sits directly on the stored ore (z <= 0)")
    w("  bottom seal: every vein cell (fill and contacts) with 0 < z <= %.1f m set to n %.0e; it is not counted in any grout or volume" % (
        Z.SEAL_THICKNESS, Z.NO_GROUT_PHI))
    w("")

    w("1.3 Hole (default 35 deg, toward the hanging wall)")
    cx, cy = Z.collar()
    for ang in (35.0, 20.0):
        cfg = Z.config(hole_angle=ang)
        d = Z.hole_direction(cfg)
        ex, ey = cx + Z.ORIFICE_DEPTH * d[0], cy + Z.ORIFICE_DEPTH * d[1]
        tx, ty = cx + cfg["hole_length"] * d[0], cy + cfg["hole_length"] * d[1]
        w("  %2.0f deg: collar (%.2f, %+.2f), sealed orifice pipe to s = %.1f m at (%.2f, %+.2f), hole end s = %.1f m at (%.2f, %+.2f)" % (
            ang, cx, rel(cy), Z.ORIFICE_DEPTH, ex, rel(ey), cfg["hole_length"], tx, rel(ty)))
    w("  source = the perforated section s %.1f .. %.1f m, constant p0 = %.0f MPa, whole hole in one pass (no stages)" % (
        Z.ORIFICE_DEPTH, Z.DEFAULT_CONFIG["hole_length"], Z.P0 / 1e6))
    w("  the hole enters the fill at the end of the orifice pipe, i.e. exactly at the footwall contact; it leaves the fill through")
    w("  the hanging-wall contact and runs on into host rock:")
    for k, lab in ZONES:
        for ang in (35.0, 20.0):
            cfg = Z.config(zone=k, hole_angle=ang)
            lf = Z.fill_crossing_length(cfg)
            w("    %-10s %2.0f deg: length in the vein %.3f m (s %.1f .. %.3f), in rock beyond the hanging wall %.3f m" % (
                lab, ang, lf, Z.ORIFICE_DEPTH, Z.ORIFICE_DEPTH + lf, cfg["hole_length"] - Z.ORIFICE_DEPTH - lf))
    w("")
    for cell, cw in ((0.5, 1.0), (0.5, 2.0), (1.0, 2.0)):
        bundle = Z.build_mesh(cell)
        state, mx, my, _, _, _ = Z.build_state(Z.config(), cell, mesh_bundle=bundle)
        w("  grid %.1f m, contact width %.1f m:" % (cell, cw))
        for k, lab in ZONES:
            for ang in (35.0, 20.0):
                cfg = Z.config(zone=k, contact_width=cw, hole_angle=ang)
                phi, region = Z.section_fields(mx, my, cfg)
                src = Z.hole_cells(mx, my, cfg, cell=cell)
                d = Z.hole_direction(cfg)
                s_along = (mx[src] - cx) * d[0] + (my[src] - cy) * d[1]
                x_fw, x_hw = Z.walls(cfg)
                fw = [i for i in src if region[i] == Z.CONTACT and mx[i] > 0.5 * (x_fw(my[i]) + x_hw(my[i]))]
                hw = [i for i in src if region[i] == Z.CONTACT and mx[i] <= 0.5 * (x_fw(my[i]) + x_hw(my[i]))]
                sa = lambda ids: ", ".join("%.2f" % ((mx[i] - cx) * d[0] + (my[i] - cy) * d[1]) for i in ids)
                kinds = dict((name, int(np.sum(region[src] == code))) for code, name in Z.REGION_NAMES.items())
                Z.set_porosity(state, phi)
                cost = np.asarray(reachable_domain(state, src, p0=Z.P0, use_gravity=True)["cost"], dtype=float) / Z.P0
                band = region == Z.CONTACT
                w("    %-10s %2.0f deg: %2d source cells (s %.2f .. %.2f) = %s; footwall band %d cells (s %s), hanging-wall band %d cells (s %s); "
                  "entry p_req min over the bands %.4f p0" % (
                      lab, ang, len(src), s_along.min(), s_along.max(),
                      ", ".join("%s %d" % (n_, v) for n_, v in sorted(kinds.items()) if v), len(fw), sa(fw), len(hw), sa(hw),
                      float(cost[band].min())))
    w("")

    w("1.5 Contact bands on the grids (zone II, 35 deg; rows from the surface to the drift floor)")
    for cell, cw in ((0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (1.0, 1.0), (1.0, 2.0)):
        bundle = Z.build_mesh(cell)
        _, x, y, _, _ = bundle
        mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cfg = Z.config(zone="z2_cuts6-10", contact_width=cw)
        parts = []
        for wall in ("fw", "hw"):
            lo, hi, nrow, corner = band_stats(band_rows(mx, my, cfg, cell, wall))
            parts.append("%s: %d..%d cells per row over %d rows, %d row pairs touch only at corners" % (
                "footwall" if wall == "fw" else "hanging wall", lo, hi, nrow, corner))
        w("  grid %.1f m, width %.1f m: %s" % (cell, cw, "; ".join(parts)))
    w("")

    w("5.3 Geometry data for figure 4.7-1 (35 deg; x from the collar, z relative to the grouting level; metres)")
    zs = (rel(Z.Y_SURFACE), Z.ORIGINAL_DESIGN["layer"], 0.0, rel(Z.Y_ORE_BOTTOM), rel(Z.Y_OUTLET), rel(Z.Y_MIN))
    for k, lab in ZONES:
        cfg = Z.config(zone=k, contact_width=2.0)
        x_fw, x_hw = Z.walls(cfg)
        half = 2.0 / math.sin(math.radians(Z.DIP_DEG))
        w("  zone %s, fill width %.1f m:" % (lab, Z.ZONES[k]["width"]))
        w("    %8s | %9s %9s | %22s %22s" % ("z", "x_hw", "x_fw", "hanging-wall contact", "footwall contact"))
        for z in zs:
            y = z + Z.Y_LEVEL
            w("    %+8.2f | %9.3f %9.3f | %9.3f .. %9.3f  %9.3f .. %9.3f" % (
                z, x_hw(y), x_fw(y), x_hw(y), x_hw(y) + half, x_fw(y) - half, x_fw(y)))
        xo = x_fw(Z.Y_OUTLET + 1.0)
        w("    outlet drift: x %.3f .. %.3f, z %+.1f .. %+.1f" % (xo, xo + 2.0, rel(Z.Y_OUTLET), rel(Z.Y_OUTLET + 2.0)))
    w("  stored ore: vein between z %+.1f and %+.1f; unmined vein (n %.0e) below z %+.1f; contacts run on to z %+.1f" % (
        rel(Z.Y_ORE_BOTTOM), 0.0, Z.NO_GROUT_PHI, rel(Z.Y_ORE_BOTTOM), rel(Z.DEFAULT_CONFIG["contact_bottom"])))
    w("  bottom seal: vein cells with 0 < z <= %.1f (full vein width, contacts included)" % Z.SEAL_THICKNESS)
    w("  hole and source: see 1.3 (collar (0, %+.1f); 35 deg end (%.3f, %+.3f); 20 deg end (%.3f, %+.3f))" % (
        rel(cy), cx - 17.5 * math.cos(math.radians(35)), rel(cy + 17.5 * math.sin(math.radians(35))),
        cx - 17.5 * math.cos(math.radians(20)), rel(cy + 17.5 * math.sin(math.radians(20)))))
    text = "\n".join(L)
    print(text)
    open(os.path.join(OUT, "section.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
