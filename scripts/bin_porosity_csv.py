# -*- coding: utf-8 -*-
"""
Offline porosity-field binning from an exported ball CSV (no itasca, no FiPy).
Use this AFTER exporting balls from PFC (see the FISH commands in the report):
the CSV has one row per ball with columns x, y, radius (a header line is ok).

It replicates the porosity_reader areal-porosity logic:
    cell porosity = 1 - (sum of pi*r^2 in cell) / cell_area
on a chosen cell size, then prints mean/std/min/max/CV and writes a 2D
filled-contour PNG to outputs/.

Run (after editing CSV_PATH / CELL below, or via env):
  powershell -File scripts\\run_local.ps1 scripts\\bin_porosity_csv.py
Env overrides: BALLS_CSV=path  CELL=2.5
"""
from __future__ import print_function
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.environ.get("BALLS_CSV", os.path.join(os.path.dirname(_HERE), "outputs", "balls.csv"))
CELL = float(os.environ.get("CELL", "2.5"))     # REV sweet spot ~10*d50
# Binning extent. Default = stated container; override Y_MAX below the rough
# pile top (settled balls reach ~38.3 m) to exclude the above-pile free-surface
# void cells, which otherwise dominate the porosity variance as an artifact.
X_MIN = float(os.environ.get("X_MIN", "-7.5"))
X_MAX = float(os.environ.get("X_MAX", "7.5"))
Y_MIN = float(os.environ.get("Y_MIN", "0.0"))
Y_MAX = float(os.environ.get("Y_MAX", "40.0"))
PHI_CLIP = (0.05, 0.95)
OUT_PNG = os.path.join(os.path.dirname(_HERE), "outputs", "viz_model1_porosity.png")


def load_balls(path):
    rows = []
    f = open(path, "r")
    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                x, y, r = float(parts[0]), float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                continue   # skip header / malformed
            rows.append((x, y, r))
    finally:
        f.close()
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found: %s" % CSV_PATH)
        print("Export balls from PFC first (see FISH commands), then re-run.")
        return
    x, y, r = load_balls(CSV_PATH)
    print("loaded %d balls; radius min/mean/max = %.4f / %.4f / %.4f m"
          % (x.size, r.min(), r.mean(), r.max()))
    print("ball extent x[%.2f,%.2f] y[%.2f,%.2f]"
          % (x.min(), x.max(), y.min(), y.max()))

    nx = int(round((X_MAX - X_MIN) / CELL))
    ny = int(round((Y_MAX - Y_MIN) / CELL))
    dx = (X_MAX - X_MIN) / nx
    dy = (Y_MAX - Y_MIN) / ny
    occ = np.zeros((ny, nx), dtype=float)
    for i in range(x.size):
        ix = int(math.floor((x[i] - X_MIN) / dx))
        iy = int(math.floor((y[i] - Y_MIN) / dy))
        if 0 <= ix < nx and 0 <= iy < ny:
            occ[iy, ix] += math.pi * r[i] * r[i]
    cell_area = dx * dy
    phi = 1.0 - occ / cell_area
    phi = np.clip(phi, PHI_CLIP[0], PHI_CLIP[1])

    # Interior stats (drop the two edge columns: container is ~0.6 m wider
    # than the balls each side, which inflates the edge porosity).
    interior = phi[:, 1:-1] if nx > 2 else phi
    print("=== porosity field (cell %.3f m, grid %dx%d) ===" % (CELL, nx, ny))
    for label, a in (("all cells", phi), ("interior (no edge cols)", interior)):
        print("  %-22s mean %.4f  std %.4f  min %.4f  max %.4f  CV(std/mean) %.4f"
              % (label, a.mean(), a.std(), a.min(), a.max(), a.std() / max(a.mean(), 1e-9)))

    fig, ax = plt.subplots(figsize=(5, 9))
    xc = X_MIN + (np.arange(nx) + 0.5) * dx
    yc = Y_MIN + (np.arange(ny) + 0.5) * dy
    Xc, Yc = np.meshgrid(xc, yc)
    levels = np.linspace(PHI_CLIP[0], PHI_CLIP[1], 19)
    cf = ax.contourf(Xc, Yc, phi, levels=levels, cmap="YlOrRd")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("areal porosity phi")
    ax.set_title("model1 porosity field (cell %.2f m)\nmean %.3f std %.3f CV %.3f"
                 % (CELL, interior.mean(), interior.std(),
                    interior.std() / max(interior.mean(), 1e-9)))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print("wrote %s" % os.path.abspath(OUT_PNG))


if __name__ == "__main__":
    main()
