# -*- coding: utf-8 -*-
"""Structure of the caved rubble in the PFC2D stope-caving run (input for section 4.8).

    powershell -File scripts\\run_local.ps1 scripts\\pfc_collapse\\analyze.py

Reads the CSV export of the PFC2D 6.0 saves (scripts/pfc_collapse/export_states.py; directory from
$PFC_COLLAPSE_EXPORT, default E:/pfc_collapse_work/export -- outside the repository) and writes
outputs/pfc_collapse/analysis.txt (all numbers) and outputs/pfc_collapse/fields_*.npz (for plot.py).

Everything is two-dimensional: porosity is the AREA porosity of discs, not a volume porosity.

Pixel classification (pixel H = 0.05 m over a box around the stope):
  rubble solid   discs of rubble balls (cumulative displacement since excavation > DISP_RUBBLE)
  rock region    morphological closing of the other discs with a disc of radius R_CLOSE (fills the
                 rock's own interstices and surface dimples, not the gap between rock and rubble)
  free space V   neither solid nor rock region
  residual void  opening of V with a disc of radius R_VOID (space a disc of that radius can sweep)
  rubble region  rubble solid + the rest of V within D_PORE of a rubble disc
  porosity       (rubble region - rubble solid) / rubble region, counted only inside the rubble region
Rock pixels carry a side: footwall (material 'footwall'), hanging wall ('hanging_wall_weak' and
'hw_interface'), ore pillar ('ore'); the side of a closed-in pixel is that of the nearest rock disc.

Coordinates are the model's (x, y in m). s is the distance normal to the orebody (dip 30 deg) from
the footwall contact (stope floor), positive towards the hanging wall; the stope roof is s = 14 m.
"""
from __future__ import print_function

import math
import os
import sys

import numpy as np
from scipy import ndimage, stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT = os.environ.get("PFC_COLLAPSE_EXPORT", "E:/pfc_collapse_work/export")
OUT = os.path.join(REPO, "outputs", "pfc_collapse")

# model geometry (from the model scripts: 1chushichengyang.dat, 4kaiwa_local_stope.dat)
ORE_REF = (-28.0, 40.0)
DIP = 30.0
THICK = 14.0
STOPE_Y = (40.0, 80.0)
COT = 1.0 / math.tan(math.radians(DIP))
HX = 0.5 * THICK / math.sin(math.radians(DIP))

# analysis settings
H = 0.05
BOX = (-48.0, 60.0, 32.0, 104.0)
DISP_RUBBLE = 0.5
D50 = None                      # set from the data (median ball diameter)
R_CLOSE_D = 0.65                # x d50 (= 0.49 m, about the largest radius)
R_VOID_D = (0.5, 1.0, 1.5)      # x d50, residual-void opening radius; 1.0 is the reference
D_PORE_D = 1.3                  # x d50 (= 0.97 m, about the largest diameter)
RVE_L_D = (1, 2, 3, 4, 6, 8, 10, 13, 16)
RVE_COVER = 0.95
SLAB_D = 2                      # profile slab thickness, x d50
BAND_D = (1, 2, 3)
STATES_TIME = (5, 10, 20, 35, 49, 50)
SIDE = {"footwall": 1, "hanging_wall_weak": 2, "hw_interface": 2, "ore": 3}
SIDE_NAME = {1: "footwall", 2: "hanging wall", 3: "ore pillar"}


def load(name):
    b = np.genfromtxt(os.path.join(EXPORT, name + "_balls.csv"), delimiter=",", names=True, dtype=None)
    meta = {}
    for line in open(os.path.join(EXPORT, name + "_meta.txt")):
        k, v = line.split(" = ", 1)
        meta[k] = v.strip()
    return b, meta


def s_coord(x, y):
    """Normal distance from the footwall contact, positive towards the hanging wall."""
    xc = ORE_REF[0] + (y - ORE_REF[1]) * COT
    return (xc - x) * math.sin(math.radians(DIP)) + 0.5 * THICK


def grid():
    nx = int(round((BOX[1] - BOX[0]) / H))
    ny = int(round((BOX[3] - BOX[2]) / H))
    xs = BOX[0] + (np.arange(nx) + 0.5) * H
    ys = BOX[2] + (np.arange(ny) + 0.5) * H
    return xs, ys


def raster(b, rubble):
    """Class image: 0 free, 1/2/3 rock side, 4 rubble solid."""
    xs, ys = grid()
    img = np.zeros((len(ys), len(xs)), dtype=np.int8)
    side = np.array([SIDE.get(m, 1) for m in b["material"]], dtype=np.int8)
    order = np.concatenate([np.where(~rubble)[0], np.where(rubble)[0]])
    for k in order:
        x, y, r = b["x"][k], b["y"][k], b["radius"][k]
        if x + r < BOX[0] or x - r > BOX[1] or y + r < BOX[2] or y - r > BOX[3]:
            continue
        i0 = max(int((x - r - BOX[0]) / H), 0)
        i1 = min(int((x + r - BOX[0]) / H) + 2, len(xs))
        j0 = max(int((y - r - BOX[2]) / H), 0)
        j1 = min(int((y + r - BOX[2]) / H) + 2, len(ys))
        sub = (xs[None, i0:i1] - x) ** 2 + (ys[j0:j1, None] - y) ** 2 <= r * r
        img[j0:j1, i0:i1][sub] = 4 if rubble[k] else side[k]
    return img


def classify(img, d50, r_void_d=1.0):
    rock_solid = (img >= 1) & (img <= 3)
    rub_solid = img == 4
    rc = R_CLOSE_D * d50 / H
    dil = ndimage.distance_transform_edt(~rock_solid) <= rc
    rock = ndimage.distance_transform_edt(dil) > rc
    rock |= rock_solid
    idx = ndimage.distance_transform_edt(~rock_solid, return_distances=False, return_indices=True)
    rock_side = np.where(rock, img[idx[0], idx[1]], 0).astype(np.int8)
    free = ~rock & ~rub_solid
    rv = r_void_d * d50 / H
    core = ndimage.distance_transform_edt(free) > rv
    void = (ndimage.distance_transform_edt(~core) <= rv) & free
    near_rub = ndimage.distance_transform_edt(~rub_solid) <= D_PORE_D * d50 / H
    region = rub_solid | (free & ~void & near_rub)
    pore = region & ~rub_solid
    return {"rock": rock, "rock_side": rock_side, "rub_solid": rub_solid, "void": void, "region": region, "pore": pore}


def porosity(c, mask=None):
    reg = c["region"] if mask is None else c["region"] & mask
    a = reg.sum()
    return (c["pore"] & reg).sum() / float(a) if a else float("nan"), a * H * H


def window_stats(c, L):
    """Square windows of side L on a stride of L/4 (>= 0.25 m); keep windows >= RVE_COVER rubble region."""
    n = int(round(L / H))
    st = max(int(round(max(L / 4.0, 0.25) / H)), 1)
    R = np.pad(np.cumsum(np.cumsum(c["region"].astype(np.int64), 0), 1), ((1, 0), (1, 0)), "constant")
    P = np.pad(np.cumsum(np.cumsum(c["pore"].astype(np.int64), 0), 1), ((1, 0), (1, 0)), "constant")
    ny, nx = c["region"].shape
    J = np.arange(0, ny - n + 1, st)
    I = np.arange(0, nx - n + 1, st)
    jj, ii = np.meshgrid(J, I, indexing="ij")
    r = R[jj + n, ii + n] - R[jj, ii + n] - R[jj + n, ii] + R[jj, ii]
    p = P[jj + n, ii + n] - P[jj, ii + n] - P[jj + n, ii] + P[jj, ii]
    ok = r >= RVE_COVER * n * n
    v = p[ok] / r[ok].astype(float)
    return v


def linfit(xv, yv, w):
    """Weighted least squares y = a + b x (weights = slab area); weighted R^2 and the p-value of the slope
    from the weighted correlation (t test, n - 2 degrees of freedom)."""
    W = np.sum(w)
    xm = np.sum(w * xv) / W
    ym = np.sum(w * yv) / W
    b = np.sum(w * (xv - xm) * (yv - ym)) / np.sum(w * (xv - xm) ** 2)
    a = ym - b * xm
    ss_res = np.sum(w * (yv - a - b * xv) ** 2)
    ss_tot = np.sum(w * (yv - ym) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    n = len(xv)
    t = math.sqrt(r2 * (n - 2) / max(1.0 - r2, 1e-12))
    return a, b, r2, 2.0 * stats.t.sf(t, n - 2)


def fragments(b, rub, name):
    """Fragments = rubble balls joined by intact parallel bonds (pb_state 3) in that state."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cc_ = np.genfromtxt(os.path.join(EXPORT, name + "_contacts.csv"), delimiter=",", names=True, dtype=None)
    pos = dict((k, i) for i, k in enumerate(b["id"]))
    i1 = np.array([pos[q] for q in cc_["id1"]])
    i2 = np.array([pos[q] for q in cc_["id2"]])
    bonded = (cc_["pb_state"] == 3) & rub[i1] & rub[i2]
    g = coo_matrix((np.ones(bonded.sum()), (i1[bonded], i2[bonded])), shape=(len(b), len(b)))
    _, lab = connected_components(g, directed=False)
    lab = np.where(rub, lab, -1)
    return lab, bonded.sum(), (rub[i1] & rub[i2]).sum()


def block_solid(b, frag, d50):
    """Rubble solid at block scale: every fragment of >= 2 balls closed with the rock closing radius
    (its own interstices count as rock), single balls as discs."""
    xs, ys = grid()
    out = np.zeros((len(ys), len(xs)), dtype=bool)
    rc = R_CLOSE_D * d50
    for f in np.unique(frag[frag >= 0]):
        k = np.where(frag == f)[0]
        x, y, r = b["x"][k], b["y"][k], b["radius"][k]
        i0 = max(int((x.min() - r.max() - 2 * rc - BOX[0]) / H), 0)
        i1 = min(int((x.max() + r.max() + 2 * rc - BOX[0]) / H) + 2, len(xs))
        j0 = max(int((y.min() - r.max() - 2 * rc - BOX[2]) / H), 0)
        j1 = min(int((y.max() + r.max() + 2 * rc - BOX[2]) / H) + 2, len(ys))
        sub = np.zeros((j1 - j0, i1 - i0), dtype=bool)
        for xx, yy, rr in zip(x, y, r):
            sub |= (xs[None, i0:i1] - xx) ** 2 + (ys[j0:j1, None] - yy) ** 2 <= rr * rr
        if len(k) > 1:
            dil = ndimage.distance_transform_edt(~sub) <= rc / H
            sub = (ndimage.distance_transform_edt(dil) > rc / H) | sub
        out[j0:j1, i0:i1] |= sub
    return out


def profile(c, coord, lo, hi, dz, mask=None):
    reg = c["region"] if mask is None else c["region"] & mask
    edges = np.arange(lo, hi + 1e-9, dz)
    zc, nv, ar = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = reg & (coord >= a) & (coord < b)
        A = m.sum()
        if A * H * H >= 2.0:
            zc.append(0.5 * (a + b))
            nv.append((c["pore"] & m).sum() / float(A))
            ar.append(A * H * H)
    return np.array(zc), np.array(nv), np.array(ar)


def interior_mask(c, d50, k=3):
    far_rock = ndimage.distance_transform_edt(~c["rock"]) > k * d50 / H
    far_void = ndimage.distance_transform_edt(~c["void"]) > k * d50 / H
    return far_rock & far_void


def components(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    return lab, n


def scale_block(c, blk):
    """Block-scale sets: fragment interiors count as solid."""
    region = c["region"] | (blk & ~c["rock"])
    return {"rock": c["rock"], "rock_side": c["rock_side"], "void": c["void"], "rub_solid": blk & region,
            "region": region, "pore": region & ~blk}


def bin_field(c, nbx, nby, nb):
    field = np.full((nby, nbx), np.nan)
    cover = np.zeros((nby, nbx))
    for j in range(nby):
        for i in range(nbx):
            reg = c["region"][j * nb:(j + 1) * nb, i * nb:(i + 1) * nb]
            if reg.size == 0:
                continue
            cover[j, i] = reg.sum() / float(nb * nb)
            if cover[j, i] >= 0.5:
                field[j, i] = c["pore"][j * nb:(j + 1) * nb, i * nb:(i + 1) * nb].sum() / float(reg.sum())
    return field, cover


SCALES = (("particle", "particle scale (all disc interstices are pore)"),
          ("block", "block scale (interstices inside bonded fragments count as solid)"))


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    xs, ys = grid()
    X, Y = np.meshgrid(xs, ys)
    S = s_coord(X, Y)

    b0, m0 = load("local_stope_result_0")
    b, m = load("local_stope_result_50")
    assert np.array_equal(b0["id"], b["id"])
    d50 = float(np.median(2 * b["radius"]))
    disp = np.hypot(b["dx"], b["dy"])
    rub = disp > DISP_RUBBLE
    w("PFC2D stope caving, rubble structure (2D AREA porosity of discs throughout)")
    w("export directory: %s" % EXPORT)
    w("final state local_stope_result_50 (model time %.4f s after excavation, step %s)"
      % (float(m["pfcx_time"]) - float(m0["pfcx_time"]), m["pfcx_step"]))
    w("balls %d; diameter min / d50 / max %.4f / %.4f / %.4f m (all balls)"
      % (len(b), 2 * b["radius"].min(), d50, 2 * b["radius"].max()))
    w("pixel %.3f m; box x %.1f..%.1f, y %.1f..%.1f; rock closing radius %.3f m; pore reach %.3f m; residual-void"
      " opening radius %.3f m (reference)" % ((H,) + BOX + (R_CLOSE_D * d50, D_PORE_D * d50, d50)))
    w("")

    # ---- 2.1 final state and stability ---------------------------------------------------------
    w("2.1 Final state and stability")
    hist = np.histogram(disp, bins=[0, 0.1, 0.5, 1e9])[0]
    w("  cumulative displacement since excavation: < 0.1 m %d, 0.1..0.5 m %d, > 0.5 m %d balls -> rubble = > %.1f m"
      % (hist[0], hist[1], hist[2], DISP_RUBBLE))
    mats = dict(zip(*np.unique(b["material"][rub], return_counts=True)))
    w("  rubble balls %d (%s); solid area sum pi r^2 = %.2f m2"
      % (rub.sum(), ", ".join("%s %d" % (k, mats[k]) for k in sorted(mats)), np.sum(np.pi * b["radius"][rub] ** 2)))
    s0 = s_coord(b0["x"], b0["y"])
    w("  rubble origin: y %.2f..%.2f m, s %.2f..%.2f m (stope roof s = 14; caved up to %.2f m above the roof)"
      % (b0["y"][rub].min(), b0["y"][rub].max(), s0[rub].min(), s0[rub].max(), s0[rub].max() - THICK))
    sf = s_coord(b["x"], b["y"])
    w("  rubble now:    x %.2f..%.2f, y %.2f..%.2f m, s %.2f..%.2f m"
      % (b["x"][rub].min(), b["x"][rub].max(), b["y"][rub].min(), b["y"][rub].max(), sf[rub].min(), sf[rub].max()))
    b49, m49 = load("local_stope_result_49")
    assert np.array_equal(b49["id"], b["id"])
    step = np.hypot(b["x"] - b49["x"], b["y"] - b49["y"])
    dt = float(m["pfcx_time"]) - float(m49["pfcx_time"])
    v = np.hypot(b["vx"], b["vy"])
    w("  state 49 -> 50 (%.4f s): movement max %.4f m; > 0.01 m: %d balls (%d rubble); > 0.1 d50: %d; > 1 d50: %d"
      % (dt, step.max(), (step > 0.01).sum(), (rub & (step > 0.01)).sum(), (step > 0.1 * d50).sum(), (step > d50).sum()))
    w("  rubble movement 49 -> 50: mean %.5f m, median %.5f m, 99th pct %.4f m"
      % (step[rub].mean(), np.median(step[rub]), np.percentile(step[rub], 99)))
    mv = step > 0.1 * d50
    w("  balls moving > 0.1 d50 in 49 -> 50: x %.1f..%.1f, y %.1f..%.1f, s %.1f..%.1f; %d of them rubble"
      % (b["x"][mv].min(), b["x"][mv].max(), b["y"][mv].min(), b["y"][mv].max(), sf[mv].min(), sf[mv].max(), (mv & rub).sum()))
    w("  speed at state 50: max %.3f m/s; > 0.1 m/s %d balls; ratio-average %s; bonds intact %s; cracks %s"
      % (v.max(), (v > 0.1).sum(), m["pfcx_ratio"], m["bonds_intact"], m["crack_num"]))
    b45, _ = load("local_stope_result_45")
    st45 = np.hypot(b["x"] - b45["x"], b["y"] - b45["y"])
    w("  state 45 -> 50: movement > 0.1 d50: %d balls (%d rubble); max %.3f m"
      % ((st45 > 0.1 * d50).sum(), (rub & (st45 > 0.1 * d50)).sum(), st45.max()))

    img = raster(b, rub)
    c = classify(img, d50)
    frag, nbond, nrr = fragments(b, rub, "local_stope_result_50")
    blk = block_solid(b, frag, d50)
    cb = scale_block(c, blk)
    sets = {"particle": c, "block": cb}
    w("  rubble region %.2f m2 (particle scale) / %.2f m2 (block scale); rubble solid (pixels) %.2f m2 vs sum pi r^2"
      " %.2f m2" % (porosity(c)[1], porosity(cb)[1], c["rub_solid"].sum() * H * H, np.sum(np.pi * b["radius"][rub] ** 2)))
    w("  porosity of the whole rubble region: particle scale %.4f, block scale %.4f" % (porosity(c)[0], porosity(cb)[0]))
    w("  rock region in box %.2f m2; residual void (opening r = d50) %.2f m2" % (c["rock"].sum() * H * H, c["void"].sum() * H * H))
    for nm, msk in (("footwall s < -5 m", S < -5.0), ("hanging wall s > %.0f m (beyond the cave)" % (s0[rub].max() + 5), S > s0[rub].max() + 5)):
        msk = msk & c["rock"]
        w("  intact rock reference, %s (area %.1f m2): disc-assembly porosity %.4f"
          % (nm, msk.sum() * H * H, (msk & (img == 0)).sum() / float(msk.sum())))
    w("")

    # ---- 2.2 RVE and porosity field ---------------------------------------------------------------
    BIN_D = 8
    BIN = BIN_D * d50
    nbx = int(math.ceil((BOX[1] - BOX[0]) / BIN))
    nby = int(math.ceil((BOX[3] - BOX[2]) / BIN))
    nb = int(round(BIN / H))
    rve, fields = {}, {}
    w("2.2 Representative window (square side L, stride max(L/4, 0.25 m), windows >= %.0f%% rubble region)" % (100 * RVE_COVER))
    for sc, desc in SCALES:
        w("  %s" % desc)
        w("   L/d50   L (m)    windows   mean      std       CV")
        rows = []
        for k in RVE_L_D:
            vv = window_stats(sets[sc], k * d50)
            if len(vv) >= 3:
                rows.append((k, k * d50, len(vv), vv.mean(), vv.std(), vv.std() / vv.mean()))
                w("   %5d  %6.3f  %8d   %.4f    %.4f    %.3f" % rows[-1])
        rve[sc] = np.array(rows)
        lg = np.polyfit(np.log(rve[sc][:, 1]), np.log(rve[sc][:, 4]), 1)
        w("   log-log slope of std against L: %.3f (uncorrelated pixels would give -1)" % lg[0])
    w("  chosen bin: %d d50 = %.3f m" % (BIN_D, BIN))
    for sc, desc in SCALES:
        field, cover = bin_field(sets[sc], nbx, nby, nb)
        fields[sc] = (field, cover)
        fv = field[np.isfinite(field)]
        w("  field, %s: %d bins with >= 50%% rubble region; porosity min / mean / max %.4f / %.4f / %.4f; std %.4f"
          % (sc, len(fv), fv.min(), fv.mean(), fv.max(), fv.std()))
    cx = BOX[0] + (np.arange(nbx) + 0.5) * BIN
    cy = BOX[2] + (np.arange(nby) + 0.5) * BIN
    w("  bins (centre x, y, s, rubble-region cover, porosity particle / block):")
    for j in range(nby):
        for i in range(nbx):
            if np.isfinite(fields["particle"][0][j, i]):
                w("    %7.2f %7.2f %7.2f  %.3f  %.4f / %.4f" % (cx[i], cy[j], s_coord(cx[i], cy[j]), fields["particle"][1][j, i],
                                                            fields["particle"][0][j, i], fields["block"][0][j, i]))
    w("")

    # ---- 2.3 structures ---------------------------------------------------------------------------------
    w("2.3 Structures")
    inner = interior_mask(c, d50)
    slab = SLAB_D * d50
    prof = {}
    for sc, desc in SCALES:
        w("  %s" % desc)
        for lab, key, coord, lo, hi in (("height y", "y", Y, 38.0, 100.0), ("normal s", "s", S, -2.0, 44.0)):
            for tag, msk in (("all", None), ("interior", inner)):
                z, nv, ar = profile(sets[sc], coord, lo, hi, slab, msk)
                a, bb, r2, p = linfit(z, nv, ar)
                for q, nm in zip((z, nv, ar, a, bb, r2, p), ("z", "n", "a", "a0", "b1", "r2", "p")):
                    prof["%s_%s_%s_%s" % (sc, key, tag, nm)] = q
                w("    along %s (%s; slabs %.3f m; %d slabs, %.2f..%.2f m): n = %.4f %+.5f %s; R2 %.3f; p %.2g;"
                  " change over the span %+.4f" % (lab, tag, slab, len(z), z.min(), z.max(), a, bb, key, r2, p,
                                                  bb * (z.max() - z.min())))
                w("       slab centre / porosity / area m2: " + "; ".join("%.2f %.4f %.1f" % t for t in zip(z, nv, ar)))
    w("  plane fit n = a + b_t t + b_s s on %.3f m cells (weights = rubble-region area per cell, cells >= 25%% covered);"
      " t = up-dip distance, s = normal distance; vertical gradient at fixed x = b_t sin30 + b_s cos30" % slab)
    sn, cs30 = math.sin(math.radians(DIP)), math.cos(math.radians(DIP))
    T = (X - ORE_REF[0]) * cs30 + (Y - ORE_REF[1]) * sn
    ncell = int(round(slab / H))
    for sc, _ in SCALES:
        for tag, msk in (("all", None), ("interior", inner)):
            reg = sets[sc]["region"] if msk is None else sets[sc]["region"] & msk
            pr_ = sets[sc]["pore"] & reg
            ny_, nx_ = reg.shape
            ny_, nx_ = ny_ // ncell * ncell, nx_ // ncell * ncell
            def agg(a):
                return a[:ny_, :nx_].reshape(ny_ // ncell, ncell, nx_ // ncell, ncell).sum(axis=(1, 3)).astype(float)
            A = agg(reg)
            P = agg(pr_)
            tt = agg(np.where(reg, T, 0.0))
            ss = agg(np.where(reg, S, 0.0))
            ok = A >= 0.25 * ncell * ncell
            nv_ = P[ok] / A[ok]
            tv = tt[ok] / A[ok]
            sv = ss[ok] / A[ok]
            wv = A[ok]
            M = np.column_stack([np.ones(ok.sum()), tv, sv]) * np.sqrt(wv)[:, None]
            coef = np.linalg.lstsq(M, nv_ * np.sqrt(wv))[0]
            res = nv_ - (coef[0] + coef[1] * tv + coef[2] * sv)
            ym = np.sum(wv * nv_) / np.sum(wv)
            r2 = 1 - np.sum(wv * res ** 2) / np.sum(wv * (nv_ - ym) ** 2)
            prof["%s_plane_%s" % (sc, tag)] = np.array([coef[0], coef[1], coef[2], r2])
            w("    %-8s %-8s cells %4d: b_t %+.5f /m, b_s %+.5f /m, R2 %.3f; vertical gradient at fixed x %+.5f /m"
              % (sc, tag, ok.sum(), coef[1], coef[2], r2, coef[1] * sn + coef[2] * cs30))
    w("  interior = farther than 3 d50 from rock and from the residual void: area %.2f m2; porosity particle %.4f,"
      " block %.4f" % (porosity(c, inner)[1], porosity(c, inner)[0], porosity(cb, inner)[0]))
    w("")
    w("  contact bands (distance from the rock region of each side; band k = (k-1)..k d50, d50 = %.3f m)" % d50)
    dist = {}
    for sd in (1, 2, 3):
        dist[sd] = ndimage.distance_transform_edt(c["rock_side"] != sd) * H
    dvoid = ndimage.distance_transform_edt(~c["void"]) * H
    bands = []
    w("    %-13s band  area m2   particle  block" % "side")
    for sd, dd in ((1, dist[1]), (2, dist[2]), (3, dist[3]), (0, dvoid)):
        name = SIDE_NAME.get(sd, "free surface")
        for k in BAND_D:
            msk = (dd > (k - 1) * d50) & (dd <= k * d50)
            n_p, a_ = porosity(c, msk)
            n_b, _ = porosity(cb, msk)
            bands.append((name, k, a_, n_p, n_b))
            w("    %-13s %d     %7.2f   %.4f    %.4f" % (name, k, a_, n_p, n_b))
    far = (dist[1] > 3 * d50) & (dist[2] > 3 * d50) & (dist[3] > 3 * d50) & (dvoid > 3 * d50)
    n_in, a_in = porosity(c, far)
    n_inb, _ = porosity(cb, far)
    bands.append(("interior", 0, a_in, n_in, n_inb))
    w("    %-13s       %7.2f   %.4f    %.4f   (> 3 d50 from every rock side and from the residual void)" % ("interior", a_in, n_in, n_inb))
    w("    sensitivity of the first band to the rock closing radius (particle scale):")
    for rcd in (0.4, 0.65, 0.9):
        saved = globals()["R_CLOSE_D"]
        globals()["R_CLOSE_D"] = rcd
        cs = classify(img, d50)
        globals()["R_CLOSE_D"] = saved
        ds = dict((sd, ndimage.distance_transform_edt(cs["rock_side"] != sd) * H) for sd in (1, 2, 3))
        farr = (ds[1] > 3 * d50) & (ds[2] > 3 * d50) & (ds[3] > 3 * d50) & (ndimage.distance_transform_edt(~cs["void"]) * H > 3 * d50)
        w("      closing %.2f d50: footwall %.4f, hanging wall %.4f, ore pillar %.4f, interior %.4f"
          % (rcd, porosity(cs, ds[1] <= d50)[0], porosity(cs, ds[2] <= d50)[0], porosity(cs, ds[3] <= d50)[0], porosity(cs, farr)[0]))
    w("")
    w("  residual voids (opening radius = d50 = %.3f m; components >= pi d50^2):" % d50)
    lab, nl = components(c["void"])
    rock_d = ndimage.binary_dilation(c["rock"], structure=np.ones((3, 3), bool))
    reg_d = ndimage.binary_dilation(c["region"], structure=np.ones((3, 3), bool))
    voids = []
    for k in range(1, nl + 1):
        mk = lab == k
        A = mk.sum() * H * H
        if A < math.pi * d50 * d50:
            continue
        jj, ii = np.where(mk)
        t_rock = bool((mk & rock_d).any())
        t_rub = bool((mk & reg_d).any())
        sides = sorted(set(c["rock_side"][ndimage.binary_dilation(mk, iterations=2) & c["rock"]].tolist()) - {0})
        kind = "between rubble and rock" if (t_rock and t_rub) else ("inside rubble" if t_rub else "in rock only")
        voids.append((A, X[mk].mean(), Y[mk].mean(), S[mk].min(), S[mk].max(), xs[ii.min()], xs[ii.max()], ys[jj.min()],
                      ys[jj.max()], kind, ",".join(SIDE_NAME[q] for q in sides) or "none"))
    voids.sort(key=lambda t: -t[0])
    for t in voids:
        w("    area %8.2f m2  centroid (%.2f, %.2f)  s %.2f..%.2f  x %.2f..%.2f  y %.2f..%.2f  %s; rock sides touched: %s" % t)
    w("    %d components, total %.2f m2 (all components incl. small: %.2f m2)"
      % (len(voids), sum(t[0] for t in voids), c["void"].sum() * H * H))
    in_stope = (S >= 0.0) & (S <= THICK) & (Y >= STOPE_Y[0]) & (Y <= STOPE_Y[1])
    w("    inside the original stope outline: void %.2f m2, rubble region %.2f m2, rock region %.2f m2 (stope %.2f m2);"
      " above the original roof (s > 14): void %.2f m2, rubble region %.2f m2"
      % ((c["void"] & in_stope).sum() * H * H, (c["region"] & in_stope).sum() * H * H, (c["rock"] & in_stope).sum() * H * H,
         in_stope.sum() * H * H, (c["void"] & (S > THICK)).sum() * H * H, (c["region"] & (S > THICK)).sum() * H * H))
    hw_ref = (S > s0[rub].max() + 5) & c["rock"]
    n_hw = (hw_ref & (img == 0)).sum() / float(hw_ref.sum())
    src_area = np.sum(np.pi * b["radius"][rub] ** 2) / (1.0 - n_hw)
    w("    2D bulking: rubble region %.2f m2 / in-place area of the same balls %.2f m2 (solid / (1 - %.4f)) = %.3f;"
      " rubble region + residual void = %.2f m2 = %.3f x in-place area + stope area"
      % (porosity(c)[1], src_area, n_hw, porosity(c)[1] / src_area, porosity(c)[1] + c["void"].sum() * H * H,
         (porosity(c)[1] + c["void"].sum() * H * H) / (src_area + in_stope.sum() * H * H)))
    big = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
    top_rub = []
    for i in range(len(xs)):
        col = big[:, i]
        if col.any():
            jlo = np.where(col)[0].min()
            jhi = np.where(col)[0].max()
            top_rub.append(((jhi - jlo + 1) * H, xs[i]))
    hmax = max(top_rub)
    w("    largest void: vertical extent at most %.2f m (at x %.2f); height between the rubble surface and the roof" % hmax)
    for rd in R_VOID_D:
        cc = classify(img, d50, rd)
        lab2, n2 = components(cc["void"])
        nbig = len([q for q in range(1, n2 + 1) if (lab2 == q).sum() * H * H >= math.pi * (rd * d50) ** 2])
        w("    sensitivity: opening radius %.1f d50 (%.3f m): void %.2f m2 in %d components (>= pi r^2);"
          " rubble region %.2f m2, porosity %.4f" % (rd, rd * d50, cc["void"].sum() * H * H, nbig,
                                                      porosity(cc)[1], porosity(cc)[0]))
    w("")

    # ---- 2.4 grain size --------------------------------------------------------------------------------
    w("2.4 Particle size (sorting check only)")
    dia = 2 * b["radius"]
    w("  ball diameter, rubble: mean %.4f, median %.4f m; all balls: mean %.4f, median %.4f m; Mann-Whitney"
      " rubble vs not, two-sided p %.3g" % (dia[rub].mean(), np.median(dia[rub]), dia.mean(), np.median(dia),
                                           2 * stats.mannwhitneyu(dia[rub], dia[~rub])[1]))
    for lab_, cv in (("y", b["y"]), ("s", sf)):
        lr = stats.linregress(cv[rub], dia[rub])
        w("  rubble ball diameter vs %s: slope %+.5f m/m (std err %.5f), r %.3f, p %.3g" % (lab_, lr[0], lr[4], lr[2], lr[3]))
    ib = np.floor((b["x"] - BOX[0]) / BIN).astype(int)
    jb = np.floor((b["y"] - BOX[2]) / BIN).astype(int)
    d50f = np.full((nby, nbx), np.nan)
    for j in range(nby):
        for i in range(nbx):
            sel = rub & (ib == i) & (jb == j)
            if sel.sum() >= 10:
                d50f[j, i] = np.median(dia[sel])
    dv = d50f[np.isfinite(d50f)]
    w("  ball d50 per bin (%d bins with >= 10 rubble balls): min / mean / max %.4f / %.4f / %.4f m; std %.4f m"
      % (len(dv), dv.min(), dv.mean(), dv.max(), dv.std()))
    fr = frag[rub]
    ids = np.unique(fr)
    areas = np.bincount(fr, weights=np.pi * b["radius"][rub] ** 2)[ids]
    counts = np.bincount(fr)[ids]
    deq = np.sqrt(4 * areas / math.pi)
    order = np.argsort(deq)
    cum = np.cumsum(areas[order]) / areas.sum()
    w("  fragments (rubble balls joined by intact parallel bonds): %d; single balls %d; largest %d balls;"
      " equivalent diameter: number median %.3f m, area-weighted d50 %.3f m, max %.3f m"
      % (len(areas), (counts == 1).sum(), counts.max(), np.median(deq), deq[order][np.searchsorted(cum, 0.5)], deq.max()))
    w("  rubble-rubble contacts with intact bond: %d of %d" % (nbond, nrr))
    w("  fragment-size classes (equivalent diameter, share of rubble solid area): " + "; ".join(
        "%s %.3f" % (nm, areas[(deq > lo) & (deq <= hi)].sum() / areas.sum()) for nm, lo, hi in
        (("single ball", 0, 0.97), ("<= 2 m", 0.97, 2.0), ("2-5 m", 2.0, 5.0), ("5-10 m", 5.0, 10.0), ("> 10 m", 10.0, 1e9))))
    fdeq = np.zeros(fr.max() + 1)
    fdeq[ids] = deq
    fball = np.where(rub, fdeq[np.maximum(frag, 0)], np.nan)
    for lab_, cv in (("y", b["y"]), ("s", sf)):
        lr = stats.linregress(cv[rub], fball[rub])
        w("  fragment equivalent diameter (per ball) vs %s: slope %+.4f m/m, r %.3f, p %.3g" % (lab_, lr[0], lr[2], lr[3]))
    fbin = np.full((nby, nbx), np.nan)
    for j in range(nby):
        for i in range(nbx):
            sel = rub & (ib == i) & (jb == j)
            if sel.sum() >= 10:
                fbin[j, i] = np.median(fball[sel])
    w("")

    # ---- 2.5 time evolution ------------------------------------------------------------------------------
    w("2.5 Time evolution (rubble = cumulative displacement > %.1f m at that state; same pixel rules)" % DISP_RUBBLE)
    w("  state  t (s)  rubble  region m2  n_part  n_block  n_int_p  dn/dy(part) R2   dn/dy(block) R2   void m2  speed>0.1")
    evo = []
    for sidx in STATES_TIME:
        nm = "local_stope_result_%d" % sidx
        bs, ms = load(nm)
        rs = np.hypot(bs["dx"], bs["dy"]) > DISP_RUBBLE
        cs = classify(raster(bs, rs), d50)
        fs_, _, _ = fragments(bs, rs, nm)
        cbs = scale_block(cs, block_solid(bs, fs_, d50))
        inn = interior_mask(cs, d50)
        fy = linfit(*profile(cs, Y, 38.0, 100.0, slab))
        fyb = linfit(*profile(cbs, Y, 38.0, 100.0, slab))
        row = (sidx, float(ms["pfcx_time"]) - float(m0["pfcx_time"]), rs.sum(), porosity(cs)[1], porosity(cs)[0],
               porosity(cbs)[0], porosity(cs, inn)[0], fy[1], fy[2], fyb[1], fyb[2], cs["void"].sum() * H * H,
               (np.hypot(bs["vx"], bs["vy"]) > 0.1).sum())
        evo.append(row)
        w("  %5d %6.3f  %6d  %8.1f   %.4f  %.4f   %.4f   %+.5f %.3f   %+.5f %.3f   %7.1f  %6d" % row)
    w("")

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "analysis.txt"), "w").write(text)
    np.savez_compressed(os.path.join(OUT, "fields_final.npz"), img=img, blk=blk, rock=c["rock"], rock_side=c["rock_side"],
                        void=c["void"], region=c["region"], pore=c["pore"], box=np.array(BOX), h=H, d50=d50,
                        field_p=fields["particle"][0], field_b=fields["block"][0], cover=fields["particle"][1],
                        d50f=d50f, fbin=fbin, bin=BIN, rve_p=rve["particle"], rve_b=rve["block"], evo=np.array(evo))
    np.savez_compressed(os.path.join(OUT, "profiles.npz"), **prof)
    np.savez_compressed(os.path.join(OUT, "balls_final.npz"), x=b["x"], y=b["y"], r=b["radius"], rub=rub, frag=frag,
                        side=np.array([SIDE.get(q, 1) for q in b["material"]]), x0=b0["x"], y0=b0["y"], dia=dia, s=sf)
    np.savez_compressed(os.path.join(OUT, "bands.npz"), names=np.array([t[0] for t in bands]),
                        k=np.array([t[1] for t in bands]), a=np.array([t[2] for t in bands]),
                        n_p=np.array([t[3] for t in bands]), n_b=np.array([t[4] for t in bands]))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
