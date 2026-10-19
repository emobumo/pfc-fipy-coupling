# -*- coding: utf-8 -*-
"""
4.8 pre-draft check, deterministic part (items 4.8.1, 4.8.2, the closed-form and reach items of 4.8.3 and of the
step-2B supplement). Reach graphs and closed forms only, no solver.

    powershell -File scripts\\run_local.ps1 scripts\\check_4_8\\prewrite.py

  1. PFC model boundary and burial depth: FISH globals of the final save (scripts/pfc_collapse/fishvars.py ->
     $PFC_COLLAPSE_EXPORT/fishvars_50.txt) and the top-wall position; depth of the stope floor / roof and of the
     cave-arch top (top of the cavity = PFC rubble region + residual voids, final state) under each reading of the
     boundary; overburden three ways (total gamma H; empirical 1 psi/ft; intact rock above the arch only).
     PFC structure numbers (cavity, in-place area, 2D bulking, stope fill, basal layer) and the three K.
  2. v4 fields (data/xz_prejudge/base_geometry_K*_v4.npz): n statistics and the share outside 0.13..0.29,
     scenario-2 clipping, global F, the four 0 deg (normal) holes: basal-entry and lower-end p_req (0.25 m),
     L_max and x_f = 0.871 L_max at the zone medians (scenario 1).
  3. Zoning areas and hard-to-grout blocks from the stored zone maps; the 0.5 m solver grid (run_cases.py):
     domain, sources, reach domains, p_req from each grouting section to the lower end of the basal layer, the
     lower-end corner under three definitions, and where the lower-end mask sits.

Outputs: outputs/check_4_8/prewrite.txt; prewrite.npz (v4 normal-hole p_req fields for plot.py --v4).
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

RC = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
S2A = RC.S2A
P, A = S2A.P, S2A.A
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.analysis.fill_diagnostics import reachable_domain

OUT = os.path.join(REPO, "outputs", "check_4_8")
DATA = P.DATA
EXPORT = A.EXPORT
CELL = P.CELL
RUB, VOID = P.RUB, P.VOID
C_XF = 0.871                     # x_f / L_max (4.6, anchor SG01)
STRIKE_W = 15.0                  # stope width along strike (framework 4.8)
PSI_FT = 6894.757 / 0.3048       # Pa/m: empirical "1 psi per foot of depth"
GAMMA_FW = 26.0e3                # N/m3, the earlier framework estimate (2.6 MPa at 100 m)
STOPE_Y = (40.0, 80.0)           # model y of the stope floor / roof (analyze.STOPE_Y)


def read_kv(path):
    d = {}
    for line in open(path):
        if " = " in line:
            k, v = line.split(" = ", 1)
            d[k.strip()] = v.strip()
    return d


def npz(K):
    return np.load(os.path.join(DATA, "base_geometry_K%.2f_v4.npz" % K))


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    a2 = CELL * CELL
    w("4.8 pre-draft check, deterministic part (prewrite.py; v0.6; 2D)")
    w("")

    # ---- 1. depth, overburden, PFC structure -------------------------------------------------------------------
    fv = read_kv(os.path.join(EXPORT, "fishvars_50.txt"))
    meta = read_kv(os.path.join(EXPORT, "local_stope_result_50_meta.txt"))
    walls = [l.strip().split(",") for l in open(os.path.join(EXPORT, "local_stope_result_50_walls.csv"))][1:]
    y_top_wall = [float(r[3]) for r in walls if r[0] == "4"][0]
    gam = float(fv["unit_weight"])
    s_top = -float(meta["exc_target_top"])
    s_ins = -float(meta["insitu_syy"])
    y_surf_map = float(fv["ground_elev"]) - float(fv["real_elev_offset"])
    w("1.1 PFC model boundary (final save 50; FISH globals from fishvars_50.txt, wall end points from the export)")
    for k in ("ground_elev", "real_elev_offset", "y_top", "model_top_depth", "unit_weight", "insitu_density", "grav_acc",
              "insitu_syy", "exc_target_top", "stope_y_bot", "stope_y_top", "stope_center_y", "stope_center_depth",
              "wnst4", "ball_y"):
        w("  %s = %s" % (k, fv[k]))
    w("  top wall (id 4) at y = %.4f m (FISH y_top = %s is the value of the in-situ script, not the model top)"
      % (y_top_wall, fv["y_top"]))
    w("  model has no free surface: the top wall is a stress-servo boundary (equivalent overburden); gravity %s"
      % fv["gravity"])
    w("  in-situ stage target %.4f MPa = unit_weight %.0f N/m3 x model_top_depth %s m; model_top_depth = ground_elev"
      " - (real_elev_offset + FISH y_top) -> ground surface at model y %.2f" % (s_ins / 1e6, gam, fv["model_top_depth"],
                                                                               y_surf_map))
    w("  collapse stage (excavation and caving) target %.4f MPa, hard-coded (exc_target_top); final servo reading %.4f MPa"
      % (s_top / 1e6, -float(fv["wnst4"]) / 1e6))
    cav_g = S2A.setup()
    G = cav_g
    c = G["c"]
    xs_p, ys_p = A.grid()
    cav_pix = c["region"] | c["void"]
    arch_pix = ys_p[np.where(cav_pix.any(axis=1))[0].max()] + A.H / 2
    arch_25 = G["Y"][G["cav"]].max() + CELL / 2
    w("  cave-arch top = top of the cavity (PFC rubble region + residual voids): model y %.2f (0.05 m pixels) / %.2f"
      " (0.25 m cells), Z %.2f above the stope floor" % (arch_pix, arch_25, arch_pix - STOPE_Y[0]))
    readings = [("A", "collapse-stage top stress %.2f MPa / gamma -> %.2f m above the top wall"
                 % (s_top / 1e6, s_top / gam), y_top_wall + s_top / gam),
                ("B", "FISH elevation mapping (ground_elev, real_elev_offset)", y_surf_map),
                ("C", "in-situ-stage top stress %.4f MPa / gamma -> %.2f m above the top wall"
                 % (s_ins / 1e6, s_ins / gam), y_top_wall + s_ins / gam)]
    w("  reading: surface y; depth of the stope roof (y %.0f) / floor (y %.0f) / arch top; top-wall stress implied (gamma H"
      " above the wall)" % (STOPE_Y[1], STOPE_Y[0]))
    for tag, txt, ys_ in readings:
        w("   %s %s: surface y %.2f; depths %.2f / %.2f / %.2f m; %.4f MPa" % (
            tag, txt, ys_, ys_ - STOPE_Y[1], ys_ - STOPE_Y[0], ys_ - arch_pix, gam * (ys_ - y_top_wall) / 1e6))
    w("  overburden (MPa) at the stope floor / roof: total gamma H (gamma %.0f N/m3 of the model; in brackets %.0f);"
      " empirical %.1f Pa/m (1 psi/ft); intact rock above the arch only gamma H_arch (independent of the level)"
      % (gam, GAMMA_FW, PSI_FT))
    for tag, txt, ys_ in readings:
        hf, hr, ha = ys_ - STOPE_Y[0], ys_ - STOPE_Y[1], ys_ - arch_pix
        w("   %s: total %.3f (%.3f) / %.3f (%.3f); empirical %.3f / %.3f; above arch %.3f (%.3f)" % (
            tag, gam * hf / 1e6, GAMMA_FW * hf / 1e6, gam * hr / 1e6, GAMMA_FW * hr / 1e6, PSI_FT * hf / 1e6,
            PSI_FT * hr / 1e6, gam * ha / 1e6, GAMMA_FW * ha / 1e6))
    w("   framework check (gamma 26 kN/m3, depth H at the floor): H 100 / 140 m -> total %.2f / %.2f, empirical %.2f / %.2f,"
      " above arch (arch %.2f m above the floor) %.2f / %.2f" % (
          GAMMA_FW * 100 / 1e6, GAMMA_FW * 140 / 1e6, PSI_FT * 100 / 1e6, PSI_FT * 140 / 1e6, arch_pix - STOPE_Y[0],
          GAMMA_FW * (100 - (arch_pix - STOPE_Y[0])) / 1e6, GAMMA_FW * (140 - (arch_pix - STOPE_Y[0])) / 1e6))
    w("")
    w("1.2 PFC structure (0.25 m cells; pixel values from outputs/pfc_collapse/analysis.txt)")
    for line in open(os.path.join(REPO, "outputs", "pfc_collapse", "analysis.txt")):
        if "inside the original stope outline" in line or "2D bulking" in line:
            w("  [analysis.txt] " + line.strip())
    A0 = G["A0"]
    w("  cavity %.2f m2; PFC rubble cells %.2f m2; in-place area A0 = %.2f m2; PFC rubble / A0 = %.4f (0.25 m)"
      % (G["cav"].sum() * a2, G["pfc_rub"].sum() * a2, A0, G["pfc_rub"].sum() * a2 / A0))
    w("  basal (loose footwall) layer in the model: s <= %.1f m (prejudge.BASAL)" % P.BASAL)
    w("  K     rubble m2   void m2    c (v4)   rubble / A0")
    for K in P.KS:
        Dk = npz(K)
        rk = Dk["cls"] == RUB
        w("  %.2f  %9.2f  %8.2f   %.4f   %.4f" % (K, rk.sum() * a2, (Dk["cls"] == VOID).sum() * a2, float(Dk["mapping_c"]),
                                             rk.sum() * a2 / A0))
    w("")
    w("1.3 v4 porosity (rubble cells): K scen: min / median / max; area share < 0.13 / > 0.29; scen 2 clipped to 0.10 /"
      " 0.35 (cells, share)")
    for K in P.KS:
        Dk = npz(K)
        rk = Dk["cls"] == RUB
        for k in (1, 2):
            n = Dk["n%d" % k][rk]
            clip = ""
            if k == 2:
                lo, hi = (n <= 0.10 + 1e-12).sum(), (n >= 0.35 - 1e-12).sum()
                clip = "; clipped %d / %d of %d (%.4f / %.4f)" % (lo, hi, n.size, lo / float(n.size), hi / float(n.size))
            w("   %.2f %d: %.4f / %.4f / %.4f; %.4f / %.4f%s" % (K, k, n.min(), np.median(n), n.max(),
                                                          (n < P.N_LO).mean(), (n > P.N_HI).mean(), clip))
    w("")

    # ---- 2. structure pre-judgement --------------------------------------------------------------------------------
    S, T = npz(P.K_BASE)["S"], npz(P.K_BASE)["T"]
    XR, ZR = npz(P.K_BASE)["X"], npz(P.K_BASE)["Z"]
    ny, nx = S.shape
    w("2.1 global F (least-squares plane of ln lambda on the rubble; F = gamma p0 / (2 rho g)): K scen: p0 1 / 2 MPa")
    for K in P.KS:
        Dk = npz(K)
        rk = Dk["cls"] == RUB
        for k in (1, 2):
            gz = S2A.global_f(rk, Dk["n%d" % k], S, T)[0]
            w("   %.2f %d: %+.4f / %+.4f" % (K, k, gz * P.P0S[0] / (2 * P.RHO * P.G), gz * P.P0S[1] / (2 * P.RHO * P.G)))
    w("")
    mesh = build_mesh_for_domain(P.DOM[0], P.DOM[1], P.DOM[2], P.DOM[3], CELL)[0]
    w("2.2 the four 0 deg (normal) holes, drifts 1..4: grouting cells in the basal layer; basal-entry p_req = min p_req"
      " over basal cells holding a grouting cell; p_req to the lower end of the basal layer (MPa)")
    costs = {}
    for K in P.KS:
        Dk = npz(K)
        bas = Dk["basal"]
        low = Dk["low_end"]
        hc_all, hp = Dk["hole_cells"], Dk["hole_ptr"]
        for k in (1, 2):
            st = P.make_state(mesh, Dk["n%d" % k].ravel())
            parts = []
            for hi in range(len(hp) - 1):
                cells = hc_all[hp[hi]:hp[hi + 1]]
                cost = S2A.cost_of(st, cells, ny, nx)
                if K == P.K_BASE:
                    costs[(k, hi)] = cost
                hc = np.zeros(ny * nx, bool)
                hc[cells] = True
                hc = hc.reshape(ny, nx)
                nb = (bas & hc).sum()
                parts.append("%d cells, entry %s, lower end %.3f" % (nb, ("%.3f" % (cost[bas & hc].min() / 1e6)) if nb else
                                                                      "n/a", cost[low].min() / 1e6))
            w("   %.2f %d: %s" % (K, k, "; ".join(parts)))
    w("  [step2b.txt] table 2-2 (all 8 holes, -30 / 0 deg per drift):")
    on = False
    for line in open(os.path.join(P.OUT, "step2b.txt")):
        if line.startswith("  p_req to the lower end"):
            on = True
        elif on and not line.strip():
            break
        if on:
            w("  " + line.rstrip())
    w("")
    w("2.3 L_max = p0 / lambda(n) and x_f = %.3f L_max at the scenario-1 zone medians (basal / interior / hanging side);"
      " 2 x_f against the %.0f m stope width; holes per row along strike = ceil(%.0f / (2 x_f))" % (C_XF, STRIKE_W,
                                                                                               STRIKE_W))
    for K in P.KS:
        Dk = npz(K)
        rk = Dk["cls"] == RUB
        zs_ = P.zones(rk, Dk["S"], Dk["T"])
        med = [float(np.median(Dk["n1"][m])) for m in zs_]
        for p0 in P.P0S:
            lm = [p0 / P.lam(n) for n in med]
            w("   K %.2f p0 %.0f MPa: n %s; L_max %s m; x_f %s m; 2 x_f %s m; holes %s" % (
                K, p0 / 1e6, " / ".join("%.4f" % n for n in med), " / ".join("%.2f" % v for v in lm),
                " / ".join("%.2f" % (C_XF * v) for v in lm), " / ".join("%.2f" % (2 * C_XF * v) for v in lm),
                " / ".join("%d" % int(math.ceil(STRIKE_W / (2 * C_XF * v))) for v in lm)))
    Db = npz(P.K_BASE)
    rb = Db["cls"] == RUB
    for k in (1, 2):
        nmin = Db["n%d" % k][rb].min()
        w("   min L_max / dx over the rubble, K 1.25 scen %d (n %.4f): p0 1 / 2 MPa: %.2f / %.2f (dx 0.25 m); %.2f / %.2f"
          " (dx 0.5 m)" % (k, nmin, P.P0S[0] / P.lam(nmin) / 0.25, P.P0S[1] / P.lam(nmin) / 0.25,
                           P.P0S[0] / P.lam(nmin) / 0.5, P.P0S[1] / P.lam(nmin) / 0.5))
    w("")

    # ---- 3. zoning, solver grid, lower end ---------------------------------------------------------------------------
    w("3.1 zoning from the stored zone maps: K scen: reliable / conditional / hard m2 (share); hanging side shares")
    for K in P.KS:
        Dk = npz(K)
        rk = Dk["cls"] == RUB
        hng = Dk["hanging"]
        for k in (1, 2):
            z = Dk["zone%d" % k]
            ar = [(z == q).sum() * a2 for q in (0, 1, 2)]
            w("   %.2f %d: %s; %s" % (K, k, " / ".join("%.2f (%.4f)" % (v, v / (rk.sum() * a2)) for v in ar),
                                     " / ".join("%.4f" % ((hng & (z == q)).sum() / float(hng.sum())) for q in (0, 1, 2))))
    zb = P.zones(rb, S, T)
    for k in (1, 2):
        comp = S2A.components(Db["zone%d" % k] == 2, XR, ZR, (zb[2], zb[1], zb[0]), a2)
        w("  K 1.25 scen %d hard-to-grout blocks (%d): area, centroid, X range, Z range, hanging side / interior / basal"
          % (k, len(comp)))
        for r_ in comp:
            w("     %7.2f m2  (%.2f, %.2f)  X %.2f..%.2f  Z %.2f..%.2f  %.2f / %.2f / %.2f" % r_[:10])
    w("")
    D, C, srcs, collar_t = RC.load()
    box = RC.crop_box(C)
    j0, j1, i0, i1 = box
    cr = lambda a: a[j0:j1, i0:i1]
    rub5, hang5, low5 = cr(C["rub"]), cr(C["hang"]), cr(C["low"])
    X5, Z5, T5 = cr(C["X"]), cr(C["Z"]), cr(C["T"])
    b2 = RC.CELL ** 2
    w("3.2 solver grid: %.1f m, crop %d x %d = %d cells, X %.2f..%.2f, Z %.2f..%.2f m; rubble %d cells = %.2f m2"
      % (RC.CELL, i1 - i0, j1 - j0, (i1 - i0) * (j1 - j0), X5.min() - RC.CELL / 2, X5.max() + RC.CELL / 2,
         Z5.min() - RC.CELL / 2, Z5.max() + RC.CELL / 2, rub5.sum(), rub5.sum() * b2))
    w("  case: sources; reach domain cells / m2 / pore m3/m; lower-end cells in the reach domain (of %d); downward"
      " basal storage zone cells (in the reach domain); p_req from the grouting section to the lower end (MPa)"
      % low5.sum())
    reach5 = {}
    for tag, label, _, p0 in RC.CASES:
        M, st, src = RC.case_masks(tag, p0, C, srcs, collar_t, box)
        cost = np.asarray(reachable_domain(st, src, p0=p0)["cost"], float)
        reach5[tag] = M["reach"].reshape(rub5.shape)
        w("   %s (%s): %d; %d / %.2f / %.2f; %d; %d (%d); %.3f" % (
            tag, label, len(src), M["nreach"], M["nreach"] * b2, (M["n"] * M["vols"])[M["reach"]].sum(),
            (M["low"] & M["reach"]).sum(), M["loss"].sum(), (M["loss"] & M["reach"]).sum(), cost[M["low"]].min() / 1e6))
    w("")
    zc1, tc1 = float(D["fan_line"][1][1]), float(D["fan_collar_t"][1])
    w("3.3 lower-end corner; drift-1 collar Z %.2f m, t %.4f m (0 deg hole; the -30 deg hole shares the collar)"
      % (zc1, tc1))
    defs = [("literal: rubble below the drift-1 collar elevation (Z < %.2f)" % zc1, lambda X_, Z_, T_: Z_ < zc1),
            ("down-dip of the drift-1 collar (t < %.4f)" % tc1, lambda X_, Z_, T_: T_ < tc1),
            ("old box X < -5, Z < 20 (not to be cited)", lambda X_, Z_, T_: (X_ < -5.0) & (Z_ < 20.0))]
    for name, f in defs:
        m25 = rb & f(XR, ZR, T)
        m5 = rub5 & f(X5, Z5, T5)
        w("   %s: 0.25 m %.2f m2 (hanging side %.2f); 0.5 m %.2f m2 (hanging side %.2f); X %.2f..%.2f, Z %.2f..%.2f;"
          " share in the reach domain S1..S5 %s" % (
              name, m25.sum() * a2, (m25 & Db["hanging"]).sum() * a2, m5.sum() * b2, (m5 & hang5).sum() * b2,
              X5[m5].min(), X5[m5].max(), Z5[m5].min(), Z5[m5].max(),
              " / ".join("%.4f" % ((reach5[t_] & m5).sum() / float(m5.sum())) for t_ in [c_[0] for c_ in RC.CASES])))
    w("")
    w("3.4 lower end of the basal layer (the mask used since step 1 v2): basal cells (s <= %.1f m) with t <= min t + 2 m"
      % P.BASAL)
    for K in P.KS:
        Dk = npz(K)
        lw, bas = Dk["low_end"], Dk["basal"]
        w("   K %.2f (0.25 m): t <= %.4f m, %d cells = %.2f m2, X %.3f..%.3f, Z %.3f..%.3f; rubble Z min %.3f"
          % (K, Dk["T"][bas].min() + 2.0, lw.sum(), lw.sum() * a2, Dk["X"][lw].min(), Dk["X"][lw].max(), Dk["Z"][lw].min(),
             Dk["Z"][lw].max(), Dk["Z"][Dk["cls"] == RUB].min() - CELL / 2))
    w("   0.5 m solver grid: t <= %.2f m, %d cells, X %.3f..%.3f, Z %.3f..%.3f (cell centres)"
      % (RC.LOW_T, low5.sum(), X5[low5].min(), X5[low5].max(), Z5[low5].min(), Z5[low5].max()))
    w("   original stope floor: Z 0 from X %.2f (hanging-wall corner) to 0 (footwall corner); below it ore pillar (rock,"
      " n = 1e-3) -- the PFC model has no drawpoint or bottom structure" % (-2 * A.HX))
    w("")

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "prewrite.txt"), "w").write(text)
    np.savez_compressed(os.path.join(OUT, "prewrite.npz"), cls=Db["cls"], X=XR, Z=ZR, n1=Db["n1"], n2=Db["n2"],
                        holes=Db["hole_line"], drifts=np.load(os.path.join(P.OUT, "step2b.npz"))["drifts"],
                        cost=np.array([costs[(k, hi)] for k in (1, 2) for hi in range(len(Db["hole_ptr"]) - 1)]))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
