# -*- coding: utf-8 -*-
"""Fill maps of the step-2B solver cases (from outputs/xz_solver, written by run_cases.py --run --summary).

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\plot_cases.py

One figure per case: S at the 0.95 reach-fill snapshot (S1: end state, no snapshot S field was stored), reach-domain
outline (same hole, same p0, 0.5 m), cells filled outside it (numerical creep) hatched, sources, lower end of the
basal layer and the loss zone. Writes figures/xz_step2b/fig_b4_<case>.png (300 dpi) and .svg.
"""
from __future__ import print_function

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "outputs", "xz_solver")
OUT = os.path.join(REPO, "figures", "xz_step2b")
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans SC", "SimHei"],
                            "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9,
                            "legend.fontsize": 7, "savefig.dpi": 300, "svg.fonttype": "none"})
CELL = 0.5
LABEL = {"S1": u"S1：巷道 1，0°，2 MPa，孔口管 2 m", "S2": u"S2：巷道 1，0°，2 MPa，孔口管至 s = 3.5 m",
         "S3": u"S3：巷道 1，0°，1 MPa，孔口管 2 m", "S4": u"S4：巷道 1，−30°，2 MPa，孔口管 2 m",
         "S5": u"S5：巷道 2，−30°，2 MPa，孔口管 2 m"}


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    Sm = np.load(os.path.join(SRC, "summary.npz"))
    X, Z, rub, cls, low = Sm["X"], Sm["Z"], Sm["rub"], Sm["cls"], Sm["low"]
    ext = [X.min() - CELL / 2, X.max() + CELL / 2, Z.min() - CELL / 2, Z.max() + CELL / 2]
    for ci, tag in enumerate(Sm["tags"]):
        fr = os.path.join(SRC, "result_%s.json" % tag)
        if not os.path.isfile(fr):
            continue
        r = json.load(open(fr))
        snap = os.path.join(SRC, "snap_%s_0.95.npz" % tag)
        if os.path.isfile(snap):
            s, when = np.load(snap)["s"], u"0.95 快照（t = %.0f s，V_in = %.1f m³/m）" % (r["snaps"]["0.95"]["t"],
                                                                                   r["snaps"]["0.95"]["v_in"])
        else:
            s, when = np.load(os.path.join(SRC, "final_%s.npz" % tag))["s"], u"终止状态（t = %.0f s，V_in = %.1f m³/m；" \
                u"本算例未存快照 S 场）" % (r["t"], r["v_in"])
        reach = Sm["reach"][ci]
        src = np.load(os.path.join(SRC, "final_%s.npz" % tag))["src"]
        fig, ax = plt.subplots(figsize=(8.6, 4.9))
        ax.imshow(np.ma.masked_where(~((cls >= 1) & (cls <= 3)), np.ones(cls.shape)), origin="lower", extent=ext,
                  cmap=ListedColormap(["#bdbdbd"]), interpolation="nearest")
        ax.imshow(np.ma.masked_where(cls != 4, np.ones(cls.shape)), origin="lower", extent=ext,
                  cmap=ListedColormap(["#cfe3fa"]), interpolation="nearest")
        ax.imshow(np.ma.masked_where(cls != 5, np.ones(cls.shape)), origin="lower", extent=ext,
                  cmap=ListedColormap(["#3d3d3d"]), interpolation="nearest")
        im = ax.imshow(np.ma.masked_where(~rub, np.clip(s, 0, 1)), origin="lower", extent=ext, cmap="YlOrBr", vmin=0,
                       vmax=1, interpolation="nearest")
        out = rub & (s >= 0.5) & ~reach
        if out.any():
            ax.contourf(X[0], Z[:, 0], out.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(X[0], Z[:, 0], reach.astype(float), levels=[0.5], colors=["#1f4e9c"], linewidths=1.0)
        ax.contour(X[0], Z[:, 0], low.astype(float), levels=[0.5], colors=["#7a3b9e"], linewidths=1.0)
        sm = np.zeros(rub.size)
        sm[src] = 1.0
        ax.contour(X[0], Z[:, 0], sm.reshape(rub.shape), levels=[0.5], colors=["k"], linewidths=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(-32, 72); ax.set_ylim(-4, 50)
        ax.set_title(LABEL.get(str(tag), str(tag)) + u"；" + when)
        ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label(u"饱和度 S", fontsize=8)
        hs = [plt.Line2D([], [], color="#1f4e9c"), plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="////"),
              plt.Line2D([], [], color="k", lw=0.8), plt.Line2D([], [], color="#7a3b9e")]
        ax.legend(hs, [u"可达域轮廓（同孔同压，0.5 m）", u"可达域外已充填（数值蠕变，不计入指标）", u"出浆段（源单元）",
                       u"底层下端（t ≤ −3.37 m）"], loc="upper left", frameon=False, fontsize=6.5)
        fig.tight_layout()
        for ext_ in ("png", "svg"):
            fig.savefig(os.path.join(OUT, "fig_b4_%s.%s" % (tag, ext_)), dpi=300)
        plt.close(fig)
        print("wrote fig_b4_%s" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
