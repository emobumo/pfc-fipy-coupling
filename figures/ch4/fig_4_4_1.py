# -*- coding: utf-8 -*-
"""
Fig. 4.4-1 solution flow chart (section 4.4.1): solver branch (stage loop, time-step loop, Picard) and the reach-domain
branch side by side, joined in post-processing. Rectangles = processes, diamonds = decisions; no parameter values.

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_4_1.py
"""
from __future__ import print_function

from matplotlib.patches import FancyBboxPatch, Polygon

import ch4fig as F
from ch4fig import K, plt

XS, WS = 3.7, 4.2          # solver column centre / box width
XR, WR = 8.15, 3.1         # reach column
H = 0.62
GREY_BG = "#f1f1f1"


def box(ax, xc, yc, text, w, h=H, fill="w"):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="square,pad=0", fc=fill, ec=K.INK, lw=0.8))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=8)
    return dict(t=yc + h / 2, b=yc - h / 2, l=xc - w / 2, r=xc + w / 2, x=xc, y=yc)


def diamond(ax, xc, yc, text, w=3.6, h=1.0):
    ax.add_patch(Polygon([(xc, yc + h / 2), (xc + w / 2, yc), (xc, yc - h / 2), (xc - w / 2, yc)], closed=True,
                         fc="w", ec=K.INK, lw=0.8))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=8)
    return dict(t=yc + h / 2, b=yc - h / 2, l=xc - w / 2, r=xc + w / 2, x=xc, y=yc)


def arrow(ax, pts, label=None, at=0, dx=0.12, dy=0.0, ha="left"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=K.INK, lw=0.8, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=K.INK, shrinkA=0, shrinkB=0, mutation_scale=8))
    if label:
        ax.text(pts[at][0] + dx, pts[at][1] + dy, label, fontsize=7.5, ha=ha, va="center")


def main():
    fig = plt.figure(figsize=(F.WIDTH, 7.4))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.6); ax.axis("off")
    b0 = box(ax, XS, 13.75, u"输入：孔隙率场 $n$，$\\lambda(n)$、$k(n)$\n$p_0$，钻孔与分段方案", WS, h=0.95)
    b1 = box(ax, XS, 12.45, u"段循环：第 $k$ 段", WS)
    b2 = box(ax, XS, 11.4, u"段开始：更新孔隙率场", WS)
    b3 = box(ax, XS, 10.1, u"压力隐式求解", WS)
    b4 = box(ax, XS, 9.1, u"更新面迁移率并松弛", WS)
    d1 = diamond(ax, XS, 7.85, u"Picard 迭代收敛\n或达上限？")
    b5 = box(ax, XS, 6.55, u"饱和度显式更新、充填闭合", WS)
    b6 = box(ax, XS, 5.55, u"按库朗条件调整时间步", WS)
    d2 = diamond(ax, XS, 4.3, u"满足停注判据\n式（4-29）？")
    d3 = diamond(ax, XS, 2.85, u"全部段结束？", h=0.9)
    post = box(ax, 5.65, 1.05, u"后处理：三层评价域判定、拓扑细分、守恒统计", 8.1, h=0.75, fill=GREY_BG)
    r1 = box(ax, XR, 11.4, u"各段开始时的 $\\lambda$ 场", WR)
    r2 = box(ax, XR, 9.1, u"16 邻域多源最短路", WR)
    r3 = box(ax, XR, 6.55, u"所需源压场 $p_{\\mathrm{req}}$", WR)
    r4 = box(ax, XR, 4.3, u"可达域 $p_{\\mathrm{req}} \\leq p_0$", WR)
    # solver chain
    for a, b in ((b0, b1), (b1, b2), (b2, b3), (b3, b4), (b4, d1), (b5, b6), (b6, d2)):
        arrow(ax, [(XS, a["b"]), (XS, b["t"])])
    arrow(ax, [(XS, d1["b"]), (XS, b5["t"])], u"是", at=0, dx=0.12, dy=-0.2)
    arrow(ax, [(XS, d2["b"]), (XS, d3["t"])], u"是：本段结束", at=0, dx=0.12, dy=-0.2)
    arrow(ax, [(XS, d3["b"]), (XS, post["t"])], u"是", at=0, dx=0.12, dy=-0.2)
    # loops on the left
    arrow(ax, [(d1["l"], d1["y"]), (1.15, d1["y"]), (1.15, b3["y"] - 0.12), (b3["l"], b3["y"] - 0.12)], u"否", at=0,
          dx=-0.1, dy=0.17, ha="right")
    arrow(ax, [(d2["l"], d2["y"]), (0.75, d2["y"]), (0.75, b3["y"] + 0.12), (b3["l"], b3["y"] + 0.12)],
          u"否：下一时间步", at=1, dx=0.08, dy=-0.22)
    arrow(ax, [(d3["l"], d3["y"]), (0.35, d3["y"]), (0.35, b1["y"]), (b1["l"], b1["y"])],
          u"否：段间凝固，下一段", at=1, dx=0.08, dy=-0.22)
    # reach branch
    arrow(ax, [(b2["r"], b2["y"]), (r1["l"], r1["y"])])
    for a, b in ((r1, r2), (r2, r3), (r3, r4)):
        arrow(ax, [(XR, a["b"]), (XR, b["t"])])
    arrow(ax, [(XR, r4["b"]), (XR, post["t"])])
    ax.text(XR, 12.7, u"可达域计算", fontsize=8.5, ha="center", va="center")
    ax.text(XS, 14.45, u"求解器", fontsize=8.5, ha="center", va="center")
    ax.plot([6.15, 6.15], [3.3, 11.75], color=F.GREY, lw=0.6)
    ax.text(XR, 12.15, u"两条途径相互独立", fontsize=7.5, color=K.INK, ha="center", va="center")
    F.save(fig, "fig_4_4_1")


if __name__ == "__main__":
    main()
