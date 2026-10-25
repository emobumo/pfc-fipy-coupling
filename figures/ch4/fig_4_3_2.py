# -*- coding: utf-8 -*-
"""
Fig. 4.3-2 corollaries of the reach criterion (analytic / schematic, three panels).

    powershell -File scripts\\run_local.ps1 figures\\ch4\\fig_4_3_2.py

(a) corollary 1: homogeneous rubble, r(theta) = L_max/(1 + Pi_g sin theta), eq. (4-17), Pi_g = 0.3 for display
    (0.03-0.13 at the base parameters); circle Pi_g = 0 for reference. Coordinates x/L_max, z/L_max.
(b) corollary 2: straight up / down rays in a vertically linear lambda field, lambda = lambda0 (1 -/+ gamma r),
    dimensionless: r/L_max, lambda/lambda0, p_req/p0 = r (1 +/- Pi_g) -/+ (gamma L_max/2) r^2;
    F = gamma p0 / (2 rho g) = gamma L_max / (2 Pi_g). Pi_g = 0.10, gamma L_max = 0.52 -> F = 2.6.
(c) corollary 3: one-dimensional, slope lambda outside the wall and lambda_w inside (schematic ratio 20, no values).
"""
from __future__ import print_function

import numpy as np

import ch4fig as F
from ch4fig import K, plt

PI_A = 0.3                               # (a)
PI_B, GL = 0.10, 0.52                    # (b): Pi_g, gamma * L_max
FF = GL / (2 * PI_B)
R_MARK = 1.0                             # (b) left: r at which the areas are shown
WALL, LW_RATIO, AVAIL = (0.42, 0.52), 20.0, 0.8   # (c) schematic


def axes_in(fig, l, b, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes([l / W, b / H, w / W, h / H])


def panel_a(ax):
    th = np.linspace(0, 2 * np.pi, 721)
    r = 1.0 / (1 + PI_A * np.sin(th))
    ax.plot(np.cos(th), np.sin(th), color=F.GREY, lw=0.8, label=u"无重力（$\\Pi_g$ = 0）：圆")
    ax.plot(r * np.cos(th), r * np.sin(th), color=K.INK, lw=1.5, label=u"可达域边界：以源点为焦点的椭圆")
    up, dn = 1 / (1 + PI_A), 1 / (1 - PI_A)
    kw = dict(arrowstyle="-|>", lw=0.8, color=K.INK, shrinkA=0, shrinkB=0, mutation_scale=7)
    for end in ((0, up), (0, -dn), (1, 0)):
        ax.annotate("", xy=end, xytext=(0, 0), arrowprops=kw)
    ax.text(-0.05, 0.25, u"$L_{\\mathrm{max}}/(1+\\Pi_g)$", fontsize=7, ha="right", va="center")
    ax.text(0.05, -1.12, u"$L_{\\mathrm{max}}/(1-\\Pi_g)$", fontsize=6.5, ha="left", va="center")
    ax.text(0.5, 0.05, u"$L_{\\mathrm{max}}$", fontsize=8, ha="center", va="bottom")
    ax.plot([0], [0], "o", color=K.INK, ms=4, zorder=5)
    ax.text(0.07, -0.08, u"源点 $x_s$（焦点）", fontsize=7.5, va="top")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.25)
    ax.set_xlabel(u"$x/L_{\\mathrm{max}}$")
    ax.set_ylabel(u"$z/L_{\\mathrm{max}}$")
    K.style(ax)


def panel_a_note(fig, ax):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.50, 0.955), frameon=False, fontsize=8)
    fig.text(0.515, 0.86, u"$r(\\theta)=L_{\\mathrm{max}}/(1+\\Pi_g\\sin\\theta)$，θ 自水平向上为正", fontsize=8)
    fig.text(0.515, 0.83, u"向上距离缩短，向下距离伸长；离心率为 $\\Pi_g$", fontsize=8)


def panel_b_left(ax):
    r = np.linspace(0, 1.5, 301)
    ax.plot(r, np.ones_like(r), color=F.GREY, lw=0.8)
    ax.plot(r, 1 - GL * r, color=K.C1, lw=1.4, label=u"向上射线")
    ax.plot(r, 1 + GL * r, color=K.C2, lw=1.4, label=u"向下射线")
    s = r <= R_MARK
    ax.fill_between(r[s], 1 - GL * r[s], 1.0, color=K.C1, alpha=0.18, lw=0)
    ax.fill_between(r[s], 0.0, PI_B, color="#c8c8c8", lw=0)
    ax.axvline(R_MARK, color=F.GREY, lw=0.6)
    ax.text(R_MARK + 0.02, 1.62, u"$r$", fontsize=8)
    ax.text(1.53, 1.0, u"$\\lambda_0$", fontsize=8, va="center")
    ax.annotate(u"结构节省的累积启动压力\n$\\frac{1}{2}\\gamma\\lambda_0 r^2$", (0.66, 0.84), xytext=(0.05, 0.36),
                fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1))
    ax.text(0.05, 0.15, u"向上须克服的重力压力 $\\rho g\\,r$", fontsize=7)
    ax.text(R_MARK + 0.03, 0.5 * PI_B, u"$\\rho g$", fontsize=7, va="center")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel(u"沿射线距离 $r/L_{\\mathrm{max}}$")
    ax.set_ylabel(u"启动压力梯度 $\\lambda/\\lambda_0$")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    K.style(ax)


def preq(r, up):
    return r * (1 + PI_B) - 0.5 * GL * r ** 2 if up else r * (1 - PI_B) + 0.5 * GL * r ** 2


def panel_b_right(ax):
    r = np.linspace(0, 1.6, 321)
    ax.plot(r, preq(r, True), color=K.C1, lw=1.4, label=u"向上射线")
    ax.plot(r, preq(r, False), color=K.C2, lw=1.4, label=u"向下射线")
    ax.axhline(1.0, color=F.GREY, lw=0.8)
    ax.text(0.03, 1.03, u"源压 $p_0$", fontsize=7.5, va="bottom")
    r_up = ((1 + PI_B) - np.sqrt((1 + PI_B) ** 2 - 2 * GL)) / GL
    r_dn = (-(1 - PI_B) + np.sqrt((1 - PI_B) ** 2 + 2 * GL)) / GL
    for rr, col, name, dx, dy, ha in ((r_up, K.C1, u"$r_{\\mathrm{up}}$", 0.03, 0.9, "left"),
                                      (r_dn, K.C2, u"$r_{\\mathrm{down}}$", -0.03, 1.04, "right")):
        ax.plot([rr, rr], [0, 1], color=col, lw=0.6)
        ax.plot([rr], [1], "o", color=col, ms=4, zorder=5)
        ax.text(rr + dx, dy, name, color=col, fontsize=8, ha=ha, va="bottom" if dy > 1 else "top")
    ax.text(0.07, 1.19, u"$F$ > 1：向上偏置（翻转）", fontsize=7.5, ha="left")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(u"沿射线距离 $r/L_{\\mathrm{max}}$")
    ax.set_ylabel(u"所需源压 $p_{\\mathrm{req}}/p_0$")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    K.style(ax)
    return r_up, r_dn


def panel_c(ax):
    x0, x1 = WALL
    x = np.linspace(0, 1, 1001)
    T = np.where(x < x0, x, np.where(x < x1, x0 + LW_RATIO * (x - x0), x0 + LW_RATIO * (x1 - x0) + (x - x1)))
    ax.axvspan(x0, x1, color=F.SOLID, lw=0)
    ax.plot(x, T, color=K.INK, lw=1.4)
    ax.axhline(AVAIL, color=F.GREY, lw=0.8)
    ax.text(0.99, AVAIL + 0.06, u"可用驱动压力", fontsize=7.5, ha="right", va="bottom")
    xc = x0 + (AVAIL - x0) / LW_RATIO
    ax.plot([xc], [AVAIL], "o", color=K.INK, ms=4, zorder=5)
    ax.text(0.20, 0.30, u"斜率 λ", fontsize=7.5, ha="center", va="bottom")
    ax.text(x1 + 0.012, 1.55, u"斜率 $\\lambda_w$", fontsize=7.5, va="center")
    ax.text(0.78, 2.75, u"斜率 λ", fontsize=7.5, ha="center")
    ax.annotate(u"胶结体（阻隔墙），厚度 $w$", (0.5 * (x0 + x1), 0.15), xytext=(0.56, 0.35), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=K.INK, shrinkA=1, shrinkB=1))
    ax.text(0.18, AVAIL + 0.35, u"可达", fontsize=8, ha="center")
    ax.text(0.78, AVAIL + 0.35, u"不可达", fontsize=8, ha="center")
    ax.text(0.04, 2.85, u"只取决于 $\\lambda_w$，与墙体渗透率无关", fontsize=8, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(u"自源点的距离")
    ax.set_ylabel(u"累积启动压力")
    K.style(ax)


def main():
    fig = plt.figure(figsize=(F.WIDTH, 7.3))
    ax = axes_in(fig, 0.45, 4.62, 2.5, 2.42)
    panel_a(ax)
    F.tag(ax, "a")
    panel_a_note(fig, ax)
    ax = axes_in(fig, 0.5, 2.38, 2.3, 1.75)
    panel_b_left(ax)
    F.tag(ax, "b")
    r_up, r_dn = panel_b_right(axes_in(fig, 3.55, 2.38, 2.3, 1.75))
    ax = axes_in(fig, 0.5, 0.42, 5.35, 1.45)
    panel_c(ax)
    F.tag(ax, "c")
    F.save(fig, "fig_4_3_2")
    print("(a) up %.4f down %.4f; (b) F %.2f r_up %.4f r_down %.4f ratio %.3f" %
          (1 / (1 + PI_A), 1 / (1 - PI_A), FF, r_up, r_dn, r_up / r_dn))


if __name__ == "__main__":
    main()
