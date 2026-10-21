# -*- coding: utf-8 -*-
"""
Fig. 4.1-3 (draft): research route and argument framework of chapter 4. A block diagram, no computed results.

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_1_3.py [work|final]

Writes fig_4_1_3_<NUMBERING>.png/.svg (the command-line argument overrides NUMBERING). All wording is fixed by the
main-line spec (2026-10-22); terms per docs/术语对照表_v11_2026-10-21.md. Chinese in SimSun (Noto Serif CJK SC if
SimSun is missing), Latin letters, digits and symbols in Times New Roman (italic symbols as mathtext with a Times
fontset; relation signs and the middle dot as plain Times, so the spacing is the spec's).
Mixed lines are drawn run by run and advanced by the fonts' own glyph advances. Width 16 cm, height <= 20 cm,
section titles 10.5 pt, items 9 pt. Layout in millimetres, y measured downward from the top edge (negative).
"""
from __future__ import print_function

import os
import re
import sys

import matplotlib
import common as K
from common import plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ft2font import FT2Font, LOAD_NO_HINTING
from matplotlib.patches import FancyBboxPatch, Rectangle

NUMBERING = "work"                       # "work" / "final"
KEYS = ("s1", "s3", "s4", "s5", "s6", "s7", "s8", "s9")
NUMBERS = {"work": ("4.1", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9"),
           "final": ("4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8")}

FONTS = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                            "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold"})

DPI = 300
W, HMAX = 160.0, 200.0                   # mm
TITLE, ITEM = 10.5, 9.0                  # pt
PT = 25.4 / 72.0                         # mm per pt
LEAD = ITEM * 1.22 * PT                  # item line pitch
TROW = 4.8                               # title row
PAD = 1.2
GAP = 5.5
XL0, XL1 = 1.0, 8.0                      # stage labels
X0, X1 = 11.0, 150.0                     # section boxes
XM = 0.5 * (X0 + X1)                     # lane divider
XR = X1 + 3.5                            # dashed return path
INSET = 2.5                              # text inset in a box
LANE_L, LANE_R = "#c6daf0", "#fdeedd"    # light blue / light orange, grey levels 0.84 / 0.94
GREY_FILL, GREY_EDGE = "#e6e6e6", "#7f7f7f"
LW_BOX, LW_ARROW = 0.6, 0.8

TEXT = {
    "input": u"第 3 章　废石散体塌落形成过程与堆积结构",
    "stages": (u"提出问题与建立方法", u"结构规律与调控", u"工程检验与应用"),
    "s1": u"工程问题与结构概化",
    "s3": u"渗流模型与可达判据",
    "s4": u"数值方法与验证",
    "s5": u"结构对扩散与充填的控制规律",
    "s6": u"分段与阻隔调控",
    "s7": u"同类灌浆工程的机理检验",
    "s8": u"研究区可灌性预测与灌浆方案",
    "s9": u"本章小结：依次回答三个问题",
    "next": u"后续章节：在可靠区内布置承载结构",
    "tag5": u"回答问题①②",
    "tag6": u"回答问题③",
    "lane_l": (u"可达层面", u"启动压力梯度的路径积分决定可达域", u"（结构与注浆压力）"),
    "lane_r": (u"过程层面", u"渗透率分布与停注规则决定充填域", u"（充填过程）"),
    "design": u"设计含义",
    "check": u"机理检验",
}
# items: a string is one line; a tuple (lines, indent prefix) wraps with a hanging indent of the prefix's width
ITEMS = {
    "s1": [((u"三个问题：① 结构如何改变浆液扩散　② 结构与重力如何共同影响充填",
             u"③ 给定施工条件与停注规则如何影响残余未充填区"), u"三个问题："),
           u"等效孔隙率场：孔隙率梯度、贯通通道、随机非均质、架空骨架"],
    "s3": [u"宾汉流体变饱和渗流模型", u"路径积分可达判据 → 可达域", u"可达域为充填域的上包络",
           u"三层评价域：设计域、可达域、充填域"],
    "s4": [u"有限体积求解器 → 充填域", u"解析基准验证", u"可达域与求解器相互印证", u"停注判据"],
    "s5": ([u"重力向下偏置与梯度翻转（判据 $F$）", u"通道入口所需源压", u"致密包裹体"],
           [u"优势通道优先分流", u"锋面截断区、绕流空区", u"随机非均质：混合成因"]),
    "s6": ([u"后续段有效：$\\Delta$ > $x_f$", u"前序胶结体的双重作用：段间阴影区／封堵",
            ((u"调控：改写可达域", u"（孔位与压力、封底与封堵、段长）"), u"调控：")],
           [u"停注锋面距离 $x_f$ = $c(\\varepsilon)$·$L_{\\mathrm{max}}$", u"分段覆盖损失两态",
            u"调控：改变截断方式（停注判据、量压双控）"]),
    "s7": ([u"钻孔穿过交界面时跑浆位置可达", u"交界面临界孔隙率反推", u"封底切断向下路径"],
           [u"按估算浆量停止与持续注浆的对比", u"现场症状与调控措施对照"]),
    "s8": ([u"注浆压力窗口与结构预判", u"可灌性分区（可靠区、条件区、难灌区）", u"分步可达域与施工顺序"],
           [u"可达域充填与出矿口开放时的流失", u"先封后注、浆量上限与施工监测"]),
}


def cjk_font():
    for name in ("simsun.ttc", "NotoSerifCJKsc-Regular.otf", "NotoSerifCJK-Regular.ttc"):
        p = os.path.join(FONTS, name)
        if os.path.isfile(p):
            return p
    raise SystemExit("neither SimSun nor Noto Serif CJK SC found in %s" % FONTS)


CJK = cjk_font()
WEST = findfont(FontProperties(family="Times New Roman"))
if "times" not in os.path.basename(WEST).lower():
    raise SystemExit("Times New Roman not found (got %s)" % WEST)
_FT = {}


def is_cjk(ch):
    o = ord(ch)
    return o >= 0x2E80 or 0x2460 <= o <= 0x24FF          # CJK, full-width forms, circled digits


def runs(s):
    """'$...$' -> mathtext run; otherwise runs of CJK / western characters."""
    out = []
    for part in re.split(r"(\$[^$]*\$)", s):
        if not part:
            continue
        if part[0] == "$":
            out.append(("m", part))
            continue
        for ch in part:
            k = "c" if is_cjk(ch) else "w"
            if out and out[-1][0] == k:
                out[-1] = (k, out[-1][1] + ch)
            else:
                out.append((k, ch))
    return out


def advance(path, s, size):
    """Unhinted advance width of a plain string, mm."""
    ft = _FT.get(path)
    if ft is None:
        ft = _FT[path] = FT2Font(path)
    ft.set_size(size, 72)
    return sum(ft.load_char(ord(c), flags=LOAD_NO_HINTING).linearHoriAdvance for c in s) / 65536.0 * PT


class Page(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 25.4, HMAX / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")
        self.R = self.fig.canvas.get_renderer()

    def text(self, x, y, s, size, ha="left", limit=None):
        """Draw a mixed line with its baseline at y; returns its width (mm). limit = right edge it must not pass."""
        arts, ws = [], []
        for k, t in runs(s):
            if k == "m":
                a = self.ax.text(0, y, t, fontsize=size, va="baseline", ha="left", color="k")
                w = a.get_window_extent(self.R).width / float(DPI) * 25.4
            else:
                path = CJK if k == "c" else WEST
                a = self.ax.text(0, y, t, fontproperties=FontProperties(fname=path, size=size), va="baseline",
                                 ha="left", color="k")
                w = advance(path, t, size)
            arts.append(a)
            ws.append(w)
        tot = sum(ws)
        x = x - {"left": 0.0, "center": 0.5 * tot, "right": tot}[ha]
        if limit is not None and x + tot > limit + 1e-6:
            raise SystemExit(u"line overflows its box by %.2f mm: %s" % (x + tot - limit, s))
        for a, w in zip(arts, ws):
            a.set_x(x)
            x += w
        return tot

    def width(self, s, size):
        n = len(self.ax.texts)
        w = self.text(0.0, 0.0, s, size)
        for a in self.ax.texts[n:]:
            a.remove()
        return w

    def vtext(self, x, ymid, s, size):
        """Characters stacked vertically, centred on (x, ymid)."""
        step = 1.15 * size * PT
        y = ymid + 0.5 * step * (len(s) - 1)
        for ch in s:
            self.text(x, y - 0.38 * size * PT, ch, size, ha="center")
            y -= step

    def box(self, x0, top, x1, bottom, fill="w", rounded=False, edge="k"):
        if rounded:
            p = FancyBboxPatch((x0, bottom), x1 - x0, top - bottom, boxstyle="round,pad=0,rounding_size=1.6",
                               fc=fill, ec=edge, lw=LW_BOX, zorder=2)
        else:
            p = Rectangle((x0, bottom), x1 - x0, top - bottom, fc=fill, ec=edge, lw=LW_BOX, zorder=2)
        self.ax.add_patch(p)

    def arrow(self, pts, dashed=False):
        """Polyline through pts with an arrowhead at the last point."""
        xs, ys = zip(*pts)
        ln, = self.ax.plot(xs, ys, color="k", lw=LW_ARROW, zorder=3, solid_capstyle="butt")
        if dashed:
            ln.set_dashes((3, 2))
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        d = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5
        tail = (xb - 1.6 * (xb - xa) / d, yb - 1.6 * (yb - ya) / d)
        self.ax.annotate("", xy=(xb, yb), xytext=tail, zorder=4,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", lw=LW_ARROW, color="k",
                                         shrinkA=0, shrinkB=0, mutation_scale=8))


def baseline(top, k):
    """Baseline of item line k (0-based) under an items area starting at top."""
    em = ITEM * PT
    return top - k * LEAD - 0.5 * (LEAD - em) - 0.88 * em


def nlines(items):
    return sum(len(it[0]) if isinstance(it, tuple) else 1 for it in items)


def draw_items(P, x, top, items, limit):
    k = 0
    for it in items:
        lines, indent = (it[0], P.width(it[1], ITEM)) if isinstance(it, tuple) else ((it,), 0.0)
        for j, s in enumerate(lines):
            P.text(x + (indent if j else 0.0), baseline(top, k), s, ITEM, limit=limit)
            k += 1


def section(P, x0, top, x1, title, items, lanes=False, tag=None):
    """Section box: title row, then items (one column, or left / right lane columns). Returns the bottom."""
    n = max(nlines(items[0]), nlines(items[1])) if lanes else nlines(items)
    bottom = top - (PAD + TROW + n * LEAD + PAD)
    if lanes:                                # lane tint shows through below the white title row
        P.box(x0, top, x1, bottom, fill="none")
        P.ax.add_patch(Rectangle((x0, top - PAD - TROW), x1 - x0, PAD + TROW, fc="w", ec="none", zorder=1.5))
    else:
        P.box(x0, top, x1, bottom)
    P.text(x0 + INSET, top - PAD - 0.88 * TITLE * PT, title, TITLE)
    if tag is not None:
        tw = P.width(tag, ITEM)
        xr = x1 - INSET
        P.ax.add_patch(Rectangle((xr - tw - 2.0, top - PAD - TROW + 0.5), tw + 2.0, TROW - 0.4, fc="w",
                                 ec=GREY_EDGE, lw=LW_BOX, zorder=2.5))
        P.text(xr - 1.0, top - PAD - 0.88 * TITLE * PT, tag, ITEM, ha="right")
    itop = top - PAD - TROW
    if lanes:
        draw_items(P, x0 + INSET, itop, items[0], XM - INSET + 0.5)
        draw_items(P, XM + INSET, itop, items[1], x1 - INSET + 0.5)
    else:
        draw_items(P, x0 + INSET, itop, items, x1 - INSET + 0.5)
    return bottom


def draw(numbering):
    num = dict(zip(KEYS, NUMBERS[numbering]))
    head = lambda k: u"%s　%s" % (num[k], TEXT[k])
    P = Page()
    ax = P.ax
    y = -2.5
    # (0) input
    bin_top, bin_bot = y, y - 7.0
    wi = P.width(TEXT["input"], TITLE) + 10.0
    P.box(XM - wi / 2, bin_top, XM + wi / 2, bin_bot, fill=GREY_FILL, rounded=True, edge=GREY_EDGE)
    P.text(XM, 0.5 * (bin_top + bin_bot) - 0.36 * TITLE * PT, TEXT["input"], TITLE, ha="center")
    # stage one: 4.1, 4.3 | 4.4, lane header
    t1 = bin_bot - GAP
    b1 = section(P, X0, t1, X1, head("s1"), ITEMS["s1"])
    t3 = b1 - GAP
    wrow = X1 - X0 - 8.0
    x3b, x4a = X0 + 0.6 * wrow, X0 + 0.6 * wrow + 8.0
    b3 = section(P, X0, t3, x3b, head("s3"), ITEMS["s3"])
    b4 = section(P, x4a, t3, X1, head("s4"), ITEMS["s4"])
    if abs(b3 - b4) > 1e-9:
        raise SystemExit("4.3 and 4.4 boxes differ in height")
    th = b3 - GAP
    bh = th - (PAD + TROW + 2 * LEAD + PAD)
    # stage two / three: 4.5, 4.6 | 4.7, 4.8 (lane tint from the header to the bottom of 4.8, drawn afterwards)
    t5 = bh - GAP
    b5 = section(P, X0, t5, X1, head("s5"), ITEMS["s5"], lanes=True, tag=TEXT["tag5"])
    t6 = b5 - GAP
    b6 = section(P, X0, t6, X1, head("s6"), ITEMS["s6"], lanes=True, tag=TEXT["tag6"])
    t7 = b6 - GAP
    b7 = section(P, X0, t7, X1, head("s7"), ITEMS["s7"], lanes=True)
    t8 = b7 - GAP
    b8 = section(P, X0, t8, X1, head("s8"), ITEMS["s8"], lanes=True)
    ax.add_patch(Rectangle((X0, b8), XM - X0, th - b8, fc=LANE_L, ec="none", zorder=0))
    ax.add_patch(Rectangle((XM, b8), X1 - XM, th - b8, fc=LANE_R, ec="none", zorder=0))
    for (xa, xb), lines in (((X0, XM - 1.5), TEXT["lane_l"]), ((XM + 1.5, X1), TEXT["lane_r"])):
        P.box(xa, th, xb, bh, fill="none")
        xc = 0.5 * (xa + xb)
        P.text(xc, th - PAD - 0.88 * TITLE * PT, lines[0], TITLE, ha="center")
        for k, s in enumerate(lines[1:]):
            P.text(xc, baseline(th - PAD - TROW, k), s, ITEM, ha="center", limit=xb - INSET)
    # (h) end row
    te, be = b8 - GAP, b8 - GAP - 7.0
    w9, wn = P.width(head("s9"), TITLE) + 8.0, P.width(TEXT["next"], TITLE) + 8.0
    x9 = XM - 0.5 * (w9 + 10.0 + wn)
    xn = x9 + w9 + 10.0
    P.box(x9, te, x9 + w9, be)
    P.box(xn, te, xn + wn, be, fill=GREY_FILL, rounded=True, edge=GREY_EDGE)
    yb = 0.5 * (te + be) - 0.36 * TITLE * PT
    P.text(x9 + w9 / 2, yb, head("s9"), TITLE, ha="center")
    P.text(xn + wn / 2, yb, TEXT["next"], TITLE, ha="center")
    H = -(be - 2.5)
    if H > HMAX:
        raise SystemExit("figure height %.1f mm > %.0f mm" % (H, HMAX))
    # stage labels
    for (st, sb), s in zip(((t1, bh), (t5, b6), (t7, b8)), TEXT["stages"]):
        ax.add_patch(Rectangle((XL0, sb), XL1 - XL0, st - sb, fc=GREY_FILL, ec="none", zorder=1))
        P.vtext(0.5 * (XL0 + XL1), 0.5 * (st + sb), s, TITLE)
    # main chain
    x3c, x4c = 0.5 * (X0 + x3b), 0.5 * (x4a + X1)
    P.arrow([(XM, bin_bot), (XM, t1)])
    P.arrow([(x3c, b1), (x3c, t3)])
    P.arrow([(x3b, 0.5 * (t3 + b3)), (x4a, 0.5 * (t3 + b3))])
    yg = b4 - 0.5 * GAP
    P.arrow([(x4c, b4), (x4c, yg), (XM, yg), (XM, t5)])              # between the two header halves
    P.arrow([(XM, b5), (XM, t6)])
    P.arrow([(XM, b6), (XM, t7)])
    P.arrow([(XM, b7), (XM, t8)])
    P.text(XM + 1.8, 0.5 * (b7 + t8) - 0.36 * ITEM * PT, TEXT["design"], ITEM)
    x9c = x9 + w9 / 2
    P.arrow([(x9c, b8), (x9c, te)])
    P.arrow([(x9 + w9, 0.5 * (te + be)), (xn, 0.5 * (te + be))])
    # dashed return: 4.7 -> right sides of 4.5 and 4.6
    y5, y6, y7 = 0.5 * (t5 + b5), 0.5 * (t6 + b6), 0.5 * (t7 + b7)
    P.arrow([(X1, y7), (XR, y7), (XR, y5), (X1, y5)], dashed=True)
    P.arrow([(XR, y6), (X1, y6)], dashed=True)
    P.vtext(XR + 3.3, 0.5 * (y6 + y7), TEXT["check"], ITEM)
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    P.fig.set_size_inches(W / 25.4, H / 25.4)
    name = "fig_4_1_3_%s" % numbering
    for ext in ("png", "svg"):
        P.fig.savefig(os.path.join(K.OUT, "%s.%s" % (name, ext)), dpi=DPI)
    print("wrote", os.path.join(K.OUT, name + ".png/.svg"), "%.1f x %.1f mm" % (W, H), "CJK font", CJK)
    plt.close(P.fig)


if __name__ == "__main__":
    for v in sys.argv[1:] or [NUMBERING]:
        draw(v)
