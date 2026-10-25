# -*- coding: utf-8 -*-
"""
Shared helper for the chapter-4 figures drawn in figures/ch4/ (4.3 mechanism figures, main line 2026-10-26).

Uses the chapter settings of scripts/figures_ch4/common.py (fonts, colours, saturation scale) and writes next to the
script: <name>.svg (vector) and <name>.png (600 dpi). Width = A4 text block, 15.5 cm.
"""
from __future__ import print_function

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "figures_ch4"))

import common as K                      # noqa: E402  (rcParams, colours)
from common import plt                  # noqa: E402

WIDTH = 15.5 / 2.54                     # inches
GREY = "#9a9a9a"                        # thin auxiliary lines (no dashes: dashes are for the design domain only)
SOLID = K.SOLID


def tag(ax, letter, dx=0.45, dy=0.08):
    """Panel label (a)(b)(c) at the upper left, dx / dy inches left of / above the axes corner."""
    fig = ax.figure
    w, h = fig.get_size_inches()
    b = ax.get_position()
    fig.text(max(b.x0 - dx / w, 0.004), b.y1 + dy / h, u"(%s)" % letter, ha="left", va="bottom", fontsize=10)


def save(fig, name):
    for ext, dpi in (("svg", None), ("png", 600)):
        fig.savefig(os.path.join(HERE, "%s.%s" % (name, ext)), dpi=dpi or 300)
    print("wrote", os.path.join(HERE, name + ".svg/.png"))
