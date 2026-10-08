# -*- coding: utf-8 -*-
"""
figures/ch4_draft/contact_sheet.pdf: every draft figure, in figure-number order, one per page.

    powershell -File scripts\run_local.ps1 scripts\figures_ch4\contact_sheet.py
"""
from __future__ import print_function

import glob
import os
import re

import common as K
from common import plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg


def key(p):
    return [int(v) for v in re.findall(r"\d+", os.path.basename(p))]


def main():
    pngs = sorted(glob.glob(os.path.join(K.OUT, "fig_4_*.png")), key=key)
    out = os.path.join(K.OUT, "contact_sheet.pdf")
    with PdfPages(out) as pdf:
        for p in pngs:
            img = mpimg.imread(p)
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(8.27, 8.27 * h / float(w) + 0.5))
            ax = fig.add_axes([0, 0, 1, 1 - 0.5 / (8.27 * h / float(w) + 0.5)])
            ax.imshow(img); ax.axis("off")
            n = re.findall(r"fig_(\d+)_(\d+)_(\d+)", os.path.basename(p))[0]
            fig.text(0.02, 0.99, u"图 %s.%s-%s" % n, va="top", fontsize=11)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print("wrote", out, len(pngs), "pages")


if __name__ == "__main__":
    main()
