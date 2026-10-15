# -*- coding: utf-8 -*-
"""
4.6 check, anchors P1b and ST01: injected volume per stage (no solver).

    powershell -File scripts\\run_local.ps1 scripts\\check_4_6\\stage_injections.py

P1b   collar channel (collar_v06, 2.5 m), staged plans: V_in, time and stop
      reason of each stage (outputs/rerun_rate/B/results.json, B_pass2).
ST01  structure fields (base = uniform 0.18, random = realisation 1, 2.5 m):
      full hole and every stage of the staged plans
      (outputs/rerun_rate/C/results.json, C_pass2).
Each line is printed from pass 1 and checked against pass 2.

Output: outputs/check_4_6/stage_injections.txt
"""
from __future__ import print_function

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RR = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(REPO, "outputs", "check_4_6")


def load(name):
    return dict((r["label"].replace("_pass2", ""), r) for r in json.load(open(os.path.join(RR, name, "results.json"))))


def stages(r):
    return [(s.get("reason"), s["v_in"], s["t"]) for s in r["stages"]]


def block(title, rows, rows2, labels):
    lines = [title]
    for lab in labels:
        st = stages(rows[lab])
        lines.append("  %-40s pass 2 identical: %s" % (lab, st == stages(rows2[lab])))
        for k, (why, v, t) in enumerate(st):
            lines.append("    stage %d: V_in %.6f m3/m, t %.4f s, stop %s" % (k + 1, v, t, why))
    return lines + [""]


def main():
    b, b2 = load("B"), load("B_pass2")
    c, c2 = load("C"), load("C_pass2")
    lines = ["4.6 check: injected volume per stage (P1b, ST01), rate stop rule eps 0.1 or quota 185.86 m3/m in total", ""]
    lines += block("P1b  collar channel, staged", b, b2,
                   ["B_p1_collar_v06/staged_7_10_17", "B_p1_collar_v06/staged_8.5_17"])
    lines += block("ST01 structure fields", c, c2,
                   ["C_struct/base_H17_c2.5/single", "C_struct/base_H17_c2.5/7_10_17", "C_struct/base_H17_c2.5/8.5_17",
                    "C_struct/random_H17_c2.5/single", "C_struct/random_H17_c2.5/7_10_17",
                    "C_struct/random_H17_c2.5/8.5_17"])
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "stage_injections.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
