# -*- coding: utf-8 -*-
"""Dump the model-setup FISH globals of one PFC2D 6.0 save (read-only use of the save), for the burial depth of
the 4.8 structure input (scripts/check_4_8/prewrite.py reads the output).

Run inside PFC2D 6.0 (bundled Python 3.6, module itasca):
    pfc2d600_console.exe call scripts/pfc_collapse/run_fishvars.p2dat
Settings from the environment: PFCX_SAVE (.sav path, default the final state 50 in E:/pfc_collapse_work/sav),
PFCX_OUT (output directory, default E:/pfc_collapse_work/export). Writes <out>/fishvars_<state>.txt:
"name = value" per FISH global (n/a when the save does not define it), gravity, wall end points and the
ball y range.
"""
import os
import traceback
import itasca as it

SAVE = os.environ.get("PFCX_SAVE", "E:/pfc_collapse_work/sav/local_stope_result_50.sav")
OUT = os.environ.get("PFCX_OUT", "E:/pfc_collapse_work/export")
NAMES = ["ground_elev", "real_elev_offset", "y_top", "y_bot", "x_left", "x_right", "model_top_depth",
         "model_top_real_elev", "stope_center_depth", "stope_center_real_elev", "stope_center_y", "stope_y_bot",
         "stope_h", "stope_y_top", "unit_weight", "insitu_density", "grav_acc", "k0", "sigma_v_stope_center",
         "sigma_h_stope_center", "insitu_sxx", "insitu_syy", "exc_target_left", "exc_target_right", "exc_target_top",
         "init_density", "rock_density", "servo_sxx", "servo_syy", "height", "width", "wall_len_v", "wall_len_h",
         "ore_thick", "wnst1", "wnst2", "wnst3", "wnst4"]

tag = os.path.splitext(os.path.basename(SAVE))[0].split("_")[-1]
out = open(os.path.join(OUT, "fishvars_%s.txt" % tag), "w")
try:
    it.command("model restore '%s'" % SAVE.replace("\\", "/"))
    out.write("save = %s\n" % SAVE)
    for n in NAMES:
        try:
            out.write("%s = %r\n" % (n, it.fish.get(n)))
        except Exception:
            out.write("%s : n/a\n" % n)
    out.write("gravity = %r\n" % (tuple(it.gravity()),))
    for w in it.wall.list():
        out.write("wall %d = %r\n" % (w.id(), [tuple(v.pos()) for f in w.facets() for v in f.vertices()]))
    ys = [b.pos()[1] for b in it.ball.list()]
    out.write("ball_y = %r / %r\n" % (min(ys), max(ys)))
except Exception:
    out.write(traceback.format_exc())
out.close()
