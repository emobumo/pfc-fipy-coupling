# -*- coding: utf-8 -*-
"""Export particle, contact and wall data from PFC2D 6.0 save files (read-only use of the saves).

Run inside PFC2D 6.0 (its bundled Python 3.6, module itasca). Settings come from the
environment so the same file serves the console batch run:
    PFCX_SAVES  = ';'-separated .sav paths
    PFCX_OUT    = output directory
Per save <name>.sav it writes
    <name>_balls.csv     id, x, y, radius, dx, dy, vx, vy, material, excavation
    <name>_contacts.csv  ball-ball contacts: id1, id2, model, pb_state
    <name>_walls.csv     wall id, facet index, x1, y1, x2, y2
    <name>_meta.txt      key = value (counts, mech time, timestep, unbalanced ratio, crack counters)
"""
import os, sys, traceback
import itasca as it
from itasca import ball, wall



def fish_or_none(name):
    try:
        return it.fish.get(name)
    except Exception:
        return None


def export(sav, out):
    name = os.path.splitext(os.path.basename(sav))[0]
    it.command("model restore '%s'" % sav.replace("\\", "/"))
    with open(os.path.join(out, name + "_balls.csv"), "w") as f:
        f.write("id,x,y,radius,dx,dy,vx,vy,material,excavation\n")
        n = 0
        for b in ball.list():
            p, d, v = b.pos(), b.disp(), b.vel()
            f.write("%d,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%s,%s\n" % (
                b.id(), p[0], p[1], b.radius(), d[0], d[1], v[0], v[1],
                b.group("material"), b.group("excavation")))
            n += 1
    nc = nb = 0
    with open(os.path.join(out, name + "_contacts.csv"), "w") as f:
        f.write("id1,id2,model,pb_state\n")
        for c in it.contact.list():
            if type(c).__name__ != "BallBallContact":
                continue
            m = c.model()
            st = c.prop("pb_state") if m == "linearpbond" else -1
            f.write("%d,%d,%s,%d\n" % (c.end1().id(), c.end2().id(), m, int(st)))
            nc += 1
            if st == 3:
                nb += 1
    with open(os.path.join(out, name + "_walls.csv"), "w") as f:
        f.write("wall,facet,x1,y1,x2,y2\n")
        for w in wall.list():
            for k, fc in enumerate(w.facets()):
                a, b2 = [v.pos() for v in fc.vertices()]
                f.write("%d,%d,%.6f,%.6f,%.6f,%.6f\n" % (w.id(), k, a[0], a[1], b2[0], b2[1]))
    for cmd in ("[global pfcx_ratio = mech.solve('ratio-average')]", "[global pfcx_unbal = mech.solve('unbalanced-maximum')]",
                "[global pfcx_time = mech.time.total]", "[global pfcx_dt = global.timestep]", "[global pfcx_step = global.step]"):
        try:
            it.command(cmd)
        except Exception:
            pass
    meta = [("save", sav), ("balls", n), ("contacts_ball_ball", nc), ("bonds_intact", nb)]
    for k in ("pfcx_ratio", "pfcx_unbal", "pfcx_time", "pfcx_dt", "pfcx_step", "crack_num", "crack_tension",
              "crack_shear", "count", "insitu_sxx", "insitu_syy", "exc_target_left", "exc_target_right",
              "exc_target_top", "err_l", "err_r", "err_t", "err_b"):
        meta.append((k, fish_or_none(k)))
    with open(os.path.join(out, name + "_meta.txt"), "w") as f:
        for k, v in meta:
            f.write("%s = %s\n" % (k, v))
    print("exported", name, n, "balls", nc, "contacts", nb, "intact bonds")


OUT = os.environ["PFCX_OUT"]
if not os.path.isdir(OUT):
    os.makedirs(OUT)
LOG = open(os.path.join(OUT, "export_log.txt"), "a")
for s in os.environ["PFCX_SAVES"].split(";"):
    if s.strip():
        try:
            export(s.strip(), OUT)
            LOG.write("ok %s\n" % s.strip())
        except Exception:
            LOG.write("FAILED %s\n%s\n" % (s.strip(), traceback.format_exc()))
        LOG.flush()
LOG.close()
