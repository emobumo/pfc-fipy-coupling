# -*- coding: utf-8 -*-
from fipy import Grid2D


def build_mesh(nx=25, ny=15, dx=0.1, dy=0.1):
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    x = mesh.cellCenters()[0]
    y = mesh.cellCenters()[1]
    fx, fy = mesh.faceCenters()

    return mesh, x, y, fx, fy