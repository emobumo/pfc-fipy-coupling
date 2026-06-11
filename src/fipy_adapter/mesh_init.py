# -*- coding: utf-8 -*-
from fipy import Grid2D


def build_mesh(nx=25, ny=15, dx=0.1, dy=0.1, x0=0.0, y0=0.0):
    """
    Build a Grid2D, optionally translated so its lower-left corner sits at
    (x0, y0). The translation makes the FiPy cell coordinates match absolute
    PFC ball positions, which is required for correct porosity binning.
    """
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    if (x0 != 0.0) or (y0 != 0.0):
        mesh = mesh + [[x0], [y0]]

    x = mesh.cellCenters()[0]
    y = mesh.cellCenters()[1]
    fx, fy = mesh.faceCenters()

    return mesh, x, y, fx, fy


def build_mesh_for_domain(x_min, x_max, y_min, y_max, cell_size):
    """
    Build a translated Grid2D covering [x_min, x_max] x [y_min, y_max] with
    near-square cells of about cell_size, aligned to the PFC model extent.
    """
    nx = max(int(round((x_max - x_min) / float(cell_size))), 1)
    ny = max(int(round((y_max - y_min) / float(cell_size))), 1)
    dx = (x_max - x_min) / float(nx)
    dy = (y_max - y_min) / float(ny)
    return build_mesh(nx=nx, ny=ny, dx=dx, dy=dy, x0=x_min, y0=y_min)
