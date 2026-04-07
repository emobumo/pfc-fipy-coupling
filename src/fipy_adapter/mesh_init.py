from fipy import Grid2D


def build_mesh(nx=25, ny=15, dx=0.1, dy=0.1):
    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

    x = mesh.cellCenters()[0].value
    y = mesh.cellCenters()[1].value
    fx, fy = mesh.faceCenters()

    state = {
        "mesh": mesh,
        "x": x,
        "y": y,
        "fx": fx,
        "fy": fy,
    }

    return state