from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy.tools import numerix
from fipy.tools.numerix import nearest
import itasca
import numpy as np
import itertools
import fipy 
import math
import itasca.ball as balls
from scipy.interpolate import Rbf
################## create mesh

nx = 25
dx = 0.1

ny = 15
dy = 0.1

mesh = Grid2D(nx=nx,ny=ny,dx=dx, dy=dy)

print("successful")

phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.5)

hz = mesh.cellCenters()[1].T

##########vg模型参数
ks = 0.35
a = 3.3
n = 4.1
m = 1.-1./n
ss = 0.420
sr = 0.033

#########初始变量
kk = CellVariable(mesh = mesh,
                   value = 0)
qq = CellVariable(mesh = mesh,
                   value = 0)
########底部总水头边界

fx,fy= mesh.faceCenters()

facesT = ((mesh.facesTop & (0.5 >= fx) & (fx >= 0.0)))





phi.grad.constrain(0., mesh.facesLeft)
phi.constrain(0.5, mesh.facesRight)
phi.grad.constrain(0., mesh.facesBottom)

x = mesh.cellCenters()[0].T
y = mesh.cellCenters()[1].T

itasca.command("""
new
restore balance
save mesh
""")