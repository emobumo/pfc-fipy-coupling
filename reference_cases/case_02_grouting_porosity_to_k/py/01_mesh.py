from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm,Gmsh2D
from fipy.tools import numerix
import itasca
import numpy as np
import math
import fipy as fp
import itertools
import itasca.ball as balls
################## create mesh
mesh = fp.Gmsh2D("""
cl__1 = 0.2;
Point(1) = {0, 0, 0, cl__1};
Point(2) = {0.85, 0, 0, cl__1};
Point(3) = {0.85, 0.85, 0, cl__1};
Point(4) = {0, 0.85, 0, cl__1};

Point(5) = {0.425, 0.425, -0, 0.07};
Point(6) = {0.3825, 0.425, -0, 0.07};
Point(7) = {0.4675, 0.425, -0, 0.07};

Line(1) = {4, 1,0.1};
Line(2) = {1, 2,0.1};
Line(3) = {2, 3,0.1};
Line(4) = {3, 4,0.1};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 6};
Line Loop(9) = {1, 2, 3, 4, -5, -6};
Plane Surface(9) = {9};
Physical Surface("face") = {9};
Physical Line("outlet") = {1, 2, 3, 4};
Physical Line("inlet") = {5,6};
          """% locals())
print("successful")


num_load = 0
kk = CellVariable(name='mobility',mesh=mesh, value=1e-6)
################## save mesh point 
bs = 0.85*0.85-3.1415926*0.085*0.085/4
m_radius = math.sqrt((bs/len(mesh.cellCenters[0]))/3.14)
with open('geo_information.txt','w') as f:
     for ii in range(int(np.shape(mesh._cellVertexIDs.T)[0])):
         id_ = ii + 1
         out1 = 'geometry set' + '  ' + str(id_)
         f.write(out1 +'\n')
         out2 = 'geo polygon positions'
         f.write(out2 +'  ')
         for jj in range(3):
             geo_posx = mesh.vertexCoords.T[int(mesh._cellVertexIDs.T[ii][jj])][0]
             geo_posy = mesh.vertexCoords.T[int(mesh._cellVertexIDs.T[ii][jj])][1]
             f.write( '(' + str(geo_posx)  + ',' +  str(geo_posy) + ')'  + '  ')
         f.write('\n')
         out4 = 'measure create radius ' + str(m_radius) +' id ' + str(id_) + ' position ' + str(mesh.cellCenters[0][ii]) +' '+ str(mesh.cellCenters[1][ii])
         f.write(out4+'\n')

with open('group_information.txt','w') as f1:
     for ii in range(int(np.shape(mesh._cellVertexIDs.T)[0])):
         id_ = ii + 1
         out3 = 'ball group' + '  ' + str(id_) +'  ' + 'range geometry' + '  ' + str(id_)+' '+ 'count odd'
         f1.write(out3+'\n')

########Variable
kk = CellVariable(name='mobility',mesh=mesh, value=1e-6)
phi = CellVariable(name = "solution variable",mesh = mesh,value = 0.)

#####boundary
phi.constrain(0, mesh.physicalFaces["outlet"])

#phi.faceGrad.constrain(0., mesh.physicalFaces["edge"])


itasca.command("""
restore bond
call geo_information.txt
save mesh
""")