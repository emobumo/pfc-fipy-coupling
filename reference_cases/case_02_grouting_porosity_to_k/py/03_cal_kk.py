import itasca as it
##### 求解裂隙产生后渗透系数的变化
kk.setValue(1e-6)
for df in it.dfn.fracture.list():
     xx = df.pos()[0]
     yy = df.pos()[1]
     eel = []
     eed = []
     for ii in range(int(np.shape(mesh._cellVertexIDs.T)[0])):
         ex = mesh.cellCenters[0][ii]
         ey = mesh.cellCenters[0][ii]
         ll = math.sqrt((xx-ex)**2+(yy-ey)**2)
         eel.append(ll)
     xyz = min(eel)
     xyzid = eel.index(xyz)
     kk[xyzid] = kk[xyzid] + '1e-8'
ca.set_extra(1,kk.value.T)