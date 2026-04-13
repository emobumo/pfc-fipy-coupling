#
#itasca.command('''
#new
# ''')
#restore mesh
#call group_information.txt


for df in itasca.dfn.fracture.list():
        xx = df.pos()[0]
        yy = df.pos()[1]
        eel = []
        eed = []
        for ii in range(int(np.shape(mesh._cellVertexIDs.T)[0])):
            ex = mesh.cellCenters[0][ii]
            ey = mesh.cellCenters[1][ii]
            ll = math.sqrt((xx-ex)**2+(yy-ey)**2)
            eel.append(ll)
        xyz = min(eel)
        xyzid = eel.index(xyz)
        kk[xyzid] = kk[xyzid] + '1e-6'
#ca.set_extra(1,kk.value.T)

#kk.value[int(ii)-1] = (B*ball_porosity**3*(2*mean_radius)**2/(1-ball_porosity)**2)

#
num_load = num_load  + 1
print(num_load)
#eq = TransientTerm() == DiffusionTerm(coeff=kk)
phi.constrain(num_load*1e2 , mesh.physicalFaces["inlet"])
eq = DiffusionTerm(coeff=kk) == 0.
eq.solve(var=phi)
#eq.solve(var=phi, dt=10000)
##########set  ball.extra value及添加渗流力
for b in balls.list():  
    group_name = b.group()
    if group_name != 'None':
        b.set_extra(1,phi.value[int(group_name)-1])#水头
        b.set_extra(2,phi.grad.value[0][int(group_name)-1])
        b.set_extra(3,phi.grad.value[1][int(group_name)-1])
        b.set_extra(5,phi.value[int(group_name)-1]*1e4)
        #b.set_extra(4,phi.grad.value.T[int(group_name)-1]) #梯度矢量
        #b.set_extra(5,-kk.value[int(group_name)-1]*phi.grad.value.T[int(group_name)-1])#流场
        b.set_extra(6,kk.value[int(group_name)-1])#流场
        ball_porosity = itasca.measure.Measure.porosity(itasca.measure.find(int(group_name)))#孔隙率
        if ball_porosity > 0.7:
            ball_porosity = 0.7
        force = -1e4*b.radius()*b.radius()*3.14*phi.grad.value.T[int(group_name)-1]/(1-ball_porosity)
        b.set_force_app(force)

for ii in range(int(np.shape(mesh._cellVertexIDs.T)[0])):
    bp = balls.near((mesh.cellCenters[0][ii],mesh.cellCenters[1][ii]))
    bp.set_extra(4,-kk.value[int(ii)]*phi.grad.value.T[int(ii)])