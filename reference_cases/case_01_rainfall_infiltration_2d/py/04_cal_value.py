z1 = phi
z2 = phi - y
z3 = se
z4 = psi
z5 = kk


z61 = -kk*phi.grad()[0]
z62 = -kk*phi.grad()[1]

func1 = Rbf(x,y,z1,function='linear')
func2 = Rbf(x,y,z2,function='linear')
func3 = Rbf(x,y,z3,function='linear')
func4 = Rbf(x,y,z4,function='linear')
func5 = Rbf(x,y,z5,function='linear')

func61 = Rbf(x,y,z61,function='linear')
func62 = Rbf(x,y,z62,function='linear')

for b in balls.list():
    bx = b.pos_x()
    by = b.pos_y()
    nz1 = float(func1(bx,by))
    nz2 = float(func2(bx,by))
    nz3 = float(func3(bx,by))
    nz4 = float(func4(bx,by))
    nz5 = float(func5(bx,by))

    nz61 = float(func61(bx,by))
    nz62 = float(func62(bx,by))
    b.set_extra(1,nz1)
    b.set_extra(2,nz2)
    b.set_extra(3,nz3)
    b.set_extra(4,nz4)
    b.set_extra(5,nz5)
    b.set_extra(6,(nz61,nz62))
#    
    
    
#for b in balls.list():
#    group_name = b.group()
#    b.set_extra(1,phi.value[int(group_name)-1])
#    b.set_extra(2,(phi.value[int(group_name)-1]-mesh.cellCenters().T[int(group_name)-1][1]))
    #b.set_extra(3,se[int(group_name)-1])
    #b.set_extra(4,psi[int(group_name)-1])
    #b.set_extra(5,-phi.grad.value.T[int(group_name)-1])