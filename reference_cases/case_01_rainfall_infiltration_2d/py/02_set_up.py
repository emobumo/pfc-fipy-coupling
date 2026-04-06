phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.5)
phi.constrain(0.5, mesh.facesRight)
brr = np.where((phi-hz) >0, 0,(phi-hz))
se = (1 + (abs(a*(brr))**n ))**(-m)
psi = se*(ss-sr)+sr
sed = m*n*a*((1 + (abs(a*(brr))**n ) )**(-m-1.))*(abs(a*(brr))**(n-1))
kk = CellVariable(mesh = mesh,
                   value = ks*(se**0.5)*(1.-(1.-se**(1./m))**m)**2.)
qq = CellVariable(mesh = mesh,
                   value = (ss-sr)*sed)