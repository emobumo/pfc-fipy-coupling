
phi.faceGrad.constrain(0.5*ks/kk[350],facesT)
eq = TransientTerm(coeff=qq) == DiffusionTerm(coeff=kk)
eq.solve(var=phi,dt=0.01)


brr = np.where((phi-hz) >0, 0,(phi-hz)) #压力水头大于=0时，使其为0，满足饱和度为1.

se = (1 + (abs(a*(brr))**n ))**(-m)

psi = se*(ss-sr)+sr

sed = m*n*a*((1 + (abs(a*(brr))**n ) )**(-m-1.))*(abs(a*(brr))**(n-1))

kk = CellVariable(mesh = mesh,
                   value = ks*(se**0.5)*(1.-(1.-se**(1./m))**m)**2.)

qq = CellVariable(mesh = mesh,
                   value = (ss-sr)*sed)
