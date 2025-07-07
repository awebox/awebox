import casadi

opti = casadi.Opti()

x = opti.variable()
y = opti.variable()

opti.minimize(  (y-x**2)**2   )
opti.subject_to( x**2+y**2==1 )
opti.subject_to(       x+y>=1 )

# opti.solver('sqpmethod')
# sol_sqp = opti.solve()
# print("sqp :   ", sol_sqp.value(x), sol_sqp.value(y))

opti.solver('ipopt')
sol_ipopt = opti.solve()
print("ipopt : ", sol_ipopt.value(x), sol_ipopt.value(y))
