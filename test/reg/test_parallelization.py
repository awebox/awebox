import casadi as cd

def test_parallelization():
    N = 300
    x = cd.MX.sym('x'); z = cd.MX.sym('z'); p = cd.MX.sym('p')
    dae = {'x': x, 'z': z, 'p': p, 'ode': 0, 'alg': z}
    func = cd.integrator('func', 'idas', dae)
    F = func.map(N, 'openmp')
    F(x0=0, z0=0, p=0)