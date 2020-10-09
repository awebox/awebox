import casadi.tools as cas

def time_derivative(expr, q_sym, dq_sym, ddq_sym=None):
    deriv = cas.mtimes(cas.jacobian(expr, q_sym), dq_sym)

    if ddq_sym is not None:
        deriv += cas.mtimes(cas.jacobian(expr, dq_sym), ddq_sym)

    return deriv
