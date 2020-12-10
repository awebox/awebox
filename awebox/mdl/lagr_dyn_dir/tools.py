import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


def time_derivative(expr, variables, architecture):

    # notice that the the chain rule
    # df/dt = partial f/partial t
    #       + (partial f/partial x1)(partial x1/partial t)
    #       + (partial f/partial x2)(partial x2/partial t)...
    # doesn't care about the scaling of the component functions, as long as
    # the units of (partial xi) will cancel

    # it is here assumed that (partial f/partial t) = 0.

    vars_scaled = variables['scaled']
    deriv = 0.

    # (partial f/partial xi) for variables with trivial derivatives
    deriv_vars = struct_op.subkeys(vars_scaled, 'xddot')
    for deriv_name in deriv_vars:

        deriv_type = struct_op.get_variable_type(variables['SI'], deriv_name)

        var_name = deriv_name[1:]
        var_type = struct_op.get_variable_type(variables['SI'], var_name)

        q_sym = vars_scaled[var_type, var_name]
        dq_sym = vars_scaled[deriv_type, deriv_name]
        deriv += cas.mtimes(cas.jacobian(expr, q_sym), dq_sym)


    # (partial f/partial xi) for kite rotation matrices
    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:
        parent = architecture.parent_map[kite]

        r_name = 'r{}{}'.format(kite, parent)
        kite_has_6dof = r_name in struct_op.subkeys(vars_scaled, 'xd')
        if kite_has_6dof:
            r_kite = vars_scaled['xd', r_name]
            dcm_kite = cas.reshape(r_kite, (3, 3))

            omega = vars_scaled['xd', 'omega{}{}'.format(kite, parent)]

            dexpr_dr = cas.jacobian(expr, r_kite)
            dr_dt = cas.reshape(cas.mtimes(vect_op.skew(omega), cas.inv(dcm_kite.T)), (9, 1))
            deriv += cas.mtimes(dexpr_dr, dr_dt)

    return deriv