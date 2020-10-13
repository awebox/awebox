import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import pdb

from awebox.logger.logger import Logger as awelogger

def time_derivative_prev(expr, q_sym, dq_sym, ddq_sym=None):
    deriv = cas.mtimes(cas.jacobian(expr, q_sym), dq_sym)

    if ddq_sym is not None:
        deriv += cas.mtimes(cas.jacobian(expr, dq_sym), ddq_sym)

    return deriv


def time_derivative(options, expr, variables, architecture=None, node=None):

    vars_scaled = variables['scaled']
    xd_subkeys = struct_op.subkeys(vars_scaled, 'xd')

    deriv = 0.

    for var_type in vars_scaled.keys():

        local_subkeys = struct_op.subkeys(vars_scaled, var_type)
        for var_name in local_subkeys:

            var_might_be_derivative = (var_name[0] == 'd') and (len(var_name) > 1)
            if var_might_be_derivative:

                poss_deriv_name = var_name[1:]
                if poss_deriv_name in local_subkeys:
                    q_sym = vars_scaled[var_type, poss_deriv_name]
                    dq_sym = vars_scaled[var_type, var_name]
                    deriv += cas.mtimes(cas.jacobian(expr, q_sym), dq_sym)

                elif poss_deriv_name in xd_subkeys:
                    q_sym = vars_scaled['xd', poss_deriv_name]
                    dq_sym = vars_scaled[var_type, var_name]
                    deriv += cas.mtimes(cas.jacobian(expr, q_sym), dq_sym)

    if (architecture is not None) and (node is not None):
        parent = architecture.parent_map[node]
        node_is_a_kite = node in architecture.kite_nodes
        kite_has_6dof = (int(options['kite_dof']) == 6)

        if node_is_a_kite and kite_has_6dof:
            jacobian_dcm = vect_op.jacobian_dcm(expr, variables['SI']['xd'], vars_scaled, node, parent)
            omega_scaled = vars_scaled['xd', 'omega' + str(node) + str(parent)]
            deriv += 2. * cas.mtimes(jacobian_dcm, omega_scaled)

    return deriv