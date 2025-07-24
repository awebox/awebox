

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


def time_derivative(expr, vars_scaled, architecture, scaling):

    # notice that the the chain rule
    # df/dt = partial f/partial t
    #       + (partial f/partial x1)(partial x1/partial t)
    #       + (partial f/partial x2)(partial x2/partial t)...
    # doesn't care about the scaling of the component functions, as long as
    # the units of (partial xi) will cancel
    # if they *don't* cancel, then they need to be made-to-cancel

    # it is here assumed that (partial f/partial t) = 0.

    deriv = cas.DM(0.)
    nx = vars_scaled['x'].shape[0]
    nv = vars_scaled.cat.shape[0]
    rdot_deriv_vars = set(['r' + str(kite) + str(architecture.parent_map[kite]) for kite in architecture.kite_nodes])
    full_jac = cas.jacobian(expr, vars_scaled.cat)

    xdot = []
    for deriv_name in struct_op.subkeys(vars_scaled, 'xdot'):
        var_name = deriv_name[1:]
        deriv_type = struct_op.get_variable_type(vars_scaled, deriv_name)
        if var_name not in rdot_deriv_vars:
            if deriv_name[:7] != 'dp_ring':
                xdot.append(vars_scaled[deriv_type, deriv_name])
            else:
                xdot.append(cas.vertcat(vars_scaled[deriv_type, deriv_name],0,0))
        else:
            kite, parent = var_name[-2], var_name[-1]
            omega_name = 'omega{}{}'.format(kite, parent)
            r_kite = vars_scaled['x', var_name]
            omega = vars_scaled['x', omega_name]
            dcm_kite = cas.reshape(r_kite, (3, 3))
            dr_dt = cas.reshape(cas.mtimes(dcm_kite, vect_op.skew(omega)), (9, 1))
            xdot.append(dr_dt)

        # build xdot
        scaling_factor_q = scaling['x', var_name]
        scaling_factor_dq = scaling[deriv_type, deriv_name]
        if not (scaling_factor_q - scaling_factor_dq).is_zero():
            xdot[-1] *= scaling_factor_dq / scaling_factor_q
    xdot = cas.vertcat(*xdot)

    deriv = cas.mtimes(full_jac, cas.vertcat(xdot, np.zeros((nv - nx, 1))))

    # # (partial f/partial xi) for variables with trivial derivatives
    # deriv_vars = struct_op.subkeys(vars_scaled, 'xdot')
    # import pdb; pdb.set_trace()
    # rdot_deriv_vars = set(['r' + str(kite) + str(architecture.parent_map[kite]) for kite in architecture.kite_nodes])
    # deriv_vars_without_rdot = set(deriv_vars) - set(rdot_deriv_vars)

    # for deriv_name in deriv_vars_without_rdot:
    #     deriv_type = struct_op.get_variable_type(vars_scaled, deriv_name)

    #     var_name = deriv_name[1:]
    #     var_type = struct_op.get_variable_type(vars_scaled, var_name)

    #     q_sym = vars_scaled[var_type, var_name]
    #     dq_sym = vars_scaled[deriv_type, deriv_name]

    #     partial_f_partial_xi = cas.jacobian(expr, q_sym)
    #     partial_xi_partial_t = dq_sym

    #     scaling_factor_q = scaling[var_type, var_name]
    #     scaling_factor_dq = scaling[deriv_type, deriv_name]

    #     if not (scaling_factor_q - scaling_factor_dq).is_zero():
    #         # [delta force / delta distance_in_km] = (1/1000) [delta force / delta distance_in_m]
    #         partial_f_partial_xi = cas.mtimes(partial_f_partial_xi, cas.inv(cas.diag(scaling_factor_q)))
    #         # [delta distance_in_km / delta t] = 1000 * (delta distance_in_m / delta t)
    #         partial_xi_partial_t = cas.mtimes(cas.diag(scaling_factor_dq), partial_xi_partial_t)

    #     local_component = cas.mtimes(partial_f_partial_xi, partial_xi_partial_t)
    #     deriv += local_component

    # # (partial f/partial xi) for kite rotation matrices
    # kite_nodes = architecture.kite_nodes
    # for kite in kite_nodes:
    #     parent = architecture.parent_map[kite]

    #     r_name = 'r{}{}'.format(kite, parent)
    #     omega_name = 'omega{}{}'.format(kite, parent)

    #     kite_has_6dof = r_name in struct_op.subkeys(vars_scaled, 'x')
    #     if kite_has_6dof:
    #         r_kite = vars_scaled['x', r_name]
    #         dcm_kite = cas.reshape(r_kite, (3, 3))
    #         omega = vars_scaled['x', omega_name]

    #         dexpr_dr = cas.jacobian(expr, r_kite)
    #         dr_dt = cas.reshape(cas.mtimes(vect_op.skew(omega), cas.inv(dcm_kite.T)), (9, 1))
    #         deriv += cas.mtimes(dexpr_dr, dr_dt)

    return deriv