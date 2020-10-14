import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import pdb


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
            r_kite = vars_scaled['xd', ]
            dcm_kite = cas.reshape(r_kite, (3, 3))

            omega = vars_scaled['xd', 'omega{}{}'.format(kite, parent)]

            dexpr_dr = cas.jacobian(expr, r_kite)
            dr_dt = cas.reshape(cas.mtimes(vect_op.skew(omega), cas.inv(dcm_kite.T)), (9, 1))
            deriv += cas.mtimes(dexpr_dr, dr_dt)

    return deriv



def get_tether_segment_properties(options, architecture, variables_si, parameters, upper_node):
    kite_nodes = architecture.kite_nodes

    xd = variables_si['xd']
    theta = variables_si['theta']
    scaling = options['scaling']

    if upper_node == 1:
        vars_containing_length = xd
        vars_sym = 'xd'
        length_sym = 'l_t'
        diam_sym = 'diam_t'

    elif upper_node in kite_nodes:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_s'
        diam_sym = 'diam_s'

    else:
        vars_containing_length = theta
        vars_sym = 'theta'
        length_sym = 'l_i'
        diam_sym = 'diam_t'

    seg_length = vars_containing_length[length_sym]
    scaling_length = scaling[vars_sym][length_sym]

    seg_diam = theta[diam_sym]
    max_diam = options['system_bounds']['theta'][diam_sym][1]
    length_scaling = scaling[vars_sym][length_sym]
    scaling_diam = scaling['theta'][diam_sym]

    cross_section_area = np.pi * (seg_diam / 2.) ** 2.
    max_area = np.pi * (max_diam / 2.) ** 2.
    scaling_area = np.pi * (scaling_diam / 2.) ** 2.

    density = parameters['theta0', 'tether', 'rho']
    seg_mass = cross_section_area * density * seg_length
    scaling_mass = scaling_area * parameters['theta0', 'tether', 'rho'] * length_scaling

    props = {}
    props['seg_length'] = seg_length
    props['scaling_length'] = scaling_length

    props['seg_diam'] = seg_diam
    props['max_diam'] = max_diam
    props['scaling_diam'] = scaling_diam

    props['cross_section_area'] = cross_section_area
    props['max_area'] = max_area
    props['scaling_area'] = scaling_area

    props['seg_mass'] = seg_mass
    props['scaling_mass'] = scaling_mass

    return props
