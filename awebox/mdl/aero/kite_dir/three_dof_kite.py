#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
specific aerodynamics for a 3dof kite with roll_control
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: jochem de schutter, rachel leuthold, alu-fr 2017-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools
import awebox.mdl.aero.indicators as indicators
import awebox.mdl.mdl_constraint as mdl_constraint
import numpy as np


from awebox.logger.logger import Logger as awelogger


def get_force_vector(options, variables, atmos, wind, architecture, parameters, kite, outputs):
    kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture)

    vec_u = tools.get_local_air_velocity_in_earth_frame(options, variables, wind, kite, kite_dcm, architecture,
                                                        parameters, outputs)

    force_found_frame = 'earth'
    force_found_vector = get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture,
                                                             parameters)

    return force_found_vector, force_found_frame, vec_u, kite_dcm

def get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs):

    cstr_list = mdl_constraint.MdlConstraintList()

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        force_found_vector, force_found_frame, vec_u, kite_dcm = get_force_vector(options, variables, atmos, wind, architecture, parameters, kite, outputs)

        forces_dict = tools.get_framed_forces(vec_u, kite_dcm, variables, kite, architecture)
        force_framed_variable = forces_dict[force_found_frame]

        f_scale = tools.get_f_scale(parameters, options)

        resi_f_kite = (force_framed_variable - force_found_vector) / f_scale

        f_kite_cstr = cstr_op.Constraint(expr=resi_f_kite,
                                        name='f_aero' + str(kite) + str(parent),
                                        cstr_type='eq')
        cstr_list.append(f_kite_cstr)

    return cstr_list




def get_force_from_u_sym_in_earth_frame(vec_u, options, variables, kite, atmos, wind, architecture, parameters):

    parent = architecture.parent_map[kite]

    # get relevant variables for kite n
    q = variables['xd']['q' + str(kite) + str(parent)]
    coeff = variables['xd']['coeff' + str(kite) + str(parent)]

    # wind parameters
    rho_infty = atmos.get_density(q[2])

    kite_dcm = get_kite_dcm(options, variables, wind, kite, architecture)
    Lhat = kite_dcm[:,2]

    # lift and drag coefficients
    CL = coeff[0]

    CD0 = 0.
    poss_drag_labels_in_order_of_increasing_preference = ['CX', 'CA', 'CD']
    for poss_drag_label in poss_drag_labels_in_order_of_increasing_preference:
        local_parameter_label = '[theta0,aero,' + poss_drag_label + ',0,0]'
        if local_parameter_label in parameters.labels():
            CD0 = vect_op.abs(parameters['theta0', 'aero', poss_drag_label, '0'][0])

    CD = CD0 + CL ** 2 / (np.pi * parameters['theta0', 'geometry', 'ar'])

    s_ref = parameters['theta0', 'geometry', 's_ref']

    # lift and drag force
    f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(vec_u.T, vec_u) * s_ref * Lhat
    f_drag = CD * 1. / 2. * rho_infty * vect_op.norm(vec_u) * s_ref * vec_u

    f_aero = f_lift + f_drag

    return f_aero











def get_planar_dmc(vec_u_eff, variables, kite, architecture):

    parent = architecture.parent_map[kite]

    # get relevant variables for kite n
    q = variables['xd']['q' + str(kite) + str(parent)]

    # in kite body:
    if parent > 0:
        grandparent = architecture.parent_map[parent]
        q_parent = variables['xd']['q' + str(parent) + str(grandparent)]
    else:
        q_parent = np.array([0., 0., 0.])

    vec_t = q - q_parent # should be roughly "up-wards", ie, act like vec_w

    vec_v = vect_op.cross(vec_t, vec_u_eff)
    vec_w = vect_op.cross(vec_u_eff, vec_v)

    uhat = vect_op.smooth_normalize(vec_u_eff)
    vhat = vect_op.smooth_normalize(vec_v)
    what = vect_op.smooth_normalize(vec_w)

    planar_dcm = cas.horzcat(uhat, vhat, what)

    return planar_dcm


def get_kite_dcm(options, variables, wind, kite, architecture):

    parent = architecture.parent_map[kite]

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables, wind, kite, architecture)

    # roll angle
    coeff = variables['xd']['coeff' + str(kite) + str(parent)]
    psi = coeff[1]

    planar_dcm = get_planar_dmc(vec_u_eff, variables, kite, architecture)
    uhat = planar_dcm[:, 0]
    vhat = planar_dcm[:, 1]
    what = planar_dcm[:, 2]

    ehat1 = uhat
    ehat2 = np.cos(psi) * vhat + np.sin(psi) * what
    ehat3 = np.cos(psi) * what - np.sin(psi) * vhat

    kite_dcm = cas.horzcat(ehat1, ehat2, ehat3)

    return kite_dcm


def get_wingtip_position(kite, options, model, variables, parameters, ext_int):

    aero_coeff_ref_velocity = options['aero']['aero_coeff_ref_velocity']

    parent_map = model.architecture.parent_map
    xd = model.variables_dict['xd'](variables['xd'])

    if ext_int == 'ext':
        span_sign = 1.
    elif ext_int == 'int':
        span_sign = -1.
    else:
        awelogger.logger.error('wing side not recognized for 3dof kite.')

    parent = parent_map[kite]

    q = xd['q' + str(kite) + str(parent)]

    kite_dcm = get_kite_dcm(options, variables, model.wind, kite, model.architecture)
    ehat_span = kite_dcm[:, 1]

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    wingtip_position = q + ehat_span * span_sign * b_ref / 2.

    return wingtip_position