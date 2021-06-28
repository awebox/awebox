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
specific aerodynamics for a 3dof soft-kite
- author: Mark Schelbergen, TU Delft 2021
- author: Jochem De Schutter, ALU Freiburg 2021
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

    kite_dcm = get_kite_dcm(options, variables, kite, architecture)

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

    label = architecture.node_label(kite)

    # get relevant variables for kite n
    q = variables['x'][f'q{label}']
    dq = variables['x'][f'dq{label}']
    yaw = variables['x'][f'yaw{label}']
    pitch = variables['x'][f'pitch{label}']

    # wind parameters
    rho_infty = atmos.get_density(q[2])

    kite_dcm = get_kite_dcm(options, variables, kite, architecture)

    # Basis of body frame expressed in earth frame
    e1 = kite_dcm[:, 0]  # Longitudinal axis of the wing
    e2 = kite_dcm[:, 1]  # Transversal axis of the wing
    e3 = kite_dcm[:, 2]  # Vertical axis of the wing

    kite_projected_area = parameters['theta0', 'geometry', 's_ref']
    wing_area_side = parameters['theta0', 'geometry', 's_ref_side']

    v_app = - vec_u

    # Flow condition at kite
    alpha = get_alpha(vec_u, kite_dcm, pitch)
    beta = get_beta(vec_u, kite_dcm)

    lift_coefficient_params = cas.vertcat(
        parameters['theta0', 'aero', 'CL', '0'][0],
        parameters['theta0', 'aero', 'CL', 'alpha'][0],
        parameters['theta0', 'aero', 'CL', 'alpha'][1]
    )

    drag_coefficient_params = cas.vertcat(
        parameters['theta0', 'aero', 'CD', '0'][0],
        parameters['theta0', 'aero', 'CD', 'alpha'][0],
        parameters['theta0', 'aero', 'CD', 'alpha'][1]
    )

    c_s = parameters['theta0', 'aero', 'CS', '0'][0]

    # Aerodynamic models
    c_l = cas.mtimes(lift_coefficient_params.T, cas.vertcat(1, alpha, alpha**2))
    c_d_kite = cas.mtimes(drag_coefficient_params.T, cas.vertcat(1, alpha, alpha**2))
    lift = .5*rho_infty*kite_projected_area*cas.norm_2(v_app) * c_l*cas.cross(v_app, e2)  # TODO: note that lift term is
    # not dependent on v_app^2, since cross(v_app, e2) does not have the same magnitude as v_app
    drag_kite = -.5*rho_infty*kite_projected_area*cas.norm_2(v_app) * c_d_kite*v_app
    side_force = .5*rho_infty*wing_area_side*cas.norm_2(v_app)**2 * c_s*beta*e2

    f_aero = lift + drag_kite + side_force

    return f_aero

def get_kite_dcm_expr(options, variables, kite, architecture):

    parent = architecture.parent_map[kite]
    q = variables['x']['q' + str(kite) + str(parent)]
    yaw = variables['x']['yaw' + str(kite) + str(parent)]

    if kite == 1:

        l = variables['x']['l_t']
        l12 = cas.norm_2(q[:2])
        dcm_tau2e = cas.blockcat([
            [q[0]*q[2]/(l*l12), -q[1]/l12,  q[0]/l],
            [q[1]*q[2]/(l*l12), q[0]/l12, q[1]/l],
            [-l12/l,              0,         q[2]/l],
        ])  # Transformation from tangential (to unit sphere surface) plane to earth frame
        dcm_b2tau = cas.blockcat([
            [cas.cos(yaw), -cas.sin(yaw), 0],
            [cas.sin(yaw), cas.cos(yaw),  0],
            [0,  0,   1],
        ])
        kite_dcm = cas.mtimes(dcm_tau2e, dcm_b2tau) 

    elif kite > 1:
        raise ValueError('Soft-kite model is only implemented for single-kite systems.')

    return kite_dcm

def get_kite_dcm(options, variables, kite, architecture):

    if kite == 1:

        label = architecture.node_label(kite)
        kite_dcm = cas.reshape(variables['z'][f'dcm{label}'], 3, 3)

    elif kite > 1:
        raise ValueError('Soft-kite model is only implemented for single-kite systems.')

    return kite_dcm

def get_dcm_cstr(options, variables, architecture, outputs):

    cstr_list = mdl_constraint.MdlConstraintList()

    for kite in architecture.kite_nodes:
        label = architecture.node_label(kite)

        dcm = get_kite_dcm(options, variables, kite, architecture)
        dcm_expr = get_kite_dcm_expr(options, variables, kite, architecture)
        cstr_expr = cas.reshape(dcm - dcm_expr, 9, 1) 
        dcm_cstr = cstr_op.Constraint(expr=cstr_expr,
                                        name='dcm{}'.format(label),
                                        cstr_type='eq')
        cstr_list.append(dcm_cstr)

    return cstr_list

def get_alpha(ua, dcm, pitch):

    v_app = -ua
    e1 = dcm[:, 0]  # Longitudinal axis of the wing
    e3 = dcm[:, 2]  # Vertical axis of the wing
    
    alpha = - cas.dot(e3, v_app) / cas.dot(e1, v_app) - pitch

    return alpha

def get_beta(ua, dcm):

    v_app = -ua
    e1 = dcm[:, 0]  # Longitudinal axis of the wing
    e2 = dcm[:, 1]  # Transversal axis of the wing

    beta = cas.dot(e2, v_app) / cas.dot(e1, v_app)

    return beta