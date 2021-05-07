#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
kite aerodynamics model of an awe system
takes states and inputs and creates aerodynamic forces and moments
dependent on the position of the kite.
_aerodynamic coefficients are assumptions.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017-2021
'''

import awebox.mdl.aero.induction_dir.induction as induction
import awebox.mdl.aero.indicators as indicators
import awebox.mdl.aero.kite_dir.three_dof_kite as three_dof_kite
import awebox.mdl.aero.kite_dir.six_dof_kite as six_dof_kite
import awebox.mdl.aero.kite_dir.frames as frames
import awebox.mdl.aero.kite_dir.tools as tools
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import casadi.tools as cas
import numpy as np
import awebox.mdl.mdl_constraint as mdl_constraint

def get_forces_and_moments(options, atmos, wind, variables_si, outputs, parameters, architecture):
    outputs = get_aerodynamic_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture)

    outputs = indicators.get_performance_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture)

    if not (options['induction_model'] == 'not_in_use'):
        outputs = induction.collect_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture)

    return outputs

def get_framed_forces_and_moments(options, variables_si, atmos, wind, architecture, parameters, kite, outputs):
    parent = architecture.parent_map[kite]

    xd = variables_si['xd']
    q = xd['q' + str(kite) + str(parent)]
    dq = xd['dq' + str(kite) + str(parent)]

    vec_u_eff = tools.get_u_eff_in_earth_frame(options, variables_si, wind, kite, architecture)
    rho = atmos.get_density(q[2])
    q_eff = 0.5 * rho * cas.mtimes(vec_u_eff.T, vec_u_eff)

    if int(options['kite_dof']) == 3:
        kite_dcm = three_dof_kite.get_kite_dcm(options, variables_si, wind, kite, architecture)
    elif int(options['kite_dof']) == 6:
        kite_dcm = six_dof_kite.get_kite_dcm(kite, variables_si, architecture)
    else:
        message = 'unsupported kite_dof chosen in options: ' + str(options['kite_dof'])
        awelogger.logger.error(message)

    if int(options['kite_dof']) == 3:

        if options['aero']['lift_aero_force']:
            framed_forces = tools.get_framed_forces(vec_u_eff, kite_dcm, variables_si, kite, architecture)
        else:
            force_found_vector, force_found_frame, vec_u, kite_dcm = three_dof_kite.get_force_vector(options,
                                                                                                     variables_si,
                                                                                                     atmos,
                                                                                                     wind, architecture,
                                                                                                     parameters, kite,
                                                                                                     outputs)
            framed_forces = tools.get_framed_forces(vec_u_eff, kite_dcm, variables_si, kite, architecture,
                                                    force_found_vector, force_found_frame)

        framed_moments = tools.get_framed_moments(vec_u_eff, kite_dcm, variables_si, kite, architecture,
                                                  cas.DM.zeros((3, 1)), 'body')

    elif int(options['kite_dof']) == 6:

        if options['aero']['lift_aero_force']:
            framed_forces = tools.get_framed_forces(vec_u_eff, kite_dcm, variables_si, kite, architecture)
            framed_moments = tools.get_framed_moments(vec_u_eff, kite_dcm, variables_si, kite, architecture)

        else:
            f_found_vector, f_found_frame, m_found_vector, m_found_frame, vec_u_earth, kite_dcm = six_dof_kite.get_force_and_moment_vector(
                options, variables_si, atmos, wind, architecture, parameters, kite, outputs)
            framed_forces = tools.get_framed_forces(vec_u_eff, kite_dcm, variables_si, kite, architecture,
                                                    f_found_vector, f_found_frame)
            framed_moments = tools.get_framed_moments(vec_u_eff, kite_dcm, variables_si, kite, architecture,
                                                      m_found_vector, m_found_frame)

    else:
        message = 'unsupported kite_dof chosen in options: ' + str(options['kite_dof'])
        awelogger.logger.error(message)

    return framed_forces, framed_moments, kite_dcm, q_eff, vec_u_eff, q, dq

def get_aerodynamic_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    c_ref = parameters['theta0', 'geometry', 'c_ref']
    s_ref = parameters['theta0', 'geometry', 's_ref']
    reference_lengths = cas.diag(cas.vertcat(b_ref, c_ref, b_ref))

    kite_nodes = architecture.kite_nodes
    for kite in kite_nodes:

        framed_forces, framed_moments, kite_dcm, q_eff, vec_u_eff, q, dq = get_framed_forces_and_moments(options, variables_si, atmos, wind, architecture, parameters, kite, outputs)
        m_aero_body = framed_moments['body']
        u_eff = vect_op.smooth_norm(vec_u_eff)

        f_aero_body = framed_forces['body']
        f_aero_wind = framed_forces['wind']
        f_aero_control = framed_forces['control']
        f_aero_earth = framed_forces['earth']

        coeff_body = f_aero_body / q_eff / s_ref
        CA = coeff_body[0]
        CY = coeff_body[1]
        CN = coeff_body[2]

        f_drag_wind = f_aero_wind[0] * vect_op.xhat()
        f_side_wind = f_aero_wind[1] * vect_op.yhat()
        f_lift_wind = f_aero_wind[2] * vect_op.zhat()

        f_drag_earth = frames.from_wind_to_earth(vec_u_eff, kite_dcm, f_drag_wind)
        f_side_earth = frames.from_wind_to_earth(vec_u_eff, kite_dcm, f_side_wind)
        f_lift_earth = frames.from_wind_to_earth(vec_u_eff, kite_dcm, f_lift_wind)

        coeff_wind = f_aero_wind / q_eff / s_ref
        CD = coeff_wind[0]
        CS = coeff_wind[1]
        CL = coeff_wind[2]

        CM = cas.mtimes(cas.inv(reference_lengths), m_aero_body) / q_eff / s_ref
        Cl = CM[0]
        Cm = CM[1]
        Cn = CM[2]

        aero_coefficients = {}
        if options['aero']['lift_aero_force']:
            f_aero_name = 'f_aero' + str(kite) + str(architecture.parent_map[kite])
            f_aero_var = variables_si['xl'][f_aero_name]
            f_aero_var_in_wind = frames.from_earth_to_wind(vec_u_eff, kite_dcm, f_aero_var)
            CD_var = f_aero_var_in_wind[0]
            CS_var = f_aero_var_in_wind[1]
            CL_var = f_aero_var_in_wind[2]
            aero_coefficients['CD_var'] = CD_var
            aero_coefficients['CS_var'] = CS_var
            aero_coefficients['CL_var'] = CL_var
        else:
            aero_coefficients['CD_var'] = f_drag_wind
            aero_coefficients['CS_var'] = f_side_wind
            aero_coefficients['CL_var'] = f_lift_wind

        aero_coefficients['CD'] = CD
        aero_coefficients['CS'] = CS
        aero_coefficients['CL'] = CL
        aero_coefficients['CA'] = CA
        aero_coefficients['CY'] = CY
        aero_coefficients['CN'] = CN
        aero_coefficients['Cl'] = Cl
        aero_coefficients['Cm'] = Cm
        aero_coefficients['Cn'] = Cn
        aero_coefficients['LoverD'] = CL/CD


        base_aerodynamic_quantities = {}
        base_aerodynamic_quantities['kite'] = kite
        base_aerodynamic_quantities['air_velocity'] = vec_u_eff
        base_aerodynamic_quantities['airspeed'] = u_eff
        base_aerodynamic_quantities['aero_coefficients'] = aero_coefficients
        base_aerodynamic_quantities['f_aero_earth'] = f_aero_earth
        base_aerodynamic_quantities['f_aero_body'] = f_aero_body
        base_aerodynamic_quantities['f_aero_control'] = f_aero_control
        base_aerodynamic_quantities['f_aero_wind'] = f_aero_wind
        base_aerodynamic_quantities['f_lift_earth'] = f_lift_earth
        base_aerodynamic_quantities['f_drag_earth'] = f_drag_earth
        base_aerodynamic_quantities['f_side_earth'] = f_side_earth
        base_aerodynamic_quantities['m_aero_body'] = m_aero_body
        base_aerodynamic_quantities['kite_dcm'] = kite_dcm
        base_aerodynamic_quantities['q'] = q
        base_aerodynamic_quantities['dq'] = dq

        outputs = indicators.collect_kite_aerodynamics_outputs(options, architecture, atmos, wind, variables_si, parameters, base_aerodynamic_quantities, outputs)
        outputs = indicators.collect_environmental_outputs(atmos, wind, base_aerodynamic_quantities, outputs)
        outputs = indicators.collect_aero_validity_outputs(options, base_aerodynamic_quantities, outputs)
        outputs = indicators.collect_local_performance_outputs(architecture, atmos, wind, variables_si, parameters,
                                                               base_aerodynamic_quantities, outputs)
        outputs = indicators.collect_power_balance_outputs(options, architecture, variables_si, base_aerodynamic_quantities, outputs)


    return outputs


def get_force_and_moment_vars(variables_si, kite, parent, options):
    f_aero = tools.get_f_aero_var(variables_si, kite, parent)

    kite_has_6dof = (int(options['kite_dof']) == 6)
    if kite_has_6dof:
        m_aero = tools.get_m_aero_var(variables_si, kite, parent)
    else:
        m_aero = cas.DM.zeros((3, 1))

    return f_aero, m_aero

def get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs):

    if int(options['kite_dof']) == 3:
        cstr_list = three_dof_kite.get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs)

    elif int(options['kite_dof']) == 6:
        cstr_list = six_dof_kite.get_force_cstr(options, variables, atmos, wind, architecture, parameters, outputs)
    else:
        raise ValueError('failure: unsupported kite_dof chosen in options: %i',options['kite_dof'])

    return cstr_list


def get_wingtip_position(kite, options, model, variables, parameters, ext_int):
    if int(options['kite_dof']) == 3:
        wingtip_pos = three_dof_kite.get_wingtip_position(kite, options, model, variables, parameters, ext_int)
    elif int(options['kite_dof']) == 6:
        wingtip_pos = six_dof_kite.get_wingtip_position(kite, model, variables, parameters, ext_int)
    else:
        raise ValueError('failure: unsupported kite_dof chosen in options: %i',options['kite_dof'])

    return wingtip_pos