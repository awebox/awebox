#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
direct_cosine_matrix aerodynamics modelling file
calculates aerodynamic outputs for dcm model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-18
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.indicators as indicators

from . import stability_derivatives

import awebox.mdl.aero.actuator_disk_dir.flow as actuator_disk_flow

def get_outputs(options, atmos, wind, variables, outputs, parameters, architecture):
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    xd = variables['xd']
    u = variables['u']

    elevation_angle = indicators.get_elevation_angle(xd)

    for n in kite_nodes:

        parent = parent_map[n]

        # get relevant variables for kite n
        q = xd['q' + str(n) + str(parent)]
        dq = xd['dq' + str(n) + str(parent)]

        r = cas.reshape(xd['r' + str(n) + str(parent)], (3, 3))
        ehat_span = r[:, 1]
        ehat_chord = r[:, 0]

        # wind parameters
        rho_infty = atmos.get_density(q[2])
        uw_infty = wind.get_velocity(q[2])

        # apparent air velocity
        if options['induction_model'] == 'actuator':
            ua = actuator_disk_flow.get_kite_effective_velocity(options, variables, wind, n, parent, architecture)
        else:
            ua = uw_infty - dq

        # relative air speed
        norm_ua_squared = cas.mtimes(ua.T, ua)
        ua_norm = norm_ua_squared ** 0.5

        # angle of attack and sideslip angle
        alpha = indicators.get_alpha(ua, r)
        beta = indicators.get_beta(ua, r)

        if int(options['surface_control']) == 0:
            delta = u['delta' + str(n) + str(parent)]
            omega = xd['omega' + str(n) + str(parent)]
            [CF, CM] = stability_derivatives.stability_derivatives(
                options, alpha, beta, ua, omega, delta, parameters)
        elif int(options['surface_control']) == 1:
            delta = xd['delta' + str(n) + str(parent)]
            omega = xd['omega' + str(n) + str(parent)]
            [CF, CM] = stability_derivatives.stability_derivatives(options, alpha, beta, ua, omega, delta, parameters)
        else:
            raise ValueError('unsupported surface_control chosen: %i', options['surface_control'])

        # body-_frameforcecomponents
        # notice that these are unusual because an apparent wind reference coordinate system is in use.
        # see below (get_coeffs_from_control_surfaces) for information
        CA = CF[0]
        CY = CF[1]
        CN = CF[2]

        Cl = CM[0]
        Cm = CM[1]
        Cn = CM[2]

        dynamic_pressure = 1. / 2. * rho_infty * norm_ua_squared
        planform_area = parameters['theta0','geometry','s_ref']
        ftilde_aero = cas.mtimes(r, CF)
        f_aero = dynamic_pressure * planform_area * ftilde_aero

        ehat_drag = vect_op.normalize(ua)
        f_drag = cas.mtimes(cas.mtimes(f_aero.T, ehat_drag), ehat_drag)

        ehat_lift = vect_op.normed_cross(ua, ehat_span)
        f_lift = cas.mtimes(cas.mtimes(f_aero.T, ehat_lift), ehat_lift)

        f_side = f_aero - f_drag - f_lift

        drag_cross_lift = indicators.convert_from_body_to_wind_axes(alpha, beta, CF)
        CD = drag_cross_lift[0]
        CS = drag_cross_lift[1]
        CL = drag_cross_lift[2]

        b_ref = parameters['theta0','geometry','b_ref']
        c_ref = parameters['theta0','geometry','c_ref']

        reference_lengths = cas.diag(cas.vertcat(b_ref, c_ref, b_ref))
        m_aero = dynamic_pressure * planform_area * cas.mtimes(reference_lengths, CM)

        aero_coefficients = {}
        aero_coefficients['CD'] = CD
        aero_coefficients['CS'] = CS
        aero_coefficients['CL'] = CL
        aero_coefficients['CA'] = CA
        aero_coefficients['CN'] = CN
        aero_coefficients['CY'] = CY
        aero_coefficients['Cl'] = Cl
        aero_coefficients['Cm'] = Cm
        aero_coefficients['Cn'] = Cn

        outputs = indicators.collect_kite_aerodynamics_outputs(options, atmos, ua, ua_norm, aero_coefficients, f_aero,
                                                               f_lift, f_drag, f_side, m_aero, ehat_chord, ehat_span, r, q, n, outputs, parameters)
        outputs = indicators.collect_environmental_outputs(atmos, wind, q, n, outputs)
        outputs = indicators.collect_aero_validity_outputs(options, xd, ua, n, parent, outputs, parameters)
        outputs = indicators.collect_local_performance_outputs(options, atmos, wind, variables, CL, CD, elevation_angle, ua, n, parent,
                                          outputs,parameters)
        outputs = indicators.collect_power_balance_outputs(variables, n, outputs, architecture)

    return outputs
