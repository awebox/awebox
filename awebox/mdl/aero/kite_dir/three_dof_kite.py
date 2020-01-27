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
roll_control aerodynamics modelling file
calculates aerodynamic outputs for roll-control model
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: jochem de schutter, rachel leuthold, alu-fr 2017-18
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import awebox.mdl.aero.indicators as indicators
import awebox.mdl.aero.induction_dir.induction as induction
import pdb

def get_outputs(options, atmos, wind, variables, outputs, parameters, architecture):
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    xd = variables['xd']

    elevation_angle = indicators.get_elevation_angle(xd)

    for kite in kite_nodes:

        parent = parent_map[kite]

        # get relevant variables for kite kite
        q = xd['q' + str(kite) + str(parent)]
        dq = xd['dq' + str(kite) + str(parent)]
        coeff = xd['coeff' + str(kite) + str(parent)]

        # wind parameters
        rho_infty = atmos.get_density(q[2])
        uw_infty = wind.get_velocity(q[2])

        # apparent air velocity
        if not (options['induction_model'] == 'not_in_use'):
            ueff = induction.get_kite_effective_velocity(options, variables, wind, kite, architecture)
        else:
            ueff = uw_infty - dq

        # relative air speed
        ueff_norm = vect_op.smooth_norm(ueff, epsilon=1e-8)

        ehat_l, ehat_span = get_ehat_l_and_span(kite, options, wind, variables, architecture)
        ehat_chord = ueff/ueff_norm

        # implicit direct cosine matrix (for plotting only)
        r = cas.horzcat(ehat_chord, ehat_span, ehat_l)

        # lift and drag coefficients
        CL = coeff[0]
        CD = parameters['theta0','aero','CD0'] + CL ** 2/ (np.pi*parameters['theta0','geometry','ar'])

        # lift and drag force
        f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(ueff.T, ueff) * parameters['theta0','geometry','s_ref'] * ehat_l
        f_drag = CD * 1. / 2. * rho_infty * ueff_norm * parameters['theta0','geometry','s_ref'] * ueff
        f_side = cas.DM(np.zeros((3, 1)))

        f_aero = f_lift + f_drag
        m_aero = cas.DM(np.zeros((3, 1)))

        CA = CD
        CN = CL
        CY = cas.DM(0.)

        aero_coefficients = {}
        aero_coefficients['CD'] = CD
        aero_coefficients['CL'] = CL
        aero_coefficients['CA'] = CA
        aero_coefficients['CN'] = CN
        aero_coefficients['CY'] = CY

        outputs = indicators.collect_kite_aerodynamics_outputs(options, atmos, ueff, ueff_norm, aero_coefficients, f_aero,
                                                               f_lift, f_drag, f_side, m_aero, ehat_chord, ehat_span, r, q, kite, outputs,parameters)
        outputs = indicators.collect_environmental_outputs(atmos, wind, q, kite, outputs)
        outputs = indicators.collect_aero_validity_outputs(options, xd, ueff, kite, parent, outputs,parameters)
        outputs = indicators.collect_local_performance_outputs(options, atmos, wind, variables, CL, CD, elevation_angle, ueff, kite, parent,
                                          outputs, parameters)
        outputs = indicators.collect_power_balance_outputs(variables, kite, outputs, architecture)

    return outputs



def get_ehat_l_and_span(kite, options, wind, variables, architecture):
    parent_map = architecture.parent_map
    xd = variables['xd']

    parent = parent_map[kite]

    # get relevant variables for kite
    q = xd['q' + str(kite) + str(parent)]
    dq = xd['dq' + str(kite) + str(parent)]
    coeff = xd['coeff' + str(kite) + str(parent)]

    # wind parameters
    uw_infty = wind.get_velocity(q[2])

    # apparent air velocity
    if not (options['induction_model'] == 'not_in_use'):
        ueff = induction.get_kite_effective_velocity(options, variables, wind, kite, architecture)
    else:
        ueff = uw_infty - dq

    # in kite body:
    if parent > 0:
        grandparent = parent_map[parent]
        qparent = xd['q' + str(parent) + str(grandparent)]
    else:
        qparent = np.array([0., 0., 0.])

    ehat_r = (q - qparent) / vect_op.norm(q - qparent)
    ehat_t = vect_op.normed_cross(ueff, ehat_r)
    ehat_s = vect_op.normed_cross(ehat_t, ueff)

    # roll angle
    psi = coeff[1]

    ehat_l = cas.cos(psi) * ehat_s + cas.sin(psi) * ehat_t
    ehat_span = cas.cos(psi) * ehat_t - cas.sin(psi) * ehat_s

    return ehat_l, ehat_span

def get_wingtip_position(kite, options, model, variables, parameters, ext_int):

    parent_map = model.architecture.parent_map
    xd = model.variables_dict['xd'](variables['xd'])

    if ext_int == 'ext':
        span_sign = 1.
    elif ext_int == 'int':
        span_sign = -1.
    else:
        pdb.set_trace()

    parent = parent_map[kite]

    q = xd['q' + str(kite) + str(parent)]

    _, ehat_span = get_ehat_l_and_span(kite, options, model.wind, variables, model.architecture)

    b_ref = parameters['theta0', 'geometry', 'b_ref']

    wingtip_position = q + ehat_span * span_sign * b_ref / 2.

    return wingtip_position