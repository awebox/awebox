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

import awebox.mdl.aero.actuator_disk_dir.flow as actuator_disk_flow

def get_outputs(options, atmos, wind, variables, outputs, parameters, architecture):
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    xd = variables['xd']

    elevation_angle = indicators.get_elevation_angle(xd)

    for n in kite_nodes:

        parent = parent_map[n]

        # get relevant variables for kite n
        q = xd['q' + str(n) + str(parent)]
        dq = xd['dq' + str(n) + str(parent)]
        coeff = xd['coeff' + str(n) + str(parent)]

        # wind parameters
        rho_infty = atmos.get_density(q[2])
        uw_infty = wind.get_velocity(q[2])

        # apparent air velocity
        if options['induction_model'] == 'actuator':
            ua = actuator_disk_flow.get_kite_effective_velocity(options, variables, wind, n, parent, architecture)
        else:
            ua = uw_infty - dq

        # relative air speed
        ua_norm = vect_op.smooth_norm(ua, epsilon=1e-8)
        # ua_norm = mtimes(ua.T, ua) ** 0.5

        # in kite body:
        if parent > 0:
            grandparent = parent_map[parent]
            qparent = xd['q' + str(parent) + str(grandparent)]
        else:
            qparent = np.array([0., 0., 0.])

        ehat_r = (q - qparent) / vect_op.norm(q - qparent)
        ehat_t = vect_op.normed_cross(ua, ehat_r)
        ehat_s = vect_op.normed_cross(ehat_t, ua)

        # roll angle
        psi = coeff[1]

        ehat_l = cas.cos(psi) * ehat_s + cas.sin(psi) * ehat_t
        ehat_span = cas.cos(psi) * ehat_t - cas.sin(psi) * ehat_s
        ehat_chord = ua/ua_norm

        # implicit direct cosine matrix (for plotting only)
        r = cas.horzcat(ehat_chord, ehat_span, ehat_l)

        # lift and drag coefficients
        CL = coeff[0]
        CD = parameters['theta0','aero','CD0'] + CL ** 2/ (np.pi*parameters['theta0','geometry','ar'])

        # lift and drag force
        f_lift = CL * 1. / 2. * rho_infty * cas.mtimes(ua.T, ua) * parameters['theta0','geometry','s_ref'] * ehat_l
        f_drag = CD * 1. / 2. * rho_infty * ua_norm * parameters['theta0','geometry','s_ref'] * ua
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

        outputs = indicators.collect_kite_aerodynamics_outputs(options, atmos, ua, ua_norm, aero_coefficients, f_aero,
                                                               f_lift, f_drag, f_side, m_aero, ehat_chord, ehat_span, r, q, n, outputs,parameters)
        outputs = indicators.collect_environmental_outputs(atmos, wind, q, n, outputs)
        outputs = indicators.collect_aero_validity_outputs(options, xd, ua, n, parent, outputs,parameters)
        outputs = indicators.collect_local_performance_outputs(options, atmos, wind, variables, CL, CD, elevation_angle, ua, n, parent,
                                          outputs, parameters)
        outputs = indicators.collect_power_balance_outputs(variables, n, outputs, architecture)

    return outputs
