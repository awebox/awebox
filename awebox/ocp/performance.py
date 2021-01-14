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
performance helping file,
finds various total-trajectory performance metrics requiring knowlege of V
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold, jochem de schutter alu-fr 2017-18
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import awebox.tools.struct_operations as struct_op

def collect_performance_outputs(nlp_options, model, V):

    outputs = {}
    outputs = get_windings(nlp_options, model, V, outputs)
    outputs = get_control_frequency(nlp_options, model, V, outputs)
    outputs = get_time_period(nlp_options, V, outputs)
    [outputs_struct, outputs_dict] = make_output_structure(outputs)

    return outputs_struct, outputs_dict

def get_time_period(nlp_options, V, outputs):

    if 'time_period' not in list(outputs.keys()):
        outputs['time_period'] = {}

    n_k = nlp_options['n_k']
    pf_reelout = nlp_options['phase_fix_reelout']

    if nlp_options['phase_fix'] == 'single_reelout':
        time_period_zeroth = V['theta', 't_f',0] * round(n_k * pf_reelout)
        time_period_first = V['theta', 't_f',1] * (n_k - round(n_k * pf_reelout))
        # average over collocation nodes
        time_period = (time_period_zeroth + time_period_first) / n_k
    else:
        time_period = V['theta','t_f']

    outputs['time_period']['val'] = time_period

    return outputs

def get_control_frequency(nlp_options, model, V, outputs):

    if 'control_freq' not in list(outputs.keys()):
        outputs['control_freq'] = {}

    nk = nlp_options['n_k']

    for delta in struct_op.subkeys(model.variables, 'u'):
        if ('delta' in delta) and (not 'ddelta' in delta):
            variable = V['u', 0, delta]

            for dim in range(variable.shape[0]):
                control_freq = vect_op.estimate_1d_frequency(cas.vertcat(*V['u', :, delta, dim]), dt=V['theta', 't_f'] / nk)
                outputs['control_freq'][delta + '_' + str(dim)] = control_freq

    for delta in struct_op.subkeys(model.variables, 'xd'):
        if ('delta' in delta) and (not 'ddelta' in delta):
            variable = V['xd', 0, delta]

            for dim in range(variable.shape[0]):
                control_freq = vect_op.estimate_1d_frequency(cas.vertcat(*V['xd', :, delta, dim]), dt=V['theta', 't_f'] / nk)
                outputs['control_freq'][delta + '_' + str(dim)] = control_freq

    return outputs

def make_output_structure(outputs):
    outputs_vec = []
    full_list = []

    outputs_dict = {}

    for output_type in list(outputs.keys()):

        local_list = []
        for name in list(outputs[output_type].keys()):
            # prepare empty entry list to generate substruct
            local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

            # generate vector with outputs - SX expressions
            outputs_vec = cas.vertcat(outputs_vec, outputs[output_type][name])

        # generate dict with sub-structs
        outputs_dict[output_type] = cas.struct_symMX(local_list)
        # prepare list with sub-structs to generate struct
        full_list += [cas.entry(output_type, struct=outputs_dict[output_type])]

        # generate "empty" structure
    out_struct = cas.struct_symMX(full_list)
    # generate structure with SX expressions
    outputs_struct = out_struct(outputs_vec)

    return [outputs_struct, outputs_dict]

def get_windings(nlp_options, model, V, outputs={}):

    if 'winding' not in list(outputs.keys()):
        outputs['winding'] = {}

    nk = nlp_options['n_k']

    parent_map = model.architecture.parent_map
    kite_nodes = model.architecture.kite_nodes

    ehat_tether_x = 0.
    ehat_tether_y = 0.
    ehat_tether_z = 0.
    total_steps = float(nk)

    # TODO: in case of collocation, include collocation node info weighted with quad_weights
    # TODO: weight with time constant in case of phase fixing!
    for kdx in range(nk):
        local_ehat = vect_op.normalize(V['xd', kdx, 'q10'])
        ehat_tether_x += local_ehat[0] / total_steps
        ehat_tether_y += local_ehat[1] / total_steps
        ehat_tether_z += local_ehat[2] / total_steps

    ehat_tether = vect_op.normalize(cas.vertcat(ehat_tether_x, ehat_tether_y, ehat_tether_z))

    outputs['winding']['ehat_tether'] = ehat_tether

    ehat_side_a = vect_op.normed_cross(vect_op.yhat_np(), ehat_tether)
    # right handed coordinate system -> x/re: _a, y/im: _b, z/out: _tether
    ehat_side_b = vect_op.normed_cross(ehat_tether, ehat_side_a)

    # now project the path onto this plane
    for n in kite_nodes:
        parent = parent_map[n]

        theta_start = 0.
        theta_end = 0.

        # find the origin of the plane
        origin = np.zeros((3, 1))
        for kdx in range(nk):
            q = V['xd', kdx, 'q' + str(n) + str(parent)]
            q_in_plane = q - vect_op.dot(q, ehat_tether) * ehat_tether

            origin = origin + q_in_plane / total_steps

        # recenter the plane about origin
        for kdx in range(nk):
            q = V['xd', kdx, 'q' + str(n) + str(parent)]
            q_in_plane = q - vect_op.dot(q, ehat_tether) * ehat_tether
            q_recentered = q_in_plane - origin

            q_next = V['xd', kdx+1, 'q' + str(n) + str(parent)]
            q_next_in_plane = q_next - vect_op.dot(q_next, ehat_tether) * ehat_tether
            q_next_recentered = q_next_in_plane - origin

            delta_q = q_next_recentered - q_recentered

            x = vect_op.dot(q_recentered, ehat_side_a)
            y = vect_op.dot(q_recentered, ehat_side_b)
            r_squared = x**2. + y**2.

            dx = vect_op.dot(delta_q, ehat_side_a)
            dy = vect_op.dot(delta_q, ehat_side_b)

                # dx = vect_op.dot(dq_in_plane, ehat_side_a)
                # dy = vect_op.dot(dq_in_plane, ehat_side_b)

            dtheta = (x * dy - y * dx) / (r_squared + 1.0e-4)
            theta_end += dtheta

        winding = (theta_end - theta_start) / 2. / np.pi
        outputs['winding']['winding' + str(n)] = winding

    return outputs

def find_phase_fix_time_period_zeroth(nlp_numerics_options, V):

    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_zeroth = V['theta', 't_f', 0] * round(nk * phase_fix_reel_out) / nk
    return time_period_zeroth

def find_phase_fix_time_period_first(nlp_numerics_options, V):
    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_first = V['theta', 't_f', 1] * (nk - round(nk * phase_fix_reel_out)) / nk
    return time_period_first

def find_time_period(nlp_numerics_options, V):

    if nlp_numerics_options['phase_fix'] == 'single_reelout':
        time_period_zeroth = find_phase_fix_time_period_zeroth(nlp_numerics_options, V)
        time_period_first = find_phase_fix_time_period_first(nlp_numerics_options, V)

        # average over collocation nodes
        time_period = (time_period_zeroth + time_period_first)
    else:
        time_period = V['theta', 't_f']

    return time_period

