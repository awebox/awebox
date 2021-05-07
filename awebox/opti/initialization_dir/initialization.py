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
simple initialization intended for awe systems
initializes to a simple uniform circle path for kites, and constant location for tether nodes
no reel-in or out, as inteded for base of tracking problem
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''

import numpy as np
import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op

import awebox.opti.initialization_dir.induction as induction
import awebox.opti.initialization_dir.landing_scenario as landing
import awebox.opti.initialization_dir.standard_scenario as standard
import awebox.opti.initialization_dir.transition_scenario as transition

def get_initial_guess(nlp, model, formulation, init_options):
    V_init_si = build_si_initial_guess(nlp, model, formulation, init_options)

    if True in np.isnan(np.array(V_init_si.cat)):
        raise ValueError('NaN detected in V_init_si')
    V_init = struct_op.si_to_scaled(V_init_si, model.scaling)

    return V_init


def initialize_multipliers_to_nonzero(V_init):
    if 'xa' in list(V_init.keys()):
        V_init['xa', :] = 1.
    if 'coll_var' in list(V_init.keys()):
        V_init['coll_var', :, :, 'xa'] = 1.

    return V_init


def build_si_initial_guess(nlp, model, formulation, init_options):
    awelogger.logger.info('build si initial guess...')

    V = nlp.V
    V_init = V(0.0)

    # set lagrange multipliers different from zero to avoid singularity
    V_init = initialize_multipliers_to_nonzero(V_init)

    if not init_options['type'] in ['nominal_landing', 'compromised_landing', 'transition']:
        init_options = standard.precompute_path_parameters(init_options, model)

    ntp_dict = get_normalized_time_param_dict(nlp, formulation, init_options, V_init)
    V_init = set_normalized_time_params(init_options, formulation, V_init)

    V_init = set_final_time(init_options, V_init, model, formulation, ntp_dict)

    V_init = extract_time_grid(model, nlp, formulation, init_options, V_init, ntp_dict)

    V_init = induction.initial_guess_induction(init_options, nlp, formulation, model, V_init)

    V_init = set_xddot(V_init, nlp)

    # specified initial values for system parameters
    V_init = set_nontime_system_parameters(init_options, model, V_init)

    # initial values for homotopy parameters
    for name in list(model.parameters_dict['phi'].keys()):
        V_init['phi', name] = 1.

    return V_init


def get_normalized_time_param_dict(nlp, formulation, init_options, V_init):
    ntp_dict = {'d': nlp.d, 'n0': -999, 'n_min': -999, 'd_min': -999, 'n_min_f': -999, 'd_min_f': -999, 'n_min_0': -999,
                'd_min_0': -999}

    if init_options['type'] in ['nominal_landing']:
        ntp_dict = landing.get_nominal_landing_normalized_time_param_dict(ntp_dict, formulation)

    elif init_options['type'] in ['compromised_landing']:
        ntp_dict = landing.get_compromised_landing_normalized_time_param_dict(ntp_dict, formulation)

    elif init_options['type'] in ['transition']:
        ntp_dict = transition.get_normalized_time_param_dict(ntp_dict, formulation)

    else:
        ntp_dict = standard.get_normalized_time_param_dict(ntp_dict, formulation)

    return ntp_dict


def set_normalized_time_params(init_options, formulation, V_init):
    if init_options['type'] in ['nominal_landing']:
        V_init = landing.set_nominal_landing_normalized_time_params(formulation, V_init)

    elif init_options['type'] in ['compromised_landing']:
        V_init = landing.set_compromised_landing_normalized_time_params(formulation, V_init)

    elif init_options['type'] in ['transition']:
        V_init = transition.set_normalized_time_params(formulation, V_init)

    else:
        V_init = standard.set_normalized_time_params(formulation, V_init)

    return V_init


def set_final_time(init_options, V_init, model, formulation, ntp_dict):
    if init_options['type'] in ['nominal_landing']:
        tf_guess = landing.guess_final_time(init_options, formulation, ntp_dict)

    elif init_options['type'] in ['compromised_landing']:
        tf_guess = landing.guess_final_time(init_options, formulation, ntp_dict)

    elif init_options['type'] in ['transition']:
        tf_guess = transition.guess_final_time(init_options, formulation, ntp_dict)

    else:
        tf_guess = standard.guess_final_time(init_options, model)

    use_phase_fixing = V_init['theta', 't_f'].shape[0] > 1
    if use_phase_fixing:
        tf_guess = cas.vertcat(tf_guess, tf_guess)

    V_init['theta', 't_f'] = tf_guess

    return V_init


def extract_time_grid(model, nlp, formulation, init_options, V_init, ntp_dict):
    tf_guess = V_init['theta', 't_f']

    # extract time grid
    tgrid_xd = nlp.time_grids['x'](tf_guess)
    if 'coll' in list(nlp.time_grids.keys()):
        tgrid_coll = nlp.time_grids['coll'](tf_guess)

    d = nlp.d
    n_k = nlp.n_k

    for ndx in range(n_k + 1):

        t = tgrid_xd[ndx]

        ret = guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict)

        for var_type in ['xd', 'xl']:
            for name in struct_op.subkeys(model.variables, var_type):
                if (name in ret.keys()) and (var_type == 'xd' or ndx < n_k) and var_type in V_init.keys():
                    V_init[var_type, ndx, name] = ret[name]

        if nlp.discretization == 'direct_collocation' and (ndx < n_k):
            for ddx in range(d):
                t = tgrid_coll[ndx, ddx]

                ret = guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict)

                for var_type in ['xd', 'xl']:
                    for name in struct_op.subkeys(model.variables, var_type):
                        if name in ret.keys():
                            V_init['coll_var', ndx, ddx, var_type, name] = ret[name]

    return V_init


def guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict):
    if init_options['type'] in ['nominal_landing']:
        ret = landing.guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict)

    elif init_options['type'] in ['compromised_landing']:
        ret = landing.guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict)

    elif init_options['type'] in ['transition']:
        ret = transition.guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict)

    else:
        ret = standard.guess_values_at_time(t, init_options, model)

    return ret


def set_nontime_system_parameters(init_options, model, V_init):
    for name in set(struct_op.subkeys(model.variables, 'theta')) - set(['t_f']):
        if name in list(init_options['theta'].keys()):
            V_init['theta', name] = init_options['theta'][name]
        elif name[:3] == 'l_c':
            layer = int(name[3:])
            kites = model.architecture.kites_map[layer]
            q_first = V_init['xd', 0, 'q{}{}'.format(kites[0], model.architecture.parent_map[kites[0]])]
            q_second = V_init['xd', 0, 'q{}{}'.format(kites[1], model.architecture.parent_map[kites[1]])]
            V_init['theta', name] = np.linalg.norm(q_first - q_second)
            if init_options['cross_tether_attachment'] == 'wing_tip':
                V_init['theta', name] += - init_options['sys_params_num']['geometry']['b_ref']
        elif name[:6] == 'diam_c':
            V_init['theta', name] = init_options['theta']['diam_c']
        else:
            raise ValueError("please specify an initial value for variable '" + name + "' of type 'theta'")

    return V_init

def set_xddot(V_init, nlp):

    if 'xddot' in list(V_init.keys()):
        Xdot_init = nlp.Xdot(nlp.Xdot_fun(V_init))
        for k in range(nlp.n_k):
            V_init['xddot',k] = Xdot_init['xd',k]
    return V_init 
