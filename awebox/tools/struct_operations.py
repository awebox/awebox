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
file to provide structure operations to the awebox,
_python-3.5 / casadi-3.4.5
- author: thilo bronnenmeyer, jochem de schutter, rachel leuthold, 2017-20
'''
from typing import Dict

import casadi as ca
import casadi.tools as cas
import numpy as np
import operator

import copy
from functools import reduce
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
from awebox.ocp.discretization import construct_time_grids
from awebox.tools.sam_functionalities import construct_time_grids_SAM_reconstruction, originalTimeToSAMTime


def subkeys(casadi_struct, key):

    if key in casadi_struct.keys():
        indices = np.array(casadi_struct.f[key])
        number_index = indices.shape[0]

        subkeys = set()
        for idx in range(number_index):
            canonical = casadi_struct.getCanonicalIndex(indices[idx])
            new_key = canonical[1]
            subkeys.add(new_key)

        subkey_list = sorted(subkeys)
    else:
        subkey_list = []

    return subkey_list

def count_shooting_nodes(nlp_options):
    return nlp_options['n_k']

def get_shooting_vars(nlp_options, V, P, Xdot, model):

    shooting_nodes = count_shooting_nodes(nlp_options)

    shooting_vars = []
    for kdx in range(shooting_nodes):
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx)
        shooting_vars = cas.horzcat(shooting_vars, var_at_time)

    return shooting_vars

def get_shooting_params(nlp_options, V, P, model):

    shooting_nodes = count_shooting_nodes(nlp_options)

    parameters = model.parameters
    coll_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, (shooting_nodes))
    return coll_params



def get_coll_vars(nlp_options, V, P, Xdot, model):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    # construct list of all collocation node variables and parameters
    coll_vars = []
    for kdx in range(n_k):
        for ddx in range(d):
            var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx, ddx)
            coll_vars = cas.horzcat(coll_vars, var_at_time)

    return coll_vars


def get_coll_params(nlp_options, V, P, model):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    N_coll = n_k * d # collocation points

    parameters = model.parameters
    coll_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, N_coll)
    return coll_params


def get_ms_vars(nlp_options, V, P, Xdot, model):

    n_k = nlp_options['n_k']

    # construct list of all multiple-shooting node variables and parameters
    ms_vars = []

    # todo: should this be range (n_k + 1), ie, include the terminal node?
    for kdx in range(n_k):
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, kdx)
        ms_vars = cas.horzcat(ms_vars, var_at_time)

    return ms_vars


def get_ms_params(nlp_options, V, P, Xdot, model):
    n_k = nlp_options['n_k']
    N_ms = n_k  # collocation points

    parameters = model.parameters

    ms_params = cas.repmat(parameters(cas.vertcat(P['theta0'], V['phi'])), 1, N_ms+1)

    return ms_params


def get_algebraics_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'z'
    coll_var_name = get_collocation_var_name(V)
    V_has_collocation_vars = (coll_var_name != 'not_in_use')

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    at_control_node = (ddx is None)
    z_in_V_keys = var_type in V.keys()

    if at_control_node and z_in_V_keys:
        if kdx < nlp_options['n_k']:
            return V[var_type, kdx]
        else:
            message = 'something went wrong with the index of an algebraic variable with kdx = ' + str(kdx)
            print_op.log_and_raise_error(message)
    elif at_control_node and V_has_collocation_vars:
        return V[coll_var_name, kdx, 0, var_type]
    elif direct_collocation and V_has_collocation_vars:
        return V[coll_var_name, kdx, ddx, var_type]
    else:
        message = 'something went wrong when returning algebraic variables'
        print_op.log_and_raise_error(message)


def get_states_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'x'

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    at_control_node = (ddx is None)
    coll_var_name = get_collocation_var_name(V)
    V_has_collocation_vars = (coll_var_name != 'not_in_use')

    if at_control_node:
        return V[var_type, kdx]
    elif direct_collocation and V_has_collocation_vars:
        return V[coll_var_name, kdx, ddx, var_type]
    else:
        message = 'something went wrong when getting the states'
        print_op.log_and_raise_error(message)


def get_controls_at_time(nlp_options, V, model_variables, kdx, ddx=None):

    var_type = 'u'

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    coll_var_name = get_collocation_var_name(V)

    piecewise_constant_controls = not (nlp_options['collocation']['u_param'] == 'poly')
    at_control_node = (ddx is None)
    before_last_node = kdx < nlp_options['n_k']

    if direct_collocation and piecewise_constant_controls and before_last_node:
        return V[var_type, kdx]

    elif direct_collocation and piecewise_constant_controls and (not before_last_node):
        return V[var_type, -1]

    elif direct_collocation and (not piecewise_constant_controls) and at_control_node and before_last_node:
        return V[coll_var_name, kdx, 0, var_type]

    elif direct_collocation and (not piecewise_constant_controls) and before_last_node:
        return V[coll_var_name, kdx, ddx, var_type]

    elif multiple_shooting and before_last_node:
        return V[var_type, kdx]

    else:
        message = 'controls unavailable'
        print_op.log_and_raise_error(message)


def get_collocation_var_name(V):
    collocation_var_name_possibilities = ['coll', 'coll_var']
    coll_var_name = 'not_in_use'
    for coll_var_name_poss in collocation_var_name_possibilities:
        if coll_var_name_poss in V.keys():
            coll_var_name = coll_var_name_poss
            return coll_var_name
    return coll_var_name


def get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_type = 'xdot'
    coll_var_name = get_collocation_var_name(V)

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    at_control_node = (ddx is None)

    passed_Xdot_is_meaningful = (Xdot is not None)

    derivs_lifted_in_V = ('xdot' in list(V.keys()))

    if at_control_node and derivs_lifted_in_V and kdx < n_k:
        return V[var_type, kdx]
    elif at_control_node and passed_Xdot_is_meaningful and kdx < n_k:
        return Xdot['x', kdx]
    elif passed_Xdot_is_meaningful and kdx < n_k:
        return Xdot['coll_x', kdx, ddx]
    else:

        if ddx == d - 1:
            kdx = kdx + 1
            ddx = None
        at_control_node = (ddx is None)

        u_is_zoh = ('u' in V.keys())
        V_has_collocation_vars = (coll_var_name != 'not_in_use')

        attempted_reassamble = []
        for idx in range(model_variables.shape[0]):
            can_index = model_variables.getCanonicalIndex(idx)
            local_variable_has_a_derivative = (can_index[0] == 'x')
            if local_variable_has_a_derivative:

                var_name = can_index[1]
                dim = can_index[2]

                deriv_name = 'd' + var_name
                deriv_name_in_states = deriv_name in subkeys(model_variables, 'x')
                deriv_name_in_controls = deriv_name in subkeys(model_variables, 'u')

                if at_control_node and deriv_name_in_states:
                    local_val = V['x', kdx, deriv_name, dim]
                elif at_control_node and deriv_name_in_controls and u_is_zoh and kdx < n_k:
                    local_val = V['u', kdx, deriv_name, dim]
                elif at_control_node and deriv_name_in_controls and not u_is_zoh:
                    kdx_local = kdx - 1
                    ddx_local = -1
                    local_val = V[coll_var_name, kdx_local, ddx_local, 'u', deriv_name, dim]
                elif deriv_name_in_states and V_has_collocation_vars and kdx < n_k:
                    local_val = V[coll_var_name, kdx, ddx, 'x', deriv_name, dim]
                elif deriv_name_in_controls and V_has_collocation_vars and not u_is_zoh and kdx < n_k:
                    local_val = V[coll_var_name, kdx, ddx, 'u', deriv_name, dim]
                else:
                    local_val = cas.DM.zeros((1, 1))

                attempted_reassamble = cas.vertcat(attempted_reassamble, local_val)
        return attempted_reassamble


def test_continuity_of_get_variables_at_time(nlp_options, V_init_si, model):

    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    coll_var_name = get_collocation_var_name(V_init_si)
    V_has_collocation_vars = (coll_var_name != 'not_in_use')

    if direct_collocation and V_has_collocation_vars:

        coll_var_name = get_collocation_var_name(V_init_si)

        thresholds = {}
        for var_type in set(V_init_si.keys()) - set(['z', 'theta', 'phi', 'xi', 'u', coll_var_name]):
            thresholds[var_type] = 1.e-5
        # notice that the algebraic variables are computed depending on the rest of the inputs, potentially including the
        # controls, so the computation might look different across the control node, where u is discontinuous.
        thresholds['z'] = 0.1

        Xdot = None

        ndx_coll = 0
        ddx_coll = nlp_options['collocation']['d'] - 1

        ndx_control = ndx_coll + 1

        extract_vars_coll = get_variables_at_time(nlp_options, V_init_si, Xdot, model.variables, ndx_coll, ddx_coll)
        extract_vars_control = get_variables_at_time(nlp_options, V_init_si, Xdot, model.variables, ndx_control)

        diff = extract_vars_control.cat - extract_vars_coll.cat
        factor = cas.inv(cas.diag(vect_op.smooth_abs(extract_vars_coll.cat)))
        normalized_diff = cas.mtimes(factor, diff)
        diff_allocated = model.variables(normalized_diff)
        listed_differences = {}
        for idx in range(diff.shape[0]):
            local_label = diff_allocated.labels()[idx]
            local_canonical = diff_allocated.getCanonicalIndex(idx)

            if local_canonical in thresholds.keys():

                is_non_u = (local_canonical[0] != 'u')
                is_non_xdot = (local_canonical[0] != 'xdot')
                is_xdot_but_known = (local_canonical[0] == 'xdot') and (local_canonical[1] in (model.variables_dict['x'].keys() + model.variables_dict['u'].keys()))
                is_reasonable_comparison = is_non_u and (is_non_xdot or is_xdot_but_known)

                if is_reasonable_comparison and vect_op.smooth_abs(normalized_diff[idx]) > thresholds[local_canonical]:
                    listed_differences[local_label] = normalized_diff[idx]

        if len(listed_differences.keys()) > 0:
            message = 'the variable slices produced by struct_op are not continuous. normalized differences are: '
            print_op.base_print(message, level='error')
            print_op.print_dict_as_table(listed_differences, level='error')
            raise Exception(message + repr(listed_differences))

    elif direct_collocation and not(V_has_collocation_vars):
        message = 'WARNING: if you are not re-initializing from multiple-shooting to direct-collocation or vice-versa, something may have gone wrong with the structuring of V.'
        print_op.base_print(message, level='warning')

    return None

def get_variables_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx=None):

    var_list = []
    # make list of variables at specific time
    for var_type in model_variables.keys():

        if var_type == 'z':
            local_var = get_algebraics_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'x':
            local_var = get_states_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'u':
            local_var = get_controls_at_time(nlp_options, V, model_variables, kdx, ddx)

        elif var_type == 'theta':
            local_var = get_V_theta(V, nlp_options, kdx)

        elif var_type == 'xdot':
            local_var = get_derivs_at_time(nlp_options, V, Xdot, model_variables, kdx, ddx)

        var_list.append(local_var)

    var_at_time = model_variables(cas.vertcat(*var_list))

    return var_at_time



def get_variables_at_final_time(nlp_options, V, Xdot, model):

    multiple_shooting = (nlp_options['discretization'] == 'multiple_shooting')

    scheme = nlp_options['collocation']['scheme']
    direct_collocation = (nlp_options['discretization'] == 'direct_collocation')
    radau_collocation = (direct_collocation and scheme == 'radau')
    other_collocation = (direct_collocation and (not scheme == 'radau'))

    terminal_constraint = nlp_options['mpc']['terminal_point_constr']

    if radau_collocation and not terminal_constraint:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1, -1)
    elif direct_collocation or multiple_shooting:
        var_at_time = get_variables_at_time(nlp_options, V, Xdot, model.variables, -1)
    else:
        message = 'unfamiliar discretization option chosen: ' + nlp_options['discretization']
        print_op.log_and_raise_error(message)

    return var_at_time

def get_parameters_at_time(V, P, model_parameters):
    param_list = []

    for var_type in list(model_parameters.keys()):
        if var_type == 'phi':
            param_list.append(V[var_type])
        if var_type == 'theta0':
            param_list.append(P[var_type])

    param_at_time = model_parameters(cas.vertcat(*param_list))

    return param_at_time

def get_var_ref_at_time(nlp_options, P, V, Xdot, model, kdx, ddx=None):
    V_from_P = V(P['p', 'ref'])
    var_at_time = get_variables_at_time(nlp_options, V_from_P, Xdot, model.variables, kdx, ddx)
    return var_at_time

def get_var_ref_at_final_time(nlp_options, P, V, Xdot, model):
    V_from_P = V(P['p', 'ref'])
    var_at_time = get_variables_at_final_time(nlp_options, V_from_P, Xdot, model)
    return var_at_time

def get_V_theta(V, nlp_numerics_options, k):

    nk = nlp_numerics_options['n_k']
    k = list(range(nk+1))[k]

    if V['theta', 't_f'].shape[0] == 1:
        theta = V['theta']
    else:
        theta = []
        tf_index = V.f['theta', 't_f']
        theta_index = V.f['theta']
        for idx in theta_index:
            if idx == tf_index[0] and k < round(nk * nlp_numerics_options['phase_fix_reelout']):
                theta.append(V.cat[idx])
            elif idx == tf_index[1] and k >= round(nk * nlp_numerics_options['phase_fix_reelout']):
                theta.append(V.cat[idx])
            elif idx not in tf_index:
                theta.append(V.cat[idx])
        theta = cas.vertcat(*theta)

    return theta


def calculate_SAM_regionIndex(nlp_options: dict, k: int) -> int:
    """ Calculate the SAM region index of the interval index k.
    The integration horizon [0,nk] is divided into `n_tf` regions.
        The first ~40% are reserved for the individual micro-integrations of the SAM scheme,
        The last ~60% is reserved for the the reel-in phase.
    Note that this 40/60 relation is also hardcoded in the initialization of the t_fs

    : param nlp_options: dictionary with the nlp options e.g option['nlp']
    : param k: the integration interval index on the awebox grid

    ToDo: this could be done more efficiently, all of this can be precomputed. This function is evaluated a lot of times.
    """

    n_k = nlp_options['n_k']  # the total number of integration intervals
    n_k_ratio = 0.4 # the ratio of the number of intervals in the reel-out phase
    n_micros = nlp_options['SAM']['d']  # the number of micro-integrations
    n_regions = n_micros + 1  # the number of time-scaling regions

    # 1. reserve (~n_k_ratio) of the ingeration intervals for reelin phase
    n_single_micro = round(n_k*n_k_ratio/n_micros) # the number of intervals in a single micro-integration
    n_RO_intervals = n_single_micro*n_micros
    n_RI_intervals = n_k - n_RO_intervals

    # the number integrations intervals per regions
    delta_ns = np.array([n_single_micro]*n_micros + [n_RI_intervals])
    assert np.sum(delta_ns) == n_k, 'sum of the rounded delta_ns must be equal to n_k'

    # 2. calculate the region index
    region_indx = np.sum(k >= np.cumsum(delta_ns))
    if region_indx > n_regions - 1:
        # last interval? -> return last region
        region_indx = n_regions - 1

    return int(region_indx)


def calculate_SAM_regions(nlp_options: dict) -> list:
    """ Returs a list of lists, where each list contains the indices of the k's that belong to the same SAM region.
        For example, nk= 10, d_SAM=1, and thus 2 regions, the output will be
        [[0,1,2,4,5,6],[7,8,9,10]]

        :param nlp_options: dictionary with the nlp options e.g option['nlp']
        :return: list of lists with of indices
    """
    assert nlp_options['SAM']['use'] or nlp_options['SAM']['flag_SAM_reconstruction']
    n_k = nlp_options['n_k']
    n_tf = nlp_options['SAM']['d'] + 1
    return_list = [[] for _ in range(n_tf)]  # generate a list with n_tf empty lists
    for k in range(n_k):
        return_list[calculate_SAM_regionIndex(nlp_options, k)].append(k)
    return return_list


def calculate_tf_index(nlp_options, k):
    """ Calculates the index of the discretization region (with a constant tf) that the index `k` belongs to """
    nk = nlp_options['n_k']
    if nlp_options['SAM']['use']:
        # SAM with (d+1) regions?
        assert nlp_options['phase_fix'] == 'single_reelout', 'phase fix must be single_reelout for SAM'
        return calculate_SAM_regionIndex(nlp_options, k)

    elif nlp_options['phase_fix'] == 'single_reelout':
        # single reel-out? -> two regions
        k_reelout = round(nk * nlp_options['phase_fix_reelout'])
        if k < k_reelout:
            return 0
        else:
            return 1
    else:
        # else -> only one region
        return 0

def calculate_tf(params, V, k):

    nk = params['n_k']

    if params['SAM']['use']:
        assert params['phase_fix'] == 'single_reelout', 'phase fix must be single_reelout for SAM'
        tf =  V['theta', 't_f', calculate_SAM_regionIndex(params, k)]
    elif params['phase_fix'] == 'single_reelout':
        if k < round(nk * params['phase_fix_reelout']):
            tf = V['theta', 't_f', 0]
        else:
            tf = V['theta', 't_f', 1]
    else:
        tf = V['theta', 't_f']

    return tf


def calculate_SAM_regionIndexArray(nlpoptions, Vopt, t: np.ndarray) -> np.ndarray:
    """
    For a given time vector t IN THE ORIGINAL AWEBOX TIME, calculate the region index for each time point.

    :param nlpoptions: dictionary with the nlp options e.g option['nlp']
    :param Vopt: the solution struct
    :param t: numpy array with the time points

    """

    assert type(t) is np.ndarray, f't must be a numpy array, but is {type(t)}'

    n_k = nlpoptions['n_k']
    indeces_regions = calculate_SAM_regions(nlpoptions)
    delta_ns = np.array([len(region) for region in indeces_regions]) # number integration intervals in each region
    tfs = Vopt['theta', 't_f',:].full().flatten()
    delta_ts = tfs/n_k * delta_ns # duration of each phase in physical time
    ts_cumsum = np.cumsum(np.append(0,delta_ts)) # cumulative sum of the phase durations

    # evaluate the region index for a numpy array of times t
    region_index = np.array([np.sum(s >= ts_cumsum) - 1 for s in t])
    assert np.all(region_index >= 0), 'region index must be positive'

    # due to numerical errors: make sure that the region index is smaller than the number of regions
    if np.any(region_index >= len(delta_ns)):
        region_index[region_index >= len(delta_ns)] = len(delta_ns) - 1
        # log a warnging
        awelogger.logger.warning('Some time points are larger than the total simulation time. The region index is set to the last region.')
    return region_index


def calculate_kdx(params, V, t):

    # do some checks on the params dictionary, since sometimes they are not passed correctly
    for needed_key in ['n_k', 'phase_fix','SAM']:
        assert needed_key in params, f'There should be a "{needed_key}" key in the params dictionary, but it is not. {params.keys()}'

    n_k = params['n_k']
    lift_mode_with_single_reelout_phase_fixing = (V['theta', 't_f'].shape[0] == 2)

    if params['SAM']['use']:
        assert params['phase_fix'] == 'single_reelout', 'phase fix must be single_reelout for SAM'
        kdx, tau = calculate_kdx_SAM(params, V, t)
    elif params['SAM']['flag_SAM_reconstruction']:
        kdx, tau = calculate_kdx_SAM_reconstruction(params, V, t)
    elif lift_mode_with_single_reelout_phase_fixing:
        k_reelout = round(n_k * params['phase_fix_reelout'])
        t_reelout = k_reelout * V['theta', 't_f', 0] / n_k
        if t <= t_reelout:
            noninteger_intervals_passed = t * n_k / V['theta', 't_f', 0]
            kdx = int(noninteger_intervals_passed)
            remainder_of_interval_since_last_control_node = noninteger_intervals_passed - kdx
            tau = remainder_of_interval_since_last_control_node
        else:
            kdx = int(k_reelout + int(n_k * (t - t_reelout) / V['theta', 't_f', 1]))
            tau = (t - t_reelout) / V['theta', 't_f', 1] * n_k - (kdx-k_reelout)
    else:
        if t != V['theta', 't_f', 0].full()[0][0]:
            t = t % V['theta', 't_f', 0].full()[0][0]
            kdx = int(n_k * t / V['theta', 't_f'])
            tau = t / V['theta', 't_f'] * n_k - kdx
        else:
            kdx = n_k - 1
            tau = 1.0

    if kdx == n_k:
        kdx = n_k - 1
        tau = 1.0

    return kdx, tau

def calculate_kdx_SAM(params, V, t) -> tuple:
    """
    calculate the index of the (micro)-integration interval for a given
    time t WHICH IS IN THE ORIGINAL AWEBOX TIME t \in [0, \sum(T_i)], not in the discontinuous SAM time.
    """
    n_k = params['n_k']
    indeces_regions = calculate_SAM_regions(params)
    delta_ns = np.array([len(region) for region in indeces_regions]) # number integration intervals in each region
    tfs = V['theta', 't_f',:].full().flatten()
    delta_ts = tfs/n_k * delta_ns # duration of each phase in physical time
    ts_cumsum = np.cumsum(np.append(0,delta_ts)) # cumulative sum of the phase durations
    region_index = np.sum(t >= ts_cumsum) - 1 # index of the region where t is located

    if region_index > tfs.shape[0] - 1:
        region_index = tfs.shape[0] - 1

    # calculate the (continuous) integration index of the given time
    n_t = np.cumsum(np.append(0,delta_ns))[region_index] + (t - ts_cumsum[region_index])*n_k/tfs[region_index]

    kdx = int(np.floor(n_t))
    tau = n_t - kdx

    # special case: last integration interval
    if kdx == n_k:
        kdx = n_k - 1
        tau = 1.0

    assert kdx < n_k, 'kdx must be smaller than n_k'
    assert kdx >= 0, 'kdx must be positive'
    assert tau <= 1.0, 'tau must be smaller than 1.0'
    assert tau >= 0.0, 'tau must be positive'

    return kdx, tau

_timegrid_reconstruct_save = None # THIS IS BAAAAAD, but so much faster

def calculate_kdx_SAM_reconstruction(nlpparams, V, t) -> tuple:
    """ Calculate the interval index kdx and the remaining relative interval tau duration of the interval in which the time t is located.
    This is valid only IF THE VARIABLES V are reconstructed versions of the SAM variables.

    TODO: This implementation is currently based on a GLOBAL variable _timegrid_reconstruct_save, which is bad practice.
    But if the timegrid is recalculated for each time point, the function is very slow. Also, there is no trivial
    way of infering the index of the reconstructed grid from the reconstructed variables V.
    Best would be to have an some class instance (e.g. a discretization instance) that stores
    the timegrid, or a casadi-function t ->(kdx, tau).
    """
    assert nlpparams['SAM']['flag_SAM_reconstruction'], 'This function is only valid for SAM reconstruction'

    # 1. build timegrid from t_f_opt
    global _timegrid_reconstruct_save
    if _timegrid_reconstruct_save is None:
        print('constructing timegrid for SAM reconstruction')
        _timegrid_reconstruct_save = construct_time_grids_SAM_reconstruction(nlpparams)
    else:
        # print('using saved timegrid for SAM reconstruction')
        pass
    timegrid_f = _timegrid_reconstruct_save
    timegrid_intervals = timegrid_f['x'](V['theta', 't_f']).full().flatten()

    # 2. find the region index using numpy.argmax(timegrid > t) - 1
    index = np.argmax(timegrid_intervals > t) - 1
    delta_t = timegrid_intervals[index + 1] - timegrid_intervals[index] # todo: this approximation could be better

    # 3. find the kdx and tau using the region index and the timegrid
    kdx = index
    tau = (t - timegrid_intervals[index]) / delta_t

    return kdx, tau



def variables_si_to_scaled(model_variables, variables_si, scaling):

    variables_scaled = copy.deepcopy(variables_si)

    for idx in range(model_variables.shape[0]):
        canonical = model_variables.getCanonicalIndex(idx)
        var_type = canonical[0]
        var_name = canonical[1]
        kdx = canonical[2]

        if kdx == 0:
            variables_scaled[var_type, var_name] = var_si_to_scaled(var_type, var_name, variables_scaled[var_type, var_name], scaling)

    return variables_scaled


def variables_scaled_to_si(model_variables, variables_scaled, scaling):

    stacked = []
    for idx in range(model_variables.shape[0]):
        canonical = model_variables.getCanonicalIndex(idx)
        var_type = canonical[0]
        var_name = canonical[1]
        kdx = canonical[2]
        if kdx == 0:
            new = var_scaled_to_si(var_type, var_name, variables_scaled[var_type, var_name], scaling)
            stacked = cas.vertcat(stacked, new)

    variables_si = model_variables(stacked)
    return variables_si


def should_variable_be_scaled(var_type, var_name, var_scaled, scaling):

    end_of_message = '. proceeding with unit scaling.'
    var_identifier = var_type + ' variable ' + var_name

    if (var_type == 'phi') or (var_type == 'xi'):
        return False, None

    scaling_defined_for_variable = hasattr(scaling, 'keys') and (var_type in scaling.keys()) and (var_name in subkeys(scaling, var_type))
    if not scaling_defined_for_variable:
        message = 'scaling information unavailable for ' + var_identifier + end_of_message
        return False, message

    scale = scaling[var_type, var_name]
    var_scaled_and_scaling_have_matching_shapes = hasattr(var_scaled, 'shape') and hasattr(scale, 'shape') and (var_scaled.shape == scale.shape)
    if (not var_scaled_and_scaling_have_matching_shapes) and (not var_name == 't_f'):
        message = 'shape mismatch between ' + var_identifier + ' value-to-be-scaled ' + repr(var_scaled.shape) + ' and scaling information ' + repr(scale.shape) + end_of_message
        print_op.log_and_raise_error(message)
        return False, message

    scaling_is_numeric_columnar = vect_op.is_numeric_columnar(scale)
    if not scaling_is_numeric_columnar:
        message = 'scaling information for ' + var_identifier + ' is not numeric and columnar' + end_of_message
        return False, message

    any_scaling_is_negative = any([scale[idx] < 0 for idx in range(scale.shape[0])])
    if any_scaling_is_negative:
        message = 'negative scaling values are not allowed' + end_of_message
        return False, message

    scaling_will_return_same_value_anyway = cas.diag(scale).is_eye()
    if scaling_will_return_same_value_anyway:
        return False, None

    return True, None


def check_and_rearrange_scaling_value_before_assignment(var_type, var_name, scaling_value, scaling):

    placeholder = scaling[var_type, var_name]

    if isinstance(scaling_value, int):
        scaling_value = float(scaling_value)

    if not vect_op.is_numeric(scaling_value):
        message = 'cannot set a non-numeric scaling value for ' + var_type + ' variable (' + var_name + '): ' + str(scaling_value)
        print_op.log_and_raise_error(message)

    if vect_op.is_numeric_scalar(scaling_value):
        scaling_value = scaling_value * cas.DM.ones(placeholder.shape)

    if not hasattr(scaling_value, 'shape'):
        scaling_value = cas.DM(scaling_value)

    if not scaling_value.shape == placeholder.shape:
        message = 'cannot set the scaling of ' + var_type + ' variable (' + var_name + '). proposed value has the wrong shape (' + str(scaling_value.shape) + ') when the expected shape is (' + str(placeholder.shape) + ')'
        print_op.log_and_raise_error(message)

    for idx in range(scaling_value.shape[0]):
        if scaling_value[idx] == cas.DM(0.):
            message = 'encountered at zero-value while trying to set the scaling of ' + var_type + ' variable (' + var_name + '). zero-values are not valid scaling entries'
            print_op.log_and_raise_error(message)

    return scaling_value


def var_si_to_scaled(var_type, var_name, var_si, scaling, check_should_multiply = True):
    
    if check_should_multiply:
        should_multiply, message = should_variable_be_scaled(var_type, var_name, var_si, scaling)
        if message is not None:
            print_op.base_print(message, level='warning')
    else:
        should_multiply = True

    if should_multiply:
        scale = scaling[var_type, var_name]
        scaling_matrix = cas.inv(cas.diag(scale))
        return cas.mtimes(scaling_matrix, var_si)
    else:
        return var_si

def var_scaled_to_si(var_type, var_name, var_scaled, scaling, check_should_multiply = True):

    if check_should_multiply:
        should_multiply, message = should_variable_be_scaled(var_type, var_name, var_scaled, scaling)
        if message is not None:
            print_op.base_print(message, level='warning')
    else:
        should_multiply = True

    if should_multiply:
        scale = scaling[var_type, var_name]
        scaling_matrix = cas.diag(scale)
        return cas.mtimes(scaling_matrix, var_scaled)
    else:
        return var_scaled


def get_set_of_canonical_names_for_V_variables_without_dimensions(V):


    set_of_canonical_names_without_dimensions = set([])
    number_V_entries = V.cat.shape[0]
    for edx in range(number_V_entries):
        canonical = V.getCanonicalIndex(edx)
        set_of_canonical_names_without_dimensions.add(canonical[:-1])
    return set_of_canonical_names_without_dimensions


def si_to_scaled(V_ori, scaling):
    V = copy.deepcopy(V_ori)

    coll_var_name = get_collocation_var_name(V_ori)

    set_of_canonical_names_without_dimensions = get_set_of_canonical_names_for_V_variables_without_dimensions(V)
    for local_canonical in set_of_canonical_names_without_dimensions:

        # with SAM we have some additional variables that might need special treatment
        if local_canonical[0] in ['v_macro_coll','lam_SAM']:
            continue

        if local_canonical[0] != 'phi':

            if len(local_canonical) == 2:
                var_type = local_canonical[0]
                var_name = local_canonical[1]
                var_si = V[var_type, var_name]
                V[var_type, var_name] = var_si_to_scaled(var_type, var_name, var_si, scaling, check_should_multiply=False)

            elif len(local_canonical) == 3:
                var_type = local_canonical[0]

                # with SAM we have some additional (state) variables that might need special treatment
                var_type_to_store = var_type # the type(name) of the variable in the dictionary
                var_type_scaling = var_type # the name that is used to look up the scaling
                if var_type_to_store in ['x_macro', 'x_macro_coll', 'x_micro_minus', 'x_micro_plus']:
                    var_type_scaling = 'x' # for SAM: the scaling is stored under 'x'

                kdx = local_canonical[1]
                var_name = local_canonical[2]
                var_si = V[var_type_to_store, kdx, var_name]
                V[var_type_to_store, kdx, var_name] = var_si_to_scaled(var_type_scaling, var_name, var_si, scaling, check_should_multiply=False)

            elif (len(local_canonical) == 5) and (local_canonical[0] == coll_var_name):
                kdx = local_canonical[1]
                ddx = local_canonical[2]
                var_type = local_canonical[3]
                var_name = local_canonical[4]
                var_si = V[coll_var_name, kdx, ddx, var_type, var_name]
                V[coll_var_name, kdx, ddx, var_type, var_name] = var_si_to_scaled(var_type, var_name, var_si, scaling, check_should_multiply=False)
            else:
                message = 'unexpected variable found at canonical index: ' + str(local_canonical) + ' while scaling variables from si'
                print_op.log_and_raise_error(message)

    return V


def scaled_to_si(V_ori, scaling):
    V = copy.deepcopy(V_ori)
    coll_var_name = get_collocation_var_name(V_ori)

    set_of_canonical_names_without_dimensions = get_set_of_canonical_names_for_V_variables_without_dimensions(V)
    for local_canonical in set_of_canonical_names_without_dimensions:

        # with SAM we have some additional variables that might need special treatment
        if local_canonical[0] in ['v_macro_coll','lam_SAM']:
            continue

        if local_canonical[0] != 'phi':
            if len(local_canonical) == 2:
                var_type = local_canonical[0]
                var_name = local_canonical[1]
                var_si = V[var_type, var_name]
                V[var_type, var_name] = var_scaled_to_si(var_type, var_name, var_si, scaling, check_should_multiply=False)

            elif len(local_canonical) == 3:
                var_type = local_canonical[0]

                # with SAM we have some additional variables that might need special treatment
                var_type_to_store = var_type # the type(name) of the variable in the dictionary
                var_type_scaling = var_type # the name that is used to look up the scaling
                if var_type_to_store in ['x_macro', 'x_macro_coll', 'x_micro_minus', 'x_micro_plus']:
                    var_type_scaling = 'x' # for SAM: the scaling is stored under 'x'

                kdx = local_canonical[1]
                var_name = local_canonical[2]
                var_si = V[var_type_to_store, kdx, var_name]
                V[var_type_to_store, kdx, var_name] = var_scaled_to_si(var_type_scaling, var_name, var_si, scaling, check_should_multiply=False)

            elif (len(local_canonical) == 5) and (local_canonical[0] == coll_var_name):
                kdx = local_canonical[1]
                ddx = local_canonical[2]
                var_type = local_canonical[3]
                var_name = local_canonical[4]
                var_si = V[coll_var_name, kdx, ddx, var_type, var_name]
                V[coll_var_name, kdx, ddx, var_type, var_name] = var_scaled_to_si(var_type, var_name, var_si, scaling, check_should_multiply=False)
            else:
                message = 'unexpected variable found at canonical index: ' + str(local_canonical) + ' while un-scaling variables to si'
                print_op.log_and_raise_error(message)

    return V


def coll_slice_to_vec(coll_slice):

    coll_list = []

    for i in range(len(coll_slice)):
        coll_list.append(cas.vertcat(*coll_slice[i]))

    coll_vec = cas.vertcat(*coll_list)

    return coll_vec

def coll_slice_to_norm(coll_slice):

    slice_norm = []
    for i in range(len(coll_slice)):
        slice_norm.append(np.linalg.norm(coll_slice[i]))

    return slice_norm

def interval_slice_to_vec(interval_slice):

    interval_vec = cas.vertcat(*interval_slice)

    return interval_vec

def get_variable_type(container, name):

    if isinstance(container, dict):
        variables_dict = container

    elif hasattr(container, 'type') and container.type == 'Model' and hasattr(container, 'variables_dict'):
        variables_dict = container.variables_dict

    else:
        variables_dict = {}
        for var_type in container.keys():
            variables_dict[var_type] = {}
            for var_name in subkeys(container, var_type):
                variables_dict[var_type][var_name] = None

    for variable_type in set(variables_dict.keys()) - set(['xdot']):
        if name in list(variables_dict[variable_type].keys()):
            return variable_type

    if name in list(variables_dict['xdot'].keys()):
        return 'xdot'

    message = 'variable ' + name + ' not found in variables dictionary'
    print_op.log_and_raise_error(message)

    return None



def convert_return_status_string_to_number(return_string):

    numeric_dictionary = get_return_status_dictionary()

    for num in list(numeric_dictionary.keys()):
        if numeric_dictionary[num] == return_string:
            return num

    return 17

def convert_return_status_number_to_string(return_number):

    numeric_dictionary = get_return_status_dictionary()

    if return_number in list(numeric_dictionary.keys()):
        return numeric_dictionary[return_number]

    return 'Other'

def get_return_status_dictionary():
    # ipopt return status described here:
    # https: // www.coin - or.org / Ipopt / documentation / node36.html

    ret_stats = {}
    ret_stats[1] = 'Solve_Succeeded'
    ret_stats[2] = 'Solved_To_Acceptable_Level'
    ret_stats[3] = 'Feasible_Point_Found'
    ret_stats[4] = 'Infeasible_Problem_Detected'
    ret_stats[5] = 'Search_Direction_Becomes_Too_Small'
    ret_stats[6] = 'Diverging_Iterates'
    ret_stats[7] = 'User_Requested_Stop'
    ret_stats[8] = 'Maximum_Iterations_Exceeded'
    ret_stats[9] = 'Maximum_CpuTime_Exceeded'
    ret_stats[10] = 'Restoration_Failed'
    ret_stats[11] = 'Error_In_Step_Computation'
    ret_stats[12] = 'Invalid_Option'
    ret_stats[13] = 'Not_Enough_Degrees_Of_Freedom'
    ret_stats[14] = 'Invalid_Problem_Definition'
    ret_stats[15] = 'NonIpopt_Exception_Thrown'
    ret_stats[16] = 'Insufficient_Memory'
    ret_stats[17] = 'IPOPT_DOA'

    return ret_stats

def get_V_index(canonical):

    coll_var_name = 'coll_var'
    var_is_coll_var = (canonical[0] == coll_var_name)

    length = len(canonical)

    if var_is_coll_var:
        # coll_var, kdx, ddx, type, name

        var_type = canonical[3]
        kdx = canonical[1]
        ddx = canonical[2]
        name = canonical[4]
        dim = None

    else:
        var_type = canonical[0]
        dim = None
        kdx = None
        ddx = None

        if length == 4:
            kdx = canonical[1]
            ddx = canonical[2]
            name = canonical[3]

        elif length == 3:
            kdx = canonical[1]
            name = canonical[2]

        elif length == 2:
            name = canonical[1]
        elif length == 1:
            name = canonical[0]
        else:
            message = 'unexpected (distinct) canonical_index handing'
            print_op.log_and_raise_error(message)

    return [var_is_coll_var, var_type, kdx, ddx, name, dim]


def construct_Xdot_struct(nlp_options, variables_dict):
    ''' Construct a symbolic structure for the
        discretized state derivatives.

    @param nlp_options - discretization options
    @param model - awebox model
    @return Vdot - discretized state derivatives sym. struct.
    '''

    # extract information
    nk = nlp_options['n_k']
    x = variables_dict['x']
    z = variables_dict['z']

    # derivatives at interval nodes
    entry_tuple = (cas.entry('x', repeat=[nk], struct=x),)

    # add derivatives on collocation nodes
    if nlp_options['discretization'] == 'direct_collocation':
        d = nlp_options['collocation']['d']
        entry_tuple += (cas.entry('coll_x', repeat=[nk, d], struct=x),)
        entry_tuple += (cas.entry('coll_z', repeat=[nk, d], struct=z),)

    # make new symbolic structure
    Xdot = cas.struct_symMX([entry_tuple])

    return Xdot


##
#  @brief Method to recursively generate a casadi structure out of a nested dict.
#  @param v A (possibly nested) dictionary
#  @return subdict_struct Casadi struct_symSX with same structure as v.
def generate_nested_dict_struct(v):

    # empty entry list
    entry_list = []

    # iterate over all dict values
    for k1, v1 in v.items():

        if isinstance(v1, dict):

            # if value is a dict, recursively generate subdict struct
            substruct = generate_nested_dict_struct(v1)
            entry_list.append(cas.entry(k1,struct= substruct))

        else:
            if isinstance(v1,float) or isinstance(v1,int):
                shape = (1,1)
            else:
                shape = v1.shape
            # append value to entry list
            entry_list.append(cas.entry(k1, shape= shape))

    # make overall structure
    subdict_struct = cas.struct_symSX(entry_list)

    return subdict_struct

def get_from_dict(data_dict, mapList):
    return reduce(operator.getitem, mapList, data_dict)

def set_in_dict(data_dict, mapList, value):
    get_from_dict(data_dict, mapList[:-1])[mapList[-1]] = value
    return None

def generate_nested_dict_keys(d):

    keys_list = []

    for k, v in d.items():
        if isinstance(v,dict) and k != 'parent_map':
            subkeys = generate_nested_dict_keys(v)
            for i in range(len(subkeys)):
                keys_list += [[k]+subkeys[i]]
        else:
            keys_list += [[k]]

    return keys_list

def initialize_nested_dict(d,keys):

    if len(keys) == 1:
        d[keys[0]] = None
    else:
        if keys[0] in list(d.keys()):
            d[keys[0]] = initialize_nested_dict(d[keys[0]],keys[1:])
        else:
            d[keys[0]] = initialize_nested_dict({},keys[1:])

    return d

def setup_warmstart_data(nlp, warmstart_solution_dict):

    options_in_keys = 'options' in warmstart_solution_dict.keys()
    if options_in_keys:
        nlp_discretization = warmstart_solution_dict['options']['nlp']['discretization']

    if options_in_keys and not (nlp.discretization == nlp_discretization):

        if nlp.discretization == 'multiple_shooting':

            # number of shooting intervals
            n_k = nlp.n_k

            # initialize and extract
            V_init_proposed = nlp.V(0.0)
            V_coll = warmstart_solution_dict['V_opt']
            Xdot_coll = warmstart_solution_dict['Xdot_opt']

            lam_x_proposed = nlp.V_bounds['ub'](0.0)
            lam_x_coll = V_coll(warmstart_solution_dict['opt_arg']['lam_x0'])

            lam_g_proposed = nlp.g(0.0)
            lam_g_coll = warmstart_solution_dict['g_opt'](warmstart_solution_dict['opt_arg']['lam_g0'])
            g_coll = warmstart_solution_dict['g_opt']

            # initialize regular variables
            for var_type in set(['x', 'theta', 'phi', 'xi', 'z', 'xdot']):
                V_init_proposed[var_type] = V_coll[var_type]
                lam_x_proposed[var_type] = lam_x_coll[var_type]

            if 'u' in list(V_coll.keys()):
                V_init_proposed['u'] = V_coll['u']
                lam_x_proposed['u'] = lam_x_coll['u']
            else:
                for i in range(n_k):
                    # note: this does not give the actual mean, implement with quadrature weights instead
                    V_init_proposed['u', i] = np.mean(cas.horzcat(*V_coll['coll_var', i, :, 'u']))
                    lam_x_proposed['u', i] = np.mean(cas.horzcat(*lam_x_coll['coll_var', i, :, 'u']))

            # initialize path constraint multipliers
            for i in range(n_k):
                lam_g_proposed['path', i] = lam_g_coll['path', i]

            # initialize periodicity multipliers
            if 'periodic' in list(lam_g_coll.keys()) and 'periodic' in list(lam_g_proposed.keys()):
                lam_g_proposed['periodic'] = lam_g_coll['periodic']

            # initialize continuity multipliers
            lam_g_proposed['continuity'] = lam_g_coll['continuity']

        else:
            raise ValueError('Warmstart from multiple shooting to collocation not supported')

    else:

        V_init_proposed = warmstart_solution_dict['V_opt']
        if 'lam_x0' in warmstart_solution_dict['opt_arg'].keys():
            lam_x_proposed  = warmstart_solution_dict['opt_arg']['lam_x0']
            lam_g_proposed  = warmstart_solution_dict['opt_arg']['lam_g0']
        else:
            lam_x_proposed = np.zeros(nlp.V_bounds['ub'].shape)
            lam_g_proposed = np.zeros(nlp.g.shape)

    V_shape_matches = (V_init_proposed.cat.shape == nlp.V.cat.shape)
    if not V_shape_matches:
        raise ValueError('Variables of specified warmstart do not correspond to NLP requirements.')

    lam_x_shape_matches = (lam_x_proposed.shape == nlp.V_bounds['ub'].shape)
    if not lam_x_shape_matches:
        raise ValueError('Variable bound multipliers of specified warmstart do not correspond to NLP requirements.')

    lam_g_shape_matches = (lam_g_proposed.shape == nlp.g.shape)
    if not lam_g_shape_matches:
        raise ValueError('Constraint multipliers of specified warmstart do not correspond to NLP requirements.')

    return [V_init_proposed, lam_x_proposed, lam_g_proposed]


def dissolve_top_layer_of_struct(struct):

    dissolved_struct = {}
    for variable_type in list(struct.keys()):
        for variable in subkeys(struct, variable_type):
            dissolved_struct[variable] = struct[variable_type,variable]

    return dissolved_struct

def recursive_strip(d):
    if type(d) != dict:
        try:
            d = d(0)
        except:
            d = None
        return d
    else:
        for key in list(d.keys()):
            d[key] = recursive_strip(d[key])

def strip_of_contents(d):

    stripped_d = copy.deepcopy(d)
    recursive_strip(stripped_d)
    
    return stripped_d


def evaluate_cost_dict(cost_fun, V_opt, p_fix_num):
    cost = {}
    for name in list(cost_fun.keys()):
        if 'problem' not in name and 'objective' not in name:
            cost[name[:-4]] = cost_fun[name](V_opt, p_fix_num)

    return cost


def split_name_and_integral_order(name):

    var_name = name
    integral_order = 0

    while var_name[0] == 'd':
        integral_order += 1
        var_name = var_name[1:]

    return var_name, integral_order


def split_name_and_node_identifier(name):

    var_name = name
    kiteparent = ''

    while var_name[-1].isdigit():

        kiteparent = var_name[-1] + kiteparent
        var_name = var_name[:-1]

    return var_name, kiteparent


def split_kite_and_parent(kiteparent, architecture):

    for idx in range(1, len(kiteparent)):

        kite_try = int(kiteparent[:idx])
        parent_try = int(kiteparent[idx:])

        parent_of_kite_try = int(architecture.parent_map[kite_try])

        if parent_of_kite_try == parent_try:
            return kite_try, parent_try

    return None, None

def generate_variable_struct(variable_list):

    structs = {}
    for name in list(variable_list.keys()):
        structs[name] = cas.struct_symSX([cas.entry(variable_list[name][i][0], shape=variable_list[name][i][1])
                        for i in range(len(variable_list[name]))])

    variable_struct = cas.struct_symSX([cas.entry(name, struct=structs[name])
                        for name in list(variable_list.keys())])

    return variable_struct, structs


def find_output_idx(outputs, output_type, output_name, output_dim = 0):

    if hasattr(outputs, 'labels'):
        search_label = '[' + output_type + ',' + output_name + ',' + str(output_dim) + ']'
        if search_label in outputs.labels():
            return outputs.labels().index(search_label)
        else:
            message = 'Search output ' + search_label + ' could not be found in passed outputs, so no index value could be returned'
            print_op.log_and_raise_error(message)

    elif hasattr(outputs, 'keys'):
        odx = 0
        for found_type in outputs.keys():
            for found_name in outputs[found_type].keys():
                local_entry = outputs[found_type][found_name]
                local_size = len(local_entry)

                for found_dim in range(local_size):
                    if (output_type == found_type) and (output_name == found_name) and (output_dim == found_dim):
                        return odx
                    else:
                        odx += 1
    return None


def sanity_check_find_output_idx(outputs):

    known_odx = 0
    if hasattr(outputs, 'getCanonicalIndex'):
        output_type, output_name, output_dim = outputs.getCanonicalIndex(known_odx)
    else:
        output_type = [ot for ot in outputs.keys()][0]
        output_name = [on for on in outputs[output_type].keys()][0]
        output_dim = 0

    start_odx = find_output_idx(outputs, output_type, output_name, output_dim)
    if not (known_odx == start_odx):
        message = 'struct_op find_output_idx does not find the correct index at the start of the outputs'
        print_op.log_and_raise_error(message)


    # at end
    if hasattr(outputs, 'getCanonicalIndex'):
        known_odx = outputs.shape[0] - 1
        output_type, output_name, output_dim = outputs.getCanonicalIndex(known_odx)
    else:

        dimensions_dict = {}
        for output_type in outputs.keys():
            if output_type not in dimensions_dict.keys():
                dimensions_dict[output_type] = {}

            for output_name in outputs[output_type].keys():
                local_out = outputs[output_type][output_name]
                dimensions_dict[output_type][output_name] = len(local_out)

        known_odx = -1
        for output_type in outputs.keys():
            for output_name in outputs[output_type].keys():
                known_odx += dimensions_dict[output_type][output_name]

        output_type = [ot for ot in outputs.keys()][-1]
        output_name = [on for on in outputs[output_type].keys()][-1]
        output_dim = dimensions_dict[output_type][output_name] - 1

    end_odx = find_output_idx(outputs, output_type, output_name, output_dim)
    if not (known_odx == end_odx):
        message = 'struct_op find_output_idx does not find the correct index at the end of the outputs'
        print_op.log_and_raise_error(message)

    return None

def get_variable_from_model_or_reconstruction(variables, var_type, name):

    has_labels = hasattr(variables, 'labels')
    combined_name_in_labels = has_labels and ('[' + var_type + ',' + name + ',0]' in variables.labels())
    if combined_name_in_labels:
        return variables[var_type, name]

    has_keys = hasattr(variables, 'keys')
    var_type_in_keys = has_keys and (var_type in variables.keys())
    has_subkeys = var_type_in_keys and hasattr(variables[var_type], 'keys')
    var_name_in_subkeys = has_subkeys and (name in variables[var_type].keys())
    if var_name_in_subkeys:
        return variables[var_type][name]

    message = 'variable ' + name + ' is not in expected position (' + var_type + ') wrt variables.'
    print_op.log_and_raise_error(message)
    return None

def interpolate_solution(local_options, time_grids, variables_dict, V_opt, P_num, model_parameters, model_scaling, outputs_fun, outputs_dict, integral_output_names, integral_outputs_opt, Collocation=None, timegrid_label='ip', n_points=None, interpolate_time_grid = True):
    '''
    Postprocess tracking reference data from V-structure to (interpolated) data vectors
        with associated time grid
    :param nlp_options: the options of the nlp. notice that this should contain information about the nlp discretization, as well as the number of points desired for interpolation
    :param V_opt: the solution that we want to interpolate. remember, that for 'si' output, the numeric V input here, should also be in 'si' units. (for interpolating over the reference V, input optimization.V_ref here)
    :param time_grids: the time grids of the discretized problem. (for interpolating over the reference V, input optimization.time_grids['ref'] here)
    :return: a dictionary with entries corresponding to the interpolated variables
    '''

    nlp_discretization = local_options['discretization']
    collocation_scheme = local_options['collocation']['scheme']
    control_parametrization = local_options['collocation']['u_param']

    if n_points is None:
        n_points = local_options['interpolation']['n_points']

    if Collocation is not None:
        collocation_interpolator = Collocation.build_interpolator(local_options, V_opt)
        integral_collocation_interpolator = Collocation.build_interpolator(local_options, V_opt, integral_outputs=integral_outputs_opt)
        if interpolate_time_grid:
            time_grid_interpolated = build_time_grid_for_interpolation(time_grids, n_points)
        else:
            time_grid_interpolated = time_grids['ip']
    else:
        control_parametrization = 'zoh'
        collocation_interpolator = None
        integral_collocation_interpolator = None
        time_grid_interpolated = time_grids['x'].full()

    # add states and outputs to plotting dict
    interpolation = {'x': {}, 'xdot': {}, 'u': {}, 'z': {}, 'theta': {}, 'time_grids': {}, 'outputs': {}, 'integral_outputs': {}}

    # time
    time_grids['ip'] = time_grid_interpolated
    interpolation['time_grids'] = time_grids

    # interpolate x, z and u values from V
    V_interpolated, V_vector_series_interpolated =  interpolate_V(
        time_grids, variables_dict, control_parametrization, V_opt,
        collocation_interpolator=collocation_interpolator,
        timegrid_label=timegrid_label)

    interpolation['x'] = V_interpolated['x']
    interpolation['z'] = V_interpolated['z']
    interpolation['u'] = V_interpolated['u']
    interpolation['xdot'] = V_interpolated['xdot']

    # theta values
    for name in list(subkeys(V_opt, 'theta')):
        interpolation['theta'][name] = V_opt['theta', name].full()[0][0]

    # output values
    interpolation['outputs'] = interpolate_outputs(V_vector_series_interpolated, V_opt, P_num, variables_dict, model_parameters, model_scaling, outputs_fun, outputs_dict)

    # integral-output values
    if integral_outputs_opt.shape[0] != 0:
        interpolation['integral_outputs'] = interpolate_integral_outputs(time_grids, integral_output_names,
                                                                        integral_outputs_opt, nlp_discretization,
                                                                        collocation_scheme=collocation_scheme,
                                                                        timegrid_label=timegrid_label,
                                                                        integral_collocation_interpolator=integral_collocation_interpolator)

    return interpolation


def build_time_grid_for_interpolation(time_grids, n_points):
    time_grid_interpolated = np.linspace(float(time_grids['x'][0]), float(time_grids['x'][-1]), n_points)
    return time_grid_interpolated

def interpolate_outputs(V_vector_series_interpolated, V_sol, P_num, variables_dict, model_parameters, model_scaling, outputs_fun, outputs_dict):

    # extra variables time series (SI units)
    x = V_vector_series_interpolated['x']
    u = V_vector_series_interpolated['u']
    z = V_vector_series_interpolated['z']
    xdot = V_vector_series_interpolated['xdot']

    # time series length
    N_ip = x.shape[1]

    # model parameters
    theta = variables_dict['theta'](1.0)
    for theta_var in variables_dict['theta'].keys():
        if theta_var != 't_f':
            theta[theta_var] = V_sol['theta', theta_var]
    theta = cas.repmat(theta.cat, 1, N_ip)

    # construct variables input time series
    variables = cas.vertcat(x, xdot, u, z, theta)

    # scale variables time series to evaluate output function
    variables = cas.mtimes(cas.diag(1./model_scaling), variables)
    parameters = cas.repmat(get_parameters_at_time(V_sol, P_num, model_parameters).cat, 1, N_ip)

    # compute outputs on interpolation grid
    outputs_fun_map = outputs_fun.map(N_ip)
    outputs_series = outputs_fun_map(variables, parameters).full()

    # distribute results in plot_dict
    outputs = {}
    output_counter = 0
    for output_type in outputs_dict.keys():
        outputs[output_type] = {}
        for output_name in outputs_dict[output_type].keys():
            outputs[output_type][output_name] = []
            for dim in range(outputs_dict[output_type][output_name].shape[0]):
                outputs[output_type][output_name] += [outputs_series[output_counter, :].squeeze()]
                output_counter += 1
    return outputs

def interpolate_integral_outputs(time_grids, integral_output_names, integral_outputs_opt, nlp_discretization, collocation_scheme='radau', timegrid_label='ip', integral_collocation_interpolator=None):

    if integral_collocation_interpolator is not None:
        integral_outputs_vector_series = integral_collocation_interpolator(time_grids[timegrid_label], 'int_out').full()
    else:
        integral_outputs_vector_series = cas.horzcat(*integral_outputs_opt['int_out', :]).full()
    integral_outputs_interpolated = {}

    # integral-output values
    integral_outputs_counter = 0
    for name in integral_output_names:
        if name not in list(integral_outputs_interpolated.keys()):
            integral_outputs_interpolated[name] = []

        integral_output_dimension = integral_outputs_opt['int_out', 0, name].shape[0]

        for dim in range(integral_output_dimension):
            integral_outputs_interpolated[name] += [integral_outputs_vector_series[integral_outputs_counter, :].squeeze()]
            integral_outputs_counter += 1

    return integral_outputs_interpolated


def get_original_time_data_for_output_interpolation(time_grids):
    use_collocation = 'coll' in time_grids.keys()
    if use_collocation:
        original_times = get_concatenated_coll_time_grid(time_grids)
    elif 'u' in time_grids.keys():
        original_times = time_grids['u']
    else:
        message = 'cannot find original time series for output interpolation.'
        print_op.log_and_raise_error(message)

    return original_times



def get_concatenated_coll_time_grid(time_grids):
    original = time_grids['coll']
    n_k, collocation_d = original.shape

    reshaped = []
    for kdx in range(n_k):
        for ddx in range(collocation_d):
            reshaped = cas.vertcat(reshaped, original[kdx, ddx])
    return reshaped

def get_output_series_with_duplicates_removed(original_times, original_series, collocation_d):

    series_without_duplicates = []
    for idx in range(original_series.shape[0]):
        if (np.mod(idx, collocation_d + 1) > 0):
            series_without_duplicates = cas.vertcat(series_without_duplicates, original_series[idx])

    if not (original_times.shape == series_without_duplicates.shape):
        message = 'something went wrong when removing duplicate entries from zoh outputs, prior to interpolation'
        message += ": series does not have correct number of entries"
        print_op.log_and_raise_error(message)

    return series_without_duplicates

def interpolate_V(time_grids, variables_dict, control_parametrization, V,  collocation_interpolator=None, timegrid_label='ip'):

    V_interpolated = {'x': {}, 'z': {}, 'u': {}, 'xdot': {}}
    V_vector_series_interpolated = {}

    for var_type in V_interpolated.keys():

        if collocation_interpolator is not None:
            # interpolate system variables in vector form
            if var_type in ['x', 'z', 'xdot']:
                V_vector_series = collocation_interpolator(time_grids[timegrid_label], var_type).full()
            elif var_type in ['u']:
                if control_parametrization == 'poly':
                    V_vector_series = collocation_interpolator(time_grids[timegrid_label], var_type).full()
                elif control_parametrization == 'zoh':
                    controls = V['u', :]
                    V_vector_series = sample_and_hold_controls(time_grids, controls, timegrid_label=timegrid_label)
        else:
            V_vector_series = cas.horzcat(*V[var_type, :]).full()
            if var_type in ['u', 'z', 'xdot']:
                V_vector_series = cas.horzcat(V_vector_series, V_vector_series[:, 0]).full()

        # distribute results into results dictionary
        variable_names = variables_dict[var_type].keys()
        var_counter = 0
        for name in variable_names:

            V_interpolated[var_type][name] = []
            variable_dimension = variables_dict[var_type][name].shape[0]

            for dim in range(variable_dimension):
                V_interpolated[var_type][name] += [V_vector_series[var_counter,:].squeeze()]
                var_counter += 1

        # save vector series for output computation
        V_vector_series_interpolated[var_type] = V_vector_series

    return V_interpolated, V_vector_series_interpolated

def sample_and_hold_controls(time_grids, controls, timegrid_label='ip'):

    tgrid_u = time_grids['u']
    tgrid_ip = time_grids[timegrid_label]
    values_ip = np.zeros((controls[0].shape[0], len(tgrid_ip)))
    for idx in range(len(tgrid_ip)):
        for jdx in range(tgrid_u.shape[0] - 1):
            if tgrid_u[jdx] < tgrid_ip[idx] and tgrid_ip[idx] < tgrid_u[jdx + 1]:
                values_ip[:, idx] = controls[jdx].full().squeeze()
                break
        if tgrid_u[-1] < tgrid_ip[idx]:
            values_ip[:, idx] = controls[-1].full().squeeze()

    values = values_ip

    return values

def test():
    # todo
    awelogger.logger.warning('no tests currently defined for struct_operations')
    return None


def eval_time_grids_SAM(nlp_options: dict, tf_opt: ca.DM) -> Dict[str, np.ndarray]:
    """
    Calculate the time grids for the SAM discretization.
    This makes use of a function that translates the original nlp time to the SAM time.

    Returns a dictionary with the time grids for the states ('x'), controls ('u'), collocation nodes ('coll') and the
    time grid for the states and collocation nodes ('x_coll').

    :param nlp_options: the nlp options, e.g. trial.options['nlp']
    :param tf_opt: the optimal time-scaling parameters, e.g. Vopt['theta', 't_f']
    :return: a dictionary of numpy arrays for the timegrids with keys ('x','u', 'coll', 'x_coll')
    """
    assert nlp_options['SAM']['use']
    assert nlp_options['discretization'] == 'direct_collocation'

    timegrid_AWEbox_f = construct_time_grids(nlp_options)
    timegrid_AWEbox_eval = {key: timegrid_AWEbox_f[key](tf_opt).full().flatten() for key in timegrid_AWEbox_f.keys()}
    timegrid_SAM = {}

    # function to go from AWEbox time to SAM time
    f_scale = originalTimeToSAMTime(nlp_options, tf_opt)

    # modify a bit for better post-processing: for x_coll timegrid
    # check if any values of t are close to any values in ts_cumsum,
    # this happens if the time points are equal, but are supposed to be in different SAM regions,
    # for example when radau collocation is used

    # find  paris of indices in time_grid_ip_original that are close to each other
    close_indices = np.where(np.isclose(np.diff(timegrid_AWEbox_eval['x_coll']), 0.0))[0]
    for first_index in close_indices:
        timegrid_AWEbox_eval['x_coll'][first_index] -= 1E-6
        timegrid_AWEbox_eval['x_coll'][first_index + 1] += 1E-6

    for key in timegrid_AWEbox_f:
        timegrid_SAM[key] = f_scale.map(timegrid_AWEbox_eval[key].size)(timegrid_AWEbox_eval[key]).full().flatten()

    return timegrid_SAM
