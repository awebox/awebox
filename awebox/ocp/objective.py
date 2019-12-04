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
objective code of the awebox
constructs an objective function from the various fictitious costs.
python-3.5 / casadi-3.4.5
- refactored from awebox code (elena malz, chalmers; jochem de schutter, alu-fr; rachel leuthold, alu-fr), 2018
- edited: rachel leuthold, jochem de schutter alu-fr 2018
'''
import casadi.tools as cas
from . import collocation
import awebox.tools.struct_operations as struct_op

def get_local_tracking_function(variables, P):
    # initialization tracking

    tracking = 0.

    for name in set(struct_op.subkeys(variables, 'xd')) - set('e'):
        difference = variables['xd', name] - P['p', 'ref', name]
        tracking += P['p', 'weights', name][0] * cas.mtimes(difference.T, difference)

    for name in set(struct_op.subkeys(variables, 'xa')):
        difference = variables['xa', name] - P['p', 'ref', name]
        tracking += P['p', 'weights', name][0] * cas.mtimes(difference.T, difference)

    for name in set(struct_op.subkeys(variables, 'xl')):
        difference = variables['xl', name] - P['p', 'ref', name]
        tracking += P['p', 'weights', name][0] * cas.mtimes(difference.T, difference)

    tracking_fun = cas.Function('tracking_fun', [variables, P], [tracking])

    return tracking_fun

def find_int_weights(nlp_numerics_options):

    nk = nlp_numerics_options['n_k']
    d = nlp_numerics_options['collocation']['d']
    scheme = nlp_numerics_options['collocation']['scheme']
    Collocation = collocation.Collocation(nk,d,scheme)
    int_weights = Collocation.quad_weights

    return int_weights

def find_tracking(nlp_numerics_options, V, P, variables):

    nk = nlp_numerics_options['n_k']
    direct_collocation, multiple_shooting, d, scheme, int_weights = extract_discretization_info(nlp_numerics_options)

    tracking = 0.

    for kdx in range(nk):

        if multiple_shooting:

            for name in set(struct_op.subkeys(variables, 'xd')) - set('e'):

                difference = V['xd', kdx, name] - P['p', 'ref', 'xd', kdx, name]
                tracking += P['p', 'weights', 'xd', name][0] * cas.mtimes(difference.T, difference)

            for name in set(struct_op.subkeys(variables, 'xa')):
                difference = V['xa', kdx, name] - P['p', 'ref', 'xa', kdx, name]
                tracking += P['p', 'weights', 'xa', name][0] * cas.mtimes(difference.T, difference)

            if 'xl' in list(variables.keys()):
                for name in set(struct_op.subkeys(variables, 'xl')):
                    difference = V['xl', kdx, name] - P['p', 'ref', 'xl', kdx, name]
                    tracking += P['p', 'weights', 'xl', name][0] * cas.mtimes(difference.T, difference)

        elif direct_collocation:

            for jdx in range(d):

                for name in set(struct_op.subkeys(variables, 'xd')) - set('e'):

                    difference = V['coll_var',kdx, jdx, 'xd', name] - P['p', 'ref', 'coll_var', kdx, jdx, 'xd', name]
                    tracking += int_weights[jdx]*P['p', 'weights', 'xd', name][0] * cas.mtimes(difference.T, difference)

                for name in set(struct_op.subkeys(variables, 'xa')):
                    difference = V['coll_var', kdx, jdx, 'xa', name] - P['p', 'ref', 'coll_var', kdx, jdx, 'xa', name]
                    tracking += int_weights[jdx]*P['p', 'weights', 'xa', name][0] * cas.mtimes(difference.T, difference)

                if 'xl' in list(variables.keys()):
                    for name in set(struct_op.subkeys(variables, 'xl')):
                        difference = V['coll_var', kdx, jdx, 'xl', name] - P['p', 'ref', 'coll_var', kdx, jdx, 'xl', name]
                        tracking += int_weights[jdx]*P['p', 'weights', 'xl', name][0] * cas.mtimes(difference.T, difference)


    return tracking

def find_regularisation(nlp_numerics_options, V, P, variables):
    nk = nlp_numerics_options['n_k']
    direct_collocation, multiple_shooting, d, scheme, int_weights = extract_discretization_info(nlp_numerics_options)


    # check control parameterization
    if (direct_collocation and nlp_numerics_options['collocation']['u_param'] == 'poly'):
        u_param = 'poly'
    else:
        u_param = 'zoh'

    regularisation = 0.

    if u_param == 'zoh':
        for kdx in range(nk):
            for name in set(struct_op.subkeys(variables, 'u')) - set(['ddl_t']):
                if not 'fict' in name:
                    difference = V['u', kdx, name] - P['p', 'ref', 'u', kdx, name]
                    regularisation += P['p', 'weights', 'u', name][0] * cas.mtimes(difference.T, difference)
    elif u_param == 'poly':
        for kdx in range(nk):
            for jdx in range(d):
                for name in set(struct_op.subkeys(variables, 'u')) - set(['ddl_t']):
                    if not 'fict' in name:
                        difference = V['coll_var', kdx, jdx, 'u', name] - P['p', 'ref', 'coll_var', kdx, jdx, 'u', name]
                        regularisation += int_weights[jdx]*P['p', 'weights', 'u', name][0] * cas.mtimes(difference.T, difference)

    return regularisation

def find_ddq_regularisation(nlp_numerics_options, V, P, xdot, outputs):
    nk = nlp_numerics_options['n_k']
    direct_collocation, multiple_shooting, d, scheme, int_weights = extract_discretization_info(nlp_numerics_options)

    ddq_regularisation = 0.

    for kdx in range(nk):

        if multiple_shooting:

            for name in set(struct_op.subkeys(outputs, 'xddot_from_var')):
                ddq_regularisation += cas.mtimes(xdot['xd',kdx,name[1:]].T, xdot['xd',kdx,name[1:]])

        elif direct_collocation:

            for jdx in range(d):
                for name in set(struct_op.subkeys(outputs, 'xddot_from_var')):
                    if 'ddq' in name:
                        ddq_regularisation += int_weights[jdx]*cas.mtimes(xdot['coll_xd',kdx,jdx,name[1:]].T, xdot['coll_xd',kdx,jdx,name[1:]])

    return ddq_regularisation

def find_fictitious(nlp_numerics_options, V, P, variables):
    nk = nlp_numerics_options['n_k']
    direct_collocation, multiple_shooting, d, scheme, int_weights = extract_discretization_info(nlp_numerics_options)

    # check control parameterization
    if (direct_collocation and nlp_numerics_options['collocation']['u_param'] == 'poly'):
        u_param = 'poly'
    else:
        u_param = 'zoh'

    fictitious = 0.

    if u_param == 'zoh':
        for kdx in range(nk):
            for name in set(struct_op.subkeys(variables, 'u')):
                if 'fict' in name:
                    difference = V['u', kdx, name] - P['p', 'ref', 'u', kdx, name]
                    fictitious += P['p', 'weights', 'u', name][0] * cas.mtimes(difference.T, difference)
    elif u_param == 'poly':
        for kdx in range(nk):
            for jdx in range(d):
                for name in set(struct_op.subkeys(variables, 'u')):
                    if 'fict' in name:
                        difference = V['coll_var', kdx, jdx, 'u', name] - P['p', 'ref', 'coll_var', kdx, jdx, 'u', name]
                        fictitious += int_weights[jdx]*P['p', 'weights', 'u', name][0] * cas.mtimes(difference.T, difference)

    return fictitious

def find_time_period(nlp_numerics_options, V):
    nk = nlp_numerics_options['n_k']

    use_phase_fix = nlp_numerics_options['phase_fix']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']

    if use_phase_fix:
        time_period_zeroth = V['theta', 't_f',0] * round(nk * phase_fix_reel_out)
        time_period_first = V['theta', 't_f',1] * (nk - round(nk * phase_fix_reel_out))

        # average over collocation nodes
        time_period = (time_period_zeroth + time_period_first) / nk
    else:
        time_period = V['theta', 't_f']

    return time_period

def find_time_cost(nlp_numerics_options, V, P):

    time_period = find_time_period(nlp_numerics_options, V)
    tf_init = find_time_period(nlp_numerics_options, P.prefix['p', 'ref'])

    time_cost = P['cost', 't_f'] * (time_period - tf_init)*(time_period - tf_init)

    return time_cost

def find_tracking_cost(nlp_numerics_options, V, P, variables):
    # tracking of initial guess

    tracking = find_tracking(nlp_numerics_options, V, P, variables)
    tracking_cost = P['cost', 'tracking'] * tracking
    tracking_cost /= nlp_numerics_options['cost']['normalization']['tracking']

    return tracking_cost

def find_regularisation_cost(nlp_numerics_options, V, P, variables):
    # regularisation of inputs

    regularisation = find_regularisation(nlp_numerics_options, V, P, variables)
    regularisation_cost = P['cost', 'regularisation'] * regularisation
    regularisation_cost /= nlp_numerics_options['cost']['normalization']['regularisation']

    return regularisation_cost

def find_ddq_regularisation_cost(nlp_numerics_options, V, P, xdot, outputs):
    # regularisation of ddq

    ddq_regularisation = find_ddq_regularisation(nlp_numerics_options, V, P, xdot, outputs)
    ddq_regularisation_cost = P['cost', 'ddq_regularisation'] * ddq_regularisation
    ddq_regularisation_cost /= nlp_numerics_options['cost']['normalization']['ddq_regularisation']

    return ddq_regularisation_cost

def find_fictitious_cost(nlp_numerics_options, V, P, variables):
    # the penalization/regularization of the fictitious forces

    fictitious = find_fictitious(nlp_numerics_options, V, P, variables)
    fictitious_cost = P['cost', 'fictitious'] * fictitious
    fictitious_cost /= nlp_numerics_options['cost']['normalization']['fictitious']

    return fictitious_cost

def find_gamma_cost(V, P):
    # homotopy parameter for the fictitious forces

    gamma_cost = P['cost', 'gamma'] * V['phi', 'gamma']

    return gamma_cost

def find_psi_cost(V, P):
    # homotopy parameter for the power vs. tracking step

    psi_cost = P['cost', 'psi'] * V['phi', 'psi']

    return psi_cost

def find_iota_cost(V, P):
    # homotopy parameter for the induction step

    iota_cost = P['cost', 'iota'] * V['phi', 'iota']

    return iota_cost

def find_tau_cost(V, P):
    # homotopy parameter for the induction step

    tau_cost = P['cost', 'tau'] * V['phi', 'tau']

    return tau_cost

def find_eta_cost(V, P):
    # homotopy parameter for the landing step

    eta_cost = P['cost', 'eta'] * V['phi', 'eta']

    return eta_cost

def find_nu_cost(V, P):
    # homotopy parameter for the compromised emergency landing step

    nu_cost = P['cost', 'nu'] * V['phi', 'nu']

    return nu_cost

def find_upsilon_cost(V, P):
    # homotopy parameter for the transition step

    upsilon_cost = P['cost', 'upsilon'] * V['phi', 'upsilon']

    return upsilon_cost

def find_power_cost(nlp_numerics_options, V, P, Integral_outputs):

    # maximization term for average power
    time_period = find_time_period(nlp_numerics_options, V)

    if not nlp_numerics_options['cost']['output_quadrature']:
        average_power = V['xd', -1, 'e'] / time_period
    else:
        average_power = Integral_outputs['int_out',-1,'e'] / time_period

    power_cost = P['cost', 'power'] * (-1.) * average_power

    return power_cost

def find_theta_regularisation_cost(V, P):

    difference = V['theta'] - P['p', 'ref', 'theta']
    theta_regularisation_cost = P['cost', 'theta'] * cas.mtimes(difference.T, difference)

    return theta_regularisation_cost

def find_transition_problem_cost(V, P, nlp_numerics_options, xdot, outputs, variables):

    ddq_regularisation = find_ddq_regularisation(nlp_numerics_options, V, P, xdot, outputs)
    ddq_regularisation /= nlp_numerics_options['cost']['normalization']['ddq_regularisation']
    regularisation = find_regularisation(nlp_numerics_options, V, P, variables)
    regularisation /= nlp_numerics_options['cost']['normalization']['regularisation']


    transition_cost = ddq_regularisation + regularisation
    transition_cost = P['cost','transition']*transition_cost

    return transition_cost

def find_nominal_landing_cost(V, P, variables):

    q_end = {}
    dq_end = {}
    for name in struct_op.subkeys(variables, 'xd'):
        if name[0] == 'q':
            q_end[name] = V['xd',-1,name]
        elif name[:2] == 'dq':
            dq_end[name] = V['xd',-1,name]
    velocity_end = 0.0
    position_end = 0.0
    for position in list(q_end.keys()):
        position_end += cas.mtimes(q_end[position].T,q_end[position])
    position_end *= 1./len(list(q_end.keys()))
    for velocity in list(dq_end.keys()):
        velocity_end += cas.mtimes(dq_end[velocity].T,dq_end[velocity])
    velocity_end *= 1./len(list(dq_end.keys()))

    nominal_landing_cost = P['cost', 'nominal_landing'] * (10*velocity_end + 0*position_end)

    return nominal_landing_cost

def find_compromised_battery_cost(nlp_numerics_options, V, P, emergency_scenario, model):
    n_k = nlp_numerics_options['n_k']
    if (len(model.architecture.kite_nodes) == 1 or nlp_numerics_options['system_model']['kite_dof'] == 6 or emergency_scenario[0] != 'broken_battery'):
        compromised_battery_cost = cas.DM(0.0)
    elif emergency_scenario[0] == 'broken_battery':
        actuator_len = V['u',0,'dcoeff21'].shape[0]
        broken_actuator = slice(0,actuator_len)
        broken_kite = emergency_scenario[1]
        broken_kite_parent = model.architecture.parent_map[broken_kite]

        compromised_battery_cost = 0.0
        for j in range(n_k):
            broken_str = 'dcoeff' + str(broken_kite) + str(broken_kite_parent)
            compromised_battery_cost += cas.mtimes(V['u', j, broken_str, broken_actuator].T,V['u', j, broken_str, broken_actuator])

        compromised_battery_cost *= 1./n_k
        compromised_battery_cost = P['cost', 'compromised_battery'] * compromised_battery_cost

    return compromised_battery_cost

def find_compromised_battery_problem_cost(nlp_numerics_options, V, P, model):
    emergency_scenario = nlp_numerics_options['landing']['emergency_scenario']
    compromised_battery_problem_cost = find_compromised_battery_cost(nlp_numerics_options, V, P, emergency_scenario, model)

    return compromised_battery_problem_cost

def find_tracking_problem_cost(nlp_numerics_options, V, P, variables, parameters):

    fictitious_cost = find_fictitious_cost(nlp_numerics_options, V, P, variables)
    tracking_cost = find_tracking_cost(nlp_numerics_options, V, P, variables)
    time_cost = find_time_cost(nlp_numerics_options, V, P)

    tracking_problem_cost = fictitious_cost + tracking_cost + time_cost

    return tracking_problem_cost

def find_power_problem_cost(nlp_numerics_options, V, P, Integral_outputs):

    power_problem_cost = find_power_cost(nlp_numerics_options, V, P, Integral_outputs)

    return power_problem_cost

def find_nominal_landing_problem_cost(nlp_numerics_options, V, P, variables):

    nominal_landing_problem_cost = find_nominal_landing_cost(V, P, variables)

    return nominal_landing_problem_cost

def find_general_problem_cost(nlp_numerics_options, V, P, variables, parameters, xdot, outputs):

    gamma_cost = find_gamma_cost(V, P)
    iota_cost = find_iota_cost(V, P)
    tau_cost = find_tau_cost(V, P)
    psi_cost = find_psi_cost(V, P)
    eta_cost = find_eta_cost(V, P)
    nu_cost = find_nu_cost(V, P)
    upsilon_cost = find_upsilon_cost(V, P)

    regularisation_cost = find_regularisation_cost(nlp_numerics_options, V, P, variables)
    ddq_regularisation_cost = find_ddq_regularisation_cost(nlp_numerics_options, V, P, xdot, outputs)
    theta_regularisation_cost = find_theta_regularisation_cost(V, P)

    general_problem_cost = regularisation_cost + theta_regularisation_cost + psi_cost + iota_cost + tau_cost + gamma_cost + eta_cost + nu_cost + upsilon_cost + ddq_regularisation_cost

    return general_problem_cost

def find_objective(nlp_numerics_options, V, P, variables, parameters, xdot, outputs, model, Integral_outputs):
    # tracking dissappears slowly in the cost function and energy maximising appears. at the final step, cost function
    # contains maximising energy, lift, sosc, and regularisation.

    tracking_problem_cost = find_tracking_problem_cost(nlp_numerics_options, V, P, variables, parameters)
    power_problem_cost = find_power_problem_cost(nlp_numerics_options, V, P, Integral_outputs)
    nominal_landing_problem_cost = find_nominal_landing_problem_cost(nlp_numerics_options, V, P, variables)
    compromised_battery_problem_cost = find_compromised_battery_problem_cost(nlp_numerics_options, V, P, model)
    transition_problem_cost = find_transition_problem_cost(V, P, nlp_numerics_options, xdot, outputs, variables)
    general_problem_cost = find_general_problem_cost(nlp_numerics_options, V, P, variables, parameters, xdot, outputs)

    objective = V['phi','upsilon'] * V['phi', 'nu'] * V['phi', 'eta'] * V['phi', 'psi'] * tracking_problem_cost + (1. - V['phi', 'psi']) * power_problem_cost + general_problem_cost + (1. - V['phi', 'eta']) * nominal_landing_problem_cost + (1. - V['phi','upsilon'])*transition_problem_cost# + (1. - V['phi', 'nu']) * compromised_battery_problem_cost

    return objective

def get_component_cost_dictionary(nlp_numerics_options, V, P, variables, parameters, xdot, outputs, model, Integral_outputs):
    component_costs = {}

    component_costs['objective'] = find_objective(nlp_numerics_options, V, P, variables, parameters, xdot, outputs, model, Integral_outputs)

    component_costs['tracking_problem_cost'] = find_tracking_problem_cost(nlp_numerics_options, V, P, variables, parameters)
    component_costs['power_problem_cost'] = find_power_problem_cost(nlp_numerics_options, V, P, Integral_outputs)
    component_costs['general_problem_cost'] = find_general_problem_cost(nlp_numerics_options, V, P, variables, parameters, xdot, outputs)

    component_costs['gamma_cost'] = find_gamma_cost(V, P)
    component_costs['fictitious_cost'] = find_fictitious_cost(nlp_numerics_options, V, P, variables)
    component_costs['tracking_cost'] = find_tracking_cost(nlp_numerics_options, V, P, variables)
    component_costs['time_cost'] = find_time_cost(nlp_numerics_options, V, P)
    component_costs['iota_cost'] = find_iota_cost(V, P)
    component_costs['tau_cost'] = find_tau_cost(V, P)
    component_costs['eta_cost'] = find_eta_cost(V, P)
    component_costs['nu_cost'] = find_nu_cost(V, P)
    component_costs['upsilon_cost'] = find_upsilon_cost(V, P)

    component_costs['u_regularisation_cost'] = find_regularisation_cost(nlp_numerics_options, V, P, variables)
    component_costs['ddq_regularisation_cost'] = find_ddq_regularisation_cost(nlp_numerics_options, V, P, xdot, outputs)
    component_costs['theta_regularisation_cost'] = find_theta_regularisation_cost(V, P)
    component_costs['psi_cost'] = find_psi_cost(V, P)

    component_costs['power_cost'] = find_power_cost(nlp_numerics_options, V, P, Integral_outputs)
    component_costs['nominal_landing_cost'] = find_nominal_landing_problem_cost(nlp_numerics_options, V, P, variables)
    component_costs['transition_cost'] = find_transition_problem_cost(V, P, nlp_numerics_options, xdot, outputs, variables)
    component_costs['compromised_battery_cost'] = find_compromised_battery_problem_cost(nlp_numerics_options, V, P, model)

    return component_costs

def get_component_cost_function(component_costs, V, P):

    component_cost_fun = {}

    for name in list(component_costs.keys()):
        component_cost_fun[name + '_fun'] = cas.Function(name + '_fun', [V, P], [component_costs[name]])

    return component_cost_fun

def get_component_cost_structure(component_costs):

    list_of_entries = []
    for name in list(component_costs.keys()):
        list_of_entries += [cas.entry(name)]

    component_cost_struct = cas.struct_symMX(list_of_entries)

    return component_cost_struct

def get_cost_function_and_structure(nlp_numerics_options, V, P, variables, parameters, xdot, outputs, model, Integral_outputs):

    component_costs = get_component_cost_dictionary(nlp_numerics_options, V, P, variables, parameters, xdot, outputs, model, Integral_outputs)

    component_cost_function = get_component_cost_function(component_costs, V, P)
    component_cost_structure = get_component_cost_structure(component_costs)
    [f_fun, f_jacobian_fun, f_hessian_fun] = make_cost_function(V, P, component_costs)

    return [component_cost_function, component_cost_structure, f_fun, f_jacobian_fun, f_hessian_fun]

def make_cost_function(V, P, component_costs):
    f = []
    for cost in list(component_costs.keys()):
        f = cas.vertcat(f, component_costs[cost])
    f = cas.sum1(f)

    f_fun = cas.Function('f', [V, P], [f])
    [H,g] = cas.hessian(f,V)
    f_jacobian_fun = cas.Function('f_jacobian', [V, P], [g])
    f_hessian_fun = cas.Function('f_hessian', [V, P], [H])

    return [f_fun, f_jacobian_fun, f_hessian_fun]

def extract_discretization_info(nlp_numerics_options):

    if nlp_numerics_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        multiple_shooting = False
        d = nlp_numerics_options['collocation']['d']
        scheme = nlp_numerics_options['collocation']['scheme']
        int_weights = find_int_weights(nlp_numerics_options)
    elif nlp_numerics_options['discretization'] == 'multiple_shooting':
        direct_collocation = False
        multiple_shooting = True
        d = None
        scheme = None
        int_weights = None

    return direct_collocation, multiple_shooting, d, scheme, int_weights
