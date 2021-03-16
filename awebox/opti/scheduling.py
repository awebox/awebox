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
update model that generates the
cost update and bounds update to be used in the homotopy process
python-3.5 / casadi-3.4.5
- authors: rachel leuthold, alu-fr 2018
- edited: jochem de schutter, alu-fr 2018-2020
'''

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

def define_homotopy_update_schedule(model, formulation, nlp, cost_solver_options):

    schedule = {}
    schedule['cost'] = define_cost_update_schedule(cost_solver_options)
    schedule['bounds'] = define_bound_update_schedule(model, nlp, formulation)
    schedule['homotopy'] = define_homotopy_schedule(formulation)
    schedule['costs_to_update'] = define_costs_to_update(nlp.P, formulation)
    schedule['bounds_to_update'] = define_bounds_to_update(model, schedule['bounds'], formulation)
    schedule['labels'] = define_step_labels(formulation)

    return schedule

def define_homotopy_schedule(formulation):

    initial_schedule = ('initial','fictitious',)
    induction_schedule = ('induction',)
    tether_release_schedule = ('tether_release',)
    power_schedule = ('power',)
    transition_schedule = ('transition',)
    nominal_landing_schedule = ('nominal_landing',)
    compromised_landing_schedule = ('compromised_landing',)
    tether_schedule = ('tether',)
    final_schedule = ('final',)

    induction_model = formulation.induction_model
    traj_type = formulation.traj_type
    tether_drag_model = formulation.tether_drag_model
    fix_tether_length = formulation.fix_tether_length

    homotopy_schedule = ()
    homotopy_schedule = homotopy_schedule + initial_schedule

    if traj_type == 'tracking' and fix_tether_length == False:
        homotopy_schedule = homotopy_schedule + tether_release_schedule

    make_induction_step = not (induction_model == 'not_in_use')
    if make_induction_step:
        homotopy_schedule = homotopy_schedule + induction_schedule

    if traj_type == 'power_cycle':
        homotopy_schedule = homotopy_schedule + power_schedule

    if traj_type == 'nominal_landing':
        homotopy_schedule = homotopy_schedule + nominal_landing_schedule

    if traj_type == 'transition':
        homotopy_schedule = homotopy_schedule + transition_schedule

    if traj_type == 'compromised_landing':
        homotopy_schedule = homotopy_schedule + nominal_landing_schedule
        homotopy_schedule = homotopy_schedule + compromised_landing_schedule

    # if tether_drag_model in set([multi']):
    #     homotopy_schedule = homotopy_schedule + tether_schedule

    homotopy_schedule = homotopy_schedule + final_schedule

    return homotopy_schedule

def define_costs_to_update(P, formulation):

    updates = {}

    initial_updates = {}
    initial_updates[0] = set(struct_op.subkeys(P, 'cost'))

    fictitious_updates = {}
    fictitious_updates[0] = ['gamma', 'fictitious']
    fictitious_updates[1] = []

    tether_updates = {}
    tether_updates[0] = ['tau']
    tether_updates[1] = []

    induction_updates = {}
    induction_updates[0] = ['iota']
    induction_updates[1] = []

    tether_release_updates = {}
    tether_release_updates[0] = []

    power_updates = {}
    power_updates[0] = ['power', 'psi', 'fictitious']
    power_updates[1] = ['tracking']

    nominal_landing_updates = {}
    nominal_landing_updates[0] = ['nominal_landing', 'eta']
    nominal_landing_updates[1] = ['tracking']

    transition_updates = {}
    transition_updates[0] = ['transition', 'upsilon']
    transition_updates[1] = ['tracking']

    compromised_landing_updates = {}
    compromised_landing_updates[0] = ['compromised_battery', 'nu']
    compromised_landing_updates[1] = []

    final_updates = {}
    final_updates[0] = []

    updates['initial'] = initial_updates
    updates['fictitious'] = fictitious_updates
    updates['tether'] = tether_updates
    updates['induction'] = induction_updates
    updates['tether_release'] = tether_release_updates
    updates['power'] = power_updates
    updates['transition'] = transition_updates
    updates['nominal_landing'] = nominal_landing_updates
    updates['compromised_landing'] = compromised_landing_updates
    updates['final'] = final_updates

    return updates

def define_bounds_to_update(model, bounds_schedule, formulation):

    updates = {}

    initial_updates = {}
    initial_updates[0] = []

    fictitious_updates = {}
    fictitious_updates[0] = ['gamma']
    fictitious_updates[1] = ['gamma']
    for name in struct_op.subkeys(model.variables, 'u'):
        if 'fict' in name:
            fictitious_updates[1] += [name] * 2

    tether_updates = {}
    tether_updates[0] = ['tau']
    tether_updates[1] = ['tau']

    induction_updates = {}
    induction_updates[0] = ['iota']
    induction_updates[1] = ['iota']

    tether_release_updates = {}
    # check which tether length variable is a control variable
    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        tether_release_updates[0] = ['ddl_t', 'ddl_t']
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        tether_release_updates[0] = ['dddl_t', 'dddl_t']

    power_updates = {}
    # check which tether length variable is a control variable
    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        power_updates[0] = ['ddl_t', 'ddl_t', 'psi'] + struct_op.subkeys(model.variables, 'theta') * 2
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        power_updates[0] = ['dddl_t', 'dddl_t', 'psi'] + struct_op.subkeys(model.variables, 'theta') * 2
    # check if phase fix
    if 'dl_t' in list(bounds_schedule.keys()):
        power_updates[0] += ['dl_t']*2
    if 'l_t' in list(bounds_schedule.keys()):
        power_updates[0] += ['l_t']*2

    power_updates[1] = ['psi']

    nominal_landing_updates = {}
    # check which tether length variable is a control variable
    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        nominal_landing_updates[0] = ['ddl_t', 'ddl_t', 'eta'] + struct_op.subkeys(model.variables, 'theta') * 2
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        nominal_landing_updates[0] = ['dddl_t', 'dddl_t', 'eta'] + struct_op.subkeys(model.variables, 'theta') * 2
    # check if phase fix
    if 'dl_t' in list(bounds_schedule.keys()):
        nominal_landing_updates[0] += ['dl_t']*2

    nominal_landing_updates[1] = ['eta']

    transition_updates = {}
    # check which tether length variable is a control variable
    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        transition_updates[0] = ['ddl_t', 'ddl_t', 'upsilon'] + struct_op.subkeys(model.variables, 'theta') * 2
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        transition_updates[0] = ['dddl_t', 'dddl_t', 'upsilon'] + struct_op.subkeys(model.variables, 'theta') * 2
    # check if phase fix
    if 'dl_t' in list(bounds_schedule.keys()):
        transition_updates[0] += ['dl_t']*2

    transition_updates[1] = ['upsilon']

    compromised_landing_updates = {}
    compromised_landing_updates[0] = ['nu']
    compromised_landing_updates[1] = ['nu']

    final_updates = {}
    final_updates[0] = []

    updates['initial'] = initial_updates
    updates['fictitious'] = fictitious_updates
    updates['tether'] = tether_updates
    updates['induction'] = induction_updates
    updates['tether_release'] = tether_release_updates
    updates['power'] = power_updates
    updates['transition'] = transition_updates
    updates['nominal_landing'] = nominal_landing_updates
    updates['compromised_landing'] = compromised_landing_updates
    updates['final'] = final_updates

    return updates

def define_step_labels(formulation):

    updates = {}

    initial_updates = {}
    initial_updates[0] = 'Initial solution...'

    fictitious_updates = {}
    fictitious_updates[0] = 'Minimize fictitious forces...'
    fictitious_updates[1] = 'Eliminate fictitious forces...'

    tether_updates = {}
    tether_updates[0] = 'Introduce physical tether drag...'
    tether_updates[1] = 'Enforce physical tether drag...'

    induction_updates = {}
    induction_updates[0] = 'Introduce induction constraints...'
    induction_updates[1] = 'Enforce induction constraints...'

    tether_release_updates = {}
    tether_release_updates[0] = 'Releasing tether constraints...'

    power_updates = {}
    power_updates[0] = 'Switch to power problem...'
    power_updates[1] = 'Maximize average power...'

    transition_updates = {}
    transition_updates[0] = 'Switch to transition problem...'
    transition_updates[1] = 'Regularize transition trajectory...'

    nominal_landing_updates = {}
    nominal_landing_updates[0] = 'Switch to landing problem...'
    nominal_landing_updates[1] = 'Minimize final tether length...'

    compromised_landing_updates = {}
    compromised_landing_updates[0] = 'Switch to compromised landing problem...'
    compromised_landing_updates[1] = 'Enforce compromised landing paradigm...'

    final_updates = {}
    final_updates[0] = 'Final solution.'

    updates['initial'] = initial_updates
    updates['fictitious'] = fictitious_updates
    updates['tether'] = tether_updates
    updates['induction'] = induction_updates
    updates['tether_release'] = tether_release_updates
    updates['power'] = power_updates
    updates['transition'] = transition_updates
    updates['nominal_landing'] = nominal_landing_updates
    updates['compromised_landing'] = compromised_landing_updates
    updates['final'] = final_updates

    return updates


def update_cost(schedule, step_name, counter, cost_update_counter, p_fix_num):

    costs_to_update = schedule['costs_to_update'][step_name][counter]
    for cost_name in costs_to_update:
        count_of_this_update = cost_update_counter[cost_name] + 1

        update = schedule['cost'][cost_name][count_of_this_update]

        p_fix_num['cost', cost_name] = update

        cost_update_counter[cost_name] = count_of_this_update

    return cost_update_counter, p_fix_num

def update_bounds(schedule, step_name, counter, bound_update_counter, V_bounds, model, nlp):
    bounds_to_update = schedule['bounds_to_update'][step_name][counter]

    for bound_name in bounds_to_update:

        count_of_this_update = bound_update_counter[bound_name] + 1
        update = schedule['bounds'][bound_name][count_of_this_update]

        new_value = update[2]

        if new_value == 'final':
            V_bounds = update_final_bounds(bound_name, V_bounds, nlp, update)
        else:
            V_bounds = update_nonfinal_bounds(bound_name, V_bounds, model, nlp, update)

        bound_update_counter[bound_name] = count_of_this_update

    return bound_update_counter, V_bounds

def update_final_bounds(bound_name, V_bounds, nlp, update):
    bound_type = update[0]
    var_type = update[1]

    if var_type == 'phi':
        V_bounds[bound_type][var_type, bound_name] = 0.

    if var_type == 'theta':
        V_bounds[bound_type][var_type, bound_name] = nlp.V_bounds[bound_type][var_type, bound_name]

    if var_type == 'u':
        if 'u' in list(V_bounds[bound_type].keys()):
            V_bounds[bound_type][var_type, :, bound_name] = nlp.V_bounds[bound_type][var_type, :, bound_name]
        else:
            V_bounds[bound_type]['coll_var', :, :, var_type, bound_name] = nlp.V_bounds[bound_type]['coll_var', :, :, var_type, bound_name]

    if var_type in {'xl', 'xa', 'xd'}:

        if var_type in list(nlp.V.keys()): # not the case for xa and xl in radau collocation
            V_bounds[bound_type][var_type, :, bound_name] = nlp.V_bounds[bound_type][var_type, :, bound_name]

        if 'coll_var' in list(nlp.V.keys()): # not the case for multiple shooting
            V_bounds[bound_type]['coll_var', :, :, var_type, bound_name] = nlp.V_bounds[bound_type]['coll_var', :, :, var_type, bound_name]

    return V_bounds

def update_nonfinal_bounds(bound_name, V_bounds, model, nlp, update):
    bound_type = update[0]
    var_type = update[1]
    new_value = update[2]

    if var_type == 'phi':
        V_bounds[bound_type][var_type, bound_name] = new_value
    else:

        scaled_value = new_value / model.scaling[var_type][bound_name]

        if var_type == 'theta':
            V_bounds[bound_type][var_type, bound_name] = scaled_value

        if var_type == 'u':
            if 'u' in list(V_bounds[bound_type].keys()):
                V_bounds[bound_type][var_type, :, bound_name] = scaled_value
            else:
                V_bounds[bound_type]['coll_var', :, :, var_type, bound_name] = scaled_value

        if var_type in {'xl', 'xa', 'xd'}:
            if var_type in list(nlp.V.keys()): # not the case for xa and xl in radau collocation
                V_bounds[bound_type][var_type, :, bound_name] = scaled_value

            if 'coll_var' in list(nlp.V.keys()): # not the case for multiple shooting
                V_bounds[bound_type]['coll_var', :, :, var_type, bound_name] = scaled_value

    return V_bounds

def create_empty_bound_update_schedule(model, nlp, formulation):
    bound_schedule = {}

    for name in list(model.parameters_dict['phi'].keys()):
        bound_schedule[name] = {}

    for name in struct_op.subkeys(model.variables, 'theta'):
        bound_schedule[name] = {}

    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        bound_schedule['ddl_t'] = {}
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        bound_schedule['dddl_t'] = {}
    # check if phase fix
    if nlp.V['theta','t_f'].shape[0] > 1:
        bound_schedule['dl_t'] = {}
        bound_schedule['l_t'] = {}

    for name in struct_op.subkeys(model.variables, 'u'):
        if 'fict' in name:
            bound_schedule[name] = {}

    return bound_schedule

def define_bound_update_schedule(model, nlp, formulation):

    bound_schedule = create_empty_bound_update_schedule(model, nlp, formulation)

    for name in list(model.parameters_dict['phi'].keys()):
        bound_schedule[name][1] = ['lb', 'phi', 'final']
        bound_schedule[name][2] = ['ub', 'phi', 'final']

    for name in struct_op.subkeys(model.variables, 'theta'):
        bound_schedule[name][1] = ['lb', 'theta', 'final']
        bound_schedule[name][2] = ['ub', 'theta', 'final']

    for name in struct_op.subkeys(model.variables, 'u'):
        if 'fict' in name:
            bound_schedule[name][1] = ['lb', 'u', 'final']
            bound_schedule[name][2] = ['ub', 'u', 'final']

    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        bound_schedule['ddl_t'][1] = ['lb', 'u', 'final']
        bound_schedule['ddl_t'][2] = ['ub', 'u', 'final']
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        bound_schedule['dddl_t'][1] = ['lb', 'u', 'final']
        bound_schedule['dddl_t'][2] = ['ub', 'u', 'final']
    if 'dl_t' in list(bound_schedule.keys()):
        bound_schedule['dl_t'][1] = ['lb','xd','final']
        bound_schedule['dl_t'][2] = ['ub','xd','final']
    if 'l_t' in list(bound_schedule.keys()):
        bound_schedule['l_t'][1] = ['lb','xd','final']
        bound_schedule['l_t'][2] = ['ub','xd','final']

    return bound_schedule

def define_cost_update_schedule(cost_solver_options):
    cost_schedule = cost_solver_options
    return cost_schedule

def initialize_cost_update_counter(P):

    cost_update_counter = {}
    for name in set(struct_op.subkeys(P, 'cost')):

        cost_update_counter[name] = -1

    return cost_update_counter

def initialize_bound_update_counter(model, schedule, formulation):
    bound_update_counter = {}

    for name in list(model.parameters_dict['phi'].keys()):
        bound_update_counter[name] = 0

    for name in struct_op.subkeys(model.variables, 'theta'):
        bound_update_counter[name] = 0

    for name in struct_op.subkeys(model.variables, 'u'):
        if 'fict' in name:
            bound_update_counter[name] = 0

    if 'ddl_t' in list(model.variables_dict['u'].keys()):
        bound_update_counter['ddl_t'] = 0
    elif 'dddl_t' in list(model.variables_dict['u'].keys()):
        bound_update_counter['dddl_t'] = 0

    if 'dl_t' in list(schedule['bounds'].keys()):
        bound_update_counter['dl_t'] = 0

    if 'l_t' in list(schedule['bounds'].keys()):
        bound_update_counter['l_t'] = 0

    return bound_update_counter

def find_current_homotopy_parameter(parameters, V_bounds):
    """ Return 'active' homotopy parameter by identifying which parameter
    has the bounds [0,1]. If no such parameter is identified, "None" is returned.
    """

    phi_name = None
    for phi in list(parameters.keys()):
        ub = V_bounds['ub']['phi',phi]
        lb = V_bounds['lb']['phi',phi]
        if ub != lb:
            phi_name = phi
    
    return phi_name