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
operation functions
to generate operation-mode specific constraints, variables, and parameters

constraints are divided as initial, terminal and periodic constraints, that are either inequalities or equalities
-- function inputs for initial and terminal constraints are (variables, pfix ref variables)
-- function inputs for periodic constraints are (initial variables, final variables)

python-3.5 / casadi-3.4.5
- authors: rachel leuthold, thilo bronnenmeyer, alu-fr 2018-20
'''

import casadi.tools as cas

import awebox.tools.vector_operations as vect_op
import awebox.tools.performance_operations as perf_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.parameterization as parameterization
import awebox.tools.constraint_operations as cstr_op

import awebox.ocp.ocp_constraint as ocp_constraint

import awebox.mdl.aero.induction_dir.vortex_dir.fixing as vortex_fix
import awebox.mdl.aero.induction_dir.vortex_dir.strength as vortex_strength


from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op


def get_operation_conditions(options):

    periodic = perf_op.determine_if_periodic(options)
    initial_conditions = determine_if_initial_conditions(options)
    param_initial_conditions = determine_if_param_initial_conditions(options)
    param_terminal_conditions = determine_if_param_terminal_conditions(options)
    terminal_inequalities = determine_if_terminal_inequalities(options)
    integral_constraints = determine_if_integral_constraints(options)

    return [periodic, initial_conditions, param_initial_conditions, param_terminal_conditions, terminal_inequalities, integral_constraints]

def determine_if_integral_constraints(options):
    compromised_landing = (options['trajectory']['type'] == 'compromised_landing')
    broken_battery = compromised_landing and (options['compromised_landing']['emergency_scenario'][0] == 'broken_battery')
    return broken_battery

def determine_if_terminal_inequalities(options):
    return (options['trajectory']['type'] in ['nominal_landing', 'compromised_landing'])


def determine_if_param_initial_conditions(options):
    return (options['trajectory']['type'] in ['transition','nominal_landing','compromised_landing'])

def determine_if_initial_conditions(options):
    return (options['trajectory']['type'] in ['launch','mpc'])

def determine_if_param_terminal_conditions(options):
    return (options['trajectory']['type'] in ['transition', 'launch'])

def get_initial_constraints(options, initial_variables, ref_variables, model, xi_dict):

    cstr_list = ocp_constraint.OcpConstraintList()

    # list all initial equalities ==> put SX expressions in dict
    if 'e' in list(model.variables_dict['xd'].keys()):
        init_energy_eq = make_initial_energy_equality(initial_variables, ref_variables)
        init_energy_cstr = cstr_op.Constraint(expr=init_energy_eq,
                                    name='initial_energy',
                                    cstr_type='eq')
        cstr_list.append(init_energy_cstr)

    _, initial_conditions, param_initial_conditions, _, _, _ = get_operation_conditions(options)

    if param_initial_conditions:
        init_param_eq = make_param_initial_conditions(initial_variables, ref_variables, xi_dict, model, options)
        init_param_cstr = cstr_op.Constraint(expr=init_param_eq,
                                    name='param_initial_conditions',
                                    cstr_type='eq')
        cstr_list.append(init_param_cstr)

    if initial_conditions:
        init_eq = make_initial_conditions(initial_variables, ref_variables, xi_dict, model, options)
        init_cstr = cstr_op.Constraint(expr=init_eq,
                                    name='initial_conditions',
                                    cstr_type='eq')
        cstr_list.append(init_cstr)

    return cstr_list


def generate_integral_constraints(options, variables, parameters, model):

    eqs_dict = {}
    ineqs_dict = {}
    constraint_list = []
    integral_constants_list = []
    ineqs_list = []
    eqs_list = []
    ineqs_constants_dict = {}

    [periodic, initial_conditions, param_initial_conditions, param_terminal_conditions, terminal_inequalities, integral_constraints] = get_operation_conditions(options)

    if integral_constraints:
        ineqs_dict['terminal_battery'] = make_terminal_battery_integrand(options, variables, parameters, model)
        constraint_list.append(ineqs_dict['terminal_battery'])
        ineqs_list.append(ineqs_dict['terminal_battery'])

    # generate integral constraints - empty struct containing both equalities and inequalitiess
    integral_constraints_struct = make_constraint_struct(eqs_dict, ineqs_dict)
    integral_constraints_eqs_struct = make_constraint_struct(eqs_dict, [])
    integral_constraints_ineqs_struct = make_constraint_struct([], ineqs_dict)

    integral_constraints = integral_constraints_struct(*constraint_list)
    integral_ineqs_constraints = integral_constraints_ineqs_struct(*ineqs_list)
    integral_eqs_constraints = integral_constraints_eqs_struct(*eqs_list)
    integral_constraints_fun = {}
    integral_constraints_fun['inequality'] = cas.Function('integral_fun', [variables, parameters], [integral_ineqs_constraints.cat])
    integral_constraints_fun['equality'] = cas.Function('integral_fun', [variables, parameters], [integral_eqs_constraints.cat])

    # I(tf) = I(0) + int(dI) < I_margin
    # I(tf) - I_margin + int(dI)
    # I(tf) - I_margin = I_const
    if list(integral_constraints.keys()):
        ineqs_constants_dict['terminal_battery'] = make_terminal_battery_constant(options)
        integral_constants_list.append(ineqs_constants_dict['terminal_battery'])

    integral_constants = integral_constraints_struct(*integral_constants_list)

    return integral_constraints_struct, integral_constraints_fun, integral_constants

def get_terminal_constraints(options, terminal_variables, ref_variables, model, xi_dict):

    cstr_list = ocp_constraint.OcpConstraintList()

    _, _, _, param_terminal_conditions, terminal_inequalities, integral_constraints = get_operation_conditions(options)

    if param_terminal_conditions:
        terminal_param_eq = make_param_terminal_conditions(terminal_variables, ref_variables, xi_dict, model, options)
        terminal_param_cstr = cstr_op.Constraint(expr=terminal_param_eq,
                                    name='param_terminal_conditions',
                                    cstr_type='eq')
        cstr_list.append(terminal_param_cstr)

    if terminal_inequalities:
        terminal_ineq = make_terminal_position_inequality(terminal_variables, model, options)
        terminal_ineq_cstr = cstr_op.Constraint(expr=terminal_ineq,
                                    name='terminal_inequalities',
                                    cstr_type='ineq')
        cstr_list.append(terminal_ineq_cstr)

    return cstr_list



def get_vortex_strength_constraints(options, variables, model):
    # this function is just the placeholder. For the applied constraint, see constraints.append_wake_fix_constraints()

    ineqs_dict = {}
    eqs_dict, constraint_list = vortex_strength.get_cstr_in_operation_format(options, variables, model)

    # generate initial constraints - empty struct containing both equalities and inequalitiess
    vortex_strength_constraints_struct = make_constraint_struct(eqs_dict, ineqs_dict)

    # fill in struct and create function
    vortex_strength_constraints = vortex_strength_constraints_struct(cas.vertcat(*constraint_list))
    vortex_strength_constraints_fun = cas.Function('vortex_strength_constraints_fun', [variables], [vortex_strength_constraints.cat])

    return vortex_strength_constraints, vortex_strength_constraints_fun


def get_wake_fix_constraints(options, variables, model):
    # this function is just the placeholder. For the applied constraint, see constraints.append_wake_fix_constraints()

    ineqs_dict = {}
    eqs_dict, constraint_list = vortex_fix.get_cstr_in_operation_format(options, variables, model, Collocation)

    # generate initial constraints - empty struct containing both equalities and inequalitiess
    wake_fix_constraints_struct = make_constraint_struct(eqs_dict, ineqs_dict)

    # fill in struct and create function
    wake_fix_constraints = wake_fix_constraints_struct(cas.vertcat(*constraint_list))
    wake_fix_constraints_fun = cas.Function('wake_fix_constraints_fun', [variables], [wake_fix_constraints.cat])

    return wake_fix_constraints, wake_fix_constraints_fun


def get_periodic_constraints(options, initial_model_variables, terminal_model_variables):
    cstr_list = ocp_constraint.OcpConstraintList()

    periodic, _, _, _, _, _ = get_operation_conditions(options)

    # list all periodic equalities ==> put SX expressions in dict
    if periodic:
        periodic_eq = make_periodicity_equality(initial_model_variables, terminal_model_variables, options)
        cstr = cstr_op.Constraint(expr=periodic_eq,
                                  name='state_periodicity',
                                  cstr_type='eq')
        cstr_list.append(cstr)

    return cstr_list

def make_initial_energy_equality(initial_model_variables, ref_variables):

    initial_energy = initial_model_variables['xd', 'e']
    e_0 = ref_variables['xd', 'e']

    initial_energy_eq = initial_energy - e_0

    return initial_energy_eq

def variable_does_not_belong_to_unselected_induction_model(name, options):
    induction_steadyness = options['induction']['steadyness']
    induction_symmetry = options['induction']['symmetry']

    induction_label = ''
    if induction_steadyness == 'steady':
        induction_label += 'q'
    elif induction_steadyness == 'unsteady':
        induction_label += 'u'

    if induction_symmetry == 'axisymmetric':
        induction_label += 'axi'
    elif induction_symmetry == 'asymmetric':
        induction_label += 'asym'

    remaining_induction_labels = ['qaxi', 'qasym', 'uaxi', 'uasym']
    if induction_label in remaining_induction_labels:
        remaining_induction_labels.remove(induction_label)

    not_unselected = True
    for label in remaining_induction_labels:
        if label in name:
            not_unselected = False

    return not_unselected



def make_periodicity_equality(initial_model_variables, terminal_model_variables, options):

    periodicity_cstr = []
    for name in struct_op.subkeys(initial_model_variables, 'xd'):

        not_unselected_induction_model = variable_does_not_belong_to_unselected_induction_model(name, options)

        if (not name[0] == 'e') and (not name[0] == 'w') and (not name[:2] == 'dw') and (not name[:3] == 'psi') and not_unselected_induction_model:

            initial_value = vect_op.columnize(initial_model_variables['xd', name])
            final_value = vect_op.columnize(terminal_model_variables['xd', name])

            difference = initial_value - final_value

            periodicity_cstr = cas.vertcat(periodicity_cstr, difference)

    periodicity_eq = periodicity_cstr

    return periodicity_eq

def make_param_initial_conditions(initial_model_variables, ref_variables, xi_dict, model,options):
    initial_states = initial_model_variables

    awelogger.logger.info('Parameterizing initial constraint...')
    xi_0 = xi_dict['xi']['xi_0']
    initial_splines = parameterization.get_splines(initial_model_variables, xi_dict, 'initial')

    xd_struct = model.variables_dict['xd']

    spline_list = []

    for i in range(xd_struct.cat.shape[0]):
        (state_name, state_dim) = xd_struct.getCanonicalIndex(i)
        spline_list += [initial_splines[state_name + '_' + str(state_dim)](xi_0)]

    var_ref_initial = xd_struct(cas.vertcat(*spline_list))

    # initializate lists
    initial_conditions_eq_list = []
    black_list = []

    # compute black list of variables that should not be constrained
    if options['trajectory']['type'] == 'compromised_landing' and options['compromised_landing']['emergency_scenario'][0] == 'structural_damages':
        broken_kite = options['compromised_landing']['emergency_scenario'][1]
        broken_parent = model.architecture.parent_map[broken_kite]
        black_list += ['coeff' + str(broken_kite) + str(broken_parent)]
    variable_list = set(xd_struct.keys()) - set(black_list)

    # iterate over variables to construct constraints
    for variable in variable_list:
        initial_conditions_eq_list += [initial_states['xd', variable] - var_ref_initial[variable] / model.scaling['xd'][variable]]
    initial_conditions_eq = cas.vertcat(*initial_conditions_eq_list)

    return initial_conditions_eq

def make_initial_conditions(initial_model_variables, ref_variables, xi_dict, model,options):
    initial_states = initial_model_variables

    awelogger.logger.info('Introducing initial constraint...')

    xd_struct = model.variables_dict['xd']

    # initializate lists
    initial_conditions_eq_list = []
    black_list = []

    variable_list = set(xd_struct.keys()) - set(black_list)

    # iterate over variables to construct constraints
    for variable in variable_list:
        initial_conditions_eq_list += [initial_states['xd', variable] - ref_variables['xd',variable]]
    initial_conditions_eq = cas.vertcat(*initial_conditions_eq_list)

    return initial_conditions_eq

def make_param_terminal_conditions(terminal_model_variables, ref_variables, xi_dict, model, options):
    terminal_states = terminal_model_variables

    awelogger.logger.info('Parameterizing terminal constraint...')
    xi_f = xi_dict['xi']['xi_f']
    terminal_splines = parameterization.get_splines(terminal_model_variables, xi_dict, 'terminal')

    xd_struct = model.variables_dict['xd']

    spline_list = []

    for i in range(xd_struct.cat.shape[0]):
        (state_name, state_dim) = xd_struct.getCanonicalIndex(i)
        spline_list += [terminal_splines[state_name + '_' + str(state_dim)](xi_f)]

    var_ref_terminal = xd_struct(cas.vertcat(*spline_list))

    # initializate lists
    terminal_conditions_eq_list = []
    black_list = []

    # compute black list of variables that should not be constrained
    variable_list = set(xd_struct.keys()) - set(black_list)

    # iterate over variables to construct constraints
    for variable in variable_list:
        terminal_conditions_eq_list += [terminal_states['xd', variable] - var_ref_terminal[variable] / model.scaling['xd'][variable]]
    terminal_conditions_eq = cas.vertcat(*terminal_conditions_eq_list)

    return terminal_conditions_eq

def make_terminal_position_inequality(terminal_variables, model, options):

    main_node_radius = options['nominal_landing']['main_node_radius']
    kite_radius = options['nominal_landing']['kite_node_radius']
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    terminal_position_inequality = (cas.mtimes(terminal_variables['xd', 'q10'].T, terminal_variables['xd', 'q10']) - main_node_radius ** 2) / main_node_radius ** 2
    for node in kite_nodes:
        parent = parent_map[node]
        terminal_position_inequality = cas.vertcat(terminal_position_inequality, (cas.mtimes(terminal_variables['xd', 'q' + str(node) + str(parent)].T,
                                                                                     terminal_variables['xd', 'q' + str(node) + str(parent)])
                                                                              - kite_radius**2)
                                               / main_node_radius**2)

    #terminal_position_inequality = vertcat(terminal_position_inequality, (mtimes(terminal_variables['xd', 'q21'].T, terminal_variables['xd', 'q21']) - (landing_radius_qn1) ** 2) / (landing_radius_q10) ** 2)
    #terminal_position_inequality = vertcat(terminal_position_inequality, (mtimes(terminal_variables['xd', 'q31'].T, terminal_variables['xd', 'q31']) - (landing_radius_qn1) ** 2) / (landing_radius_q10) ** 2)

    return terminal_position_inequality

def make_terminal_battery_integrand(options, variables, parameters, model):

    nu = parameters['phi','nu']
    broken_kite = options['compromised_landing']['emergency_scenario'][1]
    broken_kite_parent = model.architecture.parent_map[broken_kite]
    surface = options['compromised_landing']['kite']['flap_length']*options['compromised_landing']['kite']['flap_width']
    moment_arm = options['compromised_landing']['kite']['flap_length']/2.
    q_z = variables['xd','q' + str(broken_kite) + str(broken_kite_parent),2]
    dq = variables['xd','dq' + str(broken_kite) + str(broken_kite_parent)]
    C_L = variables['xd','coeff' + str(broken_kite) + str(broken_kite_parent),0]
    Phi = variables['xd','coeff' + str(broken_kite) + str(broken_kite_parent),1]
    dC_L = variables['u','dcoeff' + str(broken_kite) + str(broken_kite_parent),0]
    dPhi = variables['u','dcoeff' + str(broken_kite) + str(broken_kite_parent),1]

    density = model.atmos.get_density(q_z)
    dynamic_pressure = 0.5*cas.mtimes(dq.T,dq)*density
    c_dl = options['compromised_landing']['aero']['c_dl']
    c_dphi = options['compromised_landing']['aero']['c_dphi']
    deflection_lift_0 = options['compromised_landing']['aero']['defl_lift_0']
    deflection_roll_0 = options['compromised_landing']['aero']['defl_roll_0']
    deflection_lift = deflection_lift_0 + c_dl*C_L
    deflection_roll = deflection_roll_0 + c_dphi*Phi
    ddeflection_lift = c_dl*dC_L
    ddeflection_roll = c_dphi*dPhi
    lift_moment = dynamic_pressure*surface*moment_arm*cas.sin(deflection_lift)
    roll_moment = dynamic_pressure*surface*moment_arm*cas.sin(deflection_roll)

    terminal_battery_integrand = -(lift_moment*ddeflection_lift + roll_moment*ddeflection_roll + options['compromised_landing']['battery']['power_controller'] + options['compromised_landing']['battery']['power_electronics'])*(1. - nu)

    return terminal_battery_integrand

def make_terminal_battery_constant(options):

    voltage = options['compromised_landing']['battery']['voltage']
    charge = options['compromised_landing']['battery']['charge']
    number_of_cells = options['compromised_landing']['battery']['number_of_cells']
    conversion_efficiency = options['compromised_landing']['battery']['conversion_efficiency']
    charge_fraction = options['compromised_landing']['battery']['charge_fraction']
    terminal_battery_constant = charge_fraction*number_of_cells*voltage*charge*conversion_efficiency

    return terminal_battery_constant

def make_constraint_struct(eqs_dict, ineqs_dict):

    entry_list = make_entry_list(eqs_dict, ineqs_dict)
    constraint_struct = cas.struct_symSX(entry_list)

    return constraint_struct

def clear_empty_keys(dict):
    if bool(dict):
        for name in list(dict.keys()):
            try:
                dict[name].size()
            except:
                awelogger.logger.warning('removing constraint entry (' + name + ') from dictionary, because it appears to be empty')
                dict.pop(name)
    return dict

def make_entry_list(eqs_dict, ineqs_dict):

    eqs_dict = clear_empty_keys(eqs_dict)
    ineqs_dict = clear_empty_keys(ineqs_dict)

    # make entry list for all non-empty dicts
    entry_list = []
    if bool(eqs_dict): # check if not empty

        # equality constraint struct
        eq_struct = cas.struct_symSX([
            cas.entry(name, shape = eqs_dict[name].size()) for name in list(eqs_dict.keys())
        ])
        entry_list.append(cas.entry('equality', struct = eq_struct))

    if ineqs_dict: # check if not empty

        # inequality constraint struct
        ineq_struct = cas.struct_symSX([
            cas.entry(name, shape = ineqs_dict[name].size()) for name in list(ineqs_dict.keys())
        ])
        entry_list.append(cas.entry('inequality', struct = ineq_struct))

    return entry_list
