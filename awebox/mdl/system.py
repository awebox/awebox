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
module that describes the awe system under consideration, geometry, etc.
python-3.5 / casadi 3.0.0
- author: elena malz, chalmers 2016
- edited: rachel leuthold, alu-fr 2017-2021
          jochem de schutter, alu-fr 2017
'''



import casadi.tools as cas

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.actuator_dir.system as actuator_system
import copy
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op
import awebox.tools.struct_operations as struct_op
from awebox.logger.logger import Logger as awelogger

def generate_structure(options, architecture):

    kite_dof = options['kite_dof']
    surface_control = options['surface_control']
    tether_control_var = options['tether']['control_var']

    # _system architecture (see _zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    # _states, generalized coordinates and controls related to the tether
    # connection points
    # connection points only have position and velocity
    tether_states = [('q', (3, 1)), ('dq', (3, 1))]
    tether_gc = ['q']  # generalized coordinates
    tether_controls = []  # [('u',(3,1))]  # artificial force
    tether_multipliers = [('lambda', (1, 1))]

    # _states, generalized coordinates and controls related to kites

    kite_states = [('q', (3, 1)), ('dq', (3, 1))]
    kite_controls = [('f_fict', (3, 1))]
    kite_multipliers = [('lambda', (1, 1))]

    kite_gc = ['q']

    if int(kite_dof) == 3:
        kite_states = kite_states + [('coeff', (2, 1))]
        kite_controls = kite_controls + [('dcoeff', (2, 1))]

    elif int(kite_dof) == 6:
        kite_states = kite_states + [('omega', (3, 1)), ('r', (9, 1))]
        kite_controls = kite_controls + [('m_fict', (3, 1))]

        if int(surface_control) == 0:
            # delta:
            # (aileron left-right [right teu+, rad],
            # elevator [ted+, rad],
            # rudder [tel+, rad])
            kite_controls = kite_controls + [('delta', (3, 1))]

        if int(surface_control) == 1:
            # delta:
            # (aileron left-right [right teu+, rad],
            # elevator [ted+, rad],
            # rudder [tel+, rad])
            kite_states = kite_states + [('delta', (3, 1))]
            kite_controls = kite_controls + [('ddelta', (3, 1))]

    else:
        raise ValueError('kite dof option %s not inluded at present', str(kite_dof))

    # add drag mode states and controls
    if options['trajectory']['system_type'] == 'drag_mode':
        kite_states += [('kappa', (1, 1))]
        kite_controls += [('dkappa', (1, 1))]

    # _list states, generalized coordinates and controls of all the nodes
    # together
    system_states = []
    system_gc = []
    system_controls = []
    system_multipliers = []

    system_lifted = []

    for n in range(1, number_of_nodes):
        parent = parent_map[n]

        if n in kite_nodes:
            system_states.extend(
                [(kite_states[i][0] + str(n) + str(parent), kite_states[i][1]) for i in range(len(kite_states))])
            system_controls.extend(
                [(kite_controls[i][0] + str(n) + str(parent), kite_controls[i][1]) for i in range(len(kite_controls))])
            system_multipliers.extend(
                [(kite_multipliers[i][0] + str(n) + str(parent), kite_multipliers[i][1]) for i in
                 range(len(kite_multipliers))])

            system_gc.extend([kite_gc[i] + str(n) + str(parent)
                              for i in range(len(kite_gc))])

        else:
            system_states.extend(
                [(tether_states[i][0] + str(n) + str(parent), tether_states[i][1]) for i in range(len(tether_states))])
            system_controls.extend(
                [(tether_controls[i][0] + str(n) + str(parent), tether_controls[i][1]) for i in
                 range(len(tether_controls))])
            system_multipliers.extend(
                [(tether_multipliers[i][0] + str(n) + str(parent), tether_multipliers[i][1]) for i in
                 range(len(tether_multipliers))])

            system_gc.extend([tether_gc[i] + str(n) + str(parent)
                              for i in range(len(tether_gc))])

    # add cross-tethers
    if options['cross_tether'] and len(kite_nodes) > 1:
        for l in architecture.layer_nodes:
            kite_children = architecture.kites_map[l]
            if len(kite_children) == 2:
                system_multipliers.extend(
                        [
                            (
                            tether_multipliers[i][0] + str(kite_children[0]) + str(kite_children[1]),
                            tether_multipliers[i][1]
                            )
                            for i in range(len(tether_multipliers))
                        ]
                    )
            else:
                for k in range(len(kite_children)):
                    system_multipliers.extend(
                        [
                            (
                            tether_multipliers[i][0] + str(kite_children[k]) + str(kite_children[(k+1)%len(kite_children)]),
                            tether_multipliers[i][1]
                            )
                            for i in range(len(tether_multipliers))
                        ]
                    )

    # _add global states and controls
    if options['trajectory']['system_type'] == 'lift_mode':
        system_states.extend([('l_t', (1, 1))])
        system_states.extend([('dl_t', (1, 1))])

        # _energy + main tether length and speed
        if tether_control_var == 'ddl_t':
            system_controls.extend([('ddl_t', (1, 1))])  # main tether acceleration
        elif tether_control_var == 'dddl_t':
            system_states.extend([('ddl_t', (1, 1))])  # main tether acceleration
            system_controls.extend([('dddl_t', (1, 1))])  # main tether jerk
        else:
            raise ValueError('invalid tether control variable chosen')
        
    if options['integral_outputs']:
        pass
    else:
        system_states.extend([('e', (1, 1))])  # energy
        system_states.extend([('e_without_fictitious', (1, 1))])  # energy

        if options['integration']['include_integration_test']:
            system_states.extend([('total_time_unscaled', (1, 1))])
            system_states.extend([('total_time_scaled', (1, 1))])

    # introduce aerodynamics variables
    system_lifted, system_states = extend_aerodynamics(options, system_lifted, system_states, architecture)

    # system state derivatives
    system_derivatives = []
    for i in range(len(system_states)):
        system_derivatives.extend([('d'+system_states[i][0], system_states[i][1])])

    # system parameters
    system_parameters = [('diam_t', (1, 1)), ('t_f', (1, 1))]
    if options['trajectory']['system_type'] == 'drag_mode':
        system_parameters.extend([('l_t', (1, 1))])

    if (architecture.number_of_nodes - architecture.number_of_kites) > 1:
        system_parameters += [('l_s', (1, 1)), ('diam_s', (1, 1))]
    if architecture.get_number_intermediate_tethers() > 0:
        system_parameters += [('l_i', (1, 1)), ('diam_i', (1, 1))]

    if options['include_P_max']:
        system_parameters += [('P_max', (1, 1))] # max power

    if options['model_bounds']['ellipsoidal_flight_region']['include']:
        system_parameters += [('ell_radius', (1, 1))]

    if options['induction_model'] == 'averaged':
        system_parameters += [('a', (1, 1))]  # average induction
        system_parameters += [('ell_theta', (1, 1))]

    # add cross-tether lengths and diameters
    if options['cross_tether'] and len(kite_nodes) > 1:
        for l in architecture.layer_nodes:
            system_parameters.extend(
                [
                    ('l_c{}'.format(l), (1, 1)), ('diam_c{}'.format(l), (1, 1))
                ]
            )

    # store variables in dict
    system_variables_list = {'x': system_states,
                             'xdot': system_derivatives,
                             'u': system_controls,
                             'z': system_multipliers + system_lifted,
                             'theta': system_parameters}

    return system_variables_list, system_gc


def extend_general_induction(options, system_lifted, system_states, architecture):
    for kite in architecture.kite_nodes:
        system_lifted.extend([('wui' + str(kite), (3, 1))])

    return system_lifted, system_states


def extend_vortex_induction(options, system_lifted, system_states, architecture):
    system_lifted, system_states = vortex_tools.extend_system_variables(options, system_lifted, system_states, architecture)
    return system_lifted, system_states


def extend_actuator_induction(options, system_lifted, system_states, architecture):

    system_lifted, system_states = extend_actuator_support(options, system_lifted, system_states, architecture)

    if not options['aero']['actuator']['support_only']:
        system_lifted, system_states = extend_actuator_induction_factors(options, system_lifted, system_states,
                                                                         architecture)

    return system_lifted, system_states



def extend_actuator_induction_factors(options, system_lifted, system_states, architecture):
    return actuator_system.extend_actuator_induction_factors(options, system_lifted, system_states, architecture)

def extend_actuator_support(options, system_lifted, system_states, architecture):
    return actuator_system.extend_actuator_support(options, system_lifted, system_states, architecture)



def extend_aerodynamics(options, system_lifted, system_states, architecture):

    if options['tether']['lift_tether_force']:
        for node in range(1, architecture.number_of_nodes):
            parent = architecture.parent_map[node]
            system_lifted.extend([('f_tether' + str(node) + str(parent), (3, 1))])

    # create the lifted force and moment vars. so that the implicit
    # aerodynamic constraints (with induction correction) can be enforced
    if options['aero']['lift_aero_force']:
        kite_dof = options['kite_dof']
        for kite in architecture.kite_nodes:
            parent = architecture.parent_map[kite]
            system_lifted.extend([('f_aero' + str(kite) + str(parent), (3, 1))])
            if int(kite_dof) == 6:
                system_lifted.extend([('m_aero' + str(kite) + str(parent), (3, 1))])

    # create the induction vars. for all comparison models
    comparison_labels = options['aero']['induction']['comparison_labels']
    if comparison_labels:
        system_lifted, system_states = extend_general_induction(options, system_lifted, system_states, architecture)

    any_act = any(label[:3] == 'act' for label in comparison_labels)
    if any_act:
        system_lifted, system_states = extend_actuator_induction(options, system_lifted, system_states, architecture)

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:
        system_lifted, system_states = extend_vortex_induction(options, system_lifted, system_states, architecture)

    return system_lifted, system_states


def define_bounds(model_system_bounds_options, variables):

    variable_bounds = {}
    for variable_type in list(variables.keys()):
        variable_bounds[variable_type] = {}
        for name in struct_op.subkeys(variables, variable_type):

            variable_bounds[variable_type][name] = {}
            if variable_type in list(model_system_bounds_options.keys()):
                var_name, _ = struct_op.split_name_and_node_identifier(name)  # omit node numbers

                if name in list(model_system_bounds_options[variable_type].keys()):  # check if variable has node bounds
                    variable_bounds[variable_type][name]['lb'] = model_system_bounds_options[variable_type][name][0]
                    variable_bounds[variable_type][name]['ub'] = model_system_bounds_options[variable_type][name][1]

                elif name in list(model_system_bounds_options['x'].keys()): # relevant specifically for the ddl_t/dddl_t control values
                    variable_bounds[variable_type][name]['lb'] = model_system_bounds_options['x'][name][0]
                    variable_bounds[variable_type][name]['ub'] = model_system_bounds_options['x'][name][1]

                elif var_name in list(model_system_bounds_options[variable_type].keys()):  # check if variable has global bounds
                    variable_bounds[variable_type][name]['lb'] = model_system_bounds_options[variable_type][var_name][0]
                    variable_bounds[variable_type][name]['ub'] = model_system_bounds_options[variable_type][var_name][1]

                else:
                    variable_bounds[variable_type][name]['lb'] = -cas.inf
                    variable_bounds[variable_type][name]['ub'] = cas.inf
            else:
                variable_bounds[variable_type][name]['lb'] = -cas.inf
                variable_bounds[variable_type][name]['ub'] = cas.inf

    return variable_bounds

def report_model_bounds(options_object, variables, to_echo_or_latex='echo', latex_dict={}, nan_replacement='--', trial_name='', digits=4, save=False):
    joined_dict = {}

    if (options_object is not None) and (hasattr(options_object, 'flattened_dict')):
        for key in options_object.flattened_dict.keys():
            if 'model.system_bounds' in key:

                system_bounds = options_object.flattened_dict[key]
                for var_name, local_dict in system_bounds.items():

                    entry_name = None
                    for idx in range(len(variables.labels())):
                        var_label = variables.labels()[idx]
                        search_var_type = var_label.split(',')[0][1:]
                        search_var_name = var_label.split(',')[1]
                        search_var_base_name, _ = struct_op.split_name_and_node_identifier(search_var_name)

                        if (search_var_base_name == var_name) and (search_var_type != 'xdot'):
                            entry_name = search_var_type + '.' + var_name

                    among_current_variables = (entry_name is not None)
                    if among_current_variables:

                        if to_echo_or_latex == 'latex':
                            base_entry_name = entry_name
                            # entry_name = '$ ' + base_entry_name + ' $'
                            entry_name = base_entry_name

                        try:
                            joined_dict[entry_name] = {'lower': local_dict['value'][0],
                                                      'upper': local_dict['value'][1],
                                                      'units': local_dict['units'],
                                                      }
                            if to_echo_or_latex == 'latex':
                                # joined_dict[entry_name]['units'] = r'\unit{' + local_dict['units'] + r'}'
                                joined_dict[entry_name][r'\aweboxOptions{model.system_bounds.}'] = r'\aweboxOptions{' + base_entry_name + r'}'
                        except:
                            pass

        if len(list(joined_dict.keys())) > 0:
            caption = 'variable bounds'
            if trial_name is not None:
                caption += ' for ' + trial_name

            string_out = print_op.print_dict_as_table(joined_dict, level='info', to_echo_or_latex=to_echo_or_latex, nan_replacement=nan_replacement, transpose=True, caption=caption, latex_dict=latex_dict, sort_dim=None, digits=digits, repr_type='f', latex_symbolic_in_first_column=True)
            if save:
                save_op.write_string_to_txt_or_tex(string_out, trial_name.replace(' ', '_'),
                                                   to_echo_or_latex=to_echo_or_latex)
        return None


def scale_variable(variables, var_si, scaling):

    var_scaled = variables(copy.deepcopy(var_si))

    for variable_type in list(variables.keys()):
        subkeys = struct_op.subkeys(variables, variable_type)
        for name in subkeys:
            local_si = var_scaled[variable_type, name]
            var_scaled[variable_type, name] = struct_op.var_si_to_scaled(variable_type, name, local_si, scaling)

    return var_scaled


def scale_bounds(variable_bounds, scaling):
    for variable_type in list(variable_bounds.keys()):
        for name in list(variable_bounds[variable_type].keys()):
            for bound_type in ['lb', 'ub']:
                local_si = variable_bounds[variable_type][name][bound_type]
                if isinstance(local_si, float) or isinstance(local_si, int) or not (local_si.shape == scaling[variable_type, name].shape):
                    local_si = local_si * cas.DM.ones(scaling[variable_type, name].shape)

                variable_bounds[variable_type][name][bound_type] = struct_op.var_si_to_scaled(variable_type, name, cas.DM(local_si), scaling)

    return variable_bounds

##
#  @brief Method to construct a system parameter struct out of the model options.
#  @param options The modeling options.
#  @param architecture The system architecture
#  @return sys_params System parameters casadi struct.
def generate_system_parameters(options, architecture):

    parameters_dict = {}

    # extract parametric options
    parametric_options = options['params']
    parameters_dict['theta0'] = struct_op.generate_nested_dict_struct(parametric_options)

    # optimization parameters
    parameters_dict['phi'] = generate_optimization_parameters()
    parameters = cas.struct_symSX([
        cas.entry('theta0', struct=parameters_dict['theta0']),
        cas.entry('phi', struct=parameters_dict['phi'])
    ])

    return parameters, parameters_dict


def generate_optimization_parameters():

    # variable system parameters
    p_dec = cas.struct_symSX([(
        cas.entry('gamma'),  # force homotopy variable
        cas.entry('tau'),   # tether drag homotopy variable
        cas.entry('iota'),  # induction homotopy variable
        cas.entry('psi'),    # power homotopy variable
        cas.entry('eta'),  # nominal landing homotopy variable
        cas.entry('nu'),  # compromised landing homotopy variable
        cas.entry('upsilon'),  # transition homotopy variable
    )])

    optimization_parameters = p_dec

    return optimization_parameters

