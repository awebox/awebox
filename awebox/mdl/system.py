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
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import copy
import awebox.tools.print_operations as print_op
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
        kite_states += [('kappa', (1,1))]
        kite_controls += [('dkappa',(1,1))]

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
                [(kite_multipliers[i][0] + str(n) + str(parent), kite_multipliers[i][1]) for i in range(len(kite_multipliers))])

            system_gc.extend([kite_gc[i] + str(n) + str(parent)
                              for i in range(len(kite_gc))])

        else:
            system_states.extend(
                [(tether_states[i][0] + str(n) + str(parent), tether_states[i][1]) for i in range(len(tether_states))])
            system_controls.extend(
                [(tether_controls[i][0] + str(n) + str(parent), tether_controls[i][1]) for i in range(len(tether_controls))])
            system_multipliers.extend(
                [(tether_multipliers[i][0] + str(n) + str(parent), tether_multipliers[i][1]) for i in range(len(tether_multipliers))])

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
    system_states.extend([('l_t', (1, 1)), ('dl_t', (1, 1))]) # main tether length and speed

    # _energy + main tether length and speed
    if tether_control_var == 'ddl_t':
        system_controls.extend([('ddl_t', (1, 1))])  # main tether acceleration
    elif tether_control_var == 'dddl_t':
        system_states.extend([('ddl_t', (1, 1))]) # main tether acceleration
        system_controls.extend([('dddl_t', (1, 1))])  # main tether jerk
    else:
        raise ValueError('invalid tether control variable chosen')

    if options['integral_outputs']:
        32.0
    else:
        system_states.extend([('e', (1, 1))]) # energy

    # introduce aerodynamics variables
    system_lifted, system_states = extend_aerodynamics(options, system_lifted, system_states, architecture)

    # system state derivatives
    system_derivatives = []
    for i in range(len(system_states)):
        system_derivatives.extend([('d'+system_states[i][0], system_states[i][1])])

    # system parameters
    system_parameters = [('l_s', (1, 1)), ('l_i', (1, 1)), ('diam_s', (1, 1)), ('diam_t', (1, 1)), ('t_f',(1,1))]
    if options['tether']['use_wound_tether']:
        system_parameters += [('l_t_full', (1, 1))]


    # add cross-tether lengths and diameters
    if options['cross_tether'] and len(kite_nodes) > 1:
        for l in architecture.layer_nodes:
            system_parameters.extend(
                [
                    ('l_c{}'.format(l),(1,1)), ('diam_c{}'.format(l),(1,1))
                ]
            )

    # store variables in dict
    system_variables_list = {'xd':system_states,
                             'xddot':system_derivatives,
                             'u':system_controls,
                             'xa':system_multipliers,
                             'theta':system_parameters}

    if not system_lifted == []:
        system_variables_list['xl'] = system_lifted

    return system_variables_list, system_gc

def extend_general_induction(options, system_lifted, system_states, architecture):

    for kite in architecture.kite_nodes:
        system_lifted.extend([('ui' + str(kite), (3, 1))])

    return system_lifted, system_states

def extend_vortex_induction(options, system_lifted, system_states, architecture):

    wingtips = ['ext', 'int']
    wake_nodes = options['aero']['vortex']['wake_nodes']
    rings = wake_nodes - 1

    filaments = vortex_filament_list.expected_number_of_filaments(options, architecture)
    vortex_representation = options['aero']['vortex']['representation']

    for kite in architecture.kite_nodes:
        for ring in range(rings):
            gamma_name = 'wg_' + str(kite) + '_' + str(ring)
            system_lifted.extend([(gamma_name, (1, 1))])

        for wake_node in range(wake_nodes):
            for tip in wingtips:
                coord_name = 'wx_' + str(kite) + '_' + tip + '_' + str(wake_node)
                if vortex_representation == 'state':
                    system_states.extend([(coord_name, (3, 1))])
                elif vortex_representation == 'alg':
                    system_lifted.extend([(coord_name, (3, 1))])
                else:
                    message = 'specified vortex representation ' + vortex_representation + ' is not allowed'
                    awelogger.logger.error(message)
                    raise Exception(message)

    for kite_obs in architecture.kite_nodes:
        for fdx in range(filaments):
            ind_name = 'wu_fil_' + str(fdx) + '_' + str(kite_obs)
            system_lifted.extend([(ind_name, (3, 1))])

        ind_name = 'wu_ind_' + str(kite_obs)
        system_lifted.extend([(ind_name, (3, 1))])

    return system_lifted, system_states

def extend_actuator_induction(options, system_lifted, system_states, architecture):

    comparison_labels = options['aero']['induction']['comparison_labels']

    actuator_comp_labels = []
    for label in comparison_labels:
        if label[:3] == 'act':
            actuator_comp_labels += [label[4:]]

    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        system_lifted.extend([('local_a' + str(kite) + str(parent), (1, 1))])
        system_lifted.extend([('varrho' + str(kite) + str(parent), (1, 1))])
        system_lifted.extend([('psi' + str(kite) + str(parent), (1, 1))])
        system_lifted.extend([('cospsi' + str(kite) + str(parent), (1, 1))])
        system_lifted.extend([('sinpsi' + str(kite) + str(parent), (1, 1))])

    for layer_node in architecture.layer_nodes:

        for label in actuator_comp_labels:
            if label[0] == 'q':
                system_lifted.extend([('a_' + label + str(layer_node), (1, 1))])
            elif label[0] == 'u':
                system_states.extend([('a_' + label + str(layer_node), (1, 1))])

            if label == 'qasym':
                system_lifted.extend([('acos_' + label + str(layer_node), (1, 1))])
                system_lifted.extend([('asin_' + label + str(layer_node), (1, 1))])

            if label == 'uasym':
                system_states.extend([('acos_' + label + str(layer_node), (1, 1))])
                system_states.extend([('asin_' + label + str(layer_node), (1, 1))])

        system_lifted.extend([('bar_varrho' + str(layer_node), (1, 1))])

        system_lifted.extend([('act_dcm' + str(layer_node), (9, 1))])
        system_lifted.extend([('n_vec_length' + str(layer_node), (1, 1))])

        system_lifted.extend([('wind_dcm' + str(layer_node), (9, 1))])
        system_lifted.extend([('u_vec_length' + str(layer_node), (1, 1))])
        system_lifted.extend([('z_vec_length' + str(layer_node), (1, 1))])

        system_lifted.extend([('gamma' + str(layer_node), (1, 1))])
        system_lifted.extend([('g_vec_length' + str(layer_node), (1, 1))])
        system_lifted.extend([('cosgamma' + str(layer_node), (1, 1))])
        system_lifted.extend([('singamma' + str(layer_node), (1, 1))])

    return system_lifted, system_states

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

def define_bounds(options, variables):

    variable_bounds = {}
    for variable_type in list(variables.keys()):
        variable_bounds[variable_type] = {}
        for name in struct_op.subkeys(variables,variable_type):

            variable_bounds[variable_type][name] = {}
            if variable_type in list(options.keys()):
                var_name, _ = struct_op.split_name_and_node_identifier(name) # omit node numbers
                if name in list(options[variable_type].keys()): # check if variable has node bounds
                    variable_bounds[variable_type][name]['lb'] = options[variable_type][name][0]
                    variable_bounds[variable_type][name]['ub'] = options[variable_type][name][1]
                elif var_name in list(options[variable_type].keys()): # check if variable has global bounds
                    variable_bounds[variable_type][name]['lb'] = options[variable_type][var_name][0]
                    variable_bounds[variable_type][name]['ub'] = options[variable_type][var_name][1]
                else:
                    variable_bounds[variable_type][name]['lb'] = -cas.inf
                    variable_bounds[variable_type][name]['ub'] = cas.inf
            else:
                variable_bounds[variable_type][name]['lb'] = -cas.inf
                variable_bounds[variable_type][name]['ub'] = cas.inf
    return variable_bounds

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
                variable_bounds[variable_type][name][bound_type] = struct_op.var_si_to_scaled(variable_type, name, local_si, scaling)

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

    # define vortex induction linearization parameters
    use_vortex_linearization = options['aero']['vortex']['use_linearization']
    if use_vortex_linearization:
        vortex_linearization_list, _ = generate_structure(options, architecture)
        vortex_linearization_params, _ = struct_op.generate_variable_struct(vortex_linearization_list)
        parameters_dict['lin'] = vortex_linearization_params

    # generate nested casadi structure for system parameters
    if use_vortex_linearization:
        parameters = cas.struct_symSX([
            cas.entry('theta0', struct=parameters_dict['theta0']),
            cas.entry('phi', struct=parameters_dict['phi']),
            cas.entry('lin', struct=parameters_dict['lin'])
        ])
    else:
        parameters = cas.struct_symSX([
            cas.entry('theta0', struct= parameters_dict['theta0']),
            cas.entry('phi', struct= parameters_dict['phi'])
        ])

    return parameters, parameters_dict

def generate_optimization_parameters():

    # variable system parameters
    p_dec = cas.struct_symSX([(
        cas.entry('gamma'), # force homotopy variable
        cas.entry('tau'),   # tether drag homotopy variable
        cas.entry('iota'),  # induction homotopy variable
        cas.entry('psi'),    # power homotopy variable
        cas.entry('eta'),  # nominal landing homotopy variable
        cas.entry('nu'),  # compromised landing homotopy variable
        cas.entry('upsilon'),  # transition homotopy variable
    )])

    optimization_parameters = p_dec

    return optimization_parameters

