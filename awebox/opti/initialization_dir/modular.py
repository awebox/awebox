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
modular initial guess generation using motion primitives
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017/18)
'''

import copy
import numpy as np
import casadi.tools as ct
import casadi as cas
import collections
import awebox.tools.vector_operations as vect_op
import math
from awebox.logger.logger import Logger as awelogger
import awebox.opti.initialization_dir.interpolation as interp
import awebox.tools.struct_operations as struct_op
import awebox.mdl.wind as wind
import awebox.viz.tools as visualization_tools

def get_initial_guess(nlp, model, formulation, options):
    """
    Assemble an initial guess based on the problem formulation
    :param nlp: nlp formulation
    :param model: dynamic model
    :param formulation: problem formulation
    :param options: initialization options
    :return: initial guess in scaled units
    """
    # build SI initial guess
    V_init_si = __build_si_initial_guess(nlp, model, formulation, options)

    # check for NaNs in intial guess
    if True in np.isnan(np.array(V_init_si.cat)):
        raise ValueError('NaN detected in V_init_si')

    # scale initial guess
    V_init = struct_op.si_to_scaled(V_init_si, model.scaling)
    return V_init

def __build_si_initial_guess(nlp, model, formulation, options):
    """
    Assemble initial guess in SI units
    :param nlp: nlp formulation
    :param model: dynamic model
    :param formulation: problem formulation
    :param options: initialization options
    :return: initial guess in SI units
    """

    # logging
    awelogger.logger.info('build si initial guess...')

    # compute initial guess
    primitives = __set_primitives(options, model)
    initial_guess_schedule = __build_initial_guess_schedule(options, primitives)
    V_init = __generate_guess_from_schedule(initial_guess_schedule, primitives, nlp, model, options, formulation)

    return V_init

def __check_configuration_feasibility(configuration, options, configuration_type, model):

    ## check for simple position
    if configuration_type == 'simple_pos':

        # get parameters
        angular_looping_velocity = configuration['angular_looping_velocity']
        if options['model']['architecture'] == (1,1): #todo: still correct?
            tether_length = options['xd']['l_t']
        else:
            tether_length = options['theta']['l_s']
        cone_angle = configuration['cone_angle']
        acc_max = options['acc_max']

        groundspeed = options['groundspeed']

        # check for max acceleration
             # omega = u_a / r
             # acc = omega * u_a = u_a**2 / r = u_a**2 / (sin(phi)*l_t) < acc_max
             # u_a**2 / acc_max / l_t < sin(phi)
             # arcsin(...) < phi
             # define initial configuration
        acc = angular_looping_velocity * groundspeed
        if acc > acc_max:
            cone_angle = np.arcsin(groundspeed**2 / acc_max / tether_length)
            awelogger.logger.warning('Warning: configuration in initial guess exceeds maximum acceleration. Changing cone_angle to correspond to maximum acceleration.')
            configuration['cone_angle'] = cone_angle

        # check for min radius
            # r = sin(phi) * l_t > r_min
            # sin(phi) > r_min / l_t
            # phi > arcsin(...)
        min_rel_radius = options['min_rel_radius']
        b_ref = options['sys_params_num']['geometry']['b_ref']
        min_radius = min_rel_radius * b_ref
        radius = np.sin(cone_angle * np.pi / 180.) * tether_length
        if radius > min_radius:
            cone_angle = np.arcsin(min_radius / tether_length)
            awelogger.logger.warning('Warning: configuaration has radius that is smaller than minmum radius. Changing cone_angle to correspond to minimum radius.')
            configuration['cone_angle'] = cone_angle

    return configuration

def __check_primitives_for_feasibility(primitive, options, model):

    # todo: does this actually do anything?

    configurations = [primitive['initial_configuration'],primitive['terminal_configuration']]
    for configuration in configurations:
        if configuration['type'] == 'simple_pos':
            configuration = __check_configuration_feasibility(configuration['configuration'], options, configuration['type'], model)

    return primitive

def __set_primitives(options, model):
    """
    Create a dictionary of arguments that define motion primitives
    :return: primitive arguments dictionary
    """

    # initialize primitives dictionary
    primitives = {}

    # get trajectory type
    trajectory_type = options['type']

    ### build motion primitives

    ## build motion primitive for lift mode

    if trajectory_type == 'power_cycle' or trajectory_type == 'tracking':

        power_cycle_args = {}

        # get fixed parameters
        number_of_loopings = options['windings']
        groundspeed = options['groundspeed']
        if options['model']['architecture'] == (1,1):
            cone_angle = options['max_cone_angle_single'] * np.pi/180.
        else:
            cone_angle = options['max_cone_angle_multi'] * np.pi/180.
        inclination = options['inclination_deg'] * np.pi/180.

        # set rest of parameters
        normed_times = (0, 1)
        if options['model']['architecture'] == (1,1):
            tether_length = options['xd']['l_t']
        else:
            tether_length = options['theta']['l_s']
        radius = np.sin(cone_angle) * tether_length
        angular_looping_velocity = groundspeed/radius

        initial_configuration = {}
        initial_configuration['l_t'] = options['xd']['l_t']
        initial_configuration['inclination'] = inclination
        initial_configuration['cone_angle'] = cone_angle
        initial_configuration['angular_looping_velocity'] = angular_looping_velocity
        initial_configuration['upstream_node_velocity'] = 0.

        # define terminal configuration
        terminal_configuration = copy.deepcopy(initial_configuration)

        # build primitive
        power_cycle_args['type'] = 'goto'
        power_cycle_args['normed_times'] = normed_times
        power_cycle_args['number_of_loopings'] = number_of_loopings
        power_cycle_args['initial_configuration'] = {}
        power_cycle_args['initial_configuration']['type'] = 'simple_pos'
        power_cycle_args['initial_configuration']['configuration'] = initial_configuration
        power_cycle_args['terminal_configuration'] = {}
        power_cycle_args['terminal_configuration']['type'] = 'simple_pos'
        power_cycle_args['terminal_configuration']['configuration'] = terminal_configuration

        # add primitive to primitive args
        if trajectory_type == 'power_cycle':
            primitives['power_cycle'] = power_cycle_args
        elif trajectory_type == 'tracking':
            primitives['tracking'] = power_cycle_args

    ## build motion primitive for transition
    if trajectory_type == 'transition':
        transition_args = {}

        # build primitive
        transition_args['type'] = 'goto'
        transition_args['normed_times'] = (0,1)
        transition_args['number_of_loopings'] = 5
        transition_args['initial_configuration'] = {}
        transition_args['initial_configuration']['type'] = 'param'
        transition_args['initial_configuration']['configuration'] = 'initial'
        transition_args['terminal_configuration'] = {}
        transition_args['terminal_configuration']['type'] = 'param'
        transition_args['terminal_configuration']['configuration'] = 'terminal'

        # add primitive to primitive args
        primitives['transition'] = transition_args

    ## build motion primitive for nominal_landing
    if trajectory_type in ['nominal_landing', 'compromised_landing']:
        landing_args = {}

        # build terminal configuration
        terminal_configuration = {}
        terminal_configuration['angular_looping_velocity'] = 0.
        terminal_configuration['l_t'] = 50.
        terminal_configuration['inclination'] = 30. * np.pi / 180.
        terminal_configuration['cone_angle'] = 60. * np.pi / 180.
        terminal_configuration['upstream_node_velocity'] = 0.

        # build primitive
        landing_args['type'] = 'goto'
        landing_args['normed_times'] = (0,1)
        landing_args['number_of_loopings'] = 0.
        landing_args['initial_configuration'] = {}
        landing_args['initial_configuration']['type'] = 'param'
        landing_args['initial_configuration']['configuration'] = 'initial'
        landing_args['terminal_configuration'] = {}
        landing_args['terminal_configuration']['type'] = 'simple_pos'
        landing_args['terminal_configuration']['configuration'] = terminal_configuration

        # add primitive to primitive args
        primitives['landing'] = landing_args

    ## build motion primitive for launch
    if trajectory_type in ['launch']:

        launch_args = {}

        # build initial configuration
        initial_configuration = {}
        initial_configuration['angular_looping_velocity'] = np.pi
        initial_configuration['l_t'] = 700.
        initial_configuration['inclination'] = 20. * np.pi / 180.
        initial_configuration['cone_angle'] = 60. * np.pi / 180.
        initial_configuration['upstream_node_velocity'] = 30.

        # build primitive
        launch_args['type'] = 'goto'
        launch_args['normed_times'] = (0,1)
        launch_args['number_of_loopings'] = 0.
        launch_args['initial_configuration'] = {}
        launch_args['initial_configuration']['type'] = 'simple_pos'
        launch_args['initial_configuration']['configuration'] = initial_configuration
        launch_args['terminal_configuration'] = {}
        launch_args['terminal_configuration']['type'] = 'param'
        launch_args['terminal_configuration']['configuration'] = 'terminal'

        # add primitive to primitive args
        primitives['launch'] = launch_args

    # check for feasibility of all primitives
    for primitive in list(primitives.keys()):
        primitives[primitive] = __check_primitives_for_feasibility(primitives[primitive], options, model)

    return primitives

def __estimate_t_f(primitives, options):
    """
    Estimate final time based on trajectory type
    :param trajectory_type: type of the trajectory that is being computed
    :return: final time estimate
    """

    # get trajectory type
    trajectory_type = options['type']

    if trajectory_type == 'power_cycle':

        # get parameters
        number_of_loopings = primitives['power_cycle']['number_of_loopings']
        angular_looping_velocity = (primitives['power_cycle']['initial_configuration']['configuration']['angular_looping_velocity'] + primitives['power_cycle']['terminal_configuration']['configuration']['angular_looping_velocity'])/2.

        # compute final time
        t_f = 2 * np.pi * number_of_loopings / angular_looping_velocity

    else:
        t_f = 50.

    return t_f

def __build_initial_guess_schedule(options, primitives):
    """
    Build a schedule for the initial guess containing the motion primitives
    :param trajectory_type: type of trajectory that is being computed
    :return: initial guess schedule dictionary containing motion primitives
    """

    # get estimate for final time
    t_f = __estimate_t_f(primitives, options)

    # initialize dict
    initial_guess_schedule = {}
    initial_guess_schedule['t_f'] = t_f
    initial_guess_schedule['primitives'] = []

    # get trajectory type
    trajectory_type = options['type']

    # add primitives to schedule
    if trajectory_type == 'power_cycle':
        initial_guess_schedule['primitives'] += ['power_cycle']
    if trajectory_type in ['transition']:
        initial_guess_schedule['primitives'] += ['transition']
    if trajectory_type in ['nominal_landing', 'compromised_landing']:
        initial_guess_schedule['primitives'] += ['landing']
    if trajectory_type in ['launch']:
        initial_guess_schedule['primitives'] += ['launch']

    return initial_guess_schedule

def __generate_guess_from_schedule(initial_guess_schedule, primitives, nlp, model, initialization_options, formulation):
    """
    Take an initial guess schedule and generate an initial guess out of the motion primitives it contains
    :param initial_guess_schedule: dictionary containing motion primitives
    :param primitives: dictionary containing arguments of motion primitives
    :param nlp: nlp formulation
    :param model: system model
    :param initialization_options: initialization options
    :return:initial guess in SI units
    """

    # get interpolation scheme
    interpolation_scheme = initialization_options['interpolation_scheme']

    # initialize struct
    V_init = nlp.V
    V_init = V_init(0.0)

    # read out estimate for final time
    t_f = initial_guess_schedule['t_f']

    # add xa to initial guess
    if 'xa' in list(V_init.keys()):
        V_init['xa', :] = 1.
    if 'coll_var' in list(V_init.keys()):
        V_init['coll_var',:,:,'xa'] = 1.

    # add specified initial values for system parameters to initial guess
    for name in set(struct_op.subkeys(model.variables, 'theta')) - set(['t_f']):
        if name in list(initialization_options['theta'].keys()):
            V_init['theta', name] = initialization_options['theta'][name]
        else:
            raise ValueError("please specify an initial value for variable '" + name + "' of type 'theta'")

    # add initial time guess (same for both intervals in case of phase fixing)
    V_init['theta', 't_f'] = t_f

    # add initial guess for homotopy parameters
    for name in list(model.parameters_dict['phi'].keys()):
        V_init['phi', name] = 1.

    # add motion primitives to initial guess
    for primitive in initial_guess_schedule['primitives']:
        V_init = __add_primitive_to_guess(V_init, primitives[primitive], nlp, model, t_f, initialization_options, formulation, interpolation_scheme)

    return V_init

def __add_primitive_to_guess(V_init, primitive, nlp, model, t_f, initialization_options, formulation, interpolation_scheme):
    """
    Add primitive that is scheduled in the initial guess schedule to the initial guess at the correct time indeces
    :param V_init: partly assembled initial guess
    :param primitives: arguments of motion primitives
    :param nlp: nlp formulation
    :param model: system model
    :param t_f: final time estimate
    :param initialization_options: initialization options
    :return: partly assembled initial guess
    """

    # generate parameters that are used for interpolation
    time_grid_parameters = __generate_time_grid_parameters(t_f, primitive, nlp)
    interpolation_parameters = __generate_interpolation_parameters(time_grid_parameters, model, initialization_options, primitive, formulation, nlp, interpolation_scheme)

    # generate initial guess values based on parameters
    V_init = __generate_vals(V_init, primitive, nlp, model, time_grid_parameters, interpolation_parameters, interpolation_scheme, initialization_options)

    return V_init

def __generate_interpolation_parameters(time_grid_parameters, model, initialization_options, primitive, formulation, nlp, interpolation_scheme):
    """
    Generate the parameters used in interpolation, such as polynomial coefficients and configurations
    :param time_grid_parameters: parameters defining the discrete time grid depending on discretization scheme
    :param model: dynamic model
    :param initialization_options: initialization options
    :param primitive: arguments describing motion primitive
    :return: interpolation parameters
    """

    # read out inputs
    kite_nodes = model.architecture.kite_nodes
    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    l_s = initialization_options['theta']['l_s']
    l_i = initialization_options['theta']['l_i']

    # get boundary conditions for parameterization
    conf_0, conf_f = __get_boundary_configurations(primitive, model, initialization_options, formulation, nlp)
    boundary_conditions = __get_boundary_conditions(conf_0, conf_f, model)

    # generate polynomial coefficients for s curves
    polynomial_coeff = {}
    polynomial_coeff['l_t'] = __parameterize_curve(boundary_conditions['l_t'], time_grid_parameters, interpolation_scheme)

    for node in range(1, number_of_nodes):
        parameter_list = []
        parent = parent_nodes[node]
        node_str = str(node) + str(parent)
        if (node in kite_nodes):
            parameter_list += ['Omega', 'Phi']
            if node == 1:
                parameter_list += ['inclination']
        else:
            parameter_list += ['inclination', 'azimuth']
        for parameter in parameter_list:
            polynomial_coeff[parameter + node_str] = __parameterize_curve(boundary_conditions[parameter + node_str], time_grid_parameters, interpolation_scheme)

    # create configuration dict
    configurations = {}
    configurations['conf_0'] = conf_0
    configurations['conf_f'] = conf_f
    configurations['l_i'] = l_i
    configurations['l_s'] = l_s
    configurations['number_of_looping'] = primitive['number_of_loopings']

    # create interpolation parameters dict
    interpolation_parameters = {}
    interpolation_parameters['configurations'] = configurations
    interpolation_parameters['polynomial_coeff'] = polynomial_coeff

    # build params to create wind object
    params = ct.struct_symSX([ct.entry('theta0', struct=model.parameters_dict['theta0'])])

    params_num = params(0.0)
    for name in list(initialization_options['sys_params_num']['wind'].keys()):
        if type(initialization_options['sys_params_num']['wind'][name]) == dict:
            for sub_name in list(initialization_options['sys_params_num']['wind'][name].keys()):
                params_num['theta0','wind', name, sub_name] = initialization_options['sys_params_num']['wind'][name][sub_name]
        else:
            params_num['theta0','wind', name] = initialization_options['sys_params_num']['wind'][name]

    interpolation_parameters['wind'] = wind.Wind(model.wind_options, params_num)

    return interpolation_parameters

def __get_boundary_conditions(conf_0, conf_f, model):
    """
    Derive boundary conditions from initial and terminal configuration
    :param conf_0: initial configuration
    :param conf_f: terminal configuration
    :param model: dynamic model
    :return: boundary conditions
    """

    # read out inputs
    number_of_nodes = model.architecture.number_of_nodes
    kite_nodes = model.architecture.kite_nodes

    # compute tether conditions
    tether_conditions = {}
    tether_conditions['p_hat_0'] = conf_0['l_t']
    tether_conditions['dp_hat_0'] = conf_0['dl_t']
    tether_conditions['ddp_hat_0'] = conf_0['ddl_t']
    tether_conditions['p_hat_f'] = conf_f['l_t']
    tether_conditions['dp_hat_f'] = conf_f['dl_t']
    tether_conditions['ddp_hat_f'] = conf_f['ddl_t']

    # initialize and assemble dictionary for the rest of the boundary conditions
    boundary_conditions = {}
    boundary_conditions['l_t'] = tether_conditions

    for node in range(1,number_of_nodes):
        if (node in kite_nodes):  # kite notes parameterized with radial angle Omega and cone angle Phi
            parameter_list = ['Omega', 'Phi']
            if node == 1:
                parameter_list += ['inclination']
            for parameter_str in parameter_list:
                boundary_conditions = __get_conditions(parameter_str, node, boundary_conditions, conf_0, conf_f, model)
        else:  # intermediate nodes parameterized with inclination and azimuth angle
            for parameter_str in ['inclination', 'azimuth']:
                boundary_conditions = __get_conditions(parameter_str, node, boundary_conditions, conf_0, conf_f, model)

    return boundary_conditions

def __get_conditions(parameter_str, node, boundary_conditions, conf_0, conf_f, model):
    """
    Translate configurations into boundary conditions for a specific parameter
    :param parameter_str: name of the parameter
    :param node: number of node
    :param boundary_conditions: partly assembled boundary conditions
    :param conf_0: initial configuration
    :param conf_f: terminal configuration
    :param model: dynamic modeland
    :return: partly assembled boundary conditions
    """

    # read out inputs
    parent_nodes = model.architecture.parent_map
    parent = parent_nodes[node]
    node_str = str(node) + str(parent)

    # initialize and assemble boundary conditions dict
    boundary_conditions[parameter_str + node_str] = {}
    for derivative in ['', 'd', 'dd']:  # go thorugh all derivatives of parameter
        boundary_conditions[parameter_str + node_str][derivative + 'p_hat_0'] = conf_0[derivative + parameter_str + node_str]
        boundary_conditions[parameter_str + node_str][derivative + 'p_hat_f'] = conf_f[derivative + parameter_str + node_str]

    return boundary_conditions

def __get_boundary_configurations(primitive, model, initialization_options, formulation, nlp):
    """
    Get boundary configuration from motion primitive arguments
    :param primitives: motion primitive arguments
    :param model: dynamic model
    :param initialization_options: initialization options
    :return: boundary configurations
    """

    # read out inputs
    initial_configuration_description = primitive['initial_configuration']['configuration']
    initial_configuration_type = primitive['initial_configuration']['type']
    terminal_configuration_description = primitive['terminal_configuration']['configuration']
    terminal_configuration_type = primitive['terminal_configuration']['type']

    # get configurations based on description
    conf_0 = __interpret_specific_configuration_description(initial_configuration_description, initial_configuration_type, model, initialization_options, formulation, nlp)
    conf_f = __interpret_specific_configuration_description(terminal_configuration_description, terminal_configuration_type, model, initialization_options, formulation, nlp)

    # add multiples of 2 pi for loopings
    kite_nodes = model.architecture.kite_nodes
    parent_nodes = model.architecture.parent_map
    number_of_loopings = primitive['number_of_loopings']
    for node in kite_nodes:
        parent = parent_nodes[node]
        conf_f['Omega' + str(node) + str(parent)] += 2*np.pi*number_of_loopings

    return conf_0, conf_f

def __interpret_specific_configuration_description(configuration_description, description_type, model, initialization_options, formulation, nlp):
    """
    Take a specific configuration description and translate it into a configuration depending on the type of description
    :param configuration_description: specific configuration description
    :param model: dynamic model
    :param initialization_options: initialization options
    :return: configuration corresponding to description
    """

    # choose interpretation function based on description type
    if description_type == 'simple_pos':
        conf = __get_configuration_from_simple_pos(configuration_description, model, initialization_options)
    elif description_type == 'param':
        conf = __get_configuration_from_param(formulation, configuration_description, model, initialization_options, nlp)
    elif description_type == 'node_pos':
        conf = configuration_description

    return conf

def __generate_conf_struct(model):
    """
    Generate a casadi struct for the symbolic configuration
    :param model: system model
    :return: configuration struct
    """

    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    kite_nodes = model.architecture.kite_nodes

    interpolation_list = []
    interpolation_list += [('l_t', 1)]

    for node in range(1, number_of_nodes):
        parent = parent_nodes[node]
        node_str = str(node) + str(parent)
        interpolation_list += [('q' + node_str, 3)]
        if node in kite_nodes:
            interpolation_list += [('Omega' + node_str, 1)]
            interpolation_list += [('Phi' + node_str, 1)]
            if node == 1:
                interpolation_list += [('inclination' + node_str, 1)]
        else:
            interpolation_list += [('inclination' + node_str, 1)]
            interpolation_list += [('azimuth' + node_str, 1)]

        conf_struct = ct.struct_symSX([ct.entry(interpolation_list[i][0], shape=interpolation_list[i][1])
                        for i in range(len(interpolation_list))])

        dconf_struct = ct.struct_symSX([ct.entry('d' + interpolation_list[i][0], shape=interpolation_list[i][1])
                        for i in range(len(interpolation_list))])

        ddconf_struct = ct.struct_symSX([ct.entry('dd' + interpolation_list[i][0], shape=interpolation_list[i][1])
                        for i in range(len(interpolation_list))])

        confs_struct = ct.struct_symSX([
                    ct.entry('var', struct = conf_struct),
                    ct.entry('dvar', struct = dconf_struct),
                    ct.entry('ddvar', struct = ddconf_struct)
                ])

    return confs_struct

def __get_configuration_from_simple_pos(description, model, initialization_options):

    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    parent_nodes[0] = 0
    layer_siblings = __get_layer_siblings(model)
    kite_nodes = model.architecture.kite_nodes
    l_s = initialization_options['theta']['l_s']
    l_i = initialization_options['theta']['l_i']
    angular_looping_velocity = description['angular_looping_velocity']
    upstream_node_velocity = description['upstream_node_velocity']

    l_t = description['l_t']
    dl_t = 0.0
    ddl_t = 0.0
    inclination = description['inclination']
    dinclination = 0.0
    ddinclination = 0.0
    secondary_inclination = 0.0  #todo: usefull to set in interface?
    dsecondary_inclination = 0.0
    ddsecondary_inclination = 0.0
    azimuth = 0.0
    dazimuth = 0.0
    ddazimuth = 0.0
    cone_angle = description['cone_angle']
    dcone_angle = 0.0
    ddcone_angle = 0.0

    # define functions

    sconf = __generate_conf_struct(model)
    interpol_fun = {}
    for node in range(1, number_of_nodes):
        parent = parent_nodes[node]
        grandparent = parent_nodes[parent]
        grandgrandparent = parent_nodes[grandparent]
        node_str = str(node) + str(parent)
        parent_str = str(parent) + str(grandparent)
        grandparent_str = str(grandparent) + str(grandgrandparent)
        q_parent = np.zeros([3, 1])
        q_grandparent = np.zeros([3, 1])
        e_hat_x = ct.SX.sym('e_hat_x', 3, 1)
        e_hat_x[1] = 0.0
        if parent != 0:
            q_parent = sconf['var','q' + parent_str]
            if grandparent != 0:
                q_grandparent = sconf['var','q' + grandparent_str]
        if node in kite_nodes:

            # get rotational axis
            if node == 1:
                e_hat_x[0] = np.cos(sconf['var','inclination' + node_str])
                e_hat_x[2] = np.sin(sconf['var','inclination' + node_str])
                tether_length = sconf['var','l_t']
            else:
                tether_length = l_s
                e_hat_x = vect_op.normalize(q_parent - q_grandparent)
            radius = tether_length * np.sin(sconf['var','Phi' + node_str])
            l_x = tether_length * np.cos(sconf['var','Phi' + node_str])
            e_hat_y = vect_op.yhat() #todo: what if azimuth is non-zero?
            e_hat_z = vect_op.cross(e_hat_x, e_hat_y)
            e_hat_r = e_hat_z * np.sin(sconf['var','Omega' + node_str]) + e_hat_y * np.cos(sconf['var','Omega' + node_str])
            q = e_hat_x * l_x + e_hat_r * radius

        else:
            if node == 1:
                tether_length = sconf['var','l_t']

            else:
                tether_length = l_i
                #todo: what if intermediate inclination is not 0?

            sinclination = sconf['var','inclination' + node_str]
            e_hat_x[0] = np.cos(sinclination)
            e_hat_x[2] = np.sin(sinclination)
            q = e_hat_x * tether_length

        q += q_parent
        dq = ct.mtimes(ct.jacobian(q, sconf['var']), sconf['dvar'])
        ddq = ct.mtimes(ct.jacobian(dq, ct.vertcat(sconf['var'], sconf['dvar'])), ct.vertcat(sconf['dvar'], sconf['ddvar']))

        interpol_fun['q' + node_str] = ct.Function('q' + node_str, [sconf], [q, dq, ddq])

    # fill nconf
    nconf = sconf(0.0)
    variables_to_add = collections.OrderedDict()
    variables_to_add['l_t'] = [l_t, dl_t, ddl_t]

    nconf = __add_var_to_nconf(nconf, 'l_t', variables_to_add)

    for node in range(1, number_of_nodes):
        parent = parent_nodes[node]
        node_str = str(node) + str(parent)
        if node in kite_nodes:
            Omega = __get_Omega(layer_siblings, node, parent)
            dOmega = angular_looping_velocity
            ddOmega = 0.0
            variables_to_add['Omega' + node_str] = [Omega, dOmega, ddOmega]
            if node == 1:
                variables_to_add['inclination' + node_str] = [inclination, dinclination, ddinclination]
            variables_to_add['Phi' + node_str] = [cone_angle, dcone_angle, ddcone_angle]
        else:
            variables_to_add['azimuth' + node_str] = [azimuth, dazimuth, ddazimuth]
            if node == 1:
                variables_to_add['inclination' + node_str] = [inclination, dinclination, ddinclination]
            else:
                variables_to_add['inclination' + node_str] = [secondary_inclination, dsecondary_inclination, ddsecondary_inclination]

        for variable in list(variables_to_add.keys()):
            if variable not in ['q', 'l_t', 'inclination']:
                nconf = __add_var_to_nconf(nconf, variable, variables_to_add)

        q, dq, ddq = interpol_fun['q' + node_str](nconf)
        variables_to_add['q' + node_str] = [q, dq, ddq]
        nconf = __add_var_to_nconf(nconf, 'q' + node_str, variables_to_add)

    # put nconf together in conf
    conf = struct_op.dissolve_top_layer_of_struct(nconf)

    conf['dq10'] += ct.DM([upstream_node_velocity, 0, 0])
    conf['dq21'] += ct.DM([upstream_node_velocity, 0, 0])
    conf['dq31'] += ct.DM([upstream_node_velocity, 0, 0])
    tether_vec = conf['q10']
    conf['dl_t'] += ct.mtimes(conf['dq10'].T, (tether_vec/ ct.norm_2(tether_vec)))
    xx = conf['q10'][0]
    zz = conf['q10'][2]
    conf['dinclination10'] += - (zz / (zz**2 + xx**2)) * conf['dq10'][0]

    return conf

def __add_var_to_nconf(nconf, var_str, variables_to_add):

    var, dvar, ddvar = variables_to_add[var_str]

    nconf['var', var_str] = var
    nconf['dvar', 'd' + var_str] = dvar
    nconf['ddvar', 'dd' + var_str] = ddvar

    return nconf

def __get_configuration_from_param(formulation, description, model, initialization_options, nlp):  #todo: debug!

    xi_dict = formulation.xi_dict
    initial_or_terminal = description
    plot_dict = xi_dict['plot_dict_pickle_' + initial_or_terminal]

    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    parent_nodes[0] = 0
    kite_nodes = model.architecture.kite_nodes

    conf = {}
    conf['q00'] = np.zeros([3,1])
    conf['dq00'] = np.zeros([3,1])
    conf['ddq00'] = np.zeros([3,1])

    sconf = __generate_conf_struct(model)

    # build functions

    interpol_fun = {}
    for node in range(1, number_of_nodes):
        parent = parent_nodes[node]
        grandparent = parent_nodes[parent]
        grandgrandparent = parent_nodes[grandparent]
        node_str = str(node) + str(parent)
        parent_str = str(parent) + str(grandparent)
        grandparent_str = str(grandparent) + str(grandgrandparent)
        q_parent = np.zeros([3, 1])
        q_grandparent = np.zeros([3, 1])
        if parent != 0:
            q_parent = sconf['var','q' + parent_str]
            if grandparent != 0:
                q_grandparent = sconf['var','q' + grandparent_str]
        if node in kite_nodes:
            if node == 1:
                raise ValueError('ERROR: for single kites, parameterized initial conditions are not yet defined.') #todo: how to get rotational axis for single kites?
            else:
                x_t = sconf['var','q' + node_str] - q_parent
                e_hat_x = vect_op.normalize(q_parent - q_grandparent)
            e_hat_y = vect_op.normed_cross(e_hat_x, vect_op.zhat())
            e_hat_z = vect_op.normed_cross(e_hat_y, e_hat_x)
            e_hat_Omega = ct.mtimes(e_hat_y.T, x_t)*e_hat_y + ct.mtimes(e_hat_z.T, x_t)*e_hat_z
            Phi = vect_op.angle_between(e_hat_x, x_t)
            dPhi = ct.mtimes(ct.jacobian(Phi, sconf['var']), sconf['dvar'])
            ddPhi = ct.mtimes(ct.jacobian(dPhi, ct.vertcat(sconf['var'], sconf['dvar'])), ct.vertcat(sconf['dvar'], sconf['ddvar']))
            interpol_fun['Phi' + node_str] = ct.Function('Phi' + node_str, [sconf], [Phi, dPhi, ddPhi])
            Omega = vect_op.angle_between(e_hat_Omega, e_hat_y)
            dOmega = ct.mtimes(ct.jacobian(Omega, sconf['var']), sconf['dvar'])
            ddOmega = ct.mtimes(ct.jacobian(dOmega, ct.vertcat(sconf['var'],sconf['dvar'])), ct.vertcat(sconf['dvar'],sconf['ddvar']))
            interpol_fun['Omega' + node_str] = ct.Function('Omega' + node_str, [sconf], [Omega, dOmega, ddOmega])
        else:
            x_t = sconf['var','q' + node_str] - q_parent
            e_hat_xy = vect_op.xhat()*ct.mtimes(x_t.T, vect_op.xhat()) + vect_op.yhat()*ct.mtimes(x_t.T, vect_op.yhat())
            e_hat_xz = vect_op.xhat()*ct.mtimes(x_t.T, vect_op.xhat()) + vect_op.zhat()*ct.mtimes(x_t.T, vect_op.zhat())
            inclination = vect_op.angle_between(e_hat_xz, vect_op.xhat())
            dinclination = ct.mtimes(ct.jacobian(inclination, sconf['var']), sconf['dvar'])
            ddinclination = ct.mtimes(ct.jacobian(dinclination, ct.vertcat(sconf['var'], sconf['dvar'])), ct.vertcat(sconf['dvar'], sconf['ddvar']))
            interpol_fun['inclination' + node_str] = ct.Function('inclination' + node_str, [sconf], [inclination, dinclination, ddinclination])
            azimuth = vect_op.angle_between(e_hat_xy, vect_op.xhat())
            dazimuth = ct.mtimes(ct.jacobian(azimuth, sconf['var']), sconf['dvar'])
            ddazimuth = ct.mtimes(ct.jacobian(dazimuth, ct.vertcat(sconf['var'],sconf['dvar'])), ct.vertcat(sconf['dvar'], sconf['ddvar']))
            interpol_fun['azimuth' + node_str] = ct.Function('azimuth' + node_str, [sconf], [azimuth, dazimuth, ddazimuth])

    # fill nconf

    nconf = sconf(0.0)
    variables_to_add = collections.OrderedDict()

    parameterization_dict = __get_parameterization_dict(formulation, initial_or_terminal, initialization_options, nlp, model)
    nconf = __set_state_in_nconf(nconf, plot_dict, parameterization_dict, 'l_t')
    for node in range(1,number_of_nodes):
        parent = parent_nodes[node]
        grandparent = parent_nodes[parent]
        node_str = str(node) + str(parent)
        parent_str = str(parent) + str(grandparent)
        nconf = __set_state_in_nconf(nconf, plot_dict, parameterization_dict, 'q', node_str)

        if node in kite_nodes:
            Omega, dOmega, ddOmega = interpol_fun['Omega' + node_str](nconf)
            variables_to_add['Omega' + node_str] = [Omega, dOmega, ddOmega]
            Phi, dPhi, ddPhi = interpol_fun['Phi' + node_str](nconf)
            variables_to_add['Phi' + node_str] = [Phi, dPhi, ddPhi]

            if node == 1:
                inclination, dinclination, ddinclination = interpol_fun['inclination' + node_str](nconf)
                variables_to_add['inclination' + node_str] = [inclination, dinclination, ddinclination]

        else:
            inclination, dinclination, ddinclination = interpol_fun['inclination' + node_str](nconf)
            variables_to_add['inclination' + node_str] = [inclination, dinclination, ddinclination]

            azimuth, dazimuth, ddazimuth = interpol_fun['azimuth' + node_str](nconf)
            variables_to_add['azimuth' + node_str] = [azimuth, dazimuth, ddazimuth]

        for variable in list(variables_to_add.keys()):
            nconf = __add_var_to_nconf(nconf, variable, variables_to_add)

    # put nconf together in conf
    conf = struct_op.dissolve_top_layer_of_struct(nconf)

    # set initial guess for xi
    conf['parameterization_dict'] = parameterization_dict

    return conf

def __set_state_in_nconf(nconf, plot_dict, parameterization_dict, state, node_str = ''):

    # read in inputs
    param_constraint_index = parameterization_dict['param_constraint_index']
    xd = plot_dict['xd']
    xddot = plot_dict['outputs']['xddot_from_var']

    #get state dimensions
    state_dim = nconf['var', state + node_str].shape[0]

    # read in state
    for dim in range(state_dim):
        nconf['var', state + node_str, dim] = xd[state + node_str][dim][param_constraint_index]
        nconf['dvar','d' + state + node_str, dim] = xd['d' + state + node_str][dim][param_constraint_index]
        nconf['ddvar','dd' + state + node_str, dim] = xddot['dd' + state + node_str][dim][param_constraint_index]

    return nconf

def __get_parameterization_dict(formulation, initial_or_terminal, initialization_options, nlp, model):

    # initialize variables
    xi_dict = formulation.xi_dict
    trajectory_type = initialization_options['type']
    plot_dict = xi_dict['plot_dict_pickle_' + initial_or_terminal]
    param_constraint_index = None
    xi_param = None
    parameterization_dict = {}
    interp_N = 100 #TODO: MAGICNUMBER

    # set cosmetics
    cosmetics = {}
    cosmetics['plot_coll'] = True
    cosmetics['interpolation'] = {}
    cosmetics['interpolation']['N'] = interp_N
    cosmetics['interpolation']['type'] = 'poly'

    # parameterize V_param
    plot_dict = visualization_tools.interpolate_data(plot_dict, cosmetics)

    # check if xi is free or fixed
    if trajectory_type in ['nominal_landing','transition','launch']:
        fixed_xi = False
    elif trajectory_type in ['compromised_landing']:
        fixed_xi = True

    # get best exit/entry point
    if not fixed_xi:
        param_constraint_index = 15#__get_best_transition_point(plot_dict, model)
        time_grid_ip = plot_dict['time_grids']['ip']
        xi_param = time_grid_ip[param_constraint_index] / time_grid_ip[-1]
    elif fixed_xi:
        xi_param = initialization_options['compromised_landing']['xi_0_initial']
        param_constraint_index = int(xi_param * (interp_N - 1))

    else:
        raise ValueError('Configuration from param not supported yet for trajectory type ' + trajectory_type)

    parameterization_dict['param_constraint_index'] = param_constraint_index
    parameterization_dict['xi_' + initial_or_terminal] = xi_param

    return parameterization_dict

def __get_best_transition_point(plot_dict, model):

    # get system archtecture
    kite_nodes = model.architecture.kite_nodes
    parent_map = model.architecture.parent_map
    number_of_nodes = model.architecture.number_of_nodes

    # compute main tether direction
    main_node_position = plot_dict['xd']['q10']
    main_tether_direction = __list_DM_to_array(main_node_position)
    for i in range(main_tether_direction.shape[0]):
        main_tether_direction[i] = - vect_op.normalize(main_tether_direction[i])

    # compute projected node velocity for kite nodes
    projected_node_velocity = {}
    for node in range(1, number_of_nodes):
        parent = parent_map[node]
        node_str = str(node) + str(parent)
        projected_node_velocity['dq' + node_str] = []
        node_velocity = plot_dict['xd']['dq' + node_str]
        node_velocity = __list_DM_to_array(node_velocity)
        for i in range(node_velocity.shape[0]):
            projected_node_velocity['dq' + node_str] += [np.inner(node_velocity[i], main_tether_direction[i])]

    # add up velocities for all ktes
    projected_kite_velocity = 0.
    for key in list(projected_node_velocity.keys()):
        projected_kite_velocity += np.array(projected_node_velocity[key])

    # compute transition index
    transition_index = np.argmax(projected_kite_velocity)

    return transition_index

def __list_DM_to_array(list_DM):

    array = np.zeros([list_DM[0].shape[0], len(list_DM)])
    for j in range(len(list_DM)):
        array[:, j] = list_DM[j].full().reshape(list_DM[0].shape[0],)

    return array

def __parameterize_curve(boundary_conditions, time_grid_parameters, interpolation_scheme):
    """
    Solves linear system of equations to generate parameters (polynomial coefficients) of 7 segment s-curve s.t.
        boundary conditions are met
    :param boundary_conditions: boundary conditions for interpolation variables
    :param time_grid_parameters: dictionary of time grid parameters containing time grid for s-curve
    :return: polynomial coefficients parameterizing s-curve
    """

    if interpolation_scheme == 's_curve':
        tgrid_s_curve = time_grid_parameters['tgrid_s_curve']
        c_vec = __assemble_lse_for_s_curve(tgrid_s_curve, boundary_conditions)

    else:
        raise ValueError('Error: Interpolation scheme not supported.')

    polynomial_coeff = c_vec

    return polynomial_coeff

def __generate_time_grid_parameters(t_f, primitive, nlp):
    """
    Generate a dictionary containing parameters related to the time grids
    :param t_f: estimated final time
    :param primitives: arguments describing motion primitive
    :param nlp: NLP formulation
    :return: dictionary of time grid parameters
    """

    tgrid_xd = nlp.time_grids['x'](t_f)
    normed_duration = primitive['normed_times'][1] - primitive['normed_times'][0]
    tgrid_u = nlp.time_grids['u'](t_f)
    tgrid_s_curve = [(t_f * normed_duration) / 7. * segment for segment in range(1, 8)]

    time_grid_parameters = {}
    time_grid_parameters['n_k'] = nlp.n_k
    time_grid_parameters['tgrid_xd'] = tgrid_xd
    time_grid_parameters['tgrid_u'] = tgrid_u
    time_grid_parameters['tgrid_s_curve'] = tgrid_s_curve

    indeces = __get_indeces(primitive, nlp)
    time_grid_parameters['n_max'] = indeces['n_max']
    time_grid_parameters['n_current'] = indeces['n_current']

    if nlp.discretization == 'direct_collocation':
        time_grid_parameters['d'] = nlp.d
        time_grid_parameters['d_max'] = indeces['d_max']
        time_grid_parameters['d_current'] = indeces['d_current']
        time_grid_parameters['tgrid_coll'] = nlp.time_grids['coll'](t_f)

    return time_grid_parameters

def __get_indeces(primitives, nlp):

    n_k = nlp.n_k
    d = nlp.d
    time_current = primitives['normed_times'][0]
    normed_duration = primitives['normed_times'][1] - primitives['normed_times'][0]

    indeces = {}
    n_max = int(normed_duration*n_k)  #always round down indeces
    n_current = int(math.ceil(time_current*n_k) ) #always round up indeces

    if nlp.discretization == 'direct_collocation':
        indeces['d_max'] = int((normed_duration*n_k - n_max)*(d + 1))
        indeces['d_current'] = int(math.ceil((time_current*n_k - n_current)*(d + 1)))

    indeces['n_max'] = n_max
    indeces['n_current'] = n_current

    return indeces

def __generate_vals(V_init, primitive, nlp, model, time_grid_parameters, interpolation_parameters, interpolation_scheme, options):
    """
    Generate values for xd, u, xa, theta and phi and place them into initial guess
    :param V_init: partly assembled initial guess
    :param primitive: arguments describing motion primitive
    :param nlp: NLP formulation
    :param model: system model
    :param time_grid_parameters: dictionary containing parameters related to the time grids
    :param interpolation_parameters: parameters that define the interpolation
    :return: partly assembled initial guess
    """

    n_max = time_grid_parameters['n_max']
    n_current = time_grid_parameters['n_current']
    tgrid_xd = time_grid_parameters['tgrid_xd']

    if nlp.discretization == 'direct_collocation':
        d_max = time_grid_parameters['d_max']
        d_current = time_grid_parameters['d_current']
        tgrid_coll = time_grid_parameters['tgrid_coll']

    #set xi initial guess
    if options['type'] in ['nominal_landing', 'compromised_landing']:
        V_init['xi','xi_0'] = interpolation_parameters['configurations']['conf_0']['parameterization_dict']['xi_initial']
    if options['type'] in ['launch']:
        V_init['xi','xi_0'] = interpolation_parameters['configurations']['conf_f']['parameterization_dict']['xi_terminal']
    for k in range(n_max + 1):
        t_xd = tgrid_xd[k]
        continuous_guess, interpolation_variables = __get_continuous_guess(t_xd, time_grid_parameters, interpolation_parameters, primitive, model, interpolation_scheme)
        for name in struct_op.subkeys(model.variables, 'xd'):
            V_init['xd', k, name] = continuous_guess[name]
        if k < (n_max):
            if model.options['tether']['control_var'] == 'ddl_t':
                if 'u' in V_init.keys():
                    V_init['u', k + n_current, 'ddl_t'] = continuous_guess['ddl_t']
        if nlp.discretization == 'direct_collocation':
            if k == n_max:
                d_vals = d_max
            else:
                d_vals = nlp.d
            for j in range(d_vals):
                t_coll = tgrid_coll[k, j]
                continuous_guess, interpolation_variables = __get_continuous_guess(t_coll, time_grid_parameters, interpolation_parameters, primitive, model, interpolation_scheme)
                for name in struct_op.subkeys(model.variables, 'xd'):
                    V_init['coll_var', k, j, 'xd', name] = continuous_guess[name]
                if model.options['tether']['control_var'] == 'ddl_t':
                    if 'u' in V_init.keys():
                        V_init['coll_var', k, j, 'u', 'ddl_t'] = continuous_guess['ddl_t']
    return V_init

def __get_continuous_guess(t_cont, time_grid_parameters, interpolation_parameters, primitive, model, interpolation_scheme):
    """
    Returns initial guess for a point in (continuous) time based on the type of motion primitive
    :param t_cont: point in time
    :param time_grid_parameters: dictionary of parameters related to the time grids
    :param interpolation_parameters: dictionary of parameters defining the interpolation
    :param primitive: arguments describing motion primitive
    :param model: system model
    :return: initial guess for a point in (continuous) time
    """

    if primitive['type'] == 'goto':
        continuous_guess, interpolation_variables = interp.__goto_configuration(t_cont, time_grid_parameters, interpolation_parameters, model, interpolation_scheme)

    return continuous_guess, interpolation_variables

def __assemble_lse_for_s_curve(tgrid_s_curve, boundary_conditions):
    """
    Assembles the linear system of equations A*x = b that is used to find the polynomial coefficients that parameterize
        the s-curve
    :param tgrid_s_curve: vector containing s-curve time grid
    :param boundary_conditions: dictionary of boundary conditions
    :return: matrix A and vector b for the linear system of equations
    """

    # generate constants
    constants = [1., 1., 1./2., 1./6., 1./24.]

    # add zero to tgrid
    tgrid_s_curve = [0.] + tgrid_s_curve

    # generate structs
    b_vec = ct.struct_symSX([(
        ct.entry('p_hat_0'),
        ct.entry('dp_hat_0'),
        ct.entry('ddp_hat_0'),
        ct.entry('p_hat_f'),
        ct.entry('dp_hat_f'),
        ct.entry('ddp_hat_f'),
        )])

    c_vec = ct.struct_symSX([(
        ct.entry('poly_coeff_1', shape = (4,1)),
        ct.entry('poly_coeff_2', shape = (4,1)),
        ct.entry('poly_coeff_3', shape = (4,1)),
        ct.entry('poly_coeff_4', shape = (4,1)),
        ct.entry('poly_coeff_5', shape = (4,1)),
        ct.entry('poly_coeff_6', shape = (4,1)),
        ct.entry('poly_coeff_7', shape = (4,1)),
        )])

    t_vec = ct.struct_symSX([(
        ct.entry('t_0'),
        ct.entry('t_1'),
        ct.entry('t_2'),
        ct.entry('t_3'),
        ct.entry('t_4'),
        ct.entry('t_5'),
        ct.entry('t_6'),
        ct.entry('t_7'),
        )])

    # generate variable struct
    V = ct.struct_symSX([(ct.entry('c_vec', struct = c_vec),
                             ct.entry('t_vec', struct = t_vec),
                             ct.entry('b_vec', struct = b_vec),
                                  )])

    # generate equations

    ## continuity equations
    continuity_equations_list = []

    for segment in range(1,7):
        for deriv in range(3):
            LHS = 0.
            RHS = 0.
            for degree in range(4):
                if deriv + degree < 4:
                    LHS += V['c_vec','poly_coeff_' + str(segment), degree + deriv] * V['t_vec','t_' + str(segment)]**degree * constants[degree]
                    RHS += V['c_vec','poly_coeff_' + str(segment + 1), degree + deriv] * V['t_vec','t_' + str(segment)]**degree * constants[degree]
            continuity_equations_list += [LHS - RHS]

    continuity_equations = ct.vertcat(*continuity_equations_list)

    ## boundary_conditions
    initial_conditions_list = []
    terminal_conditions_list = []
    prefixes = ['', 'd', 'dd']

    for deriv in range(2): #todo: should actually be range 3 to include boundary conditions for acceleration. Produces weird results though
        initial_LHS = 0.
        initial_RHS = V['b_vec',prefixes[deriv] + 'p_hat_0']
        terminal_LHS = 0.
        terminal_RHS = V['b_vec',prefixes[deriv] + 'p_hat_f']
        for degree in range(4):
            if deriv + degree < 4:
                initial_LHS += V['c_vec','poly_coeff_1', degree + deriv] * V['t_vec','t_0']**degree * constants[degree]
                terminal_LHS += V['c_vec','poly_coeff_7', degree + deriv] * V['t_vec','t_7']**degree * constants[degree]
        initial_conditions_list += [initial_LHS - initial_RHS]
        terminal_conditions_list += [terminal_LHS - terminal_RHS]
    initial_conditions = ct.vertcat(*initial_conditions_list)
    terminal_conditions = ct.vertcat(*terminal_conditions_list)

    ## jerk conditions
    jerk_conditions_list = []

    for segment in [2, 4, 6]:
        RHS = 0.
        LHS = V['c_vec','poly_coeff_' + str(segment), -1]
        jerk_condition = LHS - RHS
        jerk_conditions_list += [jerk_condition]
    jerk_conditions = ct.vertcat(*jerk_conditions_list)

    ## concat equations
    equations = ct.vertcat(
                           continuity_equations,
                           initial_conditions,
                           terminal_conditions,
                           jerk_conditions
                           )

    # generate symbolic A_mat
    A_mat = ct.jacobian(equations, V['c_vec'])

    # generate function for A_mat
    A_mat_fun = ct.Function('A_mat_fun', [V], [A_mat])

    # generate numerical V
    V_num = V(0.0)
    for key in struct_op.subkeys(V, 'b_vec'):
        V_num['b_vec', key] = boundary_conditions[key]
    for i in range(len(struct_op.subkeys(V, 't_vec'))):
        V_num['t_vec','t_' + str(i)] = tgrid_s_curve[i]

    # create NLP minimizing the jerk

    ## build constraints
    eq_constraints_list = []
    for vec in ['t_vec', 'b_vec']:
        for key in struct_op.subkeys(V, vec):
            eq_constraints_list += [V[vec, key] - V_num[vec, key]]
    eq_constraints = ct.vertcat(*eq_constraints_list)
    constraints = ct.vertcat(equations,
                             eq_constraints
                             )

    ## build cost function
    cost_function = 0.
    for segment in range(1,8):
        for degree in range(1, 4):
            cost_function += (V['c_vec','poly_coeff_' + str(segment), degree])**2

    ## build nlp
    jerk_nlp = {'x': V, 'f': cost_function, 'g': constraints}

    ## set solver options
    jerk_options = {}
    if awelogger.logger.getEffectiveLevel() > 10:
        jerk_options['ipopt.print_level'] = 0
        jerk_options['print_time'] = 0

    ## build solver
    jerk_solver = cas.nlpsol('jerk_solver', 'ipopt', jerk_nlp, jerk_options)

    ## compute solution
    jerk_solution = jerk_solver(x0=V_num, lbg=0., ubg=0.)
    jerk_solution = V(jerk_solution['x'])

    # create outputs
    c_vec = jerk_solution['c_vec']

    return c_vec

def __get_Omega(layer_siblings, node, parent):
    """
    Distributes level siblings along circle and returns the correspondig angles
    :param layer_siblings: list of layer siblings of a node
    :param node: node for which the angle Omega should be computed
    :param parent: parent of the node
    :return: angle Omega for the node
    """
    number_of_siblings = len(layer_siblings[parent])
    if number_of_siblings == 1:
        Omega = 0.
    else:
        idx = layer_siblings[parent].index(node)
        Omega = np.float(idx) / np.float(number_of_siblings) * 2. * np.pi

    return Omega

def __get_layer_siblings(model):
    """
    Returns map (dictionary) of layer siblings for each node
    :param model: system model
    :return: layer siblings map
    """

    parent_nodes = model.architecture.parent_map
    kite_nodes = model.architecture.kite_nodes

    layer_siblings = {}
    for kite in kite_nodes:
        parent = parent_nodes[kite]

        if not(parent in list(layer_siblings.keys())):
            layer_siblings[parent] = []

        layer_siblings[parent] += [kite]

    return layer_siblings
