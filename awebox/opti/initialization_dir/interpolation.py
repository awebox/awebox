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
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import numpy as np
import casadi as cas
import casadi.tools as ct

def __goto_configuration(t_cont, time_grid_parameters, interpolation_parameters, model, interpolation_scheme):
    """Motion primitive for going from one configuration to another

    :type t_cont: double
    :param t_cont: point in (continuous) time

    :type time_grid_parameters: dict
    :param time_grid_parameters: parameters related to the time grids

    :type interpolation_parameters: dict
    :param interpolation_parameters: parameters that define the interpolation

    :type model: awebox.model_dir.model
    :param model: system model

    :rtype: dict, dict
    """

    wind = interpolation_parameters['wind']
    interpolation_variables = __get_interpolation_variables(t_cont, time_grid_parameters, interpolation_parameters, interpolation_scheme, model)
    configurations = interpolation_parameters['configurations']
    states = __compute_states_from_interpolation_variables(interpolation_variables, configurations, model, wind)
    continuous_guess = collect_guess(interpolation_variables, states, model)

    return continuous_guess, interpolation_variables

def __get_interpolation_variables(t_cont, time_grid_parameters, interpolation_parameters, interpolation_scheme, model):
    """Get values for the variables that are interpolated for a point in time

    :type t_cont: double
    :param t_cont: point in continuous time

    :type time_grid_parameters: dict
    :param time_grid_parameters: parameters related to the time grids

    :type interpolation_parameters: dict
    :param interpolation_parameters: parameters that define the interpolation

    :type interpolation_scheme: str
    :param interpolation_scheme: type of interpolation scheme

    :type model: awebox.model_dir.model
    :param model: system model

    :rtype: dict
    """

    kite_nodes = model.architecture.kite_nodes
    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    tgrid_s_curve = time_grid_parameters['tgrid_s_curve']
    interpolation_variables = {}

    interpolation_variables = __interpolate_specific_variable(t_cont, tgrid_s_curve, interpolation_variables, '', 'l_t', interpolation_parameters, interpolation_scheme)

    for node in range(1, number_of_nodes):  #kite nodes are parameterized in a rotational framework, intermediate nodes with inclination and azimuth
        parent = parent_nodes[node]
        node_str = str(node) + str(parent)
        if   (node in kite_nodes):
            variable_list = ['Phi','Omega']
            if node == 1:
                variable_list += ['inclination']
        else:
            variable_list = ['inclination', 'azimuth']
        for variable_name in variable_list:
            interpolation_variables = __interpolate_specific_variable(t_cont, tgrid_s_curve, interpolation_variables, node_str, variable_name, interpolation_parameters, interpolation_scheme)

    return interpolation_variables

def __interpolate_specific_variable(t_cont, tgrid_s_curve, interpolation_variables, node_str, variable_name, interpolation_parameters, interpolation_scheme):
    """Give the value of a specific interpolation variable and its derivatives at a point in time

    :type t_cont: double
    :param t_cont: point in continuous time

    :type tgrid_s_curve: list
    :param tgrid_s_curve: time grid of s-curve

    :type interpolation_variables: dict
    :param interpolation_variables: partly assel=mbled interpolation variables

    :type node_str: str
    :param node_str: node + parent string of given node

    :type variable_name: str
    :param variable_name: name of variable that is to be interpolated

    :type interpolation_parameters: dict
    :param interpolation_parameters: parameters that define the interpolation

    :type interpolation_scheme: str
    :param interpolation_scheme: type of interpolation scheme that is used

    :rtype: dict
    """

    poly_coeff = interpolation_parameters['polynomial_coeff']

    if interpolation_scheme == 's_curve':

        # todo: this is a really hackish method of "fixing" this problem, but i cannot figure out what is going on.
        #  can someone would actually uses this modular initialization pick this up?

        try:
            interpolation_variables[variable_name + node_str] = __eval_piecewise_polynomial(t_cont, poly_coeff[variable_name + node_str], tgrid_s_curve)
        except:
            interpolation_variables[variable_name + node_str] = interpolation_parameters['configurations']['conf_0'][variable_name + node_str]

        try:
            interpolation_variables['d' + variable_name + node_str] = __eval_piecewise_polynomial(t_cont, poly_coeff[variable_name + node_str], tgrid_s_curve, derivative_order=1)
        except:
            interpolation_variables['d' + variable_name + node_str] = interpolation_parameters['configurations']['conf_0']['d' + variable_name + node_str]

        try:
            interpolation_variables['dd' + variable_name + node_str] = __eval_piecewise_polynomial(t_cont, poly_coeff[variable_name + node_str], tgrid_s_curve, derivative_order=2)
        except:
            interpolation_variables['dd' + variable_name + node_str] = interpolation_parameters['configurations']['conf_0']['dd' + variable_name + node_str]

    elif interpolation_scheme == 'poly':
        interpolation_variables[variable_name + node_str] = __eval_polynomial(t_cont, poly_coeff[
            variable_name + node_str])
        interpolation_variables['d' + variable_name + node_str] = __eval_polynomial(t_cont, poly_coeff[
            variable_name + node_str], derivative_order=1)
        interpolation_variables['dd' + variable_name + node_str] = __eval_polynomial(t_cont, poly_coeff[
            variable_name + node_str], derivative_order=2)

    return interpolation_variables

def __eval_polynomial(t_cont, polynomial_coeff, derivative_order = 0):
    """Evaluate 5th degree polynomial that defines the interpolation

    :type t_cont: double
    :param t_cont: point in (continuous) time

    :type polynomial_coeff: list
    :param polynomial_coeff: coefficients for each piecewise polynomial

    :type derivative_order: int
    :param derivative_order: order of derivative of the polynomial expression that is evaluated

    :rtype double
    """

    constants = [1., 1., 1./2., 1./6., 1./24., 1./120.]
    polynomial_value = 0.
    if derivative_order == 0:
        for j in range(6):
            polynomial_value += polynomial_coeff[j] * constants[j] * t_cont**j
    elif derivative_order == 1:
        for j in range(5):
            polynomial_value += polynomial_coeff[j + 1] * constants[j] * t_cont**j
    elif derivative_order == 2:
        for j in range(4):
            polynomial_value += polynomial_coeff[j + 2] * constants[j] * t_cont**j

    return polynomial_value

def __eval_piecewise_polynomial(t_cont, polynomial_coeff, tgrid_s_curve, derivative_order = 0): #todo: use existing polynomial implementation
    """Evaluate piecewise polynomial expression that defines the s-curve

    :type t_cont: double
    :param t_cont: point in (continuous) time

    :type polynomial_coeff: list
    :param polynomial_coeff: coeffients for each piecewise polynomial

    :type tgrid_s_curve: list
    :param tgrid_s_curve: time grid of s-curve

    :type derivate_order: int
    :param derivative_order: order of derivative of the polynomial expression that is evaluated

    :rtype: double
    """

    vec_offset = compute_vec_offset(t_cont, tgrid_s_curve) # determines which polynomial coeff to use todo: more robust implementation
    if derivative_order == 0:
        polynomial_value = polynomial_coeff[0 + vec_offset] + polynomial_coeff[1 + vec_offset]*t_cont + 0.5*t_cont*t_cont*polynomial_coeff[2 + vec_offset] + 1./6.*t_cont*t_cont*t_cont*polynomial_coeff[3 + vec_offset]
    elif derivative_order == 1:
        polynomial_value = polynomial_coeff[1 + vec_offset] + t_cont*polynomial_coeff[2 + vec_offset] + 0.5*t_cont*t_cont*polynomial_coeff[3 + vec_offset]
    elif derivative_order == 2:
        polynomial_value = polynomial_coeff[2 + vec_offset] + t_cont*polynomial_coeff[3 + vec_offset]

    return polynomial_value

def compute_vec_offset(t_cont, tgrid_s_curve):
    """compute vector offset for index of polynomial coeff in s-curve

    :type t_cont: double
    :param t_cont: point in continuous time

    :type tgrid_s_curve: list
    :param tgrid_s_curve: time grid of s-curve

    :rtype: int
    """

    vec_offset = 0
    for t in tgrid_s_curve:
        if t < t_cont:
            vec_offset = 4*(tgrid_s_curve.index(t) + 1)
    vec_offset = int(vec_offset)

    return vec_offset

def __compute_states_from_interpolation_variables(interpolation_variables, configurations, model, wind):
    """Compute q and dq from the interpolation variables

    :type interpolation_variables: dict
    :param interpolation_variables: interpolation variables at a point in time that parameterize q and dq

    :type configurations: dict
    :param configurations: initial and terminal system configurations

    :type model: awebox.model_dir.model
    :param model: system model

    :rtype: dict
    """

    # process interpolation variables
    sstates_functions, sinterp = __process_interpolation_Variables(interpolation_variables, configurations, model)

    # evaluate functions
    states = __get_states(sstates_functions, model, interpolation_variables, sinterp)

    return states

def __get_states(sstates_functions, model, interpolation_variables, sinterp):
    """Get states from interpolation variables

    :type sstates_functions: dict
    :param sstates_functions: casadi functions mapping interpolation variables to states

    :type model: awebox.model_dir.model
    :param model: system model

    :type interpolation_variables: dict
    :param interpolation_variables: variables defining the interpolation

    :type sinterp: casadi.struct_symSX
    :param sinterp: interpolation variables

    :rtype: dict
    """

    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map
    kite_dof = model.kite_dof
    kite_nodes = model.architecture.kite_nodes

    # initialize dict
    states = {}

    # fill sinterp with numerical values
    ninterp = sinterp(0.0)
    for key in list(sinterp.keys()):
        for subkey in struct_op.subkeys(sinterp, key):
            ninterp[key, subkey] = interpolation_variables[subkey]

    for node in range(1, number_of_nodes):
        node_str = str(node) + str(parent_nodes[node])
        states['q' + node_str] = sstates_functions['q' + node_str](ninterp)
        states['dq' + node_str] = sstates_functions['dq' + node_str](ninterp)

        if int(kite_dof) == 6 and node in kite_nodes:
            states['omega' + node_str] = sstates_functions['omega' + node_str](ninterp)
            states['r' + node_str] = sstates_functions['r' + node_str](ninterp)

    return states

def __process_interpolation_Variables(interpolation_variables, configurations, model):
    """Create casadi functions mapping interpolation variables to states

    :type interpolation_variables: dict
    :param interpolation_variables: variables defining the interpolation

    :type configurations: dict
    :param configurations: parameters defining a given configuration

    :type model: awebox.model_dir.model
    :param model: system model

    :rtype: dict, casadi.struct_symSX
    """

    # initialize dict
    rotation_matrix = {}
    rotation_matrix['q00'] = np.eye(3)
    rotation_matrix['dq00'] = np.zeros([3,3])

    kite_nodes = model.architecture.kite_nodes
    parent_nodes = model.architecture.parent_map
    parent_nodes[0] = 0
    number_of_nodes = model.architecture.number_of_nodes
    kite_dof = model.kite_dof
    trajectory_type = model.options['trajectory']['type']

    conf_0 = configurations['conf_0']
    conf_f = configurations['conf_f']
    l_s = configurations['l_s']
    l_i = configurations['l_i']

    ## generate casadi expressions

    # generate symbolic variables
    sinterp = __get_sstruct(interpolation_variables)
    sstates = {}
    sstates['var'] = {}
    sstates['var']['q00'] = 0.
    sstates['dvar'] = {}
    sstates['dvar']['q00'] = 0.
    sstates['ddvar'] = {}
    sstates['ddvar']['q00'] = 0.

    # generate dict to store functions in
    sstates_functions = {}

    # generate casadi functions: inteprolation variables -> states
    l_t = sinterp['var', 'l_t']
    for node in range(1, number_of_nodes):
        parent_node = parent_nodes[node]
        node_str = str(node) + str(parent_node)
        grandparent_node = parent_nodes[parent_node]
        parent_str = str(parent_node) + str(grandparent_node)
        grandgrandparent_node = parent_nodes[grandparent_node]
        grandparent_str = str(grandparent_node) + str(grandgrandparent_node)

        if (node == 1) and (node not in kite_nodes):  # first node parameterized with main tether length
            a = cas.mtimes((conf_f['q' + node_str] - conf_0['q' + node_str]).T, (conf_f['q' + node_str] - conf_0['q' + node_str]))
            if a == 0:
                e_t = vect_op.normalize(conf_0['q' + node_str])
            else:
                b = 2 * cas.mtimes(conf_0['q' + node_str].T, (conf_f['q' + node_str] - conf_0['q' + node_str]))
                c = cas.mtimes(conf_0['q' + node_str].T, conf_0['q' + node_str]) - l_t ** 2
                D = b ** 2 - 4 * a * c
                x1 = (-b + np.sqrt(D)) / (2 * a)
                x2 = (-b - np.sqrt(D)) / (2 * a)
                #if x2 >= 0:
                #    s = x2
                #else:
                #    s = x1
                s = x1
                e_t = 1. / l_t * (conf_0['q' + node_str] + s * (conf_f['q' + node_str] - conf_0['q' + node_str]))
            sstates['var']['q' + node_str] = l_t * e_t
            rotation_matrix = compute_rotation_matrices(sinterp.prefix['var'], node_str, parent_str, rotation_matrix)

        else:
            if (node in kite_nodes):

                if node == 1:
                    tether_length = l_t
                else:
                    tether_length = l_s

                Phi = sinterp['var', 'Phi' + node_str]
                Omega = sinterp['var', 'Omega' + node_str]
                parent = sstates['var']['q' + parent_str]
                grandparent = sstates['var']['q' + grandparent_str]

                radius = tether_length * np.sin(Phi)
                l_x = tether_length * np.cos(Phi)

                # define axis of rotation
                if node != 1:
                    axis_of_rot = parent - grandparent
                    axis_of_rot = vect_op.normalize(axis_of_rot)
                else:
                    inclination = sinterp['var', 'inclination' + node_str]
                    axis_of_rot = np.zeros([3, 1])
                    axis_of_rot[0] = np.cos(inclination)
                    axis_of_rot[2] = np.sin(inclination)
                e_hat_x = axis_of_rot
                e_hat_y = vect_op.normed_cross(e_hat_x, vect_op.zhat())
                e_hat_z = vect_op.normed_cross(e_hat_y, e_hat_x)
                e_hat_r = e_hat_z * np.sin(Omega) + e_hat_y * np.cos(Omega)

                sstates['var']['q' + node_str] = sstates['var']['q' + parent_str] + e_hat_r*radius + e_hat_x*l_x

            else:
                tether_length = l_i
                rotation_matrix = compute_rotation_matrices(sinterp.prefix['var'], node_str, parent_str, rotation_matrix)
                tether_vector = vect_op.xhat()
                tether_vector = cas.mtimes(rotation_matrix['q' + node_str], tether_vector)
                sstates['var']['q' + node_str] = sstates['var']['q' + parent_str] + tether_vector*tether_length

        sstates['dvar']['q' + node_str] = cas.mtimes(cas.jacobian(sstates['var']['q' + node_str], sinterp['var']), sinterp['dvar'])


        # create rotational kinematics
        if int(kite_dof) == 6:

            # iterate over all kite ndoes
            if node in kite_nodes:

                # get node strings
                parent = parent_nodes[node]
                node_str = str(node) + str(parent)
                grandparent = parent_nodes[parent]
                parent_str = str(parent) + str(grandparent)
                grandgrandparent = parent_nodes[grandparent]
                grandparent_str = str(grandparent) + str(grandgrandparent)

                # compute dcm matrix for node
                e_hat_1, e_hat_2, e_hat_3 = __get_kite_axis(sstates, node_str,
                                                            parent_str,
                                                            grandparent_str, trajectory_type)
                dcm = ct.horzcat(e_hat_1, e_hat_2, e_hat_3)
                dcm_column = ct.reshape(dcm, (9, 1))

                # compute rotation around axis
                omega_norm = vect_op.norm(sstates['dvar']['q' + node_str]) / radius
                if trajectory_type == 'power_cycle':
                    omega_vector = vect_op.normalize(axis_of_rot) * omega_norm
                elif trajectory_type == 'nominal_landing':
                    omega_vector = cas.DM([0,0,0])

                # put in state dict
                sstates['omega' + node_str] = omega_vector
                sstates['r' + node_str] = dcm_column

        # generate functions
        sstates_functions['q' + node_str] = cas.Function('q' + node_str, [sinterp], [sstates['var']['q' + node_str]])
        sstates_functions['dq' + node_str] = cas.Function('dq' + node_str, [sinterp], [sstates['dvar']['q' + node_str]])
        if int(kite_dof) == 6 and node in kite_nodes:
            sstates_functions['r' + node_str] = cas.Function('r' + node_str, [sinterp], [sstates['r' + node_str]])
            sstates_functions['omega' + node_str] = cas.Function('omega' + node_str, [sinterp], [sstates['omega' + node_str]])

    return sstates_functions, sinterp

def __get_sstruct(dictionary):
    """Convert a dictionary to a casadi struct with the same structure and entries for 0th, 1st and 2nd derivatives

    :type dictionary: dict
    :param dictionary: variables to be put into the struct

    :rtype: casadi.struct_symSX
    """

    struct_list = []

    for key in list(dictionary.keys()):
        if key[0] != 'd':
            struct_list += [(key, dictionary[key].shape)]

    sub_struct = ct.struct_symSX([ct.entry(struct_list[i][0], shape=struct_list[i][1])
                    for i in range(len(struct_list))])

    dsub_struct = ct.struct_symSX([ct.entry('d' + struct_list[i][0], shape=struct_list[i][1])
                    for i in range(len(struct_list))])

    ddsub_struct = ct.struct_symSX([ct.entry('dd' + struct_list[i][0], shape=struct_list[i][1])
                    for i in range(len(struct_list))])

    sstruct = ct.struct_symSX([
                    ct.entry('var', struct = sub_struct),
                    ct.entry('dvar', struct = dsub_struct),
                    ct.entry('ddvar', struct = ddsub_struct)
                ])

    return sstruct

def __get_kite_axis(sstates, node_str, parent_str, grandparent_str, trajectory_type):
    """Compute kite axis from states for a specific node

    :type sstates: casadi.struct_symSX
    :param sstates: states and their derivatives

    :type node_str: str
    :param node_str: node index

    :type parent_str: str
    :param parent_str: parent node index

    :type grandparent_str: str
    :param grandparent_str: grandparent node index

    :type trajectory_type: str
    :param trajectory_type: type of trajectory that is being optimized

    :rtype: casadi.SX, casadi.DM, casadi.SX
    """

    if node_str == '10':
        e_hat_3 = vect_op.normalize(sstates['var']['q' + node_str] - sstates['var']['q' + parent_str])
    elif trajectory_type == 'nominal_landing':
        e_hat_3 = cas.DM([0,0,1])
    else:
        e_hat_3 = vect_op.normalize(sstates['var']['q' + parent_str] - sstates['var']['q' +
                                                                  grandparent_str])
    e_hat_1 = -vect_op.normalize(sstates['dvar']['q' + node_str])
    e_hat_2 = vect_op.normed_cross(e_hat_3, e_hat_1)

    return e_hat_1, e_hat_2, e_hat_3

def compute_rotation_matrices(interpolation_variables, node_str, parent_str, rotation_matrix):

    inclination = interpolation_variables['inclination' + node_str]
    azimuth = interpolation_variables['azimuth' + node_str]
    rotation_mat_y = get_rotation_matrix('y', inclination)
    rotation_mat_z = get_rotation_matrix('z', azimuth)
    rotation_matrix['q' + node_str] = cas.mtimes(rotation_mat_z, rotation_mat_y)
    rotation_matrix['q' + node_str] = cas.mtimes(rotation_matrix['q' + node_str], rotation_matrix['q' + parent_str])

    return rotation_matrix

def get_rotation_matrix(axis, angle):

    if axis == 'x':
        rotation_mat = np.array([[1, 0, 0],
                                 [0, np.cos(angle), -np.sin(angle)],
                                 [0, np.sin(angle), np.cos(angle)]])

    elif axis == 'dx':
        rotation_mat = np.array([[0, 0, 0],
                                 [0, -np.sin(angle)*angle, -np.cos(angle)*angle],
                                 [0, np.cos(angle)*angle, -np.sin(angle)*angle]])

    elif axis == 'y':
        rotation_mat = np.array([[np.cos(angle), 0, -np.sin(angle)],
                                 [0, 1, 0],
                                 [np.sin(angle), 0, np.cos(angle)]])

    elif axis == 'dy':
        rotation_mat = np.array([[-np.sin(angle)*angle, 0, -np.cos(angle)*angle],
                                 [0, 0, 0],
                                 [np.cos(angle)*angle, 0, -np.sin(angle)*angle]])

    elif axis == 'z':
        rotation_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0],
                                 [0, 0, 1]])

    elif axis == 'dz':
        rotation_mat = np.array([[-np.sin(angle)*angle, -np.cos(angle)*angle, 0],
                                 [np.cos(angle)*angle, -np.sin(angle)*angle, 0],
                                 [0, 0, 0]])

    return rotation_mat


def collect_guess(interpolation_variables, states, model):

    number_of_nodes = model.architecture.number_of_nodes
    parent_nodes = model.architecture.parent_map

    continuous_guess = {}
    for name in struct_op.subkeys(model.variables, 'xd'):
        continuous_guess[name] = 0.0
    continuous_guess['e'] = 0.0

    for state in list(states.keys()):
        continuous_guess[state] = states[state]

    continuous_guess['l_t'] = interpolation_variables['l_t']
    continuous_guess['dl_t'] = interpolation_variables['dl_t']
    continuous_guess['ddl_t'] = interpolation_variables['ddl_t']

    return continuous_guess
