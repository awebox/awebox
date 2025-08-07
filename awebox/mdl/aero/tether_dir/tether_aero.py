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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2020
'''
import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op

import awebox.mdl.aero.tether_dir.reynolds as reynolds
import awebox.mdl.aero.tether_dir.segment as segment
import awebox.mdl.aero.tether_dir.element as element
import awebox.mdl.aero.kite_dir.frames as kite_frames



def get_tether_model_types(options):

    selected_model = options['tether']['tether_drag']['model_type']

    if selected_model == 'not_in_use':
        tether_models = []

    elif selected_model == 'kite_only':
        tether_models = ['kite_only']

    elif selected_model == 'split':
        tether_models = ['split', 'single', 'multi']

    else:
        tether_models = ['split'] + [selected_model]

    return tether_models


def get_force_var(variables_si, upper_node, architecture):

    lower_node = architecture.parent_map[upper_node]
    name = str(upper_node) + str(lower_node)
    var = variables_si['z']['f_tether' + name]
    return var

def distribute_tether_drag_forces(options, variables_si, architecture, outputs):


    # initialize dictionary
    tether_drag_forces = {}
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        tether_drag_forces['f' + str(node) + str(parent)] = cas.SX.zeros((3, 1))

    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]

        drag_node = outputs['tether_aero']['homotopy_upper' + str(node)]
        drag_parent = outputs['tether_aero']['homotopy_lower' + str(node)]

        # attribute portion of segment drag to parent
        if node > 1:
            grandparent = architecture.parent_map[parent]
            tether_drag_forces['f' + str(parent) + str(grandparent)] += drag_parent
    
        # attribute portion of segment drag to node
        tether_drag_forces['f' + str(node) + str(parent)] += drag_node

    return tether_drag_forces


def get_tether_cstr(options, variables_si, architecture, outputs):

    tether_drag_forces = distribute_tether_drag_forces(options, variables_si, architecture, outputs)

    cstr_list = cstr_op.MdlConstraintList()
    for node in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[node]
        f_tether_var = get_force_var(variables_si, node, architecture)
        f_tether_val = tether_drag_forces['f' + str(node) + str(parent)]
        local_resi_unscaled = (f_tether_var - f_tether_val)

        scale = options['scaling']['z']['f_tether']
        local_resi = local_resi_unscaled / scale

        f_cstr = cstr_op.Constraint(expr=local_resi,
                                    name='f_tether' + str(node) + str(parent),
                                    cstr_type='eq')
        cstr_list.append(f_cstr)

    return cstr_list


def get_force_outputs(model_options, variables, parameters, atmos, wind, upper_node, tether_cd_fun, outputs, architecture):

    element_drag_fun = element.get_element_drag_fun(wind, atmos, tether_cd_fun, parameters)

    trivial_lower, trivial_upper = segment.get_trivial_segment_forces(atmos, wind, variables, upper_node, architecture, parameters)
    kite_only_lower, kite_only_upper = segment.get_kite_only_segment_forces(atmos, outputs, variables, upper_node, architecture, tether_cd_fun, parameters)

    split_lower, split_upper = segment.get_distributed_segment_forces(1, variables, upper_node, architecture, element_drag_fun, parameters)

    n_elements = model_options['tether']['aero_elements']
    multi_lower, multi_upper = segment.get_distributed_segment_forces(n_elements, variables, upper_node, architecture, element_drag_fun, parameters)

    equivalent_lower, equivalent_upper = get_physical_forces(model_options, variables, parameters, atmos, wind, upper_node, tether_cd_fun, architecture)


    re_number = segment.get_segment_reynolds_number(variables, atmos, wind, upper_node, architecture)

    if 'tether_aero' not in list(outputs.keys()):
        outputs['tether_aero'] = {}

    outputs['tether_aero']['multi_upper' + str(upper_node)] = multi_upper
    outputs['tether_aero']['multi_lower' + str(upper_node)] = multi_lower
    outputs['tether_aero']['split_upper' + str(upper_node)] = split_upper
    outputs['tether_aero']['split_lower' + str(upper_node)] = split_lower
    outputs['tether_aero']['trivial_upper' + str(upper_node)] = trivial_upper
    outputs['tether_aero']['trivial_lower' + str(upper_node)] = trivial_lower
    outputs['tether_aero']['kite_only_upper' + str(upper_node)] = kite_only_upper
    outputs['tether_aero']['kite_only_lower' + str(upper_node)] = kite_only_lower
    outputs['tether_aero']['equivalent_upper' + str(upper_node)] = equivalent_upper
    outputs['tether_aero']['equivalent_lower' + str(upper_node)] = equivalent_lower

    # homotopy parameters
    p_dec = parameters.prefix['phi']

    tether_model = model_options['tether']['tether_drag']['model_type']
    if tether_model == 'multi':
        # drag_node = p_dec['tau'] * split_upper + (1. - p_dec['tau']) * multi_upper
        # drag_parent = p_dec['tau'] * split_lower + (1. - p_dec['tau']) * multi_lower
        drag_node = multi_upper
        drag_parent = multi_lower

    elif tether_model == 'split':
        drag_node = split_upper
        drag_parent = split_lower

    elif tether_model == 'trivial':
        drag_node = trivial_upper
        drag_parent = trivial_lower

    elif tether_model == 'kite_only':
        drag_node = kite_only_upper
        drag_parent = kite_only_lower

    elif tether_model == 'equivalent':
        drag_node = equivalent_upper
        drag_parent = equivalent_lower
        message = 'The equivalent tether_model does not work as expected. It is currently offered only for backwards compatibility reasons, and is not recommended. Please be absolutely sure this model is what you would like to use.'
        print_op.base_print(message, level='warning')

    elif tether_model == 'not_in_use':
        drag_parent = cas.DM.zeros((3, 1))
        drag_node = cas.DM.zeros((3, 1))

    else:
        raise ValueError('tether drag model not supported.')

    outputs['tether_aero']['homotopy_upper' + str(upper_node)] = drag_node
    outputs['tether_aero']['homotopy_lower' + str(upper_node)] = drag_parent

    outputs['tether_aero']['reynolds' + str(upper_node)] = re_number

    return outputs


def get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, upper_node):

    lower_node = architecture.parent_map[upper_node]
    main_tether = (lower_node == 0)
    secondary_tether = (upper_node in architecture.kite_nodes)
    intermediate_tether = not (main_tether or secondary_tether)

    if main_tether:
        length_sym = 'l_t'
        diam_sym = 'diam_t'

    elif secondary_tether:
        length_sym = 'l_s'
        diam_sym = 'diam_s'

    elif intermediate_tether:
        length_sym = 'l_i'
        diam_sym = 'diam_t'

    else:
        message = 'unexpected outcome of tether-type categorization, while collecting tether-segment properties'
        print_op.log_and_raise_error(message)

    var_type_length = struct_op.get_variable_type(variables_si, length_sym)
    var_type_diam = struct_op.get_variable_type(variables_si, diam_sym)

    q_node = variables_si['x']['q' + str(upper_node) + str(lower_node)]
    if main_tether:
        q_parent = cas.DM.zeros((3, 1))
    else:
        grandparent = architecture.parent_map[lower_node]
        q_parent = variables_si['x']['q' + str(lower_node) + str(grandparent)]

    # we need this definition of the segment length (as opposed to just
    # using 'l_t') to keep the lagrangian mechanics working correctly
    seg_length = vect_op.norm(q_node - q_parent)

    scaling_length = scaling[var_type_length, length_sym]

    seg_diam = variables_si[var_type_diam][diam_sym]
    max_diam = options['system_bounds'][var_type_diam][diam_sym][1]
    scaling_diam = scaling[var_type_diam, diam_sym]

    cross_section_area = np.pi * (seg_diam / 2.) ** 2.
    max_area = np.pi * (max_diam / 2.) ** 2.
    scaling_area = np.pi * (scaling_diam / 2.) ** 2.

    density = parameters['theta0', 'tether', 'rho']
    seg_mass = cross_section_area * density * seg_length
    scaling_mass = scaling_area * density * scaling_length

    props = {}
    props['density'] = density
    props['seg_length'] = seg_length
    props['scaling_length'] = scaling_length

    loyd_reelout_factor = 1. / 3.
    u_ref = parameters['theta0', 'wind', 'u_ref']

    plausible_speed_label = '[' + var_type_length + ',d' + length_sym + ',0]'
    if plausible_speed_label in scaling.labels():
        scaling_speed = scaling[var_type_length, 'd' + length_sym]
    else:
        scaling_speed = loyd_reelout_factor * u_ref
    props['scaling_speed'] = scaling_speed

    possible_var_types = ['x', 'u']
    ddl_t_type = None
    for var_type in possible_var_types:
       if 'ddl_t' in struct_op.subkeys(scaling, var_type):
           ddl_t_type = var_type

    if ddl_t_type is None:
        scaling_acceleration = np.max(options['system_bounds']['x']['ddl_t'])/2.
    else:
        scaling_acceleration = scaling[ddl_t_type, 'ddl_t']
    props['scaling_acc'] = scaling_acceleration

    props['seg_diam'] = seg_diam
    props['max_diam'] = max_diam
    props['scaling_diam'] = scaling_diam

    props['cross_section_area'] = cross_section_area
    props['max_area'] = max_area
    props['scaling_area'] = scaling_area

    props['seg_mass'] = seg_mass
    props['scaling_mass'] = scaling_mass

    return props

def get_body_axes(q_upper, q_lower):
    # todo: remove this the moment Rachel is done with verification testing.

    tether = q_upper - q_lower

    xhat = vect_op.xhat()
    yhat = vect_op.yhat()
    ehat_z = vect_op.normalize(tether)
    ehat_x = vect_op.normed_cross(yhat, tether)
    ehat_y = vect_op.normed_cross(ehat_z, ehat_x)

    return ehat_x, ehat_y, ehat_z


def from_earthfixed_to_body(earthfixed_vector, q_upper, q_lower):
    # todo: remove this the moment Rachel is done with verification testing.
    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)
    DCM = cas.horzcat(ehat_x, ehat_y, ehat_z)
    body_vector = kite_frames.from_earth_to_body(DCM, earthfixed_vector)
    return body_vector

def from_body_to_earthfixed(body_vector, q_upper, q_lower):
    # todo: remove this the moment Rachel is done with verification testing.
    [ehat_x, ehat_y, ehat_z] = get_body_axes(q_upper, q_lower)
    DCM = cas.horzcat(ehat_x, ehat_y, ehat_z)
    earthfixed_vector = kite_frames.from_body_to_earth(DCM, body_vector)
    return earthfixed_vector

def get_physical_forces(model_options, variables, parameters, atmos, wind, upper_node, cd_tether_fun, architecture):
    # todo: remove this the moment Rachel is done with verification testing.
    q_upper, q_lower, dq_upper, dq_lower = element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                   architecture)
    diam = element.get_element_diameter(variables, upper_node, architecture)
    [force_upper, force_lower] = get_equivalent_tether_drag_forces(variables, parameters, upper_node, architecture, model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind,
                                      cd_tether_fun)

    return [force_lower, force_upper]


def get_inverse_equivalence_matrix(tether_length):
    # todo: remove this the moment Rachel is done with verification testing.
    # equivalent forces at upper node = [a, b, c]
    # equivalent forces at lower node = [d, e, f]
    # total forces = [Fx, Fy, Fz]
    # total moment = [Mx, My, 0]

    # a + d = Fx
    # b + e = Fy
    # c + f = Fz
    # (L/2) (b - e) = Mx <- this is what it should be. at present, it says L (a - d) = Mx
    # (L/2) (a - d) = My <- this is what it should be. at present, it says L (b - e) = My
    # c - f = 0 <- the line is presently multiplied by a constant L. annoying but not harmful.

    # A [a, b, c, d, e, f].T = [Fx, Fy, Fz, Mx, My, 0].T
    # [a, b, c, d, e, f].T = Ainv [Fx, Fy, Fz, Mx, My, 0].T

    L = tether_length
    # L = tether_length / 2.

    Ainv = np.matrix([[0.5, 0., 0., 0., 1. / L, 0.],
                      [0., 0.5, 0., 1. / L, 0., 0.],
                      [0., 0., 0.5, 0., 0., 0.5],
                      [0.5, 0., 0., 0., -1. / L, 0.],
                      [0., 0.5, 0., -1. / L, 0., 0.],
                      [0., 0., 0.5, 0., 0., -0.5]])

    return Ainv


def get_equivalent_tether_drag_forces(variables, parameters, upper_node, architecture, model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind,
                                      cd_tether_fun):
    # todo: remove this the moment Rachel is done with verification testing.
    tether = q_upper - q_lower

    [total_force_earthfixed, total_moment_earthfixed] = get_total_drag(variables, parameters, upper_node, architecture, model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun)

    total_force_body = from_earthfixed_to_body(total_force_earthfixed, q_upper, q_lower)
    total_moment_body = from_earthfixed_to_body(total_moment_earthfixed, q_upper, q_lower)

    total_moment_body[2] = 0.

    total_vect = cas.vertcat(total_force_body, total_moment_body)

    Ainv = get_inverse_equivalence_matrix(vect_op.norm(tether))

    equiv_vect = cas.mtimes(Ainv, total_vect)

    equiv_force_upper_body = equiv_vect[0:3]
    equiv_force_lower_body = equiv_vect[3:6]

    equiv_force_upper_earthfixed = from_body_to_earthfixed(equiv_force_upper_body, q_upper, q_lower)
    equiv_force_lower_earthfixed = from_body_to_earthfixed(equiv_force_lower_body, q_upper, q_lower)

    return [equiv_force_upper_earthfixed, equiv_force_lower_earthfixed]

def get_total_drag(variables, parameters, upper_node, architecture, model_options, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, cd_tether_fun):

    elem = model_options['tether']['aero_elements']
    q_average = (q_upper + q_lower) / 2.
    total_force = np.zeros((3, 1))
    total_moment = np.zeros((3, 1))
    for idx in range(elem):

        loc_s_upper = float(idx + 1) / float(elem)
        loc_s_lower = float(idx) / float(elem)

        q_loc_upper = q_lower + loc_s_upper * (q_upper - q_lower)
        q_loc_lower = q_lower + loc_s_lower * (q_upper - q_lower)

        q_loc_average = (q_loc_lower + q_loc_upper)/2.
        moment_arm = q_average - q_loc_average

        dq_loc_upper = dq_lower + loc_s_upper * (dq_upper - dq_lower)
        dq_loc_lower = dq_lower + loc_s_lower * (dq_upper - dq_lower)

        # this is a bug, but I'd like to leave it in until I'm done with all of my wake model verification tests
        loc_force = get_segment_force(variables, parameters, upper_node, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, architecture, cd_tether_fun)
        # it should read:
        # loc_force = get_segment_force(variables, parameters, upper_node, diam, q_loc_upper, q_loc_lower, dq_loc_upper, dq_loc_lower, atmos, wind, architecture, cd_tether_fun)

        loc_moment = vect_op.cross(moment_arm, loc_force)

        total_force = total_force + loc_force
        total_moment = total_moment + loc_moment

    return [total_force, total_moment]

def get_segment_force(variables, parameters, upper_node, diam, q_upper, q_lower, dq_upper, dq_lower, atmos, wind, architecture, cd_tether_fun):

    q_average = (q_upper + q_lower) / 2.
    zz = q_average[2]

    uw_average = wind.get_velocity(zz)
    density = atmos.get_density(zz)

    dq_average = (dq_upper + dq_lower) / 2.
    ua = uw_average - dq_average

    ua_norm = vect_op.smooth_norm(ua, 1e-6)
    ehat_ua = vect_op.smooth_normalize(ua, 1e-6)

    tether = q_upper - q_lower

    length = vect_op.norm(tether)
    length_parallel_to_wind = cas.mtimes(tether.T, ehat_ua)
    length_perp_to_wind = (length**2. - length_parallel_to_wind**2.)**0.5

    reynolds = segment.get_segment_reynolds_number(variables, atmos, wind, upper_node, architecture)

    cd = cd_tether_fun(reynolds, parameters)

    drag = cd * 0.5 * density * ua_norm * diam * length_perp_to_wind * ua

    return drag

