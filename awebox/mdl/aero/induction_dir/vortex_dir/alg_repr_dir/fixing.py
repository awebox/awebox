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
constraints to create "intermediate condition" fixing constraints on the positions of the wake nodes,
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-25
'''

import numpy as np
from pandas.core.indexes.api import union_indexes

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger


################# define the actual constraint

def get_constraint(nlp_options, V, P, Xdot, Outputs, Integral_outputs, model, time_grids):
    cstr_list = cstr_op.ConstraintList()
    abbreviated_variables = vortex_tools.get_list_of_abbreviated_variables(nlp_options)
    for abbreviated_var_name in abbreviated_variables:

        if 'wx' in abbreviated_var_name:
            cstr_list.append(get_node_position_constraint(nlp_options, V, P, Xdot, Outputs, model, time_grids))
        elif 'wg' in abbreviated_var_name:
            cstr_list.append(get_circulation_strength_constraint(nlp_options, V, Outputs, Integral_outputs, model, time_grids))
        else:
            cstr_list.append(get_specific_constraint(abbreviated_var_name, nlp_options, V, Outputs, Integral_outputs, model, time_grids))

    return cstr_list

##############################################
##############################################

def get_node_position_constraint(nlp_options, V, P, Xdot, Outputs, model, time_grids):

    wake_nodes = nlp_options['induction']['vortex_wake_nodes']

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    t_f_scaled = V['theta', 't_f']
    t_f_si = struct_op.var_scaled_to_si('theta', 't_f', t_f_scaled, model.scaling)
    tgrid_coll = time_grids['coll'](t_f_si)

    message = 'adding vortex ocp constraints: convected position constraints'
    print_op.base_print(message, level='info')
    expected_number = len(model.architecture.kite_nodes) * 2 * (wake_nodes - 1) * n_k
    edx = 0

    resi_expr = []
    for kite_shed in model.architecture.kite_nodes:
        for tip in ['int', 'ext']:
            for wake_node in range(1, wake_nodes):
                fixing_name = vortex_tools.get_wake_node_position_name(kite_shed=kite_shed, tip=tip,
                                                                       wake_node=wake_node)

                for ndx in range(n_k):

                    for ddx in [None]:
                        fixing_position_scaled = V['z', ndx, fixing_name]
                        convected_position_scaled = try_getting_convected_position_scaled(nlp_options, fixing_name, kite_shed, tip, wake_node, tgrid_coll, V, P, Xdot, model, ndx, ddx)
                        local_resi = fixing_position_scaled - convected_position_scaled
                        resi_expr = cas.vertcat(resi_expr, local_resi)

                    for ddx in range(d):
                        fixing_position_scaled = V['coll_var', ndx, ddx, 'z', fixing_name]
                        convected_position_scaled = try_getting_convected_position_scaled(nlp_options, fixing_name, kite_shed, tip, wake_node, tgrid_coll, V, P, Xdot, model, ndx, ddx)
                        local_resi = fixing_position_scaled - convected_position_scaled
                        resi_expr = cas.vertcat(resi_expr, local_resi)

                    edx += 1
                    print_op.print_progress(edx, expected_number)

    local_cstr = cstr_op.Constraint(expr=resi_expr,
                                    name='wx_convection',
                                    cstr_type='eq')

    print_op.close_progress()

    return local_cstr


def try_getting_convected_position_scaled(nlp_options, fixing_name, kite_shed, tip, wake_node, tgrid_coll, V, P, Xdot, model, ndx, ddx):
    t_period = tgrid_coll[-1, -1]

    if ddx is None:
        ndx = ndx - 1
        ddx = -1

    anchor_name = vortex_tools.get_wake_node_position_name(kite_shed=kite_shed, tip=tip, wake_node=wake_node - 1)
    anchor_ndx = ndx - 1

    anchoring_position_scaled = V['coll_var', anchor_ndx, ddx, 'z', anchor_name]
    anchoring_position_si = struct_op.var_scaled_to_si('z', anchor_name, anchoring_position_scaled, model.scaling, check_should_multiply=False)
    convection_time = tgrid_coll[ndx, ddx] - tgrid_coll[anchor_ndx, ddx]
    if anchor_ndx < 0:
        convection_time += t_period

    u_local = model.wind.get_velocity(anchoring_position_si[2])
    if nlp_options['induction']['vortex_convection_type'] == 'free':
        fixing_position_scaled = V['coll_var', ndx, ddx, 'z', fixing_name]
        fixing_position_si = struct_op.var_scaled_to_si('z', fixing_name, fixing_position_scaled, model.scaling, check_should_multiply=False)
        x_halfways = (fixing_position_si + anchoring_position_si) / 2.
        variables_scaled = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, ndx, ddx)
        parameters = struct_op.get_parameters_at_time(V, P, model.parameters)
        u_ind = model.wake.calculate_total_biot_savart_at_x_obs(variables_scaled, parameters, x_obs=x_halfways)
        u_local += u_ind

    convection_distance_si = convection_time * u_local
    convection_distance_scaled = struct_op.var_si_to_scaled('z', fixing_name, convection_distance_si, model.scaling, check_should_multiply=False)
    return anchoring_position_scaled + convection_distance_scaled


##############################################
##############################################

def get_circulation_strength_constraint(nlp_options, V, Outputs, Integral_outputs, model, time_grids):
    wake_nodes = nlp_options['induction']['vortex_wake_nodes']

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    message = 'adding vortex ocp constraints: ring strength constraints'
    print_op.base_print(message, level='info')
    expected_number = len(model.architecture.kite_nodes) * wake_nodes * n_k
    edx = 0

    resi_expr = []
    for kite_shed in model.architecture.kite_nodes:
        for ring in range(wake_nodes):
            fixing_name = vortex_tools.get_vortex_ring_strength_name(kite_shed, ring)

            for ndx in range(n_k):
                for ddx in [None]:
                    fixing_strength_scaled = V['z', ndx, fixing_name]
                    strength_si = get_shedding_circulation_value(nlp_options, V, Outputs, Integral_outputs, model, time_grids,
                                                   kite_shed, ring, ndx, ddx)
                    strength_scaled = struct_op.var_si_to_scaled('z', fixing_name, strength_si, model.scaling)
                    local_resi = fixing_strength_scaled - strength_scaled
                    resi_expr = cas.vertcat(resi_expr, local_resi)

                for ddx in range(d):
                    fixing_strength_scaled = V['coll_var', ndx, ddx, 'z', fixing_name]
                    strength_si = get_shedding_circulation_value(nlp_options, V, Outputs, Integral_outputs, model, time_grids,
                                                   kite_shed, ring, ndx, ddx)
                    strength_scaled = struct_op.var_si_to_scaled('z', fixing_name, strength_si, model.scaling)
                    local_resi = fixing_strength_scaled - strength_scaled
                    resi_expr = cas.vertcat(resi_expr, local_resi)

                edx += 1
                print_op.print_progress(edx, expected_number)

    local_cstr = cstr_op.Constraint(expr=resi_expr,
                                    name='wg_fixing',
                                    cstr_type='eq')

    print_op.close_progress()

    return local_cstr



##############################################
##############################################


def get_specific_constraint(abbreviated_var_name, nlp_options, V, Outputs, Integral_outputs, model, time_grids):

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']

    kite_shed_or_parent_shed_list, tip_list, wake_node_or_ring_list = vortex_tools.get_kite_or_parent_and_tip_and_node_or_ring_list_for_abbreviated_vars(abbreviated_var_name, nlp_options, model.architecture)

    cstr_list = cstr_op.ConstraintList()

    for kite_shed_or_parent_shed in kite_shed_or_parent_shed_list:
        for tip in tip_list:
            for wake_node_or_ring in wake_node_or_ring_list:

                for ndx in range(n_k):
                    local_cstr = get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs,
                                                               Integral_outputs, model, time_grids,
                                                               kite_shed_or_parent_shed, tip, wake_node_or_ring, ndx)
                    cstr_list.append(local_cstr)

                    for ddx in range(d):
                        local_cstr = get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs,
                                                                   Integral_outputs, model,
                                                                   time_grids, kite_shed_or_parent_shed, tip,
                                                                   wake_node_or_ring, ndx, ddx)
                        cstr_list.append(local_cstr)

    return cstr_list


def get_specific_local_constraint(abbreviated_var_name, nlp_options, V, Outputs, Integral_outputs, model, time_grids, kite_shed_or_parent_shed, tip,
                                  wake_node_or_ring, ndx, ddx=None):

    var_name = vortex_tools.get_var_name(abbreviated_var_name, kite_shed_or_parent_shed=kite_shed_or_parent_shed, tip=tip, wake_node_or_ring=wake_node_or_ring)
    cstr_name = 'fixing_' + var_name + '_' + str(ndx)

    if 'z' not in V.keys():
        message = 'vortex model does not appear to be set up yet for this sort of discetization. have you tried turning on zoh controls?'
        print_op.log_and_raise_error(message)

    if ddx is None:
        var_symbolic_scaled = V['z', ndx, var_name]
        var_val_scaled = V['coll_var', ndx - 1, -1, 'z', var_name]
        resi_scaled = var_symbolic_scaled - var_val_scaled

    else:
        cstr_name += ',' + str(ddx)
        var_symbolic_scaled = V['coll_var', ndx, ddx, 'z', var_name]
        var_symbolic_si = struct_op.var_scaled_to_si('z', var_name, var_symbolic_scaled, model.scaling)

        # look-up the actual value from the Outputs. Keep the computing here minimal.
        if abbreviated_var_name == 'wh':
            resi_scaled = get_local_cylinder_pitch_residual(nlp_options, V, Outputs, model, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
        elif abbreviated_var_name == 'wx_center':
            var_value_si = get_local_cylinder_center_value(nlp_options, Outputs, kite_shed_or_parent_shed, wake_node_or_ring, ndx, ddx)
            resi_scaled = get_simple_residual(var_name, var_symbolic_si, var_value_si, model.scaling)
        else:
            message = 'get_specific_local_constraint function is not set up for this abbreviation (' + abbreviated_var_name + ') yet.'
            print_op.log_and_raise_error(message)

    local_cstr = cstr_op.Constraint(expr=resi_scaled,
                                    name=cstr_name,
                                    cstr_type='eq')

    return local_cstr


def get_simple_residual(var_name, var_symbolic_si, var_value_si, model_scaling):
    resi_si = var_symbolic_si - var_value_si
    resi_scaled = struct_op.var_si_to_scaled('z', var_name, resi_si, model_scaling)

    return resi_scaled



def get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx=None):

    # V['coll_var', ndx-1, -1, 'z', var_name] = V['z', ndx, var_name]

    if ddx is None:
        ndx = ndx - 1
        ddx = -1

    n_k = nlp_options['n_k']
    ddx_shed = ddx

    # # if wake_node = 0, then ndx_shed = ndx
    # # if wake_node = 1, then ndx_shed = (ndx - 1)
    # # .... if ndx_shed is 1, then ndx_shed -> 1
    # # ....  if ndx_shed is 0, then ndx_shed -> n_k
    # # ....  if ndx_shed is -1, then ndx_shed -> n_k - 1
    # # .... so, ndx_shed -> np.mod(ndx - wake_node, n_k)

    subtracted_ndx = ndx - wake_node
    ndx_shed = np.mod(subtracted_ndx, n_k)
    periods_passed = np.floor(subtracted_ndx / n_k)

    return ndx_shed, ddx_shed, periods_passed


########## wake node position

def get_local_convected_position_value(nlp_options, V, Outputs, model, time_grids, kite_shed, tip, wake_node, ndx, ddx):
    t_f_scaled = V['theta', 't_f']
    t_f_si = struct_op.var_scaled_to_si('theta', 't_f', t_f_scaled, model.scaling)
    tgrid_coll = time_grids['coll'](t_f_si)
    wx_convected = get_the_convected_position_from_the_current_indices_and_wake_node(nlp_options, Outputs, model, tgrid_coll, kite_shed, tip,
                                                                      wake_node, ndx, ddx)

    return wx_convected


def get_the_convected_position_from_the_current_indices_and_wake_node(nlp_options, Outputs, model, tgrid_coll, kite, tip, wake_node, ndx, ddx=None):

    ndx_shed, ddx_shed, periods_passed = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)

    wingtip_pos = get_the_wingtip_position_at_shedding_indices(Outputs, kite, tip, ndx_shed, ddx_shed)

    delta_t = get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tgrid_coll, wake_node, ndx, ddx)

    u_local = model.wind.get_velocity(wingtip_pos[2])

    wx_convected = wingtip_pos + delta_t * u_local

    return wx_convected


def get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tcoll, wake_node, ndx, ddx=None):

    ndx_search = np.mod(ndx, nlp_options['n_k'])
    if (ndx_search == 0) and (ddx is None):
        ndx_search = nlp_options['n_k'] - 1
        ddx_search = -1
    elif (ddx is None):
        ndx_search = ndx_search - 1
        ddx_search = -1
    else:
        ddx_search = ddx

    ndx_shed, ddx_shed, periods_passed = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options,
                                                                                                         wake_node, ndx_search,
                                                                                                         ddx_search)

    t_period = tcoll[-1, -1]
    shedding_time = t_period * periods_passed + tcoll[ndx_shed, ddx_shed]
    current_time = tcoll[ndx_search, ddx_search]
    delta_t = current_time - shedding_time

    return delta_t


def get_the_wingtip_position_at_shedding_indices(Outputs, kite, tip, ndx_shed, ddx_shed):
    wingtip_pos = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'wingtip_' + tip + str(kite)]
    return wingtip_pos


############## ring strength

def get_shedding_circulation_value(nlp_options, V, Outputs, Integral_outputs, model, time_grids, kite_shed, ring, ndx, ddx):
    filament_strength_from_circulation = model.options['aero']['vortex']['filament_strength_from_circulation']
    use_circulation_equality_pattern = model.options['aero']['vortex']['use_circulation_equality_pattern']

    if use_circulation_equality_pattern and ring > 0:
        arbitrary_tip = 'ext'
        pattern_var_name = vortex_tools.get_var_name('wg', kite_shed_or_parent_shed=kite_shed,
                                             tip=arbitrary_tip, wake_node_or_ring=0)
        ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, ring,
                                                                                                ndx, ddx)
        pattern_var_symbolic_scaled = V['coll_var', ndx_shed, ddx_shed, 'z', pattern_var_name]
        var_symbolic_si = struct_op.var_scaled_to_si('z', pattern_var_name, pattern_var_symbolic_scaled, model.scaling)
        return var_symbolic_si


    if filament_strength_from_circulation == 'averaged':
        return get_local_average_circulation_value(nlp_options, V, Integral_outputs, model, time_grids, kite_shed, ring, ndx, ddx)
    elif filament_strength_from_circulation == 'instantaneous':
        return get_local_instantaneous_circulation_value(nlp_options, Outputs, kite_shed, ring, ndx, ddx)
    else:
        message = 'unfamiliar option for how to determine the vortex ring strength (' + filament_strength_from_circulation + ')'
        print_op.log_and_raise_error(message)
    return None

def get_local_instantaneous_circulation_value(nlp_options, Outputs, kite_shed, ring, ndx, ddx):

    out_name = 'circulation' + str(kite_shed)
    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, ring, ndx,
                                                                                            ddx)
    instantaneous_circulation = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', out_name]

    return instantaneous_circulation


def get_local_average_circulation_value(nlp_options, V, Integral_outputs, model, time_grids, kite_shed, ring, ndx, ddx):

    int_name = 'integrated_circulation' + str(kite_shed)
    local_scaling = nlp_options['induction'][int_name]
    t_f_scaled = V['theta', 't_f']
    t_f_si = struct_op.var_scaled_to_si('theta', 't_f', t_f_scaled, model.scaling)
    tgrid_coll = time_grids['coll'](t_f_si)

    optimization_period = tgrid_coll[-1, -1]
    total_integrated_circulation_scaled = Integral_outputs['coll_int_out', -1, -1, int_name]

    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, ring, ndx,
                                                                                            ddx)

    integrated_circulation_shed_scaled = Integral_outputs['coll_int_out', ndx_shed, ddx_shed, int_name]
    integrated_circulation_shed_si = integrated_circulation_shed_scaled * local_scaling
    time_shed = tgrid_coll[ndx_shed, ddx_shed]

    ddx_before_shed = ddx_shed
    if ndx_shed == 0:
        ndx_before_shed = -1
        time_before_shed = tgrid_coll[ndx_before_shed, ddx_before_shed] - optimization_period
        integrated_circulation_before_shed_scaled = Integral_outputs['coll_int_out',
            ndx_before_shed, ddx_before_shed, int_name] - total_integrated_circulation_scaled

    else:
        ndx_before_shed = ndx_shed - 1
        time_before_shed = tgrid_coll[ndx_before_shed, ddx_before_shed]
        integrated_circulation_before_shed_scaled = Integral_outputs[
            'coll_int_out', ndx_before_shed, ddx_before_shed, int_name]

    integrated_circulation_before_shed_si = integrated_circulation_before_shed_scaled * local_scaling

    definite_integral_circulation_si = integrated_circulation_shed_si - integrated_circulation_before_shed_si
    delta_t = time_shed - time_before_shed

    average_circulation = definite_integral_circulation_si / delta_t

    return average_circulation


################ cylinder center


def get_local_cylinder_center_value(nlp_options, Outputs, parent_shed, wake_node, ndx, ddx=None):
    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)
    wx_center = get_the_cylinder_center_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed)
    return wx_center

def get_the_cylinder_center_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed):
    wx_center = Outputs['coll_outputs', ndx_shed, ddx_shed, 'geometry', 'x_center' + str(parent_shed)]
    return wx_center


################ cylinder pitch

def get_local_cylinder_pitch_residual(nlp_options, V, Outputs, model, parent_shed, wake_node, ndx, ddx=None):
    var_name = vortex_tools.get_var_name('wh', kite_shed_or_parent_shed=parent_shed,
                                         tip=None, wake_node_or_ring=wake_node)
    var_local_scaled = V['coll_var', ndx, ddx, 'z', var_name]
    pitch_si = struct_op.var_scaled_to_si('z', var_name, var_local_scaled, model.scaling)

    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)

    l_hat = model.wind.get_wind_direction()
    vec_u_zero = Outputs['coll_outputs', ndx_shed, ddx_shed, 'geometry', 'vec_u_zero' + str(parent_shed)]
    total_circulation = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'total_circulation' + str(parent_shed)]
    average_period_of_rotation = Outputs['coll_outputs', ndx_shed, ddx_shed, 'geometry', 'average_period_of_rotation' + str(parent_shed)]
    resi = general_flow.get_far_wake_cylinder_residual(pitch_si, l_hat, vec_u_zero, total_circulation, average_period_of_rotation)

    pitch_ref = struct_op.var_scaled_to_si('z', var_name, 1., model.scaling)
    scale = pitch_ref**2.

    resi_scaled = resi / scale

    return resi_scaled


def get_local_cylinder_pitch_value(nlp_options, Outputs, parent_shed, wake_node, ndx, ddx=None):
    ndx_shed, ddx_shed, _ = get_the_shedding_indices_from_the_current_indices_and_wake_node(nlp_options, wake_node, ndx, ddx)
    wh = get_the_cylinder_pitch_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed)
    return wh


def get_the_cylinder_pitch_at_shedding_indices(Outputs, parent_shed, ndx_shed, ddx_shed):
    pitch = Outputs['coll_outputs', ndx_shed, ddx_shed, 'aerodynamics', 'far_wake_cylinder_pitch' + str(parent_shed)]
    return pitch


########## test

def test_the_convection_time(epsilon=1.e-4):

    ndx = 5 # some number larger than the number of collocation nodes

    nlp_options, architecture, wind, var_struct, param_struct, variables_dict = alg_structure.construct_test_model_variable_structures()

    n_k = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    scheme = nlp_options['collocation']['scheme']
    tau_root = cas.vertcat(cas.collocation_points(d, scheme))

    width = 1.73  # some number that's not likely to arise 'naturally'

    tcoll = []
    for ndx in range(n_k):
        for ddx in range(d):
            tcoll = cas.vertcat(tcoll, width * (ndx + tau_root[ddx]))

    tcoll = tcoll.reshape((d, n_k)).T
    optimization_period = tcoll[-1, -1]

    for ddx in [None, 2]:

        if ddx is None:
            case_description_string = ' at a shooting node'
        else:
            case_description_string = ' at a collocation node'

        wake_nodes = {}
        found = {}
        expected = {}
        conditions = {}
        total_condition = 0

        # if wake_node = 0 -> delta_t == 0
        # if wake_node = n_k -> delta_t = t_final
        # if wake_node = 2 n_k -> delta_t = 2 * t_final

        wake_nodes[0] = 0 * n_k
        expected[0] = 0

        wake_nodes[1] = 1 * n_k
        expected[1] = 1 * optimization_period

        wake_nodes[2] = 2 * n_k
        expected[2] = 2 * optimization_period

        wake_nodes['partial'] = 1
        expected['partial'] = width

        for name, wake_node in wake_nodes.items():
            found[name] = get_the_convection_time_from_the_current_indices_and_wake_node(nlp_options, tcoll, wake_node, ndx, ddx)
            diff = found[name] - expected[name]

            local_condition = (diff**2. < epsilon**2.)
            conditions[name] = local_condition
            total_condition += local_condition

        criteria = (total_condition == len(wake_nodes.keys()))

        if not criteria:
            message = 'something went wrong when computing how long a given wake node has been convecting, ' + case_description_string
            print_op.log_and_raise_error(message)

    return None

def test():
    test_the_convection_time()

# test()
