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
lagrangian dynamics auto-generation module
that generates the dynamics residual
python-3.5 / casadi-3.4.5
- authors: jochem de schutter, rachel leuthold alu-fr 2017-20
'''


import casadi.tools as cas
import numpy as np

import itertools

from collections import OrderedDict

from . import system

import awebox.mdl.aero.kite_dir.kite_aero as kite_aero
import awebox.mdl.aero.tether_dir.tether_aero as tether_aero
import awebox.mdl.aero.induction_dir.induction as induction

import awebox.mdl.lagr_dyn_dir.lagr_dyn as lagr_dyn
import awebox.mdl.lagr_dyn_dir.tools as lagr_tools

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op

from awebox.logger.logger import Logger as awelogger

def make_dynamics(options, atmos, wind, parameters, architecture):

    # system architecture (see zanon2013a)

    # --------------------------------------------------------------------------------------
    # generate system states, controls, algebraic vars, lifted vars, generalized coordinates
    # --------------------------------------------------------------------------------------
    [system_variable_list, system_gc] = system.generate_structure(options, architecture)

    # -----------------------------------
    # generate structured SX.sym objects
    # -----------------------------------
    system_variables = {}
    system_variables['scaled'], variables_dict = struct_op.generate_variable_struct(system_variable_list)
    scaling = generate_scaling(options['scaling'], system_variables['scaled'])
    system_variables['SI'] = generate_si_variables(scaling, system_variables['scaled'])

    variables_to_return_to_model = system_variables['scaled']

    print_op.print_variable_info('model', variables_to_return_to_model)

    # -----------------------------------
    # prepare empty constraints list and outputs
    # -----------------------------------
    cstr_list = cstr_op.MdlConstraintList()
    outputs = {}

    # ---------------------------------
    # define the equality constraints (aka. dynamics)
    # ---------------------------------

    wake = induction.get_wake_if_vortex_model_is_included_in_comparison(options, architecture, wind, system_variables['SI'], parameters)

    # enforce the lagrangian dynamics
    lagr_dyn_cstr, outputs = lagr_dyn.get_dynamics(options, atmos, wind, architecture, system_variables, system_gc, parameters, outputs, wake, scaling)
    cstr_list.append(lagr_dyn_cstr)


    # enforce lifted aerodynamic force <-- this must happen after lagr_dyn.get_dynamics, which determines the kite indicators
    if options['aero']['lift_aero_force']:
        aero_force_cstr = kite_aero.get_force_cstr(options, system_variables['SI'], atmos, wind, architecture, parameters, outputs)
        cstr_list.append(aero_force_cstr)

    # enforce lifted tether force
    if options['tether']['lift_tether_force']:
        tether_force_cstr = tether_aero.get_tether_cstr(options, system_variables['SI'], architecture, outputs)
        cstr_list.append(tether_force_cstr)

    # induction constraint
    if options['induction_model'] == 'not_in_use':
        pass
    else:
        induction_cstr = induction.get_model_constraints(options, wake, scaling, atmos, wind, system_variables, parameters, outputs, architecture)
        cstr_list.append(induction_cstr)

    # specify the required integrations, including that
    # the energy is the integral of the instantaneous power
    derivative_dict = get_dictionary_of_derivatives(options, system_variables, parameters, atmos, wind, outputs, architecture, scaling)
    integral_outputs, integral_outputs_fun, integral_scaling, alongside_integration_cstr = manage_alongside_integration(options,
                                                                                                                derivative_dict,
                                                                                                                system_variables,
                                                                                                                parameters)

    cstr_list.append(alongside_integration_cstr)


    # --------------------------------------------
    # define the inequality constraints
    # --------------------------------------------

    outputs, stress_cstr = tether_stress_inequality(options, system_variables['SI'], outputs, parameters, architecture, scaling)
    cstr_list.append(stress_cstr)

    airspeed_cstr = airspeed_inequality(options, outputs, parameters, wind, architecture)
    cstr_list.append(airspeed_cstr)

    aero_validity_cstr = aero_validity_inequality(options, outputs)
    cstr_list.append(aero_validity_cstr)

    anticollision_cstr = anticollision_inequality(options, system_variables['SI'], parameters, architecture)
    cstr_list.append(anticollision_cstr)

    acceleration_cstr = acceleration_inequality(options, system_variables['SI'])
    cstr_list.append(acceleration_cstr)

    outputs, rotation_cstr = rotation_inequality(options, system_variables['SI'], parameters, architecture, outputs)
    cstr_list.append(rotation_cstr)
 
    power, outputs = get_power(options, system_variables, parameters, outputs, architecture, scaling)
    max_power_cstr = max_power_inequality(options, system_variables['SI'], power)
    cstr_list.append(max_power_cstr)

    outputs, ellips_cstr = ellipsoidal_flight_constraint(options, system_variables['SI'], parameters, architecture, outputs)
    cstr_list.append(ellips_cstr)

    outputs, elev_azim_cstr = elevation_azimuth_constraint(options, system_variables['SI'], parameters, architecture, outputs)
    cstr_list.append(elev_azim_cstr)

    # ----------------------------------------
    #  sanity checking
    # ----------------------------------------

    check_that_all_xdot_vars_are_represented_in_dynamics(cstr_list, variables_dict, system_variables['scaled'])

    # ----------------------------------------
    #  construct outputs structure
    # ----------------------------------------

    # include the inequality constraints into the outputs
    outputs['model_inequalities'] = {}
    for cstr in cstr_list.get_list('ineq'):
        outputs['model_inequalities'][cstr.name] = cstr.expr

    # include the equality constraints into the outputs
    outputs['model_equalities'] = {}
    for cstr in cstr_list.get_list('eq'):
        outputs['model_equalities'][cstr.name] = cstr.expr

    # add other relevant outputs
    outputs = xdot_outputs(system_variables['SI'], outputs)
    outputs = power_balance_outputs(options, outputs, system_variables,
                                    parameters, architecture, scaling)  # power balance must be after tether stress inequality

    # system output function
    [out, out_fun, out_dict] = make_output_structure(outputs, system_variables, parameters)

    # ----------------------------------------
    #  return
    # ----------------------------------------

    return [
        variables_to_return_to_model,
        variables_dict,
        scaling,
        cstr_list,
        out,
        out_fun,
        out_dict,
        integral_outputs,
        integral_outputs_fun,
        integral_scaling,
        wake
    ]


def check_that_all_xdot_vars_are_represented_in_dynamics(cstr_list, variables_dict, variables_scaled):
    dynamics = cstr_list.get_expression_list('eq')

    for var_name in variables_dict['xdot'].keys():
        local_jac = cas.jacobian(dynamics, variables_scaled['xdot', var_name])
        if not (local_jac.nnz() > 0):
            message = 'xdot variable ' + str(var_name) + ' does not seem to be constrained in the model dynamics. expect poor sosc behavior.'
            awelogger.logger.warning(message)

    return None

def get_dictionary_of_derivatives(model_options, system_variables, parameters, atmos, wind, outputs, architecture, scaling):

    # ensure that energy matches power integration
    if model_options['trajectory']['type'] == 'power_cycle':
        power_si, _ = get_power(model_options, system_variables, parameters, outputs, architecture, scaling)
        energy_scaling = model_options['scaling']['x']['e']
        derivative_dict = {'e': (power_si, energy_scaling)}

    elif model_options['trajectory']['type'] == 'aaa':
        tether_force_si = system_variables['SI']['z']['lambda10'] * system_variables['SI']['theta']['l_t']
        force_scaling = model_options['scaling']['z']['lambda10']
        derivative_dict = {'f10': (tether_force_si, force_scaling)}

        if model_options['aero']['vortex_rings']['N_rings'] != 0:
            bref = parameters['theta0', 'geometry', 'b_ref']
            ar = parameters['theta0', 'geometry', 'ar']
            for j in [2,3]:
                va = outputs['aerodynamics']['airspeed{}'.format(j)]
                CL = system_variables['SI']['x']['coeff{}1'.format(j)][0]
                gamma = 2 * bref / (np.pi * ar * va * CL)
                derivative_dict['gamma_{}'.format(j)] = (gamma, 1)

                lift_vec = outputs['aerodynamics']['ehat_up{}'.format(j)]
                for i in range(3):
                    derivative_dict['normal_vec_{}_{}'.format(j, i)] = (lift_vec[i], 1)

    if model_options['kite_dof'] == 6 and model_options['beta_cost']:
        beta_scaling = 1.
        beta_si = 0.
        for kite in architecture.kite_nodes:
            beta_si += outputs['aerodynamics']['beta{}'.format(kite)]**2
        beta_si = beta_si / len(architecture.kite_nodes)
        derivative_dict['beta_cost'] =  (beta_si, beta_scaling)
    if model_options['trajectory']['system_type'] == 'drag_mode':
        power_derivative_sq_scaling = 1.
        power_derivative_sq = outputs['performance']['power_derivative']**2
        derivative_dict['power_derivative_sq'] = (power_derivative_sq, power_derivative_sq_scaling)

    induction_derivative_dict = induction.get_dictionary_of_derivatives(model_options, system_variables, parameters, atmos, wind, outputs, architecture)
    for local_key, local_val in induction_derivative_dict.items():
        if not local_key in derivative_dict.keys():
            derivative_dict[local_key] = local_val

    if model_options['integration']['include_integration_test']:
        derivative_dict['total_time_unscaled'] = (cas.DM(1.), cas.DM(1.))
        total_time_scaling = model_options['scaling']['x']['total_time_scaled']
        derivative_dict['total_time_scaled'] = (cas.DM(1.), total_time_scaling)  # second value is some arbitrary large number.

    return derivative_dict


def manage_alongside_integration(model_options, derivative_dict, system_variables, parameters):

    integral_outputs_expr_entries = []
    integral_outputs_struct_entries = []
    cstr_list = cstr_op.MdlConstraintList()

    integral_scaling = {}

    for integral_var_name, integral_derivative_tuple in derivative_dict.items():

        integral_derivative_expression_si = integral_derivative_tuple[0]
        local_scaling = integral_derivative_tuple[1]

        local_expression_scaled = integral_derivative_expression_si / local_scaling
        integral_scaling[integral_var_name] = local_scaling

        if model_options['integral_outputs']:
            local_expr = local_expression_scaled
            integral_outputs_expr_entries += [cas.entry(integral_var_name, expr=local_expr)]

            local_shape = local_expr.shape
            integral_outputs_struct_entries += [cas.entry(integral_var_name, shape=local_shape)]

        else:
            local_resi = (system_variables['scaled']['xdot', 'd' + integral_var_name] - local_expression_scaled)
            local_cstr = cstr_op.Constraint(expr=local_resi,
                                           name='integral_' + integral_var_name,
                                           cstr_type='eq')
            cstr_list.append(local_cstr)

    integral_outputs = cas.struct_SX(integral_outputs_expr_entries)

    # dynamics function options
    if model_options['construction']['jit_code_gen']['include']:
        opts = {'jit': True, 'compiler': model_options['construction']['jit_code_gen']['compiler']}
    else:
        opts = {}

    integral_outputs_fun = cas.Function('integral_outputs', [system_variables['scaled'], parameters],
                                        [integral_outputs], opts)

    return integral_outputs, integral_outputs_fun, integral_scaling, cstr_list


def make_output_structure(outputs, system_variables, parameters):
    outputs_vec = []
    full_list = []

    outputs_dict = {}

    for output_type in list(outputs.keys()):

        local_list = []
        for name in list(outputs[output_type].keys()):
            # prepare empty entry list to generate substruct
            local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

            # generate vector with outputs - SX expressions
            outputs_vec = cas.vertcat(outputs_vec, outputs[output_type][name])

        # generate dict with sub-structs
        outputs_dict[output_type] = cas.struct_symMX(local_list)
        # prepare list with sub-structs to generate struct
        full_list += [cas.entry(output_type, struct=outputs_dict[output_type])]

    # generate "empty" structure
    out_struct = cas.struct_symMX(full_list)
    # generate structure with SX expressions
    outputs_struct = out_struct(outputs_vec)
    # generate outputs function
    outputs_fun = cas.Function('outputs', [system_variables['scaled'], parameters], [outputs_struct.cat])

    return [out_struct, outputs_fun, outputs_dict]


def get_drag_power_from_kite(kite, variables_si, parameters, outputs, architecture):
    parent = architecture.parent_map[kite]
    kite_drag_power = parameters['theta0', 'aero', 'turbine_efficiency'] * \
        cas.mtimes(
            outputs['aerodynamics']['air_velocity{}'.format(kite)].T,
            outputs['aerodynamics']['f_gen{}'.format(kite)]
        )
    return kite_drag_power


def get_power(options, system_variables, parameters, outputs, architecture, scaling):
    variables_si = system_variables['SI']
    if options['trajectory']['system_type'] == 'drag_mode':
        power = cas.SX.zeros(1, 1)
        for kite in architecture.kite_nodes:
            power += get_drag_power_from_kite(kite, variables_si, parameters, outputs, architecture)
        outputs['performance']['p_current'] = power
        outputs['performance']['power_derivative'] = lagr_tools.time_derivative(power, system_variables['scaled'], architecture, scaling)
    else:
        if 'l_t' in variables_si['x'].keys():
            power = variables_si['z']['lambda10'] * variables_si['x']['l_t'] * variables_si['x']['dl_t']
        else:
            power = cas.SX(0.0)
        outputs['performance']['p_current'] = power

    return power, outputs


def drag_mode_outputs(variables_si, parameters, outputs, architecture):
    for kite in architecture.kite_nodes:
        outputs['power_balance']['P_gen{}'.format(kite)] = -1. * get_drag_power_from_kite(kite, variables_si, parameters, outputs, architecture)

    return outputs


def power_balance_outputs(options, outputs, system_variables, parameters, architecture, scaling):
    variables_si = system_variables['SI']

    # all aerodynamic forces have already been added to power balance, by this point.
    # outputs['power_balance'] is not empty!

    if options['trajectory']['system_type'] == 'drag_mode':
        outputs = drag_mode_outputs(variables_si, parameters, outputs, architecture)

    outputs = tether_power_outputs(variables_si, outputs, architecture)
    outputs = kinetic_power_outputs(outputs, system_variables, architecture, scaling)
    outputs = potential_power_outputs(outputs, system_variables, architecture, scaling)

    if options['test']['check_energy_summation']:
        outputs = comparison_kinetic_and_potential_power_outputs(outputs, system_variables, architecture, scaling)

    return outputs


def comparison_kinetic_and_potential_power_outputs(outputs, system_variables, architecture, scaling):

    outputs['power_balance_comparison'] = {}

    types = ['potential', 'kinetic']

    for type in types:
        dict = outputs['e_' + type]
        e_local = 0.

        for keyname in dict.keys():
            e_local += dict[keyname]

        # rate of change in kinetic energy
        P = lagr_tools.time_derivative(e_local, system_variables['scaled'], architecture, scaling)

        # convention: negative when kinetic energy is added to the system
        outputs['power_balance_comparison'][type] = -1. * P

    return outputs


def tether_power_outputs(variables_si, outputs, architecture):
    # compute instantaneous power transferred by each tether
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]

        # node positions
        q_n = variables_si['x']['q' + str(n) + str(parent)]
        if n > 1:
            grandparent = architecture.parent_map[parent]
            q_p = variables_si['x']['q' + str(parent) + str(grandparent)]
        else:
            q_p = cas.SX.zeros((3, 1))

        # node velocity
        dq_n = variables_si['x']['dq' + str(n) + str(parent)]
        # force and direction
        tether_force = outputs['local_performance']['tether_force' + str(n) + str(parent)]
        tether_direction = vect_op.normalize(q_n - q_p)
        # power transferred (convention: negative when power is transferred down the tree)
        outputs['power_balance']['P_tether' + str(n)] = -cas.mtimes(tether_force * tether_direction.T, dq_n)

    return outputs


def kinetic_power_outputs(outputs, system_variables, architecture, scaling):
    """Compute rate of change of kinetic energy for all system nodes

    @return: outputs updated outputs dict
    """

    # notice that the quality test uses a normalized value of each power source, so scaling should be irrelevant
    # but scaled variables are the decision variables, for which cas.jacobian is defined
    # whereas SI values are multiples of the base values, for which cas.jacobian cannot be computed

    # kinetic and potential energy in the system
    for n in range(1, architecture.number_of_nodes):
        for source in outputs['e_kinetic'].keys():

            # rate of change in kinetic energy
            e_local = outputs['e_kinetic'][source]
            power_local = lagr_tools.time_derivative(e_local, system_variables['scaled'], architecture, scaling)

            # convention: negative when energy is added to the system
            outputs['power_balance']['P_kin_' + source] = -1. * power_local

    return outputs


def potential_power_outputs(outputs, system_variables, architecture, scaling):
    """Compute rate of change of potential energy for all system nodes

    @return: outputs updated outputs dict
    """

    # notice that the quality test uses a normalized value of each power source, so scaling should be irrelevant
    # but scaled variables are the decision variables, for which cas.jacobian is defined
    # whereas SI values are multiples of the base values, for which cas.jacobian cannot be computed

    # kinetic and potential energy in the system
    for source in outputs['e_potential'].keys():

        # rate of change in potential energy (ignore in-outflow of potential energy)
        e_local = outputs['e_potential'][source]
        power_local = lagr_tools.time_derivative(e_local, system_variables['scaled'], architecture, scaling)

        # convention: negative when potential energy is added to the system
        outputs['power_balance']['P_pot_' + source] = -1. * power_local

    return outputs


def xdot_outputs(variables, outputs):
    outputs['xdot_from_var'] = variables['xdot']
    return outputs


def anticollision_inequality(options, variables, parameters, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['anticollision']['include']:

        safety_factor = options['model_bounds']['anticollision']['safety_factor']
        dist_min = safety_factor * parameters['theta0', 'geometry', 'b_ref']

        kite_combinations = itertools.combinations(kite_nodes, 2)
        for kite_pair in kite_combinations:
            kite_a = kite_pair[0]
            kite_b = kite_pair[1]
            parent_a = parent_map[kite_a]
            parent_b = parent_map[kite_b]
            vec_q_a = variables['x']['q' + str(kite_a) + str(parent_a)]
            vec_q_b = variables['x']['q' + str(kite_b) + str(parent_b)]
            dist = vec_q_a - vec_q_b
            dist_inequality = 1 - (cas.mtimes(dist.T, dist) / dist_min ** 2)

            anticollision_cstr = cstr_op.Constraint(expr=dist_inequality,
                                                  name='anticollision' + str(kite_a) + str(kite_b),
                                                  cstr_type='ineq')
            cstr_list.append(anticollision_cstr)

    return cstr_list


def dcoeff_actuation_inequality(options, variables_si, parameters, architecture):

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['dcoeff_actuation']['include']:

        # nu*u_max + (1 - nu)*u_compromised_max > u
        # u - nu*u_max + (1 - nu)*u_compromised_max < 0
        if int(options['kite_dof']) != 3:
            return cstr_list

        nu = parameters['phi', 'nu']
        dcoeff_max = options['aero']['three_dof']['dcoeff_max']
        dcoeff_compromised_max = parameters['theta0', 'model_bounds', 'dcoeff_compromised_max']
        dcoeff_min = options['aero']['three_dof']['dcoeff_min']
        dcoeff_compromised_min = parameters['theta0', 'model_bounds', 'dcoeff_compromised_min']
        traj_type = options['trajectory']['type']

        for variable in variables_si['u'].keys():

            var_name, kiteparent = struct_op.split_name_and_node_identifier(variable)
            dcoeff = variables_si['u'][variable]
            kite, parent = struct_op.split_kite_and_parent(kiteparent, architecture)
            scenario, broken_kite = options['compromised_landing']['emergency_scenario']

            if var_name == 'dcoeff':
                if (traj_type == 'compromised_landing') and (kite == broken_kite) and (scenario == 'broken_roll'):
                    applied_max = (nu * dcoeff_max + (1 - nu) * dcoeff_compromised_max)
                    applied_min = (nu * dcoeff_min + (1 - nu) * dcoeff_compromised_min)

                else:
                    applied_max = options['aero']['three_dof']['dcoeff_max']
                    applied_min = options['aero']['three_dof']['dcoeff_min']

                resi_max = (dcoeff - applied_max)
                resi_min = (applied_min - dcoeff)

                max_cstr = cstr_op.Constraint(expr=resi_max,
                                            name='dcoeff_max' + kiteparent,
                                            cstr_type='ineq')
                cstr_list.append(max_cstr)

                min_cstr = cstr_op.Constraint(expr=resi_min,
                                            name='dcoeff_min' + kiteparent,
                                            cstr_type='ineq')
                cstr_list.append(min_cstr)

    return cstr_list


def coeff_actuation_inequality(options, variables_si, parameters, architecture):

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['coeff_actuation']['include']:

        # nu*x_max + (1 - nu)*u_compromised_max > x
        # x - nu*x_max + (1 - nu)*x_compromised_max < 0
        if int(options['kite_dof']) != 3:
            return cstr_list

        nu = parameters['phi', 'nu']
        coeff_max = options['aero']['three_dof']['coeff_max']
        coeff_compromised_max = parameters['theta0', 'model_bounds', 'coeff_compromised_max']
        coeff_min = options['aero']['three_dof']['coeff_min']
        coeff_compromised_min = parameters['theta0', 'model_bounds', 'coeff_compromised_min']
        traj_type = options['trajectory']['type']

        for variable in list(variables_si['x'].keys()):

            var_name, kiteparent = struct_op.split_name_and_node_identifier(variable)
            if var_name == 'coeff':

                coeff = variables_si['x'][variable]
                scenario, broken_kite = options['compromised_landing']['emergency_scenario']
                kite, parent = struct_op.split_kite_and_parent(kiteparent, architecture)

                if (traj_type == 'compromised_landing') and (kite == broken_kite) and (scenario == 'structural_damages'):
                    applied_max = (nu * coeff_max + (1 - nu) * coeff_compromised_max)
                    applied_min = (nu * coeff_min + (1 - nu) * coeff_compromised_min)
                else:
                    applied_max = coeff_max
                    applied_min = coeff_min

                resi_max = (coeff - applied_max)
                resi_min = (applied_min - coeff)

                max_cstr = cstr_op.Constraint(expr=resi_max,
                                            name='coeff_max' + kiteparent,
                                            cstr_type='ineq')
                cstr_list.append(max_cstr)

                min_cstr = cstr_op.Constraint(expr=resi_min,
                                            name='coeff_min' + kiteparent,
                                            cstr_type='ineq')
                cstr_list.append(min_cstr)

    return cstr_list


def max_power_inequality(options, variables, power):

    cstr_list = cstr_op.MdlConstraintList()

    if 'P_max' in variables['theta'].keys():
        max_power_ineq = (power - variables['theta']['P_max'])/options['scaling']['theta']['P_max']

        max_power_cstr = cstr_op.Constraint(expr=max_power_ineq,
                                    name='max_power_cstr',
                                    cstr_type='ineq')
        cstr_list.append(max_power_cstr)

    return cstr_list


def ellipsoidal_flight_constraint(options, variables, parameters, architecture, outputs):

    cstr_list = cstr_op.MdlConstraintList()

    alpha = parameters['theta0', 'model_bounds', 'ellipsoidal_flight_region', 'alpha']
    if 'ell_radius' in list(variables['theta'].keys()):
        r = variables['theta']['ell_radius'] - parameters['theta0', 'geometry', 'b_ref']
    else:
        r = parameters['theta0', 'model_bounds', 'ellipsoidal_flight_region', 'radius'] - parameters['theta0', 'geometry', 'b_ref']
    if options['model_bounds']['ellipsoidal_flight_region']['include']:
        for node in range(1,architecture.number_of_nodes):
            q = variables['x']['q{}'.format(architecture.node_label(node))]

            yy = q[1]
            zz = - q[0]*np.sin(alpha) + q[2]*np.cos(alpha)
            ellipse_ineq = zz**2/(r*np.sin(alpha))**2 + yy**2/r**2 - 1

            ellipse_cstr = cstr_op.Constraint(expr=ellipse_ineq,
                                        name='ellipse_flight' + architecture.node_label(node),
                                        cstr_type='ineq')
            cstr_list.append(ellipse_cstr)

    return outputs, cstr_list

def elevation_azimuth_constraint(options, variables, parameters, architecture, outputs):


    cstr_list = cstr_op.MdlConstraintList()

    bounds = parameters['theta0', 'model_bounds', 'azimuth_elevation', 'bounds']
    el_min = bounds[0]
    el_max = bounds[1]
    az_min = bounds[2]
    if options['model_bounds']['azimuth_elevation']['include']:
        l_t = variables['theta']['l_t']
        q10 = variables['x']['q10']

        xx = q10[0]
        yy = q10[1]
        zz = q10[2]

        el_az_eq = cas.vertcat(
            l_t * np.sin(el_min) - zz,
            zz - l_t * np.sin(el_max),
            xx * np.tan(az_min) - yy
        )

        elev_azim_cstr = cstr_op.Constraint(expr=el_az_eq,
                                    name='elevation_azimuth_bounds',
                                    cstr_type='ineq')
        cstr_list.append(elev_azim_cstr)

    return outputs, cstr_list

def acceleration_inequality(options, variables):

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['acceleration']['include']:

        acc_max = options['model_bounds']['acceleration']['acc_max'] * options['scaling']['other']['g']

        for name in list(variables['xdot'].keys()):

            var_name, var_label = struct_op.split_name_and_node_identifier(name)

            if var_name == 'ddq':
                acc = variables['xdot'][name]
                acc_sq = cas.mtimes(acc.T, acc)
                acc_sq_norm = acc_sq / acc_max ** 2.

                # acc^2 < acc_max^2 -> acc^2 / acc_max^2 - 1 < 0
                local_ineq = acc_sq_norm - 1.

                acc_cstr = cstr_op.Constraint(expr=local_ineq,
                                            name='acceleration' + var_label,
                                            cstr_type='ineq')
                cstr_list.append(acc_cstr)

    return cstr_list


def airspeed_inequality(options, outputs, parameters, wind, architecture):

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['airspeed']['include']:

        # system architecture
        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map

        # constraint bounds
        airspeed_max = parameters['theta0', 'model_bounds', 'airspeed_limits'][1]
        airspeed_min = parameters['theta0', 'model_bounds', 'airspeed_limits'][0]
        airspeed_scaling = wind.get_speed_ref(options)

        for kite in kite_nodes:
            airspeed = outputs['aerodynamics']['airspeed' + str(kite)]
            parent = parent_map[kite]

            max_resi = airspeed - airspeed_max
            min_resi = airspeed_min - airspeed

            max_cstr = cstr_op.Constraint(expr=max_resi / airspeed_scaling,
                                        name='airspeed_max' + str(kite) + str(parent),
                                        cstr_type='ineq')
            cstr_list.append(max_cstr)

            min_cstr = cstr_op.Constraint(expr=min_resi / airspeed_scaling,
                                        name='airspeed_min' + str(kite) + str(parent),
                                        cstr_type='ineq')
            cstr_list.append(min_cstr)

    return cstr_list


def aero_validity_inequality(options, outputs):

    cstr_list = cstr_op.MdlConstraintList()

    if options['model_bounds']['aero_validity']['include']:
        for name in outputs['aero_validity'].keys():

            ineq = outputs['aero_validity'][name]
            valid_cstr = cstr_op.Constraint(expr=ineq,
                                            name=name,
                                            cstr_type='ineq')
            cstr_list.append(valid_cstr)

    return cstr_list


def tether_stress_inequality(options, variables_si, outputs, parameters, architecture, scaling):

    cstr_list = cstr_op.MdlConstraintList()

    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # system (scaled) variables
    z = variables_si['z']
    theta = variables_si['theta']

    tightness = options['model_bounds']['tether_stress']['scaling']

    if 'local_performance' not in outputs.keys():
        outputs['local_performance'] = {}

    # mass vector, containing the mass of all nodes
    for node in range(1, number_of_nodes):

        parent = parent_map[node]
        node_label = str(node) + str(parent)

        seg_props = tether_aero.get_tether_segment_properties(options, architecture, scaling, variables_si, parameters, upper_node=node)
        seg_length = seg_props['seg_length']
        cross_section_area = seg_props['cross_section_area']

        tension = z['lambda' + node_label] * seg_length

        min_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][0]
        max_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][1]

        maximum_allowed_stress = parameters['theta0', 'tether', 'max_stress'] / parameters['theta0', 'tether', 'stress_safety_factor']
        characteristic_tension = vect_op.smooth_abs(scaling['z', 'lambda' + node_label] * seg_props['scaling_length'])

        # outputs related to the constraints themselves
        tether_constraint_includes = options['model_bounds']['tether']['tether_constraint_includes']

        if node in tether_constraint_includes['stress']:

            # stress_max = max_tension_from_stress / A_max
            # (tension / A) < stress_max
            # tension < A * stress_max
            # tension - A * stress_max < 0
            # (tension - A * stress_max) / characteristic_tension < 0
            stress_inequality_untightened = tension - cross_section_area * maximum_allowed_stress
            stress_inequality = stress_inequality_untightened / characteristic_tension * tightness

            stress_cstr = cstr_op.Constraint(expr=stress_inequality,
                                           name='tether_stress' + node_label,
                                           cstr_type='ineq')
            cstr_list.append(stress_cstr)

        elif node in tether_constraint_includes['force']:

            lambda_scaling = scaling['z', 'lambda' + node_label]
            length_scaling = seg_props['scaling_length']
            force_scaling = lambda_scaling * length_scaling

            force_max_resi = tension - max_tension
            force_min_resi = min_tension - tension

            force_max_cstr = cstr_op.Constraint(expr=force_max_resi / force_scaling,
                                               name='tether_force_max' + node_label,
                                               cstr_type='ineq')
            cstr_list.append(force_max_cstr)

            force_min_cstr = cstr_op.Constraint(expr=force_min_resi / force_scaling,
                                               name='tether_force_min' + node_label,
                                               cstr_type='ineq')
            cstr_list.append(force_min_cstr)

        # outputs so that the user can find the stress and tension
        outputs['local_performance']['tether_stress' + node_label] = tension / cross_section_area
        outputs['local_performance']['tether_force' + node_label] = tension

    if options['cross_tether'] and len(architecture.kite_nodes) > 1:
        for l in architecture.layer_nodes:
            kites = architecture.kites_map[l]
            seg_length = theta['l_c{}'.format(l)]
            seg_diam = theta['diam_c{}'.format(l)]
            cross_section = np.pi * seg_diam ** 2. / 4.
            cross_section_max = np.pi * options['system_bounds']['theta']['diam_c'][1] ** 2.0 / 4.
            max_tension_from_stress = maximum_allowed_stress * cross_section_max

            if len(kites) == 2:
                tension = z['lambda{}{}'.format(kites[0], kites[1])] * seg_length
                outputs['local_performance']['tether_stress{}{}'.format(kites[0], kites[1])] = tension / cross_section
                outputs['local_performance']['tether_force{}{}'.format(kites[0], kites[1])] = tension

                stress_inequality_untightened = tension / max_tension_from_stress - cross_section / cross_section_max
                stress_ineq_tightened = stress_inequality_untightened * tightness

                stress_cstr = cstr_op.Constraint(expr=stress_ineq_tightened,
                                               name='tether_stress' + str(kites[0]) + str(kites[1]),
                                               cstr_type='ineq')
                cstr_list.append(stress_cstr)

            else:
                for kdx in range(len(kites)):
                    cdx = (kdx + 1) % len(kites)
                    label = '{}{}'.format(kites[kdx], kites[cdx])

                    tension = z['lambda' + label] * seg_length
                    outputs['local_performance']['tether_stress' + label] = tension / cross_section
                    outputs['local_performance']['tether_force' + label] = tension

                    stress_inequality_untightened = tension / max_tension_from_stress - cross_section / cross_section_max
                    stress_ineq_tightened = stress_inequality_untightened * tightness

                    stress_cstr = cstr_op.Constraint(expr=stress_ineq_tightened,
                                                   name='tether_stress' + label,
                                                   cstr_type='ineq')
                    cstr_list.append(stress_cstr)

    return outputs, cstr_list


def generate_scaling(scaling_options, variables):

    scaling = variables(1.)

    # set of variable labels for which no scaling is provided
    unset_set = []

    # set the non-derivative variable scalings
    for var_type in set(scaling.keys()) - set(['xdot']):
        for var_name in struct_op.subkeys(scaling, var_type):

            split_name, kiteparent = struct_op.split_name_and_node_identifier(var_name)
            integral_name, integral_order = struct_op.split_name_and_integral_order(var_name)
            split_integral_name, kiteparent = struct_op.split_name_and_node_identifier(integral_name)

            if var_name in scaling_options[var_type]:
                lookup_name = var_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif split_name in scaling_options[var_type]:
                lookup_name = split_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 3) and ('ddd' + integral_name in scaling_options[var_type]):
                lookup_name = 'ddd' + integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 3) and ('ddd' + split_integral_name in scaling_options[var_type]):
                lookup_name = 'ddd' + split_integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 2) and ('dd' + integral_name in scaling_options[var_type]):
                lookup_name = 'dd' + integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 2) and ('dd' + split_integral_name in scaling_options[var_type]):
                lookup_name = 'dd' + split_integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 1) and ('d' + integral_name in scaling_options[var_type]):
                lookup_name = 'd' + integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif (integral_order > 1) and ('d' + split_integral_name in scaling_options[var_type]):
                lookup_name = 'd' + split_integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif integral_name in scaling_options[var_type]:
                lookup_name = integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            elif split_integral_name in scaling_options[var_type]:
                lookup_name = split_integral_name
                scaling_value = scaling_options[var_type][lookup_name]

            else:
                scaling_value = cas.DM(1.)
                unset_set += [var_type + var_name]

            checked_and_rearranged_value = struct_op.check_and_rearrange_scaling_value_before_assignment(var_type, var_name, scaling_value, scaling)
            scaling[var_type, var_name] = checked_and_rearranged_value

    # set the derivative variable scalings
    var_type = 'xdot'
    for var_name in struct_op.subkeys(scaling, var_type):

        integral_name = var_name[1:]

        integral_name_in_x = integral_name in struct_op.subkeys(scaling, 'x')
        integral_name_in_u = integral_name in struct_op.subkeys(scaling, 'u')

        if integral_name_in_x:
            scaling[var_type, var_name] = scaling['x', integral_name]

        elif integral_name_in_u:
            scaling[var_type, var_name] = scaling['u', integral_name]

        else:
            message = 'unable to find the scaling information for xdot variable (' + var_name + ') in x or u'
            print_op.log_and_raise_error(message)

    # warn about potentially missing scaling information
    for local_label in unset_set:

        is_tf = 't_f' in local_label
        is_dcm = ('[x,r' in local_label) or ('dcm' in local_label)
        is_deriv_dcm = '[xdot,dr' in local_label
        is_a_cosine_or_a_sine = ('[z,cos' in local_label) or ('[z,sin' in local_label)
        leave_unscaled = is_tf or is_dcm or is_deriv_dcm or is_a_cosine_or_a_sine

        if leave_unscaled:
            unset_set.pop(local_label)

    if len(unset_set) > 0:
        message = 'only unit-scaling information found for the following variables: \n' + repr(unset_set) + '.\n' + 'Proceeding with unit scaling.'
        print_op.base_print(message, level='warning')

    return scaling


def generate_si_variables(scaling, variables):

    # scale variables
    variables_si = {}
    for var_type in list(scaling.keys()):
        subkeys = struct_op.subkeys(variables, var_type)

        variables_si[var_type] = cas.struct_SX(
            [cas.entry(var_name, expr=struct_op.var_scaled_to_si(var_type, var_name, variables[var_type, var_name], scaling)) for var_name in subkeys])

    return variables_si



def get_roll_expr(x, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle roll angle via roll = atan(expr),
    :param x: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: tan(roll)
    """

    # node + parent position
    q0 = x['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = x['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(x['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 1]) / cas.mtimes(q_hat.T, r[:, 2])


def get_pitch_expr(x, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle pitch angle via pitch = asin(expr),
    :param x: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: sin(pitch)
    """

    # node + parent position
    q0 = x['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = x['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(x['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 0]) / vect_op.norm(q_hat)


def get_span_angle_expr(options, x, n0, n1, parent_map, parameters):
    """ Return the expression that allows to compute the cross-tether vs. body span-vector angle and related inequality,
    :param x: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: span_inequality, span_angle
    """

    # node + parent position
    q0 = x['q{}{}'.format(n0, parent_map[n0])]
    r0 = cas.reshape(x['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix
    r_wtip = cas.vertcat(0.0, -parameters['theta0', 'geometry', 'b_ref'] / 2, 0.0)

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = x['q{}{}'.format(n1, parent_map[n1])]
        r1 = cas.reshape(x['r{}{}'.format(n1, parent_map[n1])], (3, 3))

    # first node
    q_first = q0 + cas.mtimes(r0, r_wtip)
    q_second = q1 + cas.mtimes(r1, r_wtip)

    # tether direction
    q_hat = q_first - q_second

    # span inequality
    span_ineq = cas.cos(parameters['theta0', 'model_bounds', 'span_angle']) * vect_op.norm(q_hat) - cas.mtimes(
        r0[:, 1].T, q_hat)

    # scale span inequality
    span_ineq = span_ineq / options['scaling']['theta']['l_s']

    # angle between aircraft span vector and cross-tether
    span_angle = cas.acos(cas.mtimes(r0[:, 1].T, q_hat) / vect_op.norm(q_hat))

    return span_ineq, span_angle


def get_yaw_expr(options, x, n0, n1, parent_map, gamma_max):
    """ Compute angle between kite yaw vector and tether, including corresponding inequality.
    :param x: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: yaw expression, yaw angle
    """
    # node + parent position
    q0 = x['q{}{}'.format(n0, parent_map[n0])]

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = x['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(x['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    yaw_angle = cas.arccos(cas.mtimes(q_hat.T, r[:, 2]) / vect_op.norm(q_hat))
    yaw_expr = (cas.mtimes(q_hat.T, r[:, 2]) - cas.cos(gamma_max) * vect_op.norm(q_hat))

    # scale yaw_expression
    if n0 == 1:
        scale = options['scaling']['x']['l_t']
    else:
        scale = options['scaling']['theta']['l_s']
    yaw_expr = yaw_expr / scale

    return yaw_expr, yaw_angle


def rotation_inequality(options, variables, parameters, architecture, outputs):

    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    x = variables['x']

    cstr_list = cstr_op.MdlConstraintList()

    kite_has_6_dof = (options['kite_dof'] == 6)
    if kite_has_6_dof:

        # create bound expressions from angle bounds
        if options['model_bounds']['rotation']['type'] == 'roll_pitch':
            max_angles = cas.vertcat(
                cas.tan(parameters['theta0', 'model_bounds', 'rot_angles', 0]),
                cas.sin(parameters['theta0', 'model_bounds', 'rot_angles', 1])
            )
            min_angles = -1. * max_angles

        for kite in kite_nodes:
            parent = parent_map[kite]

            if options['model_bounds']['rotation']['type'] == 'roll_pitch':
                rotation_angles = cas.vertcat(
                    get_roll_expr(x, kite, parent_map[kite], parent_map),
                    get_pitch_expr(x, kite, parent_map[kite], parent_map)
                )

                if options['model_bounds']['rotation']['include']:

                    expr_max = rotation_angles - max_angles
                    expr_min = min_angles - rotation_angles

                    cstr_max = cstr_op.Constraint(expr=expr_max,
                                                name='rotation_max' + str(kite) + str(parent),
                                                cstr_type='ineq')
                    cstr_list.append(cstr_max)

                    cstr_min = cstr_op.Constraint(expr=expr_min,
                                                name='rotation_min' + str(kite) + str(parent),
                                                cstr_type='ineq')
                    cstr_list.append(cstr_min)

                outputs['local_performance']['rot_angles' + str(kite) + str(parent)] = cas.vertcat(
                    cas.atan(rotation_angles[0]),
                    cas.asin(rotation_angles[1])
                )

            elif options['model_bounds']['rotation']['type'] == 'yaw':

                yaw_expr, yaw_angle = get_yaw_expr(
                    options, x, kite, parent_map[kite], parent_map,
                    parameters['theta0', 'model_bounds', 'rot_angles', 2]
                )

                if options['model_bounds']['rotation']['include']:
                    cstr_min = cstr_op.Constraint(expr=-1. * yaw_expr,
                                                name='rotation_max' + str(kite) + str(parent),
                                                cstr_type='ineq')
                    cstr_list.append(cstr_min)

                outputs['local_performance']['rot_angles' + str(kite) + str(parent)] = yaw_angle

        # cross-tether
        if options['cross_tether'] and (number_of_nodes > 2):
            for layer in architecture.layer_nodes:
                kites = architecture.kites_map[layer]
                if len(kites) == 2:
                    no_tethers = 1
                else:
                    no_tethers = len(kites)

                for k in range(no_tethers):
                    tether_name = '{}{}'.format(kites[k], kites[(k + 1) % len(kites)])
                    tether_name2 = '{}{}'.format(kites[(k + 1) % len(kites)], kites[k])

                    if options['tether']['cross_tether']['attachment'] != 'wing_tip':

                        yaw_expr, yaw_angle = get_yaw_expr(
                            options, x, kites[k], kites[(k + 1) % len(kites)], parent_map,
                            parameters['theta0', 'model_bounds', 'rot_angles_cross', 2]
                        )

                        yaw_expr2, yaw_angle2 = get_yaw_expr(
                            options, x, kites[(k + 1) % len(kites)], kites[k], parent_map,
                            parameters['theta0', 'model_bounds', 'rot_angles_cross', 2]
                        )

                        if options['model_bounds']['rotation']['include']:
                            cstr_max = cstr_op.Constraint(expr=-1. * yaw_expr,
                                                        name='rotation_max' + tether_name,
                                                        cstr_type='ineq')
                            cstr_list.append(cstr_max)

                            cstr_max = cstr_op.Constraint(expr=-1. * yaw_expr,
                                                        name='rotation_max' + tether_name2,
                                                        cstr_type='ineq')
                            cstr_list.append(cstr_max)

                        outputs['local_performance']['rot_angles' + tether_name] = yaw_angle
                        outputs['local_performance']['rot_angles' + tether_name2] = yaw_angle2

                    else:

                        # get angle between body span vector and cross-tether and related inequality
                        rotation_angle_expr, span = get_span_angle_expr(options, x, kites[k], kites[(k + 1) % len(kites)],
                                                                        parent_map, parameters)
                        rotation_angle_expr2, span2 = get_span_angle_expr(options, x, kites[(k + 1) % len(kites)],
                                                                          kites[k], parent_map, parameters)

                        if options['model_bounds']['rotation']['include']:
                            cstr_max_tether1 = cstr_op.Constraint(expr=rotation_angle_expr,
                                                                    name='rotation_max' + tether_name,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_max_tether1)

                            cstr_max_tether2 = cstr_op.Constraint(expr=rotation_angle_expr2,
                                                                    name='rotation_max' + tether_name2,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_max_tether2)

                        outputs['local_performance']['rot_angles' + tether_name] = span
                        outputs['local_performance']['rot_angles' + tether_name2] = span2

    return outputs, cstr_list
