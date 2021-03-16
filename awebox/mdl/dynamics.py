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

import awebox.mdl.aero.indicators as indicators

import awebox.mdl.mdl_constraint as mdl_constraint

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
    system_variables['SI'], options['scaling'] = generate_si_variables(options['scaling'], system_variables['scaled'])
    scaling = options['scaling']

    # -----------------------------------
    # prepare empty constraints list and outputs
    # -----------------------------------
    cstr_list = mdl_constraint.MdlConstraintList()
    outputs = {}

    # ---------------------------------
    # define the equality constraints (aka. dynamics)
    # ---------------------------------

    # enforce the lagrangian dynamics
    lagr_dyn_cstr, outputs = lagr_dyn.get_dynamics(options, atmos, wind, architecture, system_variables, system_gc, parameters, outputs)
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
    induction_cstr = get_induction_cstr(options, atmos, wind, system_variables['SI'], parameters, outputs, architecture)
    cstr_list.append(induction_cstr)

    # ensure that energy matches power integration
    power = get_power(options, system_variables['SI'], parameters, outputs, architecture)
    integral_outputs_fun, integral_outputs_struct, integral_scaling, energy_cstr = manage_power_integration(options,
                                                                                                                power,
                                                                                                                system_variables,
                                                                                                                parameters)
    cstr_list.append(energy_cstr)


    # --------------------------------------------
    # define the inequality constraints
    # --------------------------------------------


    wound_length_cstr = wound_tether_length_inequality(options, system_variables['SI'])
    cstr_list.append(wound_length_cstr)

    outputs, stress_cstr = tether_stress_inequality(options, system_variables['SI'], outputs, parameters, architecture)
    cstr_list.append(stress_cstr)

    airspeed_cstr = airspeed_inequality(options, outputs, parameters, architecture)
    cstr_list.append(airspeed_cstr)

    aero_validity_cstr = aero_validity_inequality(options, outputs)
    cstr_list.append(aero_validity_cstr)

    anticollision_cstr = anticollision_inequality(options, system_variables['SI'], parameters, architecture)
    cstr_list.append(anticollision_cstr)

    acceleration_cstr = acceleration_inequality(options, system_variables['SI'])
    cstr_list.append(acceleration_cstr)

    outputs, rotation_cstr = rotation_inequality(options, system_variables['SI'], parameters, architecture, outputs)
    cstr_list.append(rotation_cstr)

    dcoeff_cstr = dcoeff_actuation_inequality(options, system_variables['SI'], parameters, architecture)
    cstr_list.append(dcoeff_cstr)

    coeff_cstr = coeff_actuation_inequality(options, system_variables['SI'], parameters, architecture)
    cstr_list.append(coeff_cstr)

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
    outputs = xddot_outputs(system_variables['SI'], outputs)
    outputs = power_balance_outputs(options, outputs, system_variables,
                                    parameters, architecture)  # power balance must be after tether stress inequality

    # system output function
    [out, out_fun, out_dict] = make_output_structure(outputs, system_variables, parameters)

    # ----------------------------------------
    #  return
    # ----------------------------------------

    return [
        system_variables['scaled'],
        variables_dict,
        scaling,
        cstr_list,
        out,
        out_fun,
        out_dict,
        integral_outputs_struct,
        integral_outputs_fun,
        integral_scaling]


def get_induction_cstr(options, atmos, wind, variables_si, parameters, outputs, architecture):

    cstr_list = mdl_constraint.MdlConstraintList()

    if not (options['induction_model'] == 'not_in_use'):

        induction_init = induction.get_trivial_residual(options, atmos, wind, variables_si, parameters, outputs, architecture)
        induction_final = induction.get_final_residual(options, atmos, wind, variables_si, parameters, outputs, architecture)

        induction_constraint = parameters['phi', 'iota'] * induction_init \
                               + (1. - parameters['phi', 'iota']) * induction_final

        induction_cstr = cstr_op.Constraint(expr=induction_constraint,
                                                      name='induction',
                                                      cstr_type='eq')
        cstr_list.append(induction_cstr)

    return cstr_list



def manage_power_integration(options, power, system_variables, parameters):

    cstr_list = mdl_constraint.MdlConstraintList()

    # generate empty integral_outputs struct
    integral_outputs = cas.struct_SX([])
    integral_outputs_struct = cas.struct_symSX([])

    integral_scaling = {}

    if options['integral_outputs']:
        integral_outputs = cas.struct_SX([cas.entry('e', expr=power / options['scaling']['xd']['e'])])
        integral_outputs_struct = cas.struct_symSX([cas.entry('e')])

        integral_scaling['e'] = options['scaling']['xd']['e']

    else:
        energy_resi = (system_variables['SI']['xddot']['de'] - power) / options['scaling']['xd']['e']

        energy_cstr = cstr_op.Constraint(expr=energy_resi,
                                       name='energy',
                                       cstr_type='eq')
        cstr_list.append(energy_cstr)

    # dynamics function options
    if options['jit_code_gen']['include']:
        opts = {'jit': True, 'compiler': options['jit_code_gen']['compiler']}
    else:
        opts = {}

    integral_outputs_fun = cas.Function('integral_outputs', [system_variables['scaled'], parameters],
                                        [integral_outputs], opts)

    return integral_outputs_fun, integral_outputs_struct, integral_scaling, cstr_list


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


def get_power(options, variables_si, parameters, outputs, architecture):
    if options['trajectory']['system_type'] == 'drag_mode':
        power = cas.SX.zeros(1, 1)
        for kite in architecture.kite_nodes:
            power += get_drag_power_from_kite(kite, variables_si, parameters, outputs, architecture)
    else:
        power = variables_si['xa']['lambda10'] * variables_si['xd']['l_t'] * variables_si['xd']['dl_t']

    return power



def drag_mode_outputs(variables_si, parameters, outputs, architecture):
    for kite in architecture.kite_nodes:
        outputs['power_balance']['P_gen{}'.format(kite)] = -1. * get_drag_power_from_kite(kite, variables_si, parameters, outputs, architecture)

    return outputs




def power_balance_outputs(options, outputs, system_variables, parameters, architecture):
    variables_si = system_variables['SI']

    # all aerodynamic forces have already been added to power balance, by this point.
    # outputs['power_balance'] is not empty!

    if options['trajectory']['system_type'] == 'drag_mode':
        outputs = drag_mode_outputs(variables_si, parameters, outputs, architecture)

    outputs = tether_power_outputs(variables_si, outputs, architecture)
    outputs = kinetic_power_outputs(options, outputs, system_variables, architecture)
    outputs = potential_power_outputs(options, outputs, system_variables, architecture)

    if options['test']['check_energy_summation']:
        outputs = comparison_kin_and_pot_power_outputs(options, outputs, system_variables, architecture)

    return outputs


def comparison_kin_and_pot_power_outputs(options, outputs, system_variables, architecture):

    outputs['power_balance_comparison'] = {}

    types = ['potential', 'kinetic']

    for type in types:
        dict = outputs['e_' + type]
        e_local = 0.

        for keyname in dict.keys():
            e_local += dict[keyname]

        # rate of change in kinetic energy
        P = lagr_tools.time_derivative(e_local, system_variables, architecture)

        # convention: negative when kinetic energy is added to the system
        outputs['power_balance_comparison'][type] = -1. * P

    return outputs



def tether_power_outputs(variables_si, outputs, architecture):
    # compute instantaneous power transferred by each tether
    for n in range(1, architecture.number_of_nodes):
        parent = architecture.parent_map[n]

        # node positions
        q_n = variables_si['xd']['q' + str(n) + str(parent)]
        if n > 1:
            grandparent = architecture.parent_map[parent]
            q_p = variables_si['xd']['q' + str(parent) + str(grandparent)]
        else:
            q_p = cas.SX.zeros((3, 1))

        # node velocity
        dq_n = variables_si['xd']['dq' + str(n) + str(parent)]
        # force and direction
        tether_force = outputs['local_performance']['tether_force' + str(n) + str(parent)]
        tether_direction = vect_op.normalize(q_n - q_p)
        # power transferred (convention: negative when power is transferred down the tree)
        outputs['power_balance']['P_tether' + str(n)] = -cas.mtimes(tether_force * tether_direction.T, dq_n)

    return outputs


def kinetic_power_outputs(options, outputs, system_variables, architecture):
    """Compute rate of change of kinetic energy for all system nodes

    @return: outputs updated outputs dict
    """

    # notice that the quality test uses a normalized value of each power source, so scaling should be irrelevant
    # but scaled variables are the decision variables, for which cas.jacobian is defined
    # whereas SI values are multiples of the base values, for which cas.jacobian cannot be computed

    # kinetic and potential energy in the system
    for n in range(1, architecture.number_of_nodes):
        label = str(n) + str(architecture.parent_map[n])

        categories = {'q' + label: str(n)}

        if n == 1:
            categories['ground_station'] = 'groundstation1'

        for cat in categories.keys():

            # rate of change in kinetic energy
            e_local = outputs['e_kinetic'][cat]

            P = lagr_tools.time_derivative(e_local, system_variables, architecture)

            # convention: negative when energy is added to the system
            outputs['power_balance']['P_kin' + categories[cat]] = -1. * P

    return outputs

def potential_power_outputs(options, outputs, system_variables, architecture):
    """Compute rate of change of potential energy for all system nodes

    @return: outputs updated outputs dict
    """

    # notice that the quality test uses a normalized value of each power source, so scaling should be irrelevant
    # but scaled variables are the decision variables, for which cas.jacobian is defined
    # whereas SI values are multiples of the base values, for which cas.jacobian cannot be computed

    # kinetic and potential energy in the system
    for n in range(1, architecture.number_of_nodes):
        label = str(n) + str(architecture.parent_map[n])

        # rate of change in potential energy (ignore in-outflow of potential energy)
        e_local = outputs['e_potential']['q' + label]
        P = lagr_tools.time_derivative(e_local, system_variables, architecture)

        # convention: negative when potential energy is added to the system
        outputs['power_balance']['P_pot' + str(n)] = -1. * P

    return outputs





def xddot_outputs(variables, outputs):
    outputs['xddot_from_var'] = variables['xddot']
    return outputs



def anticollision_inequality(options, variables, parameters, architecture):
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    cstr_list = mdl_constraint.MdlConstraintList()

    if options['model_bounds']['anticollision']['include']:

        safety_factor = options['model_bounds']['anticollision']['safety_factor']
        dist_min = safety_factor * parameters['theta0', 'geometry', 'b_ref']

        kite_combinations = itertools.combinations(kite_nodes, 2)
        for kite_pair in kite_combinations:
            kite_a = kite_pair[0]
            kite_b = kite_pair[1]
            parent_a = parent_map[kite_a]
            parent_b = parent_map[kite_b]
            dist = variables['xd']['q' + str(kite_a) + str(parent_a)] - variables['xd']['q' + str(kite_b) + str(parent_b)]
            dist_inequality = 1 - (cas.mtimes(dist.T, dist) / dist_min ** 2)

            anticollision_cstr = cstr_op.Constraint(expr=dist_inequality,
                                                  name='anticollision' + str(kite_a) + str(kite_b),
                                                  cstr_type='ineq')
            cstr_list.append(anticollision_cstr)

    return cstr_list


def dcoeff_actuation_inequality(options, variables_si, parameters, architecture):

    cstr_list = mdl_constraint.MdlConstraintList()

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

    cstr_list = mdl_constraint.MdlConstraintList()

    if options['model_bounds']['coeff_actuation']['include']:

        # nu*xd_max + (1 - nu)*u_compromised_max > xd
        # xd - nu*xd_max + (1 - nu)*xd_compromised_max < 0
        if int(options['kite_dof']) != 3:
            return cstr_list

        nu = parameters['phi', 'nu']
        coeff_max = options['aero']['three_dof']['coeff_max']
        coeff_compromised_max = parameters['theta0', 'model_bounds', 'coeff_compromised_max']
        coeff_min = options['aero']['three_dof']['coeff_min']
        coeff_compromised_min = parameters['theta0', 'model_bounds', 'coeff_compromised_min']
        traj_type = options['trajectory']['type']

        for variable in list(variables_si['xd'].keys()):

            var_name, kiteparent = struct_op.split_name_and_node_identifier(variable)
            if var_name == 'coeff':

                coeff = variables_si['xd'][variable]
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


def acceleration_inequality(options, variables):

    cstr_list = mdl_constraint.MdlConstraintList()

    if options['model_bounds']['acceleration']['include']:

        acc_max = options['model_bounds']['acceleration']['acc_max'] * options['scaling']['other']['g']

        for name in list(variables['xddot'].keys()):

            var_name, var_label = struct_op.split_name_and_node_identifier(name)

            if var_name == 'ddq':
                acc = variables['xddot'][name]
                acc_sq = cas.mtimes(acc.T, acc)
                acc_sq_norm = acc_sq / acc_max ** 2.

                # acc^2 < acc_max^2 -> acc^2 / acc_max^2 - 1 < 0
                local_ineq = acc_sq_norm - 1.

                acc_cstr = cstr_op.Constraint(expr=local_ineq,
                                            name='acceleration' + var_label,
                                            cstr_type='ineq')
                cstr_list.append(acc_cstr)

    return cstr_list


def airspeed_inequality(options, outputs, parameters, architecture):

    cstr_list = mdl_constraint.MdlConstraintList()

    if options['model_bounds']['airspeed']['include']:

        # system architecture
        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map

        # constraint bounds
        airspeed_max = parameters['theta0', 'model_bounds', 'airspeed_limits'][1]
        airspeed_min = parameters['theta0', 'model_bounds', 'airspeed_limits'][0]

        for kite in kite_nodes:
            airspeed = outputs['aerodynamics']['airspeed' + str(kite)]
            parent = parent_map[kite]

            max_resi = airspeed / airspeed_max - 1.
            min_resi = - airspeed / airspeed_min + 1.

            max_cstr = cstr_op.Constraint(expr=max_resi,
                                        name='airspeed_max' + str(kite) + str(parent),
                                        cstr_type='ineq')
            cstr_list.append(max_cstr)

            min_cstr = cstr_op.Constraint(expr=min_resi,
                                        name='airspeed_min' + str(kite) + str(parent),
                                        cstr_type='ineq')
            cstr_list.append(min_cstr)

    return cstr_list


def aero_validity_inequality(options, outputs):

    cstr_list = mdl_constraint.MdlConstraintList()

    if options['model_bounds']['aero_validity']['include']:
        for name in outputs['aero_validity'].keys():

            ineq = outputs['aero_validity'][name]
            valid_cstr = cstr_op.Constraint(expr=ineq,
                                            name=name,
                                            cstr_type='ineq')
            cstr_list.append(valid_cstr)

    return cstr_list

def tether_stress_inequality(options, variables_si, outputs, parameters, architecture):

    cstr_list = mdl_constraint.MdlConstraintList()

    # system architecture (see zanon2013a)
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map

    # system (scaled) variables
    xa = variables_si['xa']
    theta = variables_si['theta']

    tightness = options['model_bounds']['tether_stress']['scaling']

    tether_constraints = ['tether_stress', 'tether_force_max', 'tether_force_min', 'tether_tension']

    if 'local_performance' not in outputs.keys():
        outputs['local_performance'] = {}

    # mass vector, containing the mass of all nodes
    for n in range(1, number_of_nodes):

        parent = parent_map[n]

        seg_props = tether_aero.get_tether_segment_properties(options, architecture, variables_si, parameters, upper_node=n)
        seg_length = seg_props['seg_length']
        cross_section_area = seg_props['cross_section_area']
        max_area = seg_props['max_area']

        tension = xa['lambda' + str(n) + str(parent)] * seg_length

        min_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][0]
        max_tension = parameters['theta0', 'model_bounds', 'tether_force_limits'][1]

        maximum_allowed_stress = parameters['theta0', 'tether', 'max_stress'] / parameters['theta0', 'tether', 'stress_safety_factor']
        max_tension_from_stress = maximum_allowed_stress * max_area

        # stress_max = max_tension_from_stress / A_max
        # (tension / A) < stress_max
        # tension / A < max_tension_from_stress / Amax
        # tension / max_tension_from_stress < A / Amax
        # tension / max_tension_from_stress - A / Amax < 0
        stress_inequality_untightened = tension / max_tension_from_stress - cross_section_area / max_area
        stress_inequality = stress_inequality_untightened * tightness

        # outputs related to the constraints themselves
        tether_constraint_includes = options['model_bounds']['tether']['tether_constraint_includes']

        if n in tether_constraint_includes['stress']:
            stress_cstr = cstr_op.Constraint(expr=stress_inequality,
                                           name='tether_stress' + str(n) + str(parent),
                                           cstr_type='ineq')
            cstr_list.append(stress_cstr)

        if n in tether_constraint_includes['force']:

            force_max_resi = (tension - max_tension) / vect_op.smooth_abs(max_tension)
            force_min_resi = -(tension - min_tension) / vect_op.smooth_abs(min_tension)

            force_max_cstr = cstr_op.Constraint(expr=force_max_resi,
                                               name='tether_force_max' + str(n) + str(parent),
                                               cstr_type='ineq')
            cstr_list.append(force_max_cstr)

            force_min_cstr = cstr_op.Constraint(expr=force_min_resi,
                                               name='tether_force_min' + str(n) + str(parent),
                                               cstr_type='ineq')
            cstr_list.append(force_min_cstr)

        # outputs so that the user can find the stress and tension
        outputs['local_performance']['tether_stress' + str(n) + str(parent)] = tension / cross_section_area
        outputs['local_performance']['tether_force' + str(n) + str(parent)] = tension

    if options['cross_tether'] and len(architecture.kite_nodes) > 1:
        for l in architecture.layer_nodes:
            kites = architecture.kites_map[l]
            seg_length = theta['l_c{}'.format(l)]
            seg_diam = theta['diam_c{}'.format(l)]
            cross_section = np.pi * seg_diam ** 2. / 4.
            cross_section_max = np.pi * options['system_bounds']['theta']['diam_c'][1] ** 2.0 / 4.
            max_tension_from_stress = maximum_allowed_stress * cross_section_max

            if len(kites) == 2:
                tension = xa['lambda{}{}'.format(kites[0], kites[1])] * seg_length
                outputs['local_performance']['tether_stress{}{}'.format(kites[0], kites[1])] = tension / cross_section

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

                    tension = xa['lambda' + label] * seg_length
                    outputs['local_performance']['tether_stress' + label] = tension / cross_section

                    stress_inequality_untightened = tension / max_tension_from_stress - cross_section / cross_section_max
                    stress_ineq_tightened = stress_inequality_untightened * tightness

                    stress_cstr = cstr_op.Constraint(expr=stress_ineq_tightened,
                                                   name='tether_stress' + label,
                                                   cstr_type='ineq')
                    cstr_list.append(stress_cstr)


    return outputs, cstr_list


def wound_tether_length_inequality(options, variables):

    cstr_list = mdl_constraint.MdlConstraintList()


    if options['model_bounds']['wound_tether_length']['include']:

        length_scaling = options['scaling']['xd']['l_t']

        if options['tether']['use_wound_tether']:
            l_t_full = variables['theta']['l_t_full'] / options['tether']['wound_tether_safety_factor']
            l_t = variables['xd']['l_t']

            expr = (l_t - l_t_full) / length_scaling

        cstr = cstr_op.Constraint(expr=expr,
                                name='wound_tether_length',
                                cstr_type='ineq')
        cstr_list.append(cstr)

    return cstr_list


def generate_si_variables(scaling_options, variables):

    scaling = {}

    prepared_xd_names = scaling_options['xd'].keys()

    if 'xddot' not in scaling_options.keys():
        scaling_options['xddot'] = {}

    for var_type in variables.keys():
        scaling[var_type] = {}

        for var_name in struct_op.subkeys(variables, var_type):

            stripped_name, _ = struct_op.split_name_and_node_identifier(var_name)
            prepared_names = scaling_options[var_type].keys()

            var_might_be_derivative = (stripped_name[0] == 'd') and (len(stripped_name) > 1)
            poss_deriv_name = stripped_name[1:]

            var_might_be_sec_derivative = var_might_be_derivative and (stripped_name[1] == 'd') and (len(stripped_name) > 2)
            poss_sec_deriv_name = stripped_name[2:]

            var_might_be_third_derivative = var_might_be_sec_derivative and (stripped_name[2] == 'd') and (len(stripped_name) > 3)
            poss_third_deriv_name = stripped_name[3:]



            if var_name in prepared_names:
                scaling[var_type][var_name] = cas.DM(scaling_options[var_type][var_name])

            elif stripped_name in prepared_names:
                scaling[var_type][var_name] = cas.DM(scaling_options[var_type][stripped_name])

            elif var_might_be_derivative and (poss_deriv_name in prepared_names):
                scaling[var_type][var_name] = cas.DM(scaling_options[var_type][poss_deriv_name])

            elif var_might_be_sec_derivative and (poss_sec_deriv_name in prepared_names):
                scaling[var_type][var_name] = cas.DM(scaling_options[var_type][poss_sec_deriv_name])

            elif var_might_be_third_derivative and (poss_third_deriv_name in prepared_names):
                scaling[var_type][var_name] = cas.DM(scaling_options[var_type][poss_third_deriv_name])



            elif var_name in prepared_xd_names:
                scaling[var_type][var_name] = cas.DM(scaling_options['xd'][var_name])

            elif stripped_name in prepared_xd_names:
                scaling[var_type][var_name] = cas.DM(scaling_options['xd'][stripped_name])

            elif var_might_be_derivative and (poss_deriv_name in prepared_xd_names):
                scaling[var_type][var_name] = cas.DM(scaling_options['xd'][poss_deriv_name])

            elif var_might_be_sec_derivative and (poss_sec_deriv_name in prepared_xd_names):
                scaling[var_type][var_name] = cas.DM(scaling_options['xd'][poss_sec_deriv_name])

            elif var_might_be_third_derivative and (poss_third_deriv_name in prepared_xd_names):
                scaling[var_type][var_name] = cas.DM(scaling_options['xd'][poss_third_deriv_name])

            else:
                message = 'no scaling information provided for variable ' + var_name + ', expected in ' + var_type + '. Proceeding with unit scaling.'
                awelogger.logger.warning(message)
                scaling[var_type][var_name] = cas.DM(1.)

        scaling_options[var_type].update(scaling[var_type])

    # scale variables
    variables_si = {}
    for var_type in list(scaling.keys()):
        subkeys = struct_op.subkeys(variables, var_type)

        variables_si[var_type] = cas.struct_SX(
            [cas.entry(var_name, expr=struct_op.var_scaled_to_si(var_type, var_name, variables[var_type, var_name], scaling)) for var_name in subkeys])

    return variables_si, scaling_options



def get_roll_expr(xd, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle roll angle via roll = atan(expr),
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: tan(roll)
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 1] / cas.mtimes(q_hat.T, r[:, 2]))


def get_pitch_expr(xd, n0, n1, parent_map):
    """ Return the expression that allows to compute the bridle pitch angle via pitch = asin(expr),
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node 
    :param parent_map: architecture parent map
    :return: sin(pitch)
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    return cas.mtimes(q_hat.T, r[:, 0] / vect_op.norm(q_hat))


def get_span_angle_expr(options, xd, n0, n1, parent_map, parameters):
    """ Return the expression that allows to compute the cross-tether vs. body span-vector angle and related inequality,
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: span_inequality, span_angle
    """

    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]
    r0 = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix
    r_wtip = cas.vertcat(0.0, -parameters['theta0', 'geometry', 'b_ref'] / 2, 0.0)

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]
        r1 = cas.reshape(xd['r{}{}'.format(n1, parent_map[n1])], (3, 3))

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


def get_yaw_expr(options, xd, n0, n1, parent_map, gamma_max):
    """ Compute angle between kite yaw vector and tether, including corresponding inequality.
    :param xd: system variables
    :param n0: node number of kite node
    :param n1: node number of tether attachment node
    :param parent_map: architecture parent map
    :return: yaw expression, yaw angle
    """
    # node + parent position
    q0 = xd['q{}{}'.format(n0, parent_map[n0])]

    if n1 == 0:
        q1 = np.zeros((3, 1))
    else:
        q1 = xd['q{}{}'.format(n1, parent_map[n1])]

    q_hat = q0 - q1  # tether direction
    r = cas.reshape(xd['r{}{}'.format(n0, parent_map[n0])], (3, 3))  # rotation matrix

    yaw_angle = cas.arccos(cas.mtimes(q_hat.T, r[:, 2]) / vect_op.norm(q_hat))
    yaw_expr = (cas.mtimes(q_hat.T, r[:, 2]) - cas.cos(gamma_max) * vect_op.norm(q_hat))

    # scale yaw_expression
    if n0 == 1:
        scale = options['scaling']['xd']['l_t']
    else:
        scale = options['scaling']['theta']['l_s']
    yaw_expr = yaw_expr / scale

    return yaw_expr, yaw_angle


def rotation_inequality(options, variables, parameters, architecture, outputs):

    number_of_nodes = architecture.number_of_nodes
    kite_nodes = architecture.kite_nodes
    parent_map = architecture.parent_map

    xd = variables['xd']

    cstr_list = mdl_constraint.MdlConstraintList()

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
                    get_roll_expr(xd, kite, parent_map[kite], parent_map),
                    get_pitch_expr(xd, kite, parent_map[kite], parent_map)
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
                    options, xd, kite, parent_map[kite], parent_map,
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

                    if options['tether']['cross_tether']['attachment'] is not 'wing_tip':

                        # get roll and pitch expressions at each end of the cross-tether
                        rotation_angles = cas.vertcat(
                            get_roll_expr(xd, kites[k], kites[(k + 1) % len(kites)], parent_map),
                            get_pitch_expr(xd, kites[k], kites[(k + 1) % len(kites)], parent_map)
                        )
                        rotation_angles2 = cas.vertcat(
                            get_roll_expr(xd, kites[(k + 1) % len(kites)], kites[k], parent_map),
                            get_pitch_expr(xd, kites[(k + 1) % len(kites)], kites[k], parent_map)
                        )

                        if options['model_bounds']['rotation']['include']:
                            expr_max_tether1 = rotation_angles - max_angles
                            expr_max_tether2 = rotation_angles2 - max_angles
                            expr_min_tether1 = min_angles - rotation_angles
                            expr_min_tether2 = min_angles - rotation_angles2

                            cstr_max_tether1 = cstr_op.Constraint(expr=expr_max_tether1,
                                                                    name='rotation_max' + tether_name,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_max_tether1)

                            cstr_max_tether2 = cstr_op.Constraint(expr=expr_max_tether2,
                                                                    name='rotation_max' + tether_name2,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_max_tether2)

                            cstr_min_tether1 = cstr_op.Constraint(expr=expr_min_tether1,
                                                                    name='rotation_min' + tether_name,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_min_tether1)

                            cstr_min_tether2 = cstr_op.Constraint(expr=expr_min_tether2,
                                                                    name='rotation_min' + tether_name2,
                                                                    cstr_type='ineq')
                            cstr_list.append(cstr_min_tether2)

                        outputs['local_performance']['rot_angles' + tether_name] = cas.vertcat(
                            cas.atan(rotation_angles[0]),
                            cas.asin(rotation_angles[1])
                        )
                        outputs['local_performance']['rot_angles' + tether_name2] = cas.vertcat(
                            cas.atan(rotation_angles2[0]),
                            cas.asin(rotation_angles2[1])
                        )

                    else:

                        # get angle between body span vector and cross-tether and related inequality
                        rotation_angle_expr, span = get_span_angle_expr(options, xd, kites[k], kites[(k + 1) % len(kites)],
                                                                        parent_map, parameters)
                        rotation_angle_expr2, span2 = get_span_angle_expr(options, xd, kites[(k + 1) % len(kites)],
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
