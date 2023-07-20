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
"""
induction and local flow manager
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020-21
"""
import pdb

import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.actuator_dir.actuator as actuator
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex
import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.averaged as averaged
import awebox.tools.print_operations as print_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

def get_wake_if_vortex_model_is_included_in_comparison(model_options, architecture, wind, variables_si, parameters):
    if not (model_options['induction_model'] == 'not_in_use'):
        if vortex.model_is_included_in_comparison(model_options):
            return vortex.build(model_options, architecture, wind, variables_si, parameters)

    return None

def get_model_constraints(model_options, wake, scaling, atmos, wind, system_variables, parameters, outputs, architecture):

    variables_si = system_variables['SI']

    cstr_list = cstr_op.ConstraintList()

    if model_options['induction_model'] is 'averaged':
        ellipse_half_cstr = averaged.get_ellipse_half_constraint(model_options, variables_si, parameters, architecture, outputs)
        cstr_list.append(ellipse_half_cstr)

    else:
        induction_cstr = get_induction_cstr(model_options, wind, system_variables, parameters, architecture, scaling)
        cstr_list.append(induction_cstr)

        if actuator.model_is_included_in_comparison(model_options):
            actuator_cstr = actuator.get_model_constraints(model_options, atmos, wind, variables_si, parameters, outputs,
                                                       architecture, scaling)
            cstr_list.append(actuator_cstr)

        if vortex.model_is_included_in_comparison(model_options):
            vortex_cstr = vortex.get_model_constraints(model_options, wake, system_variables, parameters, architecture, scaling)
            cstr_list.append(vortex_cstr)

    return cstr_list

def get_induction_cstr(options, wind, system_variables, parameters, architecture, scaling):

    variables_si = system_variables['SI']
    variables_scaled = system_variables['scaled']

    iota = parameters['phi', 'iota']

    cstr_list = cstr_op.ConstraintList()
    for kite in architecture.kite_nodes:

        vec_u_ind_var = get_kite_induced_velocity_var(variables_si, kite)

        vec_u_ind_trivial = cas.DM.zeros((3, 1))
        resi_trivial = (vec_u_ind_var - vec_u_ind_trivial)

        vec_u_ind_final = get_induced_velocity_at_kite_si(options, wind, variables_si, kite, architecture, parameters)
        resi_final = (vec_u_ind_var - vec_u_ind_final)

        print_op.warn_about_temporary_functionality_alteration()
        resi_final *= 1.e-3

        resi_homotopy = (iota * resi_trivial + (1. - iota) * resi_final)

        # resi_scaled = []
        # for dim in range(resi_homotopy.shape[0]):
        #     local_scale = vect_op.find_jacobian_based_scalar_expression_scaling(resi_homotopy[dim], variables_scaled,
        #                                                                         parameters)
        #     resi_scaled = cas.vertcat(resi_scaled, resi_homotopy[dim] / local_scale)

        print_op.warn_about_temporary_functionality_alteration()
        # resi_scaled = struct_op.var_si_to_scaled('z', 'ui' + str(kite), resi_homotopy, scaling)

        general_cstr = cstr_op.Constraint(expr=resi_homotopy,
                                          name='induction_' + str(kite),
                                          cstr_type='eq')
        cstr_list.append(general_cstr)

    return cstr_list

def log_and_raise_unknown_induction_model_error(induction_model):
    message = 'induction model (' + induction_model + ') is not recognized'
    print_op.log_and_raise_error(message)
    return None

## velocities

def get_kite_induced_velocity_var(variables, kite):
    ind_var = variables['z']['ui' + str(kite)]
    return ind_var

def get_induced_velocity_at_kite_si(model_options, wind, variables_si, kite, architecture, parameters):
    induction_model = model_options['induction_model']
    parent = architecture.parent_map[kite]

    force_zero = model_options['aero']['induction']['force_zero']

    if induction_model == 'not_in_use' or force_zero:
        vec_u_ind_kite = cas.DM.zeros((3, 1))
    elif induction_model == 'actuator':
        vec_u_ind_kite = actuator_flow.get_kite_induced_velocity(model_options, variables_si, parameters, architecture, wind, kite, parent)
    elif induction_model == 'vortex':
        vec_u_ind_kite = vortex.get_induced_velocity_at_kite_si(variables_si, kite)
    else:
        log_and_raise_unknown_induction_model_error(induction_model)

    return vec_u_ind_kite


def get_kite_effective_velocity(variables, wind, kite, architecture):

    parent = architecture.parent_map[kite]

    u_app_kite = general_flow.get_kite_apparent_velocity(variables, wind, kite, parent)
    u_ind_kite = get_kite_induced_velocity_var(variables, kite)
    u_eff_kite = u_app_kite + u_ind_kite

    return u_eff_kite


#### outputs

def collect_outputs(options, atmos, wind, wake, variables_si, outputs, parameters, architecture, scaling):

    if actuator.model_is_included_in_comparison(options):
        outputs = actuator.collect_actuator_outputs(options, atmos, wind, variables_si, outputs, parameters, architecture, scaling)

    if vortex.model_is_included_in_comparison(options):
        outputs = vortex.collect_vortex_outputs(options, wind, wake, variables_si, outputs, architecture, scaling)

    return outputs

def get_dictionary_of_derivatives(model_options, system_variables, parameters, atmos, wind, outputs, architecture):
    derivative_dict = {}

    if model_options['induction_model'] == 'averaged':
        averaged_derivative_dict = averaged.get_dictionary_of_derivatives(system_variables, parameters, atmos, wind, outputs, architecture)
        for local_key, local_val in averaged_derivative_dict.items():
            if not local_key in derivative_dict.keys():
                derivative_dict[local_key] = local_val

    if vortex.model_is_included_in_comparison(model_options):
        vortex_derivative_dict = vortex.get_dictionary_of_derivatives(outputs, architecture)
        for local_key, local_val in vortex_derivative_dict.items():
            if not local_key in derivative_dict.keys():
                derivative_dict[local_key] = local_val

    return derivative_dict