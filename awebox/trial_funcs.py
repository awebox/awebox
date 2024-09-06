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
"""
Collection of trial functions
python version 3.5 / casadi 3.4.5
- author:
    Thilo Bronnenmeyer, kiteswarms 2018
    Jochem De Schutter, alu-fr 2018
"""
import os.path


import awebox.tools.vector_operations as vect_op
import awebox.viz.tools as tools
import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.save_operations as save_op
import copy

def is_possibly_an_already_loaded_seed(loaded_dict):

    loaded_dict_is_a_dict = isinstance(loaded_dict, dict)
    if not loaded_dict_is_a_dict:
        return False

    contains_expected = ('plot_dict' in loaded_dict.keys()) and ('solution_dict' in loaded_dict.keys())
    if not contains_expected:
        return False

    return True



def generate_trial_data_and_write_to_csv(trial, filename, freq, rotation_representation, interpolate = False):
    """
    Generate an output .csv file containing all information from the trial
    :param trial: trial whose data is to be stored in the .csv
    :param name: name of the .csv
    :param freq: sampling frequency for output
    :return: None
    """

    if not interpolate:
        if 'interpolation_si' in trial.visualization.plot_dict.keys():
            data_dict = trial.visualization.plot_dict['interpolation_si']
        else:
            data_dict = trial.visualization.plot_dict['interpolation_scaled']
    else:
        data_dict = interpolate_data(trial, freq)
    data_dict['architecture'] = trial.model.architecture
    data_dict['options'] = trial.options
    save_op.write_csv_data(data_dict=data_dict, filename=filename, rotation_representation=rotation_representation)
    return None

def interpolate_data(trial, freq):
    """
    Interpolate data trial data with a given sampling frequency
    :param trial: trial whose data is to be interpolated
    :param freq: sampling frequency
    :return: dictionary with trial data, interpolation time grid
    """

    tf = trial.optimization.V_final_si['theta', 't_f', 0]  # TODO: phase fix tf
    n_points = int(freq * tf) # number of interpolating points

    parametric_options = trial.options['visualization']['cosmetics']
    parametric_options['interpolation']['n_points'] = n_points
    time_grids = copy.deepcopy(trial.optimization.time_grids)
    variables_dict = trial.model.variables_dict
    outputs_fun = trial.model.outputs_fun
    P_fix_num = trial.optimization.p_fix_num
    V_opt = trial.optimization.V_opt
    model_scaling = trial.model.scaling.cat
    model_parameters = struct_op.strip_of_contents(trial.model.parameters)
    outputs_dict = trial.model.outputs_dict
    outputs_opt = trial.optimization.outputs_opt
    integral_output_names = trial.model.integral_scaling.keys()
    integral_outputs_opt = trial.optimization.integral_outputs_opt

    if trial.options['nlp']['discretization'] == 'direct_collocation':
        Collocation = trial.nlp.Collocation
    else:
        Collocation = None

    interpolation = struct_op.interpolate_solution(parametric_options, time_grids, variables_dict, V_opt,
        P_fix_num, model_parameters, model_scaling, outputs_fun, outputs_dict, 
        integral_output_names, integral_outputs_opt, Collocation=Collocation)
    return interpolation


def generate_optimal_model(trial, param_options=None):

    """
    Generate optimal model dict based on both optimized parameter values
    as numerical values of constant parameters.
    :param trial: trial containing OCP solution and model information
    :return: model dict containing all relevant functions.
    """

    # fill in variables structure
    variables  = []
    for var_type in list(trial.model.variables.keys()):
        if var_type != 'theta':
            variables.append(cas.SX.sym(var_type,trial.model.variables[var_type].shape[0]))
        else:
            for var in list(trial.model.variables_dict['theta'].keys()):
                if var != 't_f':
                    variables.append(cas.SX(trial.optimization.V_opt['theta',var]))
                else:
                    if trial.optimization.V_opt['theta','t_f'].shape[0] == 1:
                        t_f = trial.optimization.V_opt['theta','t_f']
                        variables.append(cas.SX(t_f))
                    else:
                        t_f = trial.visualization.plot_dict['output_vals'][1]['final', 'time_period', 'val']
                        variables.append(cas.SX(t_f))
    variables = trial.model.variables(cas.vertcat(*variables))

    # fill in parameters structure
    parameters = trial.model.parameters(0.0)
    if param_options is None:
        param_options = trial.options['solver']['initialization']['sys_params_num']
    for param_type in list(param_options.keys()):
        if isinstance(param_options[param_type],dict):
            for param in list(param_options[param_type].keys()):
                if isinstance(param_options[param_type][param],dict):
                    for subparam in list(param_options[param_type][param].keys()):
                        parameters['theta0',param_type,param,subparam] = param_options[param_type][param][subparam]

                else:
                    parameters['theta0',param_type,param] = param_options[param_type][param]
        else:
            parameters['theta0', param_type] = param_options[param_type]

    # create stage cost function
    import awebox.ocp.objective as obj
    import awebox.ocp.discretization as discr
    reg_costs_fun, reg_costs_struct = obj.get_general_reg_costs_function(trial.nlp.options, trial.model.variables, trial.nlp.V)
    weights = obj.get_regularization_weights(trial.model.variables, trial.optimization.p_fix_num, trial.options['nlp']).cat
    refs = struct_op.get_variables_at_time(trial.options['nlp'], trial.nlp.V(trial.optimization.p_fix_num['p','ref']), trial.nlp.Xdot(0.0), trial.model.variables, 0, 0)
    var = trial.model.variables
    xdot_reg = reg_costs_fun(var, refs, weights)[1]
    u_reg = reg_costs_fun(var, refs, weights)[2]
    beta_reg = 0.0
    for kite in trial.model.architecture.kite_nodes:
        beta_sq = trial.model.outputs(trial.model.outputs_fun(var, trial.model.parameters))['aerodynamics', 'beta{}'.format(kite)]**2
        beta_reg += trial.optimization.p_fix_num['cost', 'beta']*beta_sq / trial.options['nlp']['cost']['normalization']['beta']
    if not 'e' in trial.model.variables_dict['x'].keys():
        power = trial.model.integral_outputs_fun(var, trial.model.parameters)
    else:
        outputs_eval = trial.model.outputs(trial.model.outputs_fun(var, trial.model.parameters))
        power = outputs_eval['performance','p_current'] / trial.model.scaling['x']['e']
    cost_weighting = discr.setup_nlp_cost()(trial.optimization.p_fix_num['cost'])
    stage_cost = - cost_weighting['power']*power / t_f.full()[0][0] + u_reg + xdot_reg + beta_reg
    quadrature = cas.Function('quad', [var, trial.model.parameters], [stage_cost])

    # create dae object based on numerical parameters
    import awebox.mdl.dae as dae
    model_dae = dae.Dae(
        variables,
        parameters,
        trial.model.dynamics,
        quadrature,
        param = 'num')

    # build model rootfinder
    model_dae.build_rootfinder()

    # create function arguments
    f_args = [model_dae.dae['x'], model_dae.dae['p'], model_dae.dae['z']]

    # create model dict
    model = {}
    model['dae'] = model_dae.dae
    model['constraints'] = cas.Function('constraints', [*f_args], [trial.model.constraints_fun(variables, parameters)])
    model['outputs'] = cas.Function('outputs', [*f_args], [trial.model.outputs_fun(variables, parameters)])
    model['scaling'] = trial.model.scaling
    model['var_bounds'] = trial.model.variable_bounds
    model['var_bounds_fun'] = cas.Function(
        'var_bounds',
        [*f_args],
        [generate_var_bounds_fun(trial.model)[0](variables)]
        )
    model['var_constr_str'] = generate_var_bounds_fun(trial.model)[1]
    model['t_f'] = t_f.full()[0][0]
    model['rootfinder'] = model_dae.rootfinder

    return model


def generate_var_bounds_fun(model):

    var_constraints = []
    var_constr_str = []
    var_bounds = model.variable_bounds
    for var_type in list(model.variables.keys()):

        if var_type in ['x', 'u', 'z']:

            for var in list(model.variables_dict[var_type].keys()):

                variables_sym = model.variables[var_type, var]
                upper_bounds = vect_op.columnize(var_bounds[var_type][var]['ub'])
                lower_bounds = vect_op.columnize(var_bounds[var_type][var]['lb'])

                if (upper_bounds.shape != variables_sym.shape) and vect_op.is_numeric_scalar(upper_bounds):
                    upper_bounds = upper_bounds * cas.DM.ones(variables_sym.shape)
                if (lower_bounds.shape != variables_sym.shape) and vect_op.is_numeric_scalar(lower_bounds):
                    lower_bounds = lower_bounds * cas.DM.ones(variables_sym.shape)
                if (upper_bounds.shape != lower_bounds.shape) or (upper_bounds.shape != variables_sym.shape) or (lower_bounds.shape != variables_sym.shape):
                    message = 'something went wrong with the dimension of the variable '
                    message += 'bounds for ' + var_type + ' variable ' + var + ' with shape ' + str(variables_sym.shape) + '. '
                    message += 'ub shape is ' + str(upper_bounds.shape) + ', while lb shape is ' + str(lower_bounds.shape)
                    print_op.log_and_raise_error(message)

                for dim in range(variables_sym.shape[0]):
                    if upper_bounds[dim] != np.inf:
                        var_constraints.append(
                            model.variables[var_type, var, dim] - upper_bounds[dim]
                        )
                        var_constr_str.append(var_type + ' ' + var + ' ' + str(dim) + ' ub')

                    if lower_bounds[dim] != -np.inf:
                        var_constraints.append(
                            - model.variables[var_type, var, dim] + lower_bounds[dim]
                        )
                        var_constr_str.append(var_type + ' ' + var + ' ' + str(dim) + ' lb')

    var_bounds_fun = cas.Function('var_bounds', [model.variables], [cas.vertcat(*var_constraints)])
    return [var_bounds_fun, var_constr_str]
