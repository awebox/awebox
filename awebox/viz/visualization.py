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
##################################
# Class Visualization contains plotting functions to visualize data
# of trials and sweeps
###################################
from . import tools
from . import trajectory
from . import variables
from . import animation
from . import output
from . import wake

import os


import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)
import matplotlib.pyplot as plt
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger

from typing import Dict, Tuple
from awebox.tools import struct_operations as struct_op
import numpy as np
import casadi as ca
import casadi.tools as cas
from awebox.tools.sam_functionalities import reconstruct_full_from_SAM, \
    originalTimeToSAMTime, CollocationIRK, constructPiecewiseCasadiExpression
from awebox.opti import diagnostics
from awebox.tools.struct_operations import calculate_SAM_regions, calculate_SAM_regionIndexArray, eval_time_grids_SAM


#todo: compare to initial guess for all plots as option
#todo: options for saving plots


class Visualization(object):

    def __init__(self):
        self._plot_dict = None
        self.__has_been_initially_calibrated = False
        self.__has_been_recalibrated = False

    def build(self, model, nlp, name, options):
        """
        Generate plot dictionary with all relevant plot information.
        :param model: system model
        :param nlp: NLP formulation
        :param visualization_options: visualization related options
        :return: None
        """

        self._plot_dict = tools.calibrate_visualization(model, nlp, name, options)
        self.__has_been_initially_calibrated = True

        self.create_plot_logic_dict()
        self._options = options

        return None

    def recalibrate(self, V_plot_scaled, P_fix_num, plot_dict, output_vals, integral_output_vals, parametric_options, time_grids, cost, name, V_ref_scaled, global_outputs):
        print_op.base_print('recalibrating visualization...')
        self._plot_dict = tools.recalibrate_visualization(V_plot_scaled, P_fix_num, plot_dict, output_vals, integral_output_vals,
                                                          parametric_options, time_grids, cost, name, V_ref_scaled,
                                                          global_outputs)
        self.__has_been_recalibrated = True

        return None

    def printCosts(self):
        """
        Print the indiviudal costs of the solution, if >0
        """
        print_op.base_print('======================================')
        print_op.base_print('Costs:')
        for key in self._plot_dict['cost'].keys():
            value = float(self._plot_dict['cost'][key])
            if np.abs(value) > 0:
                print_op.base_print(f'\t{key}: {value:0.4f}')
        print_op.base_print('======================================')

    def plot(self, V_plot_scaled, P_fix_num, parametric_options, output_vals, integral_output_vals, flags, time_grids, cost, name, sweep_toggle, V_ref_scaled, global_outputs, fig_name='plot', fig_num=None, recalibrate = True):
        """
        Generate plots with given parametric and visualization options
        :param V_plot_scaled: plot data (scaled)
        :param parametric_options: values for parametric options
        :param visualization_options: visualization related options
        :return: None
        """

        has_not_been_recalibrated = (not self.__has_been_recalibrated)

        interpolation_in_plot_dict = 'interpolation_si' in self._plot_dict.keys()
        if interpolation_in_plot_dict:
            ip_time_length = len(self._plot_dict['interpolation_si']['time_grids']['ip'])
            ip_vars_length = len(self._plot_dict['interpolation_si']['x']['q10'][0])
        interpolation_length_is_inconsistent = (not interpolation_in_plot_dict) or (ip_time_length != ip_vars_length)

        threshold = 1e-2
        V_plot_scaled_in_plot_dict = ('V_plot_scaled' in self._plot_dict.keys()) and (self._plot_dict['V_plot_scaled'] is not None)
        V_plot_scaled_is_same = False
        if V_plot_scaled_in_plot_dict:
            V_plot_scaled_is_same = vect_op.norm(self._plot_dict['V_plot_scaled'].cat - V_plot_scaled.cat) / V_plot_scaled.cat.shape[0] < threshold
        if V_plot_scaled is None:
            using_new_V_plot = False
        else:
            using_new_V_plot = (not V_plot_scaled_in_plot_dict) or (not V_plot_scaled_is_same)

        if has_not_been_recalibrated or interpolation_length_is_inconsistent or using_new_V_plot:
            if recalibrate:
                self.recalibrate(V_plot_scaled, P_fix_num, self._plot_dict, output_vals, integral_output_vals, parametric_options, time_grids, cost, name, V_ref_scaled, global_outputs)

        if type(flags) is not list:
            flags = [flags]

        # define special flags
        if 'all' in flags:
            flags = list(self._plot_logic_dict.keys())
            flags.remove('animation')
            flags.remove('animation_snapshot')
            flags = [flag for flag in flags if 'outputs:' not in flag]

        level_1 = ['states', 'controls', 'isometric']
        level_2 = level_1 + ['invariants', 'algebraic_variables', 'lifted_variables', 'constraints']
        level_3 = level_2 + ['aero_dimensionless', 'aero_coefficients', 'projected_xy', 'projected_xz', 'projected_yz']

        if 'level_1' in flags:
            flags.remove('level_1')
            flags += level_1

        if 'level_2' in flags:
            flags.remove('level_2')
            flags += level_2

        if 'level_3' in flags:
            flags.remove('level_3')
            flags += level_3

        # iterate over flags
        for flag in flags:
            if flag[:5] == 'comp_':
                awelogger.logger.warning('Comparison plots are only supported for sweeps. Flag "' + flag + '" ignored.')
            else:
                self.__produce_plot(flag, fig_name, parametric_options['visualization']['cosmetics'], fig_num)

        if parametric_options['visualization']['cosmetics']['show_when_ready'] == True and sweep_toggle == False:
            plt.show()

        return None

    def create_plot_logic_dict(self):
        """
        Create a dict for selecting the correct plotting function for a given flag.
        Notation for adding entries:
        (FUNCTION, TUPLE_WITH_ADDITIONAL_ARGS/None)
        :return: dictionary for plot function selection
        """

        outputs = self.plot_dict['outputs_dict']
        variables_dict = self.plot_dict['variables_dict']
        integral_variables = self.plot_dict['integral_output_names']

        plot_logic_dict = {}
        plot_logic_dict['isometric'] = (trajectory.plot_trajectory, {'side':'isometric'})
        plot_logic_dict['projected_xy'] = (trajectory.plot_trajectory, {'side':'xy'})
        plot_logic_dict['projected_yz'] = (trajectory.plot_trajectory, {'side':'yz'})
        plot_logic_dict['projected_xz'] = (trajectory.plot_trajectory, {'side':'xz'})
        plot_logic_dict['quad'] = (trajectory.plot_trajectory, {'side':'quad'})
        plot_logic_dict['animation'] = (animation.animate_monitor_plot, None)
        plot_logic_dict['animation_snapshot'] = (animation.animate_snapshot, None)
        plot_logic_dict['vortex_haas_verification'] = (wake.plot_haas_verification_test, None)
        plot_logic_dict['local_induction_factor'] = (output.plot_local_induction_factor, None)
        plot_logic_dict['average_induction_factor'] = (output.plot_annulus_average_induction_factor, None)
        plot_logic_dict['relative_radius'] = (output.plot_relative_radius, None)
        plot_logic_dict['loyd_comparison'] = (output.plot_loyd_comparison, None)
        plot_logic_dict['aero_coefficients'] = (output.plot_aero_coefficients, None)
        plot_logic_dict['aero_dimensionless'] = (output.plot_aero_validity, None)
        plot_logic_dict['actuator_isometric'] = (wake.plot_actuator, {'side':'isometric'})
        plot_logic_dict['actuator_xy'] = (wake.plot_actuator, {'side':'xy'})
        plot_logic_dict['actuator_yz'] = (wake.plot_actuator, {'side':'yz'})
        plot_logic_dict['actuator_xz'] = (wake.plot_actuator, {'side':'xz'})
        plot_logic_dict['wake_isometric'] = (wake.plot_wake, {'side':'isometric'})
        plot_logic_dict['wake_xy'] = (wake.plot_wake, {'side':'xy'})
        plot_logic_dict['wake_yz'] = (wake.plot_wake, {'side':'yz'})
        plot_logic_dict['wake_xz'] = (wake.plot_wake, {'side':'xz'})
        plot_logic_dict['circulation'] = (output.plot_circulation, None)
        plot_logic_dict['states'] = (variables.plot_states, None)
        plot_logic_dict['wake_states'] = (variables.plot_wake_states, None)
        for variable in list(variables_dict['x'].keys()) + integral_variables:
            plot_logic_dict['states:' + variable] = (variables.plot_states, {'individual_state':variable})
        plot_logic_dict['controls'] = (variables.plot_controls, None)
        for control in list(variables_dict['u'].keys()):
            plot_logic_dict['controls:' + control] = (variables.plot_controls, {'individual_control':control})
        plot_logic_dict['invariants'] = (variables.plot_invariants, None)
        plot_logic_dict['algebraic_variables'] = (variables.plot_algebraic_variables, None)
        plot_logic_dict['wake_lifted_variables'] = (variables.plot_wake_lifted, None)
        plot_logic_dict['lifted_variables'] = (variables.plot_lifted, None)
        plot_logic_dict['constraints'] = (output.plot_constraints, None)

        for output_top_name in list(outputs.keys()):
            plot_logic_dict['outputs:' + output_top_name] = (output.plot_outputs, {'output_top_name': output_top_name})

        self._plot_logic_dict = plot_logic_dict
        self._plot_dict['plot_logic_dict'] = plot_logic_dict

    def __produce_plot(self, flag, fig_name, cosmetics, fig_num = None):
        """
        Produce the plot for a given flag, fig_num and cosmetics.
        :param flag: string identifying the kind of plot that should be produced
        :param fig_num: number of the figure that the plot should be displayed in
        :param cosmetics: cosmetic options for the plot
        :return: updated fig_num
        """

        # map flag to function
        fig_name = self._plot_dict['name'] + '_' + flag + '_' + fig_name

        if fig_num is not None:
            self._plot_logic_dict[flag][1]['fig_num'] = fig_num

        tools.map_flag_to_function(flag, self._plot_dict, cosmetics, fig_name, self._plot_logic_dict)

        if fig_num is not None:
            del self._plot_logic_dict[flag][1]['fig_num']

        # save figures
        if cosmetics['save_figs']:
            name_rep = self._plot_dict['name']
            for char in ['(', ')', '_', ' ']:
                name_rep = name_rep.replace(char, '')

            directory = "./figures"
            directory_exists = os.path.isdir(directory)
            if not directory_exists:
                os.mkdir(directory)

            save_name = directory + '/' + name_rep + '_' + flag
            plt.savefig(save_name + '.eps', bbox_inches='tight', format='eps', dpi=1000)
            plt.savefig(save_name + '.pdf', bbox_inches='tight', format='pdf', dpi=1000)

        return None

    @property
    def plot_dict(self):
        return self._plot_dict

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, value):
       self._options = value

    @property
    def has_been_initially_calibrated(self):
        return self.__has_been_initially_calibrated

    @property
    def has_been_recalibrated(self):
        return self.__has_been_recalibrated

    @property
    def plot_logic_dict(self):
        return self._plot_logic_dict

    @plot_logic_dict.setter
    def plot_logic_dict(self, value):
        print('Cannot set plot_logic_dict object.')


class VisualizationSAM(Visualization):

    def __init__(self):
        super().__init__()
        self._plot_dict_SAM: dict = None
        self._options: dict = None

    def build(self, model, nlp, name, options):
        """
        Generate plot dictionary with all relevant plot information.
        :param model: system model
        :param nlp: NLP formulation
        :param visualization_options: visualization related options
        :return: None
        """

        self._plot_dict = tools.calibrate_visualization(model, nlp, name, options)
        self._plot_dict_SAM = tools.calibrate_visualization(model, nlp, name, options)
        self.create_plot_logic_dict()
        self._options = options

    def recalibrate(self, V_plot_scaled, P_fix_num, plot_dict, output_vals, integral_output_vals, parametric_options, time_grids, cost, name, V_ref_scaled, global_outputs):
        """ Recalibrate both the SAM and the RECONSTRUCTED plot dictionaries. """

        # in the original (SAM) dictionary, only the timegrid is different
        plot_dict_Awebox = tools.recalibrate_visualization(V_plot_scaled, P_fix_num, plot_dict, output_vals, integral_output_vals,
                                                           parametric_options, time_grids, cost, name, V_ref_scaled,
                                                           global_outputs)
        self.plot_dict_SAM = plot_dict_Awebox.copy()

        # replace the interpolating grid with the SAM grid
        time_grid_ip_original: np.ndarray = self.plot_dict_SAM['time_grids']['ip']
        time_grid_xcoll_original: np.ndarray = self.plot_dict_SAM['time_grids']['x_coll'].full().flatten()

        # modify a bit for better post-processing: for x_coll timegrid
        # check if any values of t are close to any values in ts_cumsum,
        # this happens if the time points are equal, but are supposed to be in different SAM regions,
        # for example when radau collocation is used

        # find  pairs of indices in time_grid_ip_original that are close to each other
        close_indices = np.where(np.isclose(np.diff(time_grid_xcoll_original), 0.0))[0]
        for first_index in close_indices:
            time_grid_xcoll_original[first_index] -= 1E-6
            time_grid_xcoll_original[first_index + 1] += 1E-6

        originalTimeToSAMTime_f = originalTimeToSAMTime(self.options['nlp'], V_plot_scaled['theta', 't_f'])
        time_grid_SAM_eval = eval_time_grids_SAM(self.options['nlp'], V_plot_scaled['theta', 't_f'])
        time_grid_SAM_eval['ip'] = originalTimeToSAMTime_f.map(time_grid_ip_original.size)(time_grid_ip_original).full().flatten()

        # add the region indices to the SAM plot dictionary
        self.plot_dict_SAM['SAM_regions_x_coll'] = calculate_SAM_regionIndexArray(self.options['nlp'],
                                                                                  V_plot_scaled,
                                                                                  time_grid_xcoll_original)
        self.plot_dict_SAM['SAM_regions_ip'] = calculate_SAM_regionIndexArray(self.options['nlp'],
                                                                              V_plot_scaled,
                                                                              time_grid_ip_original)
        self.plot_dict_SAM['time_grids'] = time_grid_SAM_eval  # we do this AFTER we calculate the region indices

        # evaluate the polynomial of the average state trajectory

        time_grid_X, X, time_grid_X_nodes, X_coll = self.interpolate_average_state_trajectory(self.options['nlp'], V_plot_scaled)
        self.plot_dict_SAM['time_grids']['ip_X'] = time_grid_X
        self.plot_dict_SAM['X'] = X
        self.plot_dict_SAM['time_grids']['X_coll'] = time_grid_X_nodes
        self.plot_dict_SAM['X_coll'] = X_coll

        # BUILD THE RECONSTURCTED PLOT DICT
        # and replace the plot_dict with it
        self._plot_dict = self.create_reconstructed_plot_dict(V_plot_scaled, output_vals['opt'], global_outputs, integral_output_vals)

    def create_reconstructed_plot_dict(self, V_plot_scaled, output_vals, global_outputs,integral_outputs_final) -> dict:
        """ Create the plot dictionary for the RECONSTRUCTED variables and outputs. """

        # extract information
        plot_dict = self.plot_dict  # get the existing plot dict, it already contains some information
        nlp_options = self.options['nlp']

        # reconstruct the full trajectory
        awelogger.logger.info('Reconstructing the full trajectory from the SAM solution..')
        V_reconstructed_scaled, time_grid_reconstructed, output_vals_reconstructed = reconstruct_full_from_SAM(
            nlpoptions=nlp_options, V_opt_scaled=V_plot_scaled, output_vals_opt=output_vals)

        # undo the scaling of the reconstructed variables
        scaling = plot_dict['model_variables'](plot_dict['model_scaling'])
        V_reconstructed_si = struct_op.scaled_to_si(V_reconstructed_scaled, scaling)  # convert V_plot to SI units

        # interpolate the reconstructed trajectory
        n_ip = self.options['visualization']['cosmetics']['interpolation']['n_points']
        awelogger.logger.info(f'Interpolating reconstruted trajectory with {n_ip} points  ..')
        funcs_ip = build_interpolate_functions_full_solution(V_reconstructed_si, time_grid_reconstructed, nlp_options,
                                                             output_vals_reconstructed)

        # evaluate states, controls, algebraic variables, outputs at the interpolated points
        # todo: use the existing awebox interpolation, or replace
        t_ip = np.linspace(0, float(time_grid_reconstructed['x'][-1]), n_ip)
        x_ip = funcs_ip['x'].map(t_ip.size)(t_ip)
        x_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['x'], x_ip)
        u_ip = funcs_ip['u'].map(t_ip.size)(t_ip)
        u_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['u'], u_ip)
        z_ip = funcs_ip['z'].map(t_ip.size)(t_ip)
        z_ip_dict = dict_from_repeated_struct(plot_dict['variables_dict']['z'], z_ip)
        y_ip = funcs_ip['y'].map(t_ip.size)(t_ip)
        y_ip_dict = dict_from_repeated_struct(plot_dict['model_outputs'], y_ip)

        # build the output dict
        awelogger.logger.info('Building plot ditionary for the reconstructed trajectory..')
        plot_dict['z'] = z_ip_dict
        plot_dict['x'] = x_ip_dict
        plot_dict['u'] = u_ip_dict
        plot_dict['outputs'] = y_ip_dict
        plot_dict['output_vals'] = [output_vals_reconstructed,output_vals_reconstructed] # TODO: this is not the intended functionality
        plot_dict['time_grids'] = time_grid_reconstructed
        plot_dict['time_grids']['ip'] = t_ip
        plot_dict['global_outputs'] = global_outputs
        plot_dict['V_plot'] = V_reconstructed_scaled
        plot_dict['V_plot_si'] = V_reconstructed_si
        n_k_total = len(V_reconstructed_scaled['x']) - 1
        plot_dict['n_k'] = n_k_total
        plot_dict['interpolation_si'] = {'x': x_ip_dict,
                                         'u': u_ip_dict,
                                         'z': z_ip_dict,
                                         'outputs': y_ip_dict,
                                         'time_grids': time_grid_reconstructed}

        # fill theta
        plot_dict['theta'] = {}
        variables_dict = plot_dict['variables_dict']
        for name in variables_dict['theta'].keys():
            plot_dict['theta'][name] = plot_dict['V_plot']['theta', name].full()[0][0]

        plot_dict['integral_outputs_final'] = integral_outputs_final
        plot_dict['power_and_performance'] = diagnostics.compute_power_and_performance(plot_dict)

        awelogger.logger.info('... Done!')

        return plot_dict


    @property
    def plot_dict(self) -> dict:
        """ The interpolated RECONSTRUCTED trajectory and data it contains the same variables as the
        original plot_dict, expect:
            - the nlp variables V_plot now are the RECONSTRUCTED variables
            - the interpolated trajectories ('x', 'u', 'z', 'outputs') are the RECONSTRUCTED trajectories
            - the time grid is the RECONSTRUCTED time grid ('time_grids')
            - the raw output vals ('output_vals') are the RECONSTRUCTED output values, but both entries [0] and [1] are
              the same, since there are no initial reconstructed trajectories.

        Since this is the reconstructed trajectory from a SAM problem,
        the trajectory is only an approximation of a physical trajectory (!).
        """
        awelogger.logger.warning('`plot_dict` - You are accessing the RECONSTRUCTED results from a SAM problem. These '
                                 'results are only an approximation of a physical trajectory.')
        return self._plot_dict

    @property
    def plot_dict_SAM(self):
        """ The plot dictionary for the original SAM problem and its outputs. It contains:

                - the same variables as the original plot_dict, expect:
                - the time grid is the SAM time grid ('time_grids')
                - the SAM regions are calculated and stored ('SAM_regions_x_coll', 'SAM_regions_ip')
        """
        return self._plot_dict_SAM

    @plot_dict_SAM.setter
    def plot_dict_SAM(self, value):
        self._plot_dict_SAM = value

    def interpolate_average_state_trajectory(self, nlp_options: dict, V_plot_scaled: cas.struct) -> Tuple[np.ndarray, dict, np.ndarray, dict]:
        """ Interpolate the average state trajectory of the SAM solution.

        Returns the time grid and the interpolated average state trajectory.

        :param nlp_options: the nlp options e.g. trial.options['nlp']
        :param V_plot_scaled: the scaled optimal variables solution
        """

        N = 100  # number of points for the interpolation (more is not needed, since very little is happening)

        # undo scaling
        scaling = self.plot_dict['model_variables'](self.plot_dict['model_scaling'])
        V_plot_si = struct_op.scaled_to_si(V_plot_scaled, scaling)  # convert V_plot to SI units

        # interpolate the average polynomials
        from awebox.tools.sam_functionalities import CollocationIRK
        d_SAM = nlp_options['SAM']['d']
        coll_points = np.array(ca.collocation_points(d_SAM, nlp_options['SAM']['MaInt_type']))
        interpolator_average_integrator = CollocationIRK(coll_points)
        interpolator_average = interpolator_average_integrator.getPolyEvalFunction(
            shape=self.plot_dict['variables_dict']['x'].cat.shape, includeZero=True)
        tau_average = np.linspace(0, 1, N)

        # compute the average polynomials and fill the dataframe
        X_average = interpolator_average.map(tau_average.size)(tau_average, V_plot_si['x_macro', 0],
                                                               *[V_plot_si['x_macro_coll', i] for i in range(d_SAM)])
        X_average = self.plot_dict['variables_dict']['x'].repeated(X_average)
        X_dict = {}
        for entry_name in self.plot_dict['variables_dict']['x'].keys():
            X_dict[entry_name] = []
            for index_dim in range(self.plot_dict['variables_dict']['x'][entry_name].shape[0]):
                # we evaluate on the AWEBox time grid, not the SAM time grid!
                values = ca.vertcat(*X_average[:, entry_name, index_dim]).full().flatten()
                X_dict[entry_name].append(values)

        # X collocation Nodes
        X_average_coll = interpolator_average.map(coll_points.size)(coll_points, V_plot_si['x_macro', 0],
                                                               *[V_plot_si['x_macro_coll', i] for i in range(d_SAM)])
        X_average_coll = self.plot_dict['variables_dict']['x'].repeated(X_average_coll)
        X_coll_dict = {}
        for entry_name in self.plot_dict['variables_dict']['x'].keys():
            X_coll_dict[entry_name] = []
            for index_dim in range(self.plot_dict['variables_dict']['x'][entry_name].shape[0]):
                # we evaluate on the AWEBox time grid, not the SAM time grid!
                values = ca.vertcat(*X_average_coll[:, entry_name, index_dim]).full().flatten()
                X_coll_dict[entry_name].append(values)


        # find the duration of the regions
        n_k = nlp_options['n_k']
        regions_indeces = calculate_SAM_regions(nlp_options)
        regions_deltans = np.array([region.__len__() for region in regions_indeces])
        N_regions = nlp_options['SAM']['d'] + 1
        assert len(regions_indeces) == N_regions
        T_regions = (V_plot_si['theta', 't_f'] / n_k * regions_deltans).full().flatten()[0:-1]

        # construct the time grid for the average polynomials
        tau = ca.SX.sym('tau')
        b_tau = ca.vertcat(*[poly(tau) for poly in interpolator_average_integrator.polynomials_int])
        interpolator_time = ca.Function('interpolator_time', [tau], [b_tau.T@T_regions])
        time_grid_X = interpolator_time.map(tau_average.size)(tau_average).full().flatten()
        time_grid_X_coll = interpolator_time.map(coll_points.size)(coll_points).full().flatten()

        return time_grid_X, X_dict, time_grid_X_coll, X_coll_dict


def build_interpolate_functions_full_solution(V: cas.struct, tgrid: dict , nlpoptions: dict, output_vals: np.ndarray) -> Dict[str, ca.Function]:
    """ Build functions that interpolate the full solution from a given V structure and with nodes on timegrid['x'].

    Returns a dictionary of casadi functions that interpolate the state, control,algebraic variables and the outputs
    for a given time, i.e. x(t), u(t), z(t), y(t)

    :param V: the solution structure, containing 'x' (nx, n_k+1), 'u' (nu, n_k), 'z' (nz, n_k)
    :param tgrid: the time grid structure, containing 'x' (n_k+1)
    :param nlpoptions: the nlp options e.g. trial.options['nlp']
    :param output_vals: the output values, containing the structures of the model outputs e.g. plot_dict['output_vals'][1]
    """

    assert {'x', 'u', 'z'}.issubset(V.keys())

    # build micro-integration interpolation functions
    d_micro = nlpoptions['collocation']['d']
    coll_points = np.array(ca.collocation_points(d_micro,nlpoptions['collocation']['scheme']))
    coll_integrator = CollocationIRK(coll_points)
    intpoly_x_f = coll_integrator.getPolyEvalFunction(V.getStruct('x')(0).shape, includeZero=True)
    intpoly_z_f = coll_integrator.getPolyEvalFunction(V.getStruct('z')(0).shape, includeZero=False)
    intpoly_outputs_f = coll_integrator.getPolyEvalFunction(output_vals[:, 0].shape, includeZero=True)

    # number of intervals & edges
    n_k = len(V['u'])
    assert tgrid['x'].shape[0] == n_k + 1, (f'The number of edges in the time grid'
                                            f' should be equal to n_k + 1 = {n_k + 1}, but is {tgrid["x"].shape[0]}')
    edges = tgrid['x'].full().flatten()

    # iterate over the intervals
    express_x = []
    express_u = []
    express_z = []
    express_y = []
    t = ca.SX.sym('t')
    for n in range(n_k):
        t_n = t - edges[n]  # remove the offset from the time
        delta_t = edges[n + 1] - edges[n]  # duration of the interval

        # build casadi expressions for the local interpolations of state, control and algebraic variables
        express_x.append(intpoly_x_f(t_n / delta_t, V['x', n], *V['coll_var', n, :, 'x']))
        express_z.append(intpoly_z_f(t_n / delta_t, *V['coll_var', n, :, 'z']))
        express_u.append(V['u', n])
        express_y.append(intpoly_outputs_f(t_n / delta_t, *[output_vals[:, n*(d_micro+1)+i] for i in range(d_micro+1)]))

    # shift the final edge a bit to avoid numerical issues
    edges[-1] = edges[-1] + 1e-6

    # combine into single function
    express_x = constructPiecewiseCasadiExpression(t, edges.tolist(), express_x)
    express_z = constructPiecewiseCasadiExpression(t, edges.tolist(), express_z)
    express_u = constructPiecewiseCasadiExpression(t, edges.tolist(), express_u)
    express_y = constructPiecewiseCasadiExpression(t, edges.tolist(), express_y)

    return {'x': ca.Function('interpolated_x', [t], [express_x]),
            'u': ca.Function('interpolated_u', [t], [express_u]),
            'z': ca.Function('interpolated_z', [t], [express_z]),
            'y': ca.Function('interpolated_y', [t], [express_y])}


def dict_from_repeated_struct(struct: ca.tools.struct, values: ca.DM) -> dict:
    """ Create a nested dictionary from values of a repeated
    casadi structure (n_struct, n_vals) that contains the values.

    Args:
        struct: casadi struct
        values: casadi DM of shape (n_struct, n_vals) with the values
    """

    assert struct.shape[0] == values.shape[0]

    # cast into repeated structure
    struct_repeated = struct.repeated(values)

    dict_out = {}
    for canon in struct.canonicalIndices():
        vals = ca.vertcat(*struct_repeated[(slice(None),) + canon]).full().flatten()
        assign_nested_dict(dict_out, canon, vals)

    return dict_out


def assign_nested_dict(dictionary: dict, keys: list, value):
    """ (from CHATGPT)
    Indexes a nested dictionary with a list of keys and assigns a value to the final key.
    If a key is not present in the dictionary, it is created and assigned an empty dictionary.

    Args:
        dictionary (dict): The nested dictionary to be indexed.
        keys (list): A list of keys to traverse the nested dictionary.
        value: The value to be assigned at the final key.
    """
    # Navigate through the dictionary using the keys
    for index,key in enumerate(keys[:-1]):
        dictionary = dictionary.setdefault(key, {})

    # Assign the value to the final key
    dictionary[keys[-1]] = value
