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
# Class Optimization solves the NLP of the multi-kite system
###################################
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pdb
import pickle
from . import scheduling
from . import preparation
from . import diagnostics

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op
import awebox.tools.callback as callback

from numpy import linspace

import resource
import copy

from awebox.logger.logger import Logger as awelogger

import time

class Optimization(object):
    def __init__(self):
        self.__status = 'Optimization not yet built.'
        self.__V_opt = None
        self.__timings = {}
        self.__cumulative_max_memory = {}
        self.__iterations = {}
        self.__t_wall = {}
        self.__return_status_numeric = {}
        self.__outputs_init = None
        self.__outputs_opt = None
        self.__outputs_ref = None
        self.__time_grids = None
        self.__debug_fig_num = 1000

        plt.close('all')

    def build(self, options, nlp, model, formulation, name):

        awelogger.logger.info('Building NLP solver...')

        self.__name = name

        if self.__status == 'I am an optimization.':
            return None
        elif nlp.status == 'I am an NLP.':

            self.__options = options
            self.print_optimization_info()

            timer = time.time()

            # prepare callback
            self.__awe_callback = self.initialize_callback('awebox_callback', nlp, model, options)

            # generate solvers
            if options['generate_solvers']:
                self.generate_solvers(model, nlp, formulation, options, self.__awe_callback)

            # record set-up time
            self.__timings['setup'] = time.time() - timer
            self.__cumulative_max_memory['setup'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            self.__status = 'I am an optimization.'
            awelogger.logger.info('')

        else:
            raise ValueError('Cannot build optimization without building NLP.')

        return None

    def solve(self, trial_name, options, nlp, model, formulation, visualization,
              final_homotopy_step='final', warmstart_file=None, debug_flags=[],
              debug_locations=[], intermediate_solve=False):

        self.__debug_flags = debug_flags
        if debug_flags != [] and debug_locations == []:
            self.__debug_locations = 'all'
        else:
            self.__debug_locations = debug_locations

        if self.__status in ['I am an optimization.', 'I am a solved optimization.', 'I am a failed optimization.']:

            print_op.base_print('Setting up homotopy schedule...', level='info')

            # save final homotopy step
            self.__final_homotopy_step = final_homotopy_step
            self.__intermediate_solve = intermediate_solve

            # reset timings / iteration counters
            self.reset_timings_and_counters()

            # schedule the homotopy steps
            self.define_homotopy_update_schedule(model, formulation, nlp, options)

            # prepare problem
            self.define_standard_args(nlp, formulation, model, options, visualization)

            # restart the counter through the homotopy steps
            self.define_update_counter(nlp, formulation, model)

            # classifications
            use_warmstart = not (warmstart_file == None)
            make_steps = not (final_homotopy_step == 'initial_guess')

            # solve the problem
            if make_steps:
                if use_warmstart:
                    self.solve_from_warmstart(trial_name, nlp, formulation, model, options, warmstart_file, final_homotopy_step, visualization)
                else:
                    self.solve_homotopy(trial_name, nlp, model, options, final_homotopy_step, visualization)

            else:
                self.__generate_outputs_from_V(nlp, self.__V_init)
                self.__solve_succeeded = True
                self.__stats = None

            # process the solution
            self.process_solution(options, nlp, model, final_homotopy_step)

            if self.solve_succeeded:
                awelogger.logger.info('Optimization solved.')
                self.__status = 'I am a solved optimization.'
                awelogger.logger.info('Optimization solving time: %s', print_op.print_single_timing(self.__timings['optimization']))
            else:
                self.__status = 'I am a failed optimization.'

            awelogger.logger.info('')
        else:
            raise ValueError('Cannot solve optimization without building it.')

        return None

    def reset_timings_and_counters(self):

        for step in (set(self.__timings.keys()) - set(['setup']) | set(['optimization'])):
            self.__timings[step] = 0.

        for step in (set(self.__cumulative_max_memory.keys()) - set(['setup']) | set(['optimization'])):
            self.__cumulative_max_memory[step] = 0

        for step in (set(self.__iterations.keys()) - set(['setup']) | set(['optimization'])):
            self.__iterations[step] = 0.
            self.__t_wall[step] = 0.

        for step in (set(self.__return_status_numeric.keys()) - set(['setup']) | set(['optimization'])):
            self.__return_status_numeric[step] = 17

        self.__awe_callback.reset()

        return None


    ### interactive functions

    def __make_debug_plot(self, V_plot_scaled, nlp, visualization, location):

        if location == 'initial_guess':
            self.generate_outputs(nlp, {'x': self.__V_init})
        fig_name = 'debug_plot_' + location
        sweep_toggle = False
        cost_fun = nlp.cost_components[0]
        cost = struct_op.evaluate_cost_dict(cost_fun, V_plot_scaled, self.__p_fix_num)
        V_ref_scaled = self.__V_ref
        visualization.plot(V_plot_scaled, visualization.options, self.output_vals,
                           self.integral_output_vals, self.__debug_flags, self.__time_grids, cost,
                           self.__name, sweep_toggle, V_ref_scaled, self.__global_outputs_opt, fig_name=fig_name)

        return None


    def update_runtime_info(self, timer, step_name):

        self.__timings[step_name] = time.time() - timer
        self.__cumulative_max_memory[step_name] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.__return_status_numeric[step_name] = struct_op.convert_return_status_string_to_number(
            self.stats['return_status'])

        if step_name not in list(self.__iterations.keys()):
            self.__iterations[step_name] = 0.

        self.__iterations['optimization'] = self.__iterations['optimization'] + self.__iterations[step_name]
        self.__t_wall['optimization'] = self.__t_wall['optimization'] + self.__t_wall[step_name]
        self.__return_status_numeric['optimization'] = self.__return_status_numeric[step_name]
        self.__timings['optimization'] = self.__timings['optimization'] + self.__timings[step_name]
        self.__cumulative_max_memory['optimization'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def initialize_callback(self, name, nlp, model, options):

        V = nlp.V
        P = nlp.P

        nx = V.cat.shape[0]
        ng = nlp.g.shape[0]
        np = P.cat.shape[0]
        awe_callback = callback.awebox_callback(name, model, nlp, options, V, P, nx, ng, np, record_states = options['record_states'])

        return awe_callback




    ### solvers

    def generate_solvers(self, model, nlp, formulation, options, awe_callback):

        self.__solvers = preparation.generate_solvers(awe_callback, nlp, options)

        return None


    def solve_from_warmstart(self, trial_name, nlp, formulation, model, solver_options, warmstart_file, final_homotopy_step, visualization):

        self.__solve_succeeded = True

        warmstart_trial = self.extract_warmstart_trial(warmstart_file)
        self.set_warmstart_args(warmstart_trial, nlp)

        self.define_homotopy_update_schedule(model, formulation, nlp, solver_options)
        self.modify_schedule_for_warmstart(final_homotopy_step, warmstart_trial, nlp, model)

        # solve homotopy with warmstart
        self.solve_homotopy(trial_name, nlp, model, solver_options, final_homotopy_step, visualization)

        awelogger.logger.info(print_op.hline('#'))

        return None


    def solve_homotopy(self, trial_name, nlp, model, options, final_homotopy_step, visualization):

        print_op.base_print('Proceeding into homotopy...', level='info')

        available_homotopy_steps = self.__schedule['homotopy']
        if final_homotopy_step not in available_homotopy_steps:
            message = 'final_homotopy_step (' + final_homotopy_step + ') not recognized. available scheduling steps are: ' + repr(available_homotopy_steps)
            print_op.log_and_raise_error(message)

        # do not consider homotopy steps after specified final_homotopy_step
        final_index = self.__schedule['homotopy'].index(final_homotopy_step)
        homotopy_schedule = self.__schedule['homotopy'][:final_index+1]

        self.__solve_succeeded = True

        # iterate over homotopy schedule
        for step_name in homotopy_schedule:

            if self.__solve_succeeded:

                timer = time.time()
                self.solve_specific_homotopy_step(trial_name, step_name, final_homotopy_step, nlp, model, options, visualization)
                self.update_runtime_info(timer, step_name)

        awelogger.logger.info(print_op.hline('#'))

        return None

    def get_appropriate_solver_for_step(self, step_name):

        if 'all' in self.__solvers.keys():
            return self.__solvers['all']
        elif step_name == 'initial':
            return self.__solvers['initial']
        elif (step_name == 'final') and not (self.__intermediate_solve):
            return self.__solvers['final']
        else:
            return self.__solvers['middle']


    def solve_specific_homotopy_step(self, trial_name, step_name, final_homotopy_step, nlp, model, options, visualization):

        local_solver = self.get_appropriate_solver_for_step(step_name)

        if (step_name == 'initial') or (step_name == 'final'):
            self.solve_general_homotopy_step(trial_name, step_name, final_homotopy_step, 0, options, nlp, model, local_solver, visualization)

        else:
            number_of_steps = len(list(self.__schedule['bounds_to_update'][step_name].keys()))
            for homotopy_part in range(number_of_steps):
                self.solve_general_homotopy_step(trial_name, step_name, final_homotopy_step, homotopy_part, options, nlp, model, local_solver, visualization)

        return None

    def solve_general_homotopy_step(self, trial_name, step_name, final_homotopy_step, counter, solver_options, nlp, model, solver, visualization):

        if self.__solve_succeeded:

            awelogger.logger.info(print_op.hline("#"))
            awelogger.logger.info(self.__schedule['labels'][step_name][counter])
            awelogger.logger.info('')

            [self.__cost_update_counter, self.__p_fix_num] = scheduling.update_cost(self.__schedule, step_name, counter, self.__cost_update_counter, self.__p_fix_num)

            [self.__bound_update_counter, self.__V_bounds] = scheduling.update_bounds(self.__schedule, step_name, counter, self.__bound_update_counter, self.__V_bounds, model, nlp)

            # hand over the parameters to the solver
            self.__arg['p'] = self.__p_fix_num
            self.__awe_callback.update_P(self.__p_fix_num)

            # bounds on x
            self.__arg['ubx'] = self.__V_bounds['ub']
            self.__arg['lbx'] = self.__V_bounds['lb']

            # find current homotopy parameter
            if solver_options['homotopy_method']['type'] == 'single':
                phi_name = 'middle'
                solver_options['homotopy_method']['middle'] = 'penalty'
            else:
                phi_name = scheduling.find_current_homotopy_parameter(model.parameters_dict['phi'], self.__V_bounds)

            # solve
            step_has_defined_method = phi_name in solver_options['homotopy_method'].keys()
            if (phi_name != None) and step_has_defined_method and (solver_options['homotopy_method'][phi_name] == 'classic') and (counter == 0):
                if (solver_options['homotopy_step'][phi_name] < 1.0):
                    self.__perform_classic_continuation(step_name, phi_name, solver_options, solver)

            else:
                print_op.base_print('Calling the solver...', level='info')

                self.__solution = solver(**self.__arg)
                self.__stats = solver.stats()
                self.__save_stats(step_name)

            self.generate_outputs(nlp, self.__solution)

            diagnostics.print_runtime_values(self.__stats)
            diagnostics.print_homotopy_values(nlp, self.__solution, self.__p_fix_num)

            problem_is_healthy_or_unchecked = diagnostics.health_check(trial_name, step_name, final_homotopy_step, nlp, model, self.__solution, self.__arg, solver_options, self.__stats, self.__iterations, self.__cumulative_max_memory)
            if (not problem_is_healthy_or_unchecked) and (not self.__options['homotopy_method']['advance_despite_ill_health']):
                self.__solve_succeeded = False

            self.allow_next_homotopy_step()

            if step_name in self.__debug_locations or self.__debug_locations == 'all':
                V_plot_scaled = nlp.V(self.__solution['x'])
                self.__make_debug_plot(V_plot_scaled, nlp, visualization, step_name)

        return None


    def __perform_classic_continuation(self, step_name, phi_name, options, solver):

        # define parameter path
        step = options['homotopy_step'][phi_name]
        parameter_path = linspace(1-step, step, int(1/step)-1)

        # update fixed params
        self.__p_fix_num['cost', phi_name] = 0.0
        self.__arg['p'] = self.__p_fix_num

        # follow parameter path
        for phij in parameter_path:
            self.__V_bounds['ub']['phi',phi_name] = phij
            self.__V_bounds['lb']['phi',phi_name] = phij
            self.__arg['ubx'] = self.__V_bounds['ub']
            self.__arg['lbx'] = self.__V_bounds['lb']
            self.__solution = solver(**self.__arg)
            self.__stats = solver.stats()

            self.__arg['lam_x0'] = self.__solution['lam_x']
            self.__arg['lam_g0'] = self.__solution['lam_g']
            self.__arg['x0'] = self.__solution['x']

            # add up iterations of multi-step homotopies
            self.__save_stats(step_name)

        # prepare for second part of homotopy step
        self.__V_bounds['ub']['phi',phi_name] = 0
        self.__V_bounds['lb']['phi',phi_name] = 0

        return None

    def __save_stats(self, step_name):

        # add up iterations of multi-step homotopies
        if step_name not in list(self.__iterations.keys()):
            self.__iterations[step_name] = 0.

        if step_name not in list(self.__t_wall.keys()):
            self.__t_wall[step_name] = 0.

        self.__iterations[step_name] += self.__stats['iter_count']
        self.__t_wall[step_name] += self.__stats['t_wall_total']
        if 't_wall_callback_fun' in self.__stats.keys():
            self.__t_wall[step_name] -= self.__stats['t_wall_callback_fun']

        return None


    ### arguments

    def define_standard_args(self, nlp, formulation, model, options, visualization):

        self.__arg = preparation.initialize_arg(nlp, formulation, model, options, self.schedule)
        self.__arg_initial = {}
        self.__arg_initial['x0'] = nlp.V(self.__arg['x0'])

        self.__V_init = nlp.V(self.__arg['x0'])

        self.__p_fix_num = nlp.P(self.__arg['p'])

        self.__V_ref = nlp.V(self.__p_fix_num['p', 'ref'])

        if 'initial_guess' in self.__debug_locations or self.__debug_locations == 'all':
            self.__make_debug_plot(self.__V_init, nlp, visualization, 'initial_guess')

        self.__g_bounds = {}
        self.__g_bounds['lb'] = self.__arg['lbg']
        self.__g_bounds['ub'] = self.__arg['ubg']

        self.__V_bounds = {}
        self.__V_bounds['lb'] = self.__arg['lbx']
        self.__V_bounds['ub'] = self.__arg['ubx']

        self.__cost_update_counter = scheduling.initialize_cost_update_counter(nlp.P)

        return None

    def extract_warmstart_trial(self, warmstart_file):
        if type(warmstart_file) == str:
            try:
                filehandler = open(warmstart_file, 'r')
                load_trial = pickle.load(filehandler)
                warmstart_trial = load_trial.generate_solution_dict()
            except:
                raise ValueError('Specified warmstart trial does not exist.')
        elif type(warmstart_file) == dict:
            warmstart_trial = warmstart_file
        else:
            warmstart_trial = warmstart_file.generate_solution_dict()

        return warmstart_trial


    def set_warmstart_args(self, warmstart_trial, nlp):

        # set up warmstart
        [V_init_proposed,
        lam_x_proposed,
        lam_g_proposed] = struct_op.setup_warmstart_data(nlp, warmstart_trial)

        V_shape_matches = (V_init_proposed.cat.shape == nlp.V.cat.shape)
        if V_shape_matches:
            self.__V_init = V_init_proposed
            self.__arg['x0'] = self.__V_init.cat
        else:
            raise ValueError('Variables of specified warmstart do not correspond to NLP requirements.')

        lam_x_shape_matches = (lam_x_proposed.shape == self.__V_bounds['ub'].shape)
        if lam_x_shape_matches:
            self.__arg['lam_x0'] = lam_x_proposed
        else:
            raise ValueError('Variable bound multipliers of specified warmstart do not correspond to NLP requirements.')

        lam_g_shape_matches = (lam_g_proposed.shape == nlp.g.shape)
        if lam_g_shape_matches:
            self.__arg['lam_g0'] = lam_g_proposed
        else:
            raise ValueError('Constraint multipliers of specified warmstart do not correspond to NLP requirements.')

        # hand over the parameters to the solver
        self.__arg['p'] = self.__p_fix_num

        # bounds on x
        self.__arg['ubx'] = self.__V_bounds['ub']
        self.__arg['lbx'] = self.__V_bounds['lb']

        return None






    ### scheduling

    def define_homotopy_update_schedule(self, model, formulation, nlp, solver_options):
        self.__schedule = scheduling.define_homotopy_update_schedule(model, formulation, nlp, solver_options)
        return None

    def modify_schedule_for_warmstart(self, final_homotopy_step, warmstart_trial, nlp, model):

        # final homotopy step of warmstart file
        warmstart_step = warmstart_trial['final_homotopy_step']
        initial_index = self.__schedule['homotopy'].index(warmstart_step)

        # check if schedule is still consistent
        final_index = self.__schedule['homotopy'].index(final_homotopy_step)
        if final_index < initial_index:
            raise ValueError('Final homotopy step has a lower schedule index than final step of warmstart file')

        # adjust homotopy schedule
        homotopy_schedule = self.__schedule['homotopy'][initial_index:]

        self.__solve_succeeded = True

        # ensure that problem is the correct problem
        for step_name in self.__schedule['homotopy'][:initial_index]:
            if step_name == 'initial' or step_name == 'final':
                self.advance_counters_for_warmstart(step_name, 0, nlp, model)
            else:
                self.advance_counters_for_warmstart(step_name, 0, nlp, model)
                self.advance_counters_for_warmstart(step_name, 1, nlp, model)

        self.__schedule['homotopy'] = homotopy_schedule

        return None

    def define_update_counter(self, nlp, formulation, model):
        self.__bound_update_counter = scheduling.initialize_bound_update_counter(model, self.__schedule, formulation)
        self.__cost_update_counter = scheduling.initialize_cost_update_counter(nlp.P)
        return None

    def advance_counters_for_warmstart(self, step_name, counter, nlp, model):

        [self.__cost_update_counter, self.__p_fix_num] = scheduling.update_cost(self.__schedule, step_name, counter,
                                                                                self.__cost_update_counter,
                                                                                self.__p_fix_num)
        [self.__bound_update_counter, self.__V_bounds] = scheduling.update_bounds(self.__schedule, step_name, counter,
                                                                                  self.__bound_update_counter,
                                                                                  self.__V_bounds, model, nlp)
        return None

    def allow_next_homotopy_step(self):

        max_cpu_time_reached = (self.__stats['return_status'] == 'Maximum_CpuTime_Exceeded')
        if max_cpu_time_reached and self.__options['raise_error_at_max_time']:
            message = 'max_cpu_time limit reached'
            print_op.log_and_raise_error(message)

        return_status_number = struct_op.convert_return_status_string_to_number(self.__stats['return_status'])

        previous_step_failed = (return_status_number > 3)
        failure_due_to_max_iter = (return_status_number == 8)
        advance_despite_max_iter = self.__options['homotopy_method']['advance_despite_max_iter']
        excuse_failure_due_to_max_iter = failure_due_to_max_iter and advance_despite_max_iter

        should_not_advance = previous_step_failed and not excuse_failure_due_to_max_iter
        if should_not_advance:

            self.__solve_succeeded = False
            awelogger.logger.info('')
            awelogger.logger.info('ERROR: Solver FAILED, not moving on to next step...')

        else:

            try:
                self.arg_initial['lam_x0'] = self.__arg['lam_x0']
                self.arg_initial['lam_g0'] = self.__arg['lam_g0']
            except:
                awelogger.logger.info('no initial multipliers to be stored.')

            # retrieve and update
            self.__arg['lam_x0'] = self.__solution['lam_x']
            self.__arg['lam_g0'] = self.__solution['lam_g']
            self.__arg['x0'] = self.__solution['x']

        return None



    ### outputs

    def generate_outputs(self, nlp, solution):

        # extract solutions
        V_opt = nlp.V(solution['x'])

        # get outputs from V
        self.__generate_outputs_from_V(nlp, V_opt)

        return None

    def __generate_outputs_from_V(self, nlp, V_opt):

        V_initial = self.__V_init

        # general outputs
        _, nlp_output_fun = nlp.output_components
        outputs_init = nlp_output_fun(V_initial, self.__p_fix_num)
        outputs_opt = nlp_output_fun(V_opt, self.__p_fix_num)
        outputs_ref = nlp_output_fun(self.__V_ref, self.__p_fix_num)

        # integral outputs
        [nlp_integral_outputs, nlp_integral_outputs_fun] = nlp.integral_output_components
        integral_outputs_init = nlp_integral_outputs(nlp_integral_outputs_fun(V_initial, self.__p_fix_num))
        integral_outputs_opt = nlp_integral_outputs(nlp_integral_outputs_fun(V_opt, self.__p_fix_num))
        integral_outputs_ref = nlp_integral_outputs(nlp_integral_outputs_fun(self.__V_ref, self.__p_fix_num))

        # global outputs
        global_outputs_opt = nlp.global_outputs(nlp.global_outputs_fun(V_opt, self.__p_fix_num))

        # time grids
        time_grids = {'ref': {}}
        for grid in nlp.time_grids:

            # forcibly prevent decreasing time_grids for bad solution
            safe_t_f = copy.deepcopy(V_opt['theta', 't_f'])
            safe_minimum = 1.  # [seconds]
            for idx in range(safe_t_f.shape[0]):
                t_f_entry = safe_t_f[idx]
                if t_f_entry < 0.:
                    safe_t_f[idx] = safe_minimum
                    message = 'V_final includes negative time-periods, leading to ' \
                              'non-increasing time-grids that will cause later interpolations ' \
                              'to fail. computing time-grids with a small positive time, instead.'
                    print_op.base_print(message, level='warning')

            time_grids[grid] = nlp.time_grids[grid](safe_t_f)
            time_grids['ref'][grid] = nlp.time_grids[grid](self.__V_ref['theta', 't_f'])

        # set properties
        self.__outputs_opt = outputs_opt
        self.__outputs_init = outputs_init
        self.__outputs_ref = outputs_ref
        self.__global_outputs_opt = global_outputs_opt
        self.__integral_outputs_init = integral_outputs_init
        self.__integral_outputs_opt = integral_outputs_opt
        self.__integral_outputs_ref = integral_outputs_ref
        self.__integral_outputs_fun = nlp_integral_outputs_fun
        self.__time_grids = time_grids

        return None

    def process_solution(self, options, nlp, model, final_homotopy_step):

        awelogger.logger.info('')
        awelogger.logger.info('process the solution...')

        if final_homotopy_step == 'initial_guess':
            self.__V_opt = self.__V_init
        else:
            self.__V_opt = nlp.V(self.__solution['x'])

        self.__V_final_si = struct_op.scaled_to_si(self.__V_opt, model.scaling)
        self.__integral_outputs_final = self.scaled_to_si_integral_outputs(nlp, model)

        return None


    def scaled_to_si_integral_outputs(self, nlp, model):

        nk = nlp.n_k
        if nlp.discretization == 'direct_collocation':
            direct_collocation = True
            d = nlp.d
        elif nlp.discretization == 'multiple_shooting':
            direct_collocation = False

        integral_outputs_si = copy.deepcopy(self.__integral_outputs_opt)

        for ndx in range(nk+1):
            for name in list(model.integral_outputs.keys()):
                integral_outputs_si['int_out', ndx, name] = self.__integral_outputs_opt['int_out', ndx, name] * model.integral_scaling[name]

            if direct_collocation and (ndx < nk):
                for ddx in range(d):
                    for name in list(model.integral_outputs.keys()):
                        integral_outputs_si['coll_int_out', ndx, ddx, name] = self.__integral_outputs_opt['coll_int_out', ndx, ddx, name] * model.integral_scaling[name]

        return integral_outputs_si

    def print_optimization_info(self):

        awelogger.logger.info('')
        awelogger.logger.info('Solver options:')

        options_dict = {
            'NLP solver': self.__options['nlp_solver'],
            'Linear solver': self.__options['linear_solver'],
            'Final max. iterations': self.__options['max_iter'],
            'Homotopy max. iterations': self.__options['max_iter_hippo'],
            'Homotopy barrier param': self.__options['mu_hippo'],
            'Homotopy method': self.__options['homotopy_method']
        }

        if self.__options['homotopy_method'] == 'classic':
            options_dict['Homotopy step'] = self.__options['homotopy_step']

        print_op.print_dict_as_table(options_dict)

        return None

    def get_failure_step(self):
        homotopy_schedule_copy = copy.deepcopy(self.schedule['homotopy'])
        homotopy_schedule_reverse = homotopy_schedule_copy[::-1]
        for step_name in homotopy_schedule_reverse:
            if (step_name in self.iterations.keys()) and not self.solve_succeeded:
                return step_name

        return None

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, value):
        awelogger.logger.warning('Cannot set status object.')

    @property
    def solvers(self):
        return self.__solvers

    @solvers.setter
    def solvers(self, value):
        awelogger.logger.warning('Cannot set solvers object.')

    @property
    def V_opt(self):
        return self.__V_opt

    @V_opt.setter
    def V_opt(self, value):
        awelogger.logger.warning('Cannot set V_opt object.')

    @property
    def V_ref(self):
        return self.__V_ref

    @V_ref.setter
    def V_ref(self, value):
        awelogger.logger.warning('Cannot set V_ref object.')

    @property
    def V_final_si(self):
        return self.__V_final_si

    @V_final_si.setter
    def V_final_si(self, value):
        awelogger.logger.warning('Cannot set V_final_si object.')

    @property
    def V_init(self):
        return self.__V_init

    @V_init.setter
    def V_init(self, value):
        awelogger.logger.warning('Cannot set V_init object.')

    @property
    def V_bounds(self):
        return self.__V_bounds

    @V_bounds.setter
    def V_bounds(self, value):
        awelogger.logger.warning('Cannot set V_bounds object.')

    @property
    def p_fix_num(self):
        return self.__p_fix_num

    @p_fix_num.setter
    def p_fix_num(self, value):
        awelogger.logger.warning('Cannot set p_fix_num object.')

    @property
    def timings(self):
        return self.__timings

    @timings.setter
    def timings(self, value):
        awelogger.logger.warning('Cannot set timings object.')

    @property
    def cumulative_max_memory(self):
        return self.__cumulative_max_memory

    @cumulative_max_memory.setter
    def cumulative_max_memory(self, value):
        awelogger.logger.warning('Cannot set cumulative_max_memory object.')

    @property
    def arg(self):
        return self.__arg

    @arg.setter
    def arg(self, value):
        awelogger.logger.warning('Cannot set arg object.')

    @property
    def arg_initial(self):
        return self.__arg_initial

    @arg_initial.setter
    def arg_initial(self, value):
        awelogger.logger.warning('Cannot set arg_initial object.')

    @property
    def output_vals(self):
        return {'init': self.__outputs_init,
                'opt': self.__outputs_opt,
                'ref': self.__outputs_ref}

    @property
    def outputs_init(self):
        return self.__outputs_init

    @property
    def outputs_opt(self):
        return self.__outputs_opt

    @property
    def outputs_ref(self):
        return self.__outputs_ref

    @property
    def global_outputs_opt(self):
        return self.__global_outputs_opt

    @property
    def integral_output_vals(self):
        return {'init': self.__integral_outputs_init,
                'opt': self.__integral_outputs_opt,
                'ref': self.__integral_outputs_ref}

    @property
    def integral_outputs_init(self):
        return self.__integral_outputs_init

    @property
    def integral_outputs_opt(self):
        return self.__integral_outputs_opt

    @output_vals.setter
    def output_vals(self, value):
        awelogger.logger.warning('Cannot set output_vals object.')

    @property
    def integral_outputs_final(self):
        return self.__integral_outputs_final

    @integral_outputs_final.setter
    def integral_outputs_final(self, value):
        awelogger.logger.warning('Cannot set integral_outputs_final object.')

    @property
    def solve_succeeded(self):
        return self.__solve_succeeded

    @solve_succeeded.setter
    def solve_succeeded(self, value):
        awelogger.logger.warning('Cannot set solve_succeeded object.')

    @property
    def solution(self):
        return self.__solution

    @solution.setter
    def solution(self, value):
        awelogger.logger.warning('Cannot set solution object.')

    @property
    def stats(self):
        return self.__stats

    @stats.setter
    def stats(self, value):
        awelogger.logger.warning('Cannot set stats object.')

    @property
    def iterations(self):
        return self.__iterations

    @iterations.setter
    def iterations(self, value):
        awelogger.logger.warning('Cannot set iterations object.')

    @property
    def t_wall(self):
        return self.__t_wall

    @t_wall.setter
    def t_wall(self, value):
        awelogger.logger.warning('Cannot set t_wall object.')

    @property
    def return_status_numeric(self):
        return self.__return_status_numeric

    @return_status_numeric.setter
    def return_status_numeric(self, value):
        awelogger.logger.warning('Cannot set return_status_numeric object.')

    @property
    def schedule(self):
        return self.__schedule

    @schedule.setter
    def schedule(self, value):
        awelogger.logger.warning('Cannot set schedule object.')

    @property
    def time_grids(self):
        return self.__time_grids

    @time_grids.setter
    def time_grids(self, value):
        awelogger.logger.warning('Cannot set time_grids object.')

    @property
    def final_homotopy_step(self):
        return self.__final_homotopy_step

    @final_homotopy_step.setter
    def final_homotopy_step(self, value):
        awelogger.logger.warning('Cannot set final_homotopy_step object.')

    @property
    def integral_outputs_opt(self):
        return self.__integral_outputs_opt

    @integral_outputs_opt.setter
    def integral_outputs_opt(self, value):
        awelogger.logger.warning('Cannot set integral_outputs_opt object.')

    @property
    def awe_callback(self):
        return self.__awe_callback

    @awe_callback.setter
    def awe_callback(self, value):
        awelogger.logger.warning('Cannot set awe_callback object.')
