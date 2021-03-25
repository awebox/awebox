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

from . import scheduling

from . import preparation

from . import diagnostics

import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op
import awebox.tools.callback as callback

from numpy import linspace

import matplotlib.pyplot as plt

import copy

from awebox.logger.logger import Logger as awelogger

import time

class Optimization(object):
    def __init__(self):
        self.__status = 'Optimization not yet built.'
        self.__V_opt = None
        self.__timings = {}
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

        awelogger.logger.info('Building optimization...')

        self.__name = name

        if self.__status == 'I am an optimization.':
            awelogger.logger.info('Optimization already built.')
            return None
        elif nlp.status == 'I am an NLP.':

            timer = time.time()

            # prepare callback
            self.__awe_callback = self.initialize_callback('awebox_callback', nlp, model, options)

            # generate solvers
            self.generate_solvers(model, nlp, formulation, options, self.__awe_callback)

            # record set-up time
            self.__timings['setup'] = time.time() - timer

            self.__status = 'I am an optimization.'
            awelogger.logger.info('Optimization built.')
            awelogger.logger.info('Optimization construction time: %s', print_op.print_single_timing(self.__timings['setup']))
            awelogger.logger.info('')

        else:
            raise ValueError('Cannot build optimization without building NLP.')

        return None

    def solve(self, options, nlp, model, formulation, visualization,
              final_homotopy_step='final', warmstart_file = None, vortex_linearization_file = None, debug_flags =
              [], debug_locations = []):

        self.__debug_flags = debug_flags
        if debug_flags != [] and debug_locations == []:
            self.__debug_locations = 'all'
        else:
            self.__debug_locations = debug_locations

        if self.__status in ['I am an optimization.','I am a solved optimization.', 'I am a failed optimization.']:
            awelogger.logger.info('Solving optimization...')

            # save final homotopy step
            self.__final_homotopy_step = final_homotopy_step

            # reset timings / iteration counters
            self.reset_timings_and_counters()

           # schedule the homotopy steps
            self.define_homotopy_update_schedule(model, formulation, nlp, options['cost'])

            # prepare problem
            self.define_standard_args(nlp, formulation, model, options, visualization)

            # restart the counter through the homotopy steps
            self.define_update_counter(nlp, formulation, model)

            # classifications
            use_warmstart = not (warmstart_file == None)
            use_vortex_linearization = 'lin' in model.parameters_dict.keys()
            make_steps = not (final_homotopy_step == 'initial_guess')

            # solve the problem
            if make_steps:
                if use_warmstart:
                    self.solve_from_warmstart(nlp, formulation, model, options, warmstart_file, final_homotopy_step, visualization)
                else:
                    if use_vortex_linearization:
                        self.solve_with_vortex_linearization(nlp, model, formulation, options, vortex_linearization_file, final_homotopy_step, visualization)
                    else:
                        self.solve_homotopy(nlp, model, options, final_homotopy_step, visualization)

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

        self.__timings['optimization'] = 0.
        self.__iterations['optimization'] = 0.
        self.__t_wall['optimization'] = 0.
        self.__return_status_numeric['optimization'] = 17

        for step in self.__timings.keys():
            if not (step == 'setup'):
                self.__timings[step] = 0.

        for step in self.__iterations.keys():
            if not (step == 'setup'):
                self.__iterations[step] = 0.
                self.__t_wall[step] = 0.

        for step in self.__return_status_numeric.keys():
            if not (step == 'setup'):
                self.__return_status_numeric[step] = 17

        self.__awe_callback.reset()

        return None


    ### interactive functions

    def __make_debug_plot(self, V_plot, nlp, visualization, location):

        if location == 'initial_guess':
            self.generate_outputs(nlp, {'x': self.__V_init})
        fig_name = 'debug_plot_' + location
        sweep_toggle = False
        cost_fun = nlp.cost_components[0]
        cost = struct_op.evaluate_cost_dict(cost_fun, V_plot, self.__p_fix_num)
        V_ref = self.__V_ref
        visualization.plot(V_plot, visualization.options, [self.__outputs_init,
                                                           self.__outputs_opt, self.__outputs_ref],
                           self.__integral_outputs_opt, self.__debug_flags, self.__time_grids, cost, self.__name, sweep_toggle, V_ref, fig_name=fig_name)

        return None

    def update_runtime_info(self, timer, step_name):

        self.__timings[step_name] = time.time() - timer

        self.__return_status_numeric[step_name] = struct_op.convert_return_status_string_to_number(
            self.stats['return_status'])

        if step_name not in list(self.__iterations.keys()):
            self.__iterations[step_name] = 0.

        self.__iterations['optimization'] = self.__iterations['optimization'] + self.__iterations[step_name]
        self.__t_wall['optimization'] = self.__t_wall['optimization'] + self.__t_wall[step_name]
        self.__return_status_numeric['optimization'] = self.__return_status_numeric[step_name]
        self.__timings['optimization'] = self.__timings['optimization'] + self.__timings[step_name]

    def initialize_callback(self, name, nlp, model, options):

        awelogger.logger.info('initialize callback...')

        V = nlp.V
        P = nlp.P

        nx = V.cat.shape[0]
        ng = nlp.g.shape[0]
        np = P.cat.shape[0]
        awe_callback = callback.awebox_callback(name, model, nlp, options, V, P, nx, ng, np)

        return awe_callback




    ### solvers

    def generate_solvers(self, model, nlp, formulation, options, awe_callback):

        awelogger.logger.info('generate solvers...')

        self.__solvers = preparation.generate_solvers(awe_callback, model, nlp, formulation, options)

        return None


    def solve_from_warmstart(self, nlp, formulation, model, options, warmstart_file, final_homotopy_step, visualization):

        awelogger.logger.info('solve from warmstart...')
        awelogger.logger.info('')

        self.__solve_succeeded = True

        warmstart_solution_dict = save_op.extract_solution_dict_from_file(warmstart_file)
        self.modify_args_for_warmstart(nlp, formulation, model, options, visualization, warmstart_solution_dict = warmstart_solution_dict)
        self.modify_schedule_for_warmstart(final_homotopy_step, warmstart_solution_dict, nlp, model)

        # solve homotopy with warmstart
        self.solve_homotopy(nlp, model, options, final_homotopy_step, visualization)

        awelogger.logger.info(print_op.hline('#'))

        return None

    def solve_with_vortex_linearization(self, nlp, model, formulation, options, vortex_linearization_file, final_homotopy_step, visualization):

        if vortex_linearization_file == None:
            self.solve_with_vortex_linearization_setup(nlp, model, options, final_homotopy_step, visualization)
        else:
            self.solve_with_vortex_linearization_iterative(nlp, formulation, model, options, vortex_linearization_file,
                                                      final_homotopy_step, visualization)

        return None

    def solve_with_vortex_linearization_setup(self, nlp, model, options, final_homotopy_step, visualization):

        awelogger.logger.info('solve set-up problem with vortex linearization...')
        awelogger.logger.info('')

        self.__solve_succeeded = True

        # solve set-up problem with homotopy (omitting the induction steps)
        self.solve_homotopy(nlp, model, options, final_homotopy_step, visualization)

        awelogger.logger.info(print_op.hline('#'))

        return None

    def solve_with_vortex_linearization_iterative(self, nlp, formulation, model, options, vortex_linearization_file, final_homotopy_step, visualization):

        awelogger.logger.info('solve iterative problem with vortex linearization...')
        awelogger.logger.info('')

        self.__solve_succeeded = True

        warmstart_solution_dict = save_op.extract_solution_dict_from_file(vortex_linearization_file)
        self.modify_args_for_warmstart(nlp, formulation, model, options, visualization, warmstart_solution_dict=warmstart_solution_dict)
        self.modify_schedule_for_vortex_linearization_iterative(final_homotopy_step, nlp, model)

        # solve homotopy with warmstart
        self.solve_homotopy(nlp, model, options, final_homotopy_step, visualization)

        awelogger.logger.info(print_op.hline('#'))

        return None

    def solve_homotopy(self, nlp, model, options, final_homotopy_step, visualization):

        awelogger.logger.info('solve with homotopy procedure...')
        awelogger.logger.info('')

        # do not consider homotopy steps after specified final_homotopy_step
        final_index = self.__schedule['homotopy'].index(final_homotopy_step)
        homotopy_schedule = self.__schedule['homotopy'][:final_index+1]

        self.__solve_succeeded = True

        # iterate over homotopy schedule
        for step_name in homotopy_schedule:

            if self.__solve_succeeded:

                timer = time.time()
                self.solve_specific_homotopy_step(step_name, final_homotopy_step, nlp, model, options, visualization)
                self.update_runtime_info(timer, step_name)

        awelogger.logger.info(print_op.hline('#'))

        return None

    def solve_specific_homotopy_step(self, step_name, final_homotopy_step, nlp, model, options, visualization):

        initial_solver = self.__solvers['initial']
        middle_solver = self.__solvers['middle']
        final_solver = self.__solvers['final']

        if step_name == 'initial':
            self.solve_general_homotopy_step(step_name, final_homotopy_step, 0, options, nlp, model, initial_solver, visualization)

        elif step_name == 'final':
            self.solve_general_homotopy_step(step_name, final_homotopy_step, 0, options, nlp, model, final_solver, visualization)

        else:
            number_of_steps = len(list(self.__schedule['bounds_to_update'][step_name].keys()))
            for homotopy_part in range(number_of_steps):
                self.solve_general_homotopy_step(step_name, final_homotopy_step, homotopy_part, options, nlp, model, middle_solver, visualization)

        return None

    def solve_general_homotopy_step(self, step_name, final_homotopy_step, counter, options, nlp, model, solver, visualization):

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
            phi_name = scheduling.find_current_homotopy_parameter(model.parameters_dict['phi'], self.__V_bounds)

            # solve
            if options['homotopy_method'] == 'classic' and (counter == 0) and (phi_name != None):
                
                self.__perform_classic_continuation(step_name, phi_name, options, solver)

            else:

                self.__solution = solver(**self.__arg)
                self.__stats = solver.stats()
                self.__save_stats(step_name)

            self.generate_outputs(nlp, self.__solution)

            self.allow_next_homotopy_step()

            diagnostics.print_runtime_values(self.__stats)
            diagnostics.print_homotopy_values(nlp, self.__solution, self.__p_fix_num)
            diagnostics.health_check(step_name, final_homotopy_step, nlp, self.__solution, self.__arg, options, self.__solve_succeeded, self.__stats, self.__iterations)

            if step_name in self.__debug_locations or self.__debug_locations == 'all':
                V_plot = nlp.V(self.__solution['x'])
                self.__make_debug_plot(V_plot, nlp, visualization, step_name)

        return None


    def __perform_classic_continuation(self, step_name, phi_name, options, solver):

        # define parameter path
        step = options['homotopy_step']
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
            self.__t_wall[step_name] = 0.
        self.__iterations[step_name] += self.__stats['iter_count']
        self.__t_wall[step_name] += self.__stats['t_wall_total']
        if 't_wall_callback_fun' in self.__stats.keys():
            self.__t_wall[step_name] -= self.__stats['t_wall_callback_fun']

        return None


    ### arguments

    def define_standard_args(self, nlp, formulation, model, options, visualization, warmstart_solution_dict = None):

        awelogger.logger.info('define args...')

        self.__arg = preparation.initialize_arg(nlp, formulation, model, options, warmstart_solution_dict = warmstart_solution_dict)
        self.__arg_initial = {}
        self.__arg_initial['x0'] = nlp.V(self.__arg['x0'])

        self.__V_init = nlp.V(self.__arg['x0'])

        self.__p_fix_num = nlp.P(self.__arg['p'])

        self.__V_ref = nlp.V(self.__p_fix_num['p','ref'])

        if 'initial_guess' in self.__debug_locations or self.__debug_locations == 'all':
            self.__make_debug_plot(self.__V_init, nlp, visualization, 'initial_guess')

        self.__g_bounds = {}
        self.__g_bounds['lb'] = self.__arg['lbg']
        self.__g_bounds['ub'] = self.__arg['ubg']

        self.__V_bounds = {}
        self.__V_bounds['lb'] = self.__arg['lbx']
        self.__V_bounds['ub'] = self.__arg['ubx']

        return None

    def modify_args_for_warmstart(self, nlp, formulation, model, options, visualization, warmstart_solution_dict):

        awelogger.logger.info('modify args for warmstart...')

        use_vortex_linearization = 'lin' in nlp.P.keys()

        # set up warmstart
        [V_init_proposed,
        lam_x_proposed,
        lam_g_proposed] = struct_op.setup_warmstart_data(nlp, warmstart_solution_dict)

        V_shape_matches = (V_init_proposed.cat.shape == nlp.V.cat.shape)
        if V_shape_matches:
            self.__V_init = V_init_proposed
            self.__arg['x0'] = self.__V_init.cat

            if use_vortex_linearization:
                self.__p_fix_num['lin'] = V_init_proposed

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

    def define_homotopy_update_schedule(self, model, formulation, nlp, cost_options):
        awelogger.logger.info('define homotopy update schedule...')
        self.__schedule = scheduling.define_homotopy_update_schedule(model, formulation, nlp, cost_options)
        return None

    def modify_schedule_for_warmstart(self, final_homotopy_step, warmstart_solution_dict, nlp, model):

        awelogger.logger.info('modify schedule for warmstart...')

        # final homotopy step of warmstart file
        warmstart_step = warmstart_solution_dict['final_homotopy_step']
        initial_index = self.__schedule['homotopy'].index(warmstart_step)

        # check if schedule is still consistent
        final_index = self.__schedule['homotopy'].index(final_homotopy_step)
        if final_index < initial_index:
            raise ValueError('Final homotopy step has a lower schedule index than specified initial (warmstart) step')

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

    def modify_schedule_for_vortex_linearization_iterative(self, final_homotopy_step, nlp, model):

        awelogger.logger.info('modify schedule for vortex linearization iterative problem...')

        # starting homotopy step for iterative problem
        initial_step = 'final'
        initial_index = self.__schedule['homotopy'].index(initial_step)

        # check if schedule is still consistent
        final_index = self.__schedule['homotopy'].index(final_homotopy_step)
        if final_index < initial_index:
            raise ValueError('Final homotopy step has a lower schedule index than specified initial (warmstart) step')

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

        return_status_number = struct_op.convert_return_status_string_to_number(self.__stats['return_status'])

        # check if optimization was successful
        if return_status_number > 3:

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
        V_final = nlp.V(solution['x'])

        # get outputs from V
        self.__generate_outputs_from_V(nlp, V_final)

        return None

    def __generate_outputs_from_V(self, nlp, V_final):

        # V_initial = nlp.V(self.__arg['x0']) # todo: needed elsewhere?
        V_initial = self.__V_init

        # general outputs
        [nlp_outputs, nlp_output_fun] = nlp.output_components
        outputs_init = nlp_outputs(nlp_output_fun(V_initial, self.__p_fix_num))
        outputs_opt = nlp_outputs(nlp_output_fun(V_final, self.__p_fix_num))
        outputs_ref = nlp_outputs(nlp_output_fun(self.__V_ref, self.__p_fix_num))

        # integral outputs
        [nlp_integral_outputs, nlp_integral_outputs_fun] = nlp.integral_output_components
        integral_outputs_init = nlp_integral_outputs(nlp_integral_outputs_fun(V_initial, self.__p_fix_num))
        integral_outputs_opt = nlp_integral_outputs(nlp_integral_outputs_fun(V_final, self.__p_fix_num))

        # time grids
        time_grids = {'ref':{}}
        for grid in nlp.time_grids:
            time_grids[grid] = nlp.time_grids[grid](V_final['theta','t_f'])
            time_grids['ref'][grid] = nlp.time_grids[grid](self.__V_ref['theta','t_f'])

        # set properties
        self.__outputs_opt = outputs_opt
        self.__outputs_init = outputs_init
        self.__outputs_ref = outputs_ref
        self.__integral_outputs_init = integral_outputs_init
        self.__integral_outputs_opt = integral_outputs_opt
        self.__integral_outputs_fun = nlp_integral_outputs_fun
        self.__time_grids = time_grids


    def process_solution(self, options, nlp, model, final_homotopy_step):

        awelogger.logger.info('')
        awelogger.logger.info('process the solution...')

        if final_homotopy_step == 'initial_guess':
            self.__V_opt = self.__V_init
        else:
            self.__V_opt = nlp.V(self.__solution['x'])
        self.__V_final = struct_op.scaled_to_si(self.__V_opt, model.scaling)
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

        for k in range(nk+1):
            for name in list(model.integral_outputs.keys()):
                integral_outputs_si['int_out',k,name] = self.__integral_outputs_opt['int_out',k,name]*model.integral_scaling[name]

            if direct_collocation and (k < nk):
                for j in range(d):
                    for name in list(model.integral_outputs.keys()):
                        integral_outputs_si['coll_int_out',k,j,name] = self.__integral_outputs_opt['coll_int_out',k,j,name]*model.integral_scaling[name]

        return integral_outputs_si

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
    def V_final(self):
        return self.__V_final

    @V_final.setter
    def V_final(self, value):
        awelogger.logger.warning('Cannot set V_final object.')

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
        return [self.__outputs_init, self.__outputs_opt, self.__outputs_ref]

    @property
    def integral_output_vals(self):
        return [self.__integral_outputs_init, self.__integral_outputs_opt]

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
