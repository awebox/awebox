#!/usr/bin/python3
from platform import architecture

import matplotlib
matplotlib.use('TkAgg')

import awebox as awe

import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import os

from datetime import date
import random

import awebox.trial as awe_trial
import haas2019_settings as haas2019_settings

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

import awebox.viz.wake as wake_viz
import awebox.opti.initialization_dir.initialization as initialization
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.initialization as alg_initialization

import helpful_operations as help_op


from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

awelogger.logger.setLevel(10)


# there's still a large amount of 'apathy' in the azimuthal orientation of the trajectory, which will hopefully be resolved in the future. 
# as is, you may need to run this script (to the end of the baseline OCP section) three-four times before you get the solution shared in the paper.


def run(inputs={}):

    base_name = 'haas2019'

    # basic options
    options = {}

    # allow a reduction of the problem for testing purposed

    # base problem definition
    options['user_options.system_model.architecture'] = {1: 0}
    options = haas2019_settings.set_settings(options, inputs)
    
    windings = options['user_options.trajectory.lift_mode.windings']
    periods_tracked = 2.

    n_k = options['nlp.n_k']
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    options['model.aero.vortex.far_wake_element_type'] = 'semi_infinite_filament'
    options['model.aero.vortex.representation'] = 'alg'
    options['model.aero.vortex.convection_type'] = 'rigid'

    options['solver.max_iter'] = 2e6
    options['solver.max_iter_hippo'] = options['solver.max_iter']
    options['solver.max_cpu_time'] = 1e6 * 60. * 60. #put an unreasonably high value here, so that the problem won't accidentally get killed half-ways

    # visualization
    options['visualization.cosmetics.save_figs'] = True
    options['visualization.cosmetics.save.format_list'] = ['pdf']
    options['visualization.cosmetics.trajectory.body_cross_sections_per_meter'] = 10 / options['user_options.kite_standard']['geometry']['b_ref']
    options['visualization.cosmetics.trajectory.wake_nodes'] = True
    options['visualization.cosmetics.trajectory.kite_aero_dcm'] = True
    options['visualization.cosmetics.trajectory.trajectory_rotation_dcm'] = True
    options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    options['visualization.cosmetics.trajectory.kite_bodies'] = True
    options['visualization.cosmetics.plot_ref'] = False
    options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '--'
    options['visualization.cosmetics.trajectory.temporal_epigraph_length_to_span'] = 1.
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.475] # this is the best-value we've found for the black dot on Haas2019's virtual wind tunnel position

    options['model.aero.actuator.normal_vector_model'] = 'xhat'
    options['model.aero.vortex.induction_factor_normalizing_speed'] = 'u_ref'


    ######## baseline OCP - find a reference trajectory

    options = help_op.toggle_baseline_options(options)

    # build trial and optimize
    trial_name_baseline = help_op.build_unique_trial_name(base_name, inputs)
        	
    trial_baseline = awe_trial.Trial(options, trial_name_baseline)
    trial_baseline.build()
    trial_baseline.optimize(final_homotopy_step='final')

    trial_baseline.print_cost_information()
    help_op.save_results_including_figures(trial_baseline, options)


    if trial_baseline.optimization.solve_succeeded:
        ######## simulation OCP - simulate the RLL model on the reference trajectory

        options = help_op.toggle_vortex_options(options)

        options = help_op.turn_off_inequalities_except_time(options)

        #adjust_weights_for_tracking
        options['solver.cost.tracking.0'] = 1e1 #1
        options['solver.cost.t_f.0'] = 1e2
        options['solver.cost.fictitious.0'] = 1.e-10
        options['solver.cost.u_regularisation.0'] = 1.e-1
        options['solver.cost.xdot_regularisation.0'] = 1.e-2
        options['solver.weights.vortex'] = 0.
        options['solver.weights.q'] = 1e1
        options['solver.weights.dq'] = 1e1
        options['solver.weights.coeff'] = 1e1

        options = help_op.fix_params_to_baseline(trial_baseline, options)

        ## the commented out lines here were useful when tuning the weights of the problem
        #options['user_options.induction_model'] = 'not_in_use'
        #final_homotopy_step = 'initial'
        final_homotopy_step = 'induction'

        # build trial and optimize
        trial_name_vortex = trial_name_baseline + '_vortex'
        trial_vortex = awe_trial.Trial(options, trial_name_vortex)
        trial_vortex.build()

        warmstart_and_reference = help_op.construct_vortex_initial_guess(trial_baseline, trial_vortex, inequalities_are_off=True)
        trial_vortex.optimize(final_homotopy_step=final_homotopy_step, warmstart_file=warmstart_and_reference, reference_file=warmstart_and_reference)

        trial_vortex.print_cost_information()

        help_op.save_results_including_figures(trial_vortex, options)

    return None




if __name__ == "__main__":

    inputs = {}
    #inputs['nlp.n_k'] = 10, # when testing

    trial = run(inputs)
