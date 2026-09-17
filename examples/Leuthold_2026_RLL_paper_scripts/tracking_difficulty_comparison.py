#!/usr/bin/python3
from platform import architecture

import matplotlib
# matplotlib.use('TkAgg')

import awebox as awe

import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import os

from datetime import date
import random

import awebox.trial as awe_trial
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

base_name = 'trackdiff'

def run(ratio_power_to_tracking=1., inputs={}):

    # basic options
    options = {}
    options = help_op.get_basic_options_for_convergence_expense_and_comparison(options)

    # allow a reduction of the problem for testing purposed
    if 'nlp.n_k' in inputs.keys():
        n_k = inputs['nlp.n_k']
    else:
        n_k = options['nlp.n_k']
    periods_tracked = inputs['periods_tracked']
    
    wake_nodes = help_op.from_periods_tracked_to_wake_nodes(n_k, periods_tracked)
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    if ('solver.hippo_strategy' in inputs.keys()) and (inputs['solver.hippo_strategy'] == False):
        inputs['solver.mu_hippo'] = 1e-2
        inputs['solver.hippo_strategy'] = False

    for name, val in inputs.items():
        if '.' in name:
            options[name] = inputs[name]

    # visualization
    options['visualization.cosmetics.save_figs'] = True
    options['visualization.cosmetics.save.format_list'] = ['pdf']
    options['visualization.cosmetics.animation.snapshot_index'] = -1
    options['visualization.cosmetics.trajectory.body_cross_sections_per_meter'] = 10 / options['user_options.kite_standard']['geometry']['b_ref']
    options['visualization.cosmetics.trajectory.wake_nodes'] = True
    options['visualization.cosmetics.trajectory.kite_aero_dcm'] = True
    options['visualization.cosmetics.trajectory.trajectory_rotation_dcm'] = True
    options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    options['visualization.cosmetics.trajectory.kite_bodies'] = True
    options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '--'  
    options['visualization.cosmetics.trajectory.temporal_epigraph_length_to_span'] = 5.
    
    options['model.aero.vortex.induction_factor_normalizing_speed'] = 'u_ref'  
    options['model.aero.actuator.normal_vector_model'] = 'dual'
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.32, 'switch', 1.0] 

    ######## baseline OCP - find a reference trajectory

    options = help_op.toggle_baseline_options(options)

    # build trial and optimize
    trial_name_baseline = help_op.build_unique_trial_name(base_name, inputs) + 'phi' + str(ratio_power_to_tracking)
        	
    trial_baseline = awe_trial.Trial(options, trial_name_baseline)
    trial_baseline.build()
    trial_baseline.optimize(final_homotopy_step='final')
    latex_dict = help_op.get_latex_dict() 
    trial_baseline.make_report(to_echo_or_latex='latex', latex_dict=latex_dict, save=True)
    trial_baseline.print_cost_information()
    help_op.save_results_including_figures(trial_baseline, options)


    if trial_baseline.optimization.solve_succeeded:
        ######## simulation OCP - simulate the RLL model on the reference trajectory

        options = help_op.toggle_tracking_options(options)
        options = help_op.adjust_weights_for_tracking(trial_baseline, options, ratio_power_to_tracking=ratio_power_to_tracking)
        options = help_op.fix_params_to_baseline(trial_baseline, options)

        ## the commented out lines here were useful when tuning the weights of the problem
        #options['user_options.induction_model'] = 'not_in_use'
        #final_homotopy_step = 'initial'
        final_homotopy_step = 'induction'

        # build trial and optimize
        trial_name_vortex = trial_name_baseline + '_vortex'
        trial_vortex = awe_trial.Trial(options, trial_name_vortex)
        trial_vortex.build()

        warmstart_and_reference = help_op.construct_vortex_initial_guess(trial_baseline, trial_vortex, inequalities_are_off=False)
        trial_vortex.optimize(final_homotopy_step=final_homotopy_step, warmstart_file=warmstart_and_reference, reference_file=warmstart_and_reference)

        trial_vortex.print_cost_information()

        help_op.save_results_including_figures(trial_vortex, options)

    return None

   
def call_by_pt(n_k, pt, ratio_power_to_tracking=1., inputs={}):
    import gc
    from glob import glob
    print('n_k: ' + str(n_k) + '; pt: ' + str(pt))

    if pt > -1e-10:
        inputs['nlp.n_k'] = n_k
        inputs['periods_tracked'] = pt

        trial_name = help_op.build_unique_trial_name(base_name, inputs) + 'phi' + str(ratio_power_to_tracking)
        if not glob('*' + trial_name + '*'):
            trial = run(ratio_power_to_tracking=ratio_power_to_tracking, inputs=inputs)
            del trial
        gc.collect()
        
    return None

if __name__ == "__main__":
    #1., 1e-1, 1e-2
    n_k = 20
    pt = 1.5
    rpt_list = [1e10] #[1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
    for ratio_power_to_tracking in rpt_list:
        call_by_pt(n_k, pt, ratio_power_to_tracking=ratio_power_to_tracking)
