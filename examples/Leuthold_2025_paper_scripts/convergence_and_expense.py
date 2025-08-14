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
import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

import awebox.viz.wake as wake_viz
import awebox.opti.initialization_dir.initialization as initialization


import helpful_operations as help_op

from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

awelogger.logger.setLevel(10)


def run(inputs={}):

    n_k = inputs['n_k']
    periods_tracked = inputs['periods_tracked']

    base_name = 'convergence'
    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))

    # basic options
    options = {}

    # allow a reduction of the problem for testing purposed
    options['nlp.n_k'] = n_k
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    # base problem definition
    options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    options['user_options.system_model.kite_dof'] = 6
    options['model.system_bounds.theta.t_f'] = [5., 35.]  # [s]
    options['user_options.trajectory.lift_mode.windings'] = 1

    options['user_options.induction_model'] = 'vortex'
    options['model.aero.vortex.far_wake_element_type'] = 'semi_infinite_filament'
    options['model.aero.vortex.representation'] = 'alg'
    options['model.aero.vortex.wake_nodes'] = wake_nodes
    options['model.aero.vortex.convection_type'] = 'rigid'
    options['model.aero.vortex.core_to_chord_ratio'] = 0.05

    options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.tether_force.include'] = False
    options['user_options.trajectory.fixed_params'] = {}

    options['nlp.n_k'] = n_k
    
    options['solver.linear_solver'] = 'ma86'
    options['solver.hippo_strategy'] = True
    options['solver.homotopy_method.advance_despite_max_iter'] = False
    options['solver.raise_error_at_max_time'] = True
    options['solver.max_iter'] = 2e4
    options['solver.max_iter_hippo'] = options['solver.max_iter']
    options['solver.max_cpu_time'] = 1e10 * 60. * 60. # set the max cpu time high enough that it won't kill the large problems

    # these turn out not to be the best-performing scaling options (see awebox opts/default for the latest recommended scaling options), but they were the ones used when producing the convergence and expense plots, and therefore also problem A1 of the results comparison. changing the scaling to the performance-improving settings changes the results a small amount, but (so far) not the the conclusions.
    options['model.scaling.other.position_scaling_method'] = 'altitude_and_radius'
    options['model.scaling.other.force_scaling_method'] = 'synthesized'
    options['model.scaling.other.flight_radius_estimate'] = 'synthesized'
    options['model.scaling.other.tension_estimate'] = 'synthesized'
    options['model.aero.vortex.position_scaling_method'] = 'average'
    options['model.aero.vortex.wu_ind_scaling_method'] = 'ref_betz'

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
    options['model.aero.actuator.normal_vector_model'] = 'xhat'
    options['visualization.cosmetics.temporal_epigraph_locations'] = ['switch', 1.0] 

    # use these options only in final comparison. the convergence plots were made with the xhat normal_vector
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.3, 'switch']
    options['model.aero.actuator.normal_vector_model'] = 'dual'
    
    
    # build trial and optimize
    trial_name_vortex = help_op.build_unique_trial_name(base_name, inputs)
    trial_vortex = awe_trial.Trial(options, trial_name_vortex)
    trial_vortex.build()

    trial_vortex.optimize(final_homotopy_step='final')

    trial_vortex.print_cost_information()
    help_op.save_results_including_figures(trial_vortex, options)

    return None


if __name__ == "__main__":

    inputs = {}
    inputs['n_k'] = 30
    inputs['periods_tracked'] = 1.5

    trial = run(inputs)



