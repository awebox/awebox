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
import c5a_data as c5a_data

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.tools.save_operations as save_op

import helpful_operations as help_op

import awebox.viz.wake as wake_viz
import awebox.opti.initialization_dir.initialization as initialization

from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas

awelogger.logger.setLevel(10)


def run(inputs={}):

    base_name = 'haas2017'
    
    n_k = inputs['n_k']
    periods_tracked = inputs['periods_tracked']

    wake_nodes = int(np.ceil(n_k * periods_tracked + 1))

    # basic options
    options = {}

    # allow a reduction of the problem for testing purposed
    options['nlp.n_k'] = n_k
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    # cut-off
    options['solver.max_iter'] = 20000
    options['solver.max_cpu_time'] = 30 * 60. * 60.

    # base problem definition
    options['user_options.system_model.architecture'] = {1: 0, 2: 1, 3: 1, 4: 1}
    options['user_options.kite_standard'] = c5a_data.data_dict()
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.tether_drag_model'] = 'not_in_use'

    # prescribe axisymmetric conditions
    options['user_options.wind.model'] = 'uniform'
    options['user_options.atmosphere'] = 'uniform'
    options['model.system_bounds.x.q'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
    options['params.atmosphere.g'] = 0.
    options['solver.initialization.inclination_deg'] = 0.
    options['quality.test_param.z_min'] = -cas.inf
    options['model.aero.actuator.normal_vector_model'] = 'xhat'

    options['model.scaling.other.position_scaling_method'] = 'radius_and_tether'
    options['model.scaling.other.force_scaling_method'] = 'centripetal'
    options['model.scaling.other.flight_radius_estimate'] = 'cone'
    options['model.scaling.other.tension_estimate'] = 'force_summation'

    # simplify the problem
    options = help_op.turn_off_inequalities_except_time(options)

    # describe any details of the tested wake model
    options['user_options.induction_model'] = 'vortex'
    options['model.aero.vortex.far_wake_element_type'] = 'semi_infinite_filament'
    options['model.aero.vortex.representation'] = 'alg'
    options['model.aero.vortex.convection_type'] = 'rigid'
    options['model.aero.vortex.core_to_chord_ratio'] = 0.05

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

    options['quality.test_param.vortex_truncation_error_thresh'] = 1e20

    options['model.aero.actuator.normal_vector_model'] = 'xhat'
    options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '-'
    options['visualization.cosmetics.temporal_epigraph_locations'] = [1.]
    options['model.aero.vortex.ratio_circulation_max_estimate_to_scaling_estimate'] = 2.
    options['visualization.cosmetics.trajectory.temporal_epigraph_length_to_span'] = 0.


    ## specify the problem inputs
    #######################################################
    options['solver.initialization.clockwise_rotation_about_xhat'] = False
    options['solver.initialization.check_rotational_axes.perform_check'] = True
    options['solver.initialization.kite_dcm'] = 'circular'

    tangential_force = 1.15031e5
    radial_force = 0.
    normal_force = 1.24617e6
    options['model.aero.overwrite.f_aero_rot'] = cas.DM([radial_force, tangential_force, normal_force])

    wingspan = options['user_options.kite_standard']['geometry']['b_ref']
    radius = 155.77
    l_s = 400.
    u_infty = 10.
    kite_speed_ratio = 7.
    psi0_deg = -40.

    cone_rad = cas.arcsin(radius / l_s)
    cone_deg = cone_rad * 180. / np.pi
    omega = kite_speed_ratio * u_infty / radius
    groundspeed = omega * radius
    t_f = (2. * np.pi * radius) / groundspeed
    options['solver.initialization.theta.l_s'] = l_s
    options['solver.initialization.l_t'] = l_s
    options['solver.initialization.cone_deg'] = cone_deg
    options['user_options.wind.u_ref'] = u_infty
    options['solver.initialization.groundspeed'] = groundspeed
    options['solver.initialization.psi0_rad'] = psi0_deg * np.pi / 180. 

    options['user_options.trajectory.fixed_params'] = {'t_f': t_f, 'l_s': l_s, 'diam_t': 5e-3, 'diam_s': 5e-3}

    # make sure you don't accidentally clip away the initialization to fit the inequalities
    options['solver.initialization.init_clipping'] = False

    # keep the semi-infinite part of the wake from overwhelming the plotting
    options['model.aero.vortex.far_wake_convection_time'] = t_f * 2

    # tracking
    options['solver.weights.q'] = 100.
    options['solver.weights.dq'] = 1.
    options['solver.weights.r'] = 10.
    options['solver.weights.omega'] = 1.
    options['solver.cost.tracking.0'] = 1e2 #1
    options['solver.cost.fictitious.0'] = 1.e-15
    # options['solver.cost.beta.0'] = 1.e-8
    options['solver.cost.u_regularisation.0'] = 1.e-8
    options['solver.cost.xdot_regularisation.0'] = 1.e-8
    # options['solver.weights.vortex'] = 1.e-8

    options = help_op.toggle_vortex_options(options)
    
    # build trial and optimize
    trial_name = help_op.build_unique_trial_name(base_name, inputs)
    trial = awe_trial.Trial(options, trial_name)
    trial.build()
    trial.optimize(final_homotopy_step='induction')
    
    trial.print_cost_information()
    
    help_op.save_results_including_figures(trial, options)

    # double-check that the tracking worked
    report = {}
    criteria = make_comparison(trial)
    for name_1, val_1 in criteria.items():
        for name_2, val_2 in val_1.items():
            report['criteria_' + name_1 + '_' + name_2] = val_2
    print_op.print_dict_as_table(report, level='info')
    filename = "comparison_" + trial.name + ".csv"
    save_op.write_or_append_two_column_dict_to_csv(report, filename)

    return None




def make_comparison(trial):

    radius_target = 155.77
    tsr = 7
    u_infty = 10.
    omega_target = tsr * u_infty / radius_target
    period_target = 2. * np.pi / omega_target

    plot_dict = trial.visualization.plot_dict

    criteria = {'return_status': {},
                'average_radius_m': {},
                'time_period_s': {},
                'norm_omega_radps': {},
                'stdev_omega_radps': {},
                'ehat_omega_up': {},
                'stdev_ehat_omega_up': {}
                }

    criteria['return_status']['found'] = trial.return_status_numeric
    criteria['return_status']['expected'] = 1
    criteria['return_status']['tol'] = 0.25

    for test_avg_radius_name in ['average_radius1', 'average_radius0']:
        if test_avg_radius_name in plot_dict['interpolation_si']['outputs']['geometry'].keys():
            criteria['average_radius_m']['found'] = np.mean(
                np.array(plot_dict['interpolation_si']['outputs']['geometry'][test_avg_radius_name]))
    criteria['average_radius_m']['expected'] = radius_target
    criteria['average_radius_m']['tol'] = 0.05

    criteria['time_period_s']['found'] = trial.optimization.global_outputs_opt['time_period'].full()[0][0]
    criteria['time_period_s']['expected'] = period_target
    criteria['time_period_s']['tol'] = 0.05

    norm_omega = []
    ehat_omega_up = []
    for test_omega_name in ['omega10', 'omega21', 'omega31', 'omega41']:
        if test_omega_name in plot_dict['interpolation_si']['x'].keys():
            interp_omega = plot_dict['interpolation_si']['x'][test_omega_name]
            for jdx in range(len(interp_omega)):
                local_omega = []
                for dim in range(3):
                    local_omega = cas.vertcat(local_omega, interp_omega[dim][jdx])
                norm_omega = cas.vertcat(norm_omega, vect_op.norm(local_omega))
                ehat_omega_up = cas.vertcat(ehat_omega_up, vect_op.normalize(local_omega)[2])
    average_omega = np.mean(np.array(norm_omega))
    criteria['norm_omega_radps']['found'] = average_omega
    criteria['norm_omega_radps']['expected'] = omega_target
    criteria['norm_omega_radps']['tol'] = 0.5

    stdev_omega = np.std(np.array(norm_omega))
    criteria['stdev_omega_radps']['found'] = stdev_omega
    criteria['stdev_omega_radps']['expected'] = omega_target
    criteria['stdev_omega_radps']['tol'] = 1e8

    average_ehat_omega_up = np.mean(np.array(ehat_omega_up))
    criteria['ehat_omega_up']['found'] = average_ehat_omega_up
    criteria['ehat_omega_up']['expected'] = -1
    criteria['ehat_omega_up']['tol'] = 1e8

    stdev_omega = np.std(np.array(norm_omega))
    criteria['stdev_ehat_omega_up']['found'] = stdev_omega
    criteria['stdev_ehat_omega_up']['expected'] = 1.
    criteria['stdev_ehat_omega_up']['tol'] = 1e8

    for name, val in criteria.items():
        criteria[name]['error'] = (val['found'] - val['expected']) / val['expected']

    print_op.print_dict_as_table(criteria, level='info')

    return criteria



if __name__ == "__main__":

    inputs = {}
    inputs['n_k'] = 30
    inputs['periods_tracked'] = 2

    trial = run(inputs)

