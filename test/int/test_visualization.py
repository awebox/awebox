#!/usr/bin/python3
"""Test to check whether any plot function produces an error.

@author: Thilo Bronnenmeyer
"""
import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)
import matplotlib.pyplot as plt
import numpy as np

import os
import awebox as awe
import logging
import awebox.viz.tools as viz_tools
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

plt.rcParams.update({'figure.max_open_warning': 0})

logging.basicConfig(filemode='w',format='%(message)s', level=logging.DEBUG)

def make_dummy_trial_options():

        # basic options
    options = {}
    options['user_options.system_model.architecture'] = {1:0}
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.induction_model'] = 'not_in_use'
    options['nlp.n_k'] = 2
    options['solver.max_iter'] = 0
    options['visualization.cosmetics.plot_ref'] = True
    options['visualization.cosmetics.trajectory.kite_bodies'] = True
    n_points = 5
    options['visualization.cosmetics.interpolation.n_points'] = n_points

    return options

def construct_and_solve_trial():

    options = make_dummy_trial_options()

    # build trial and optimize
    trial = awe.Trial(options, 'trial1')
    trial.build()
    trial.optimize(final_homotopy_step='initial', debug_flags='all')

    return trial

def check_no_failure_in_plotting(trial):

    # set flags and plot
    trial_flags = ['all']
    trial.plot(trial_flags)

    return None

def check_output_interpolation(trial):

    original_times = struct_op.get_original_time_data_for_output_interpolation(trial.visualization.plot_dict['time_grids'])
    odx = struct_op.find_output_idx(trial.model.outputs, 'local_performance', 'tether_stress10', 0)
    original_series = trial.optimization.outputs_opt[odx, :].T

    collocation_entries = original_times.shape[0] * original_times.shape[1]
    collocation_d = int(collocation_entries / trial.visualization.plot_dict['n_k'])
    original_series = struct_op.get_output_series_with_duplicates_removed(original_times, original_series, collocation_d)

    output_on_collocation_grid = original_series
    output_on_interpolated_grid = trial.visualization.plot_dict['outputs']['local_performance']['tether_stress10'][0]

    msg = 'Interpolated outputs do not coincide with outputs on collocation grid for identical time point!'
    assert np.abs(output_on_collocation_grid[-1] - output_on_interpolated_grid[-1]) < 1e-7, msg

    return None

def test_trial_visualization():

    trial = construct_and_solve_trial()

    check_no_failure_in_plotting(trial)
    check_output_interpolation(trial)
    check_animation(trial)

def test_sweep_visualization():

    options = make_dummy_trial_options()

    # build sweep and run
    sweep_opts = [('user_options.wind.u_ref', [5.,5.5])]
    sweep = awe.Sweep(name = 'sweep_viz_test', options = options, seed = sweep_opts)
    sweep.build()
    sweep.run(final_homotopy_step='initial', debug_flags='all')

    # set flags and plot
    sweep_flags = ['all', 'comp_all', 'outputs:tether_length', 'comp_outputs:tether_length']
    sweep.plot(sweep_flags)

    return None

def check_animation(trial, threshold=1e-4):

    # check that able to plot without errors
    trial.plot('animation')

    # check that save worked correctly
    os.remove('trial1.mp4')

    return None


def test_components():
    viz_tools.test_naca_coordinates()
    return None


if __name__ == "__main__":
    test_trial_visualization()
    test_sweep_visualization()
    test_components()
