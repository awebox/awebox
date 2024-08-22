#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""
import pdb

import awebox as awe

import ampyx_ap2_examples_settings as ampyx_ap2
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import logging

print_op.warn_about_temporary_functionality_alteration()
logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)

def run(plot_show_block=True, overwrite_options={}):
    # indicate desired system architecture
    # here: single kite with 6DOF Ampyx AP2 model
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2.set_ampyx_ap2_settings(options)

    # indicate desired operation mode
    # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1

    print_op.warn_about_temporary_functionality_alteration()
    options['solver.cost_factor.power'] = 1e1

    # indicate desired environment
    # here: wind velocity profile according to power-law
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and
    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = 40
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'

    for option_name, option_val in overwrite_options.items():
        options[option_name] = option_val

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Ampyx_AP2')
    trial.build()

    print(trial.options['model']['system_bounds']['theta']['t_f'])

    trial.optimize()

    # draw some of the pre-coded plots for analysis
    trial.plot(['states', 'controls', 'constraints', 'quad', 'outputs:local_performance'])

    # extract information from the solution for independent plotting or post-processing
    # here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
    plot_dict = trial.visualization.plot_dict
    outputs = plot_dict['interpolation_si']['outputs']
    time = plot_dict['time_grids']['ip']
    avg_power = plot_dict['power_and_performance']['avg_power']/1e3

    print('======================================')
    print('Average power: {} kW'.format(avg_power))
    print('======================================')

    plt.subplots(5, 1, sharex=True)
    plt.subplot(511)
    plt.plot(time, plot_dict['interpolation_si']['x']['l_t'][0], label='Tether Length')
    plt.ylabel('[m]')
    plt.legend()
    plt.grid(True)

    plt.subplot(512)
    plt.plot(time, plot_dict['interpolation_si']['x']['dl_t'][0], label='Tether Reel-out Speed')
    plt.ylabel('[m/s]')
    plt.legend()
    plt.hlines([20, -15], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(513)
    plt.plot(time, outputs['aerodynamics']['airspeed1'][0], label='Airspeed')
    plt.ylabel('[m/s]')
    plt.legend()
    plt.hlines([10, 32], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(514)
    plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['alpha1'][0], label='Angle of Attack')
    plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['beta1'][0], label='Side-Slip Angle')
    plt.ylabel('[deg]')
    plt.legend()
    plt.hlines([9, -6], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(515)
    plt.plot(time, outputs['local_performance']['tether_force10'][0], label='Tether Force Magnitude')
    plt.ylabel('[kN]')
    plt.xlabel('t [s]')
    plt.legend()
    plt.hlines([50, 1800], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    # a block=False argument will automatically close the figures after they've been created
    plt.show(block=plot_show_block)

    return trial

def make_comparison(trial):
    plot_dict = trial.visualization.plot_dict

    criteria = {'winding_period_s': {},
                'avg_power_kw': {}}

    criteria['avg_power_kw']['found'] = plot_dict['power_and_performance']['avg_power']/1e3
    criteria['avg_power_kw']['expected'] = 4.4

    criteria['winding_period_s']['found'] = plot_dict['time_grids']['ip'][-1]
    criteria['winding_period_s']['expected'] = 39.6

    return criteria

if __name__ == "__main__":
    overwrite_options = {}
    overwrite_options['quality.raise_exception'] = True
    overwrite_options['solver.cost_factor.power'] = 1
    overwrite_options['model.tether.control_var'] = 'ddl_t'
    overwrite_options['model.scaling.other.flight_radius_estimate'] = 'anticollision'
    overwrite_options['model.scaling.other.tension_estimate'] = 'power'
    overwrite_options['model.scaling.other.force_scaling_method'] = 'max_acceleration'
    overwrite_options['solver.cost.beta.0'] = 1.
    overwrite_options['solver.cost.tracking.0'] = 1.
    overwrite_options['solver.cost.u_regularisation.0'] = 0.0001
    overwrite_options['model.system_bounds.theta.t_f'] = [20., 50.]
    # overwrite_options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
    # overwrite_options['solver.cost.u_regularisation.0'] = 1e-3
    trial = run(overwrite_options=overwrite_options)
    pdb.set_trace()


