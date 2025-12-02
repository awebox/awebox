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

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import awebox.tools.print_operations as print_op

def general_options():
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    options['user_options.system_model.kite_dof'] = 3

    # indicate desired operation mode
    options['user_options.trajectory.type'] = 'power_cycle'

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
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
    options['solver.cost.theta_regularisation.0'] = 1e-8  # Default of 1 barely optimizes the parameters

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False
    return options

def drag_mode_options(options):
    options['user_options.trajectory.system_type'] = 'drag_mode'
    # More than 25 seconds allows for ill solutions
    options['model.system_bounds.theta.t_f'] = [10, 25]  # This needs to be adjusted quite often
    return options

def lift_mode_options(options):
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['model.system_bounds.theta.t_f'] = [10, 35]  # This needs to be adjusted quite often
    return options


def plot_states(plot_dict):
    outputs = plot_dict['outputs']
    time = plot_dict['time_grids']['ip']
    plt.subplots(3, 1, sharex=True)

    plt.subplot(311)
    plt.plot(time, outputs['aerodynamics']['airspeed1'][0], label='Airspeed')
    plt.ylabel('[m/s]')
    plt.legend()
    plt.hlines([10, 32], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(312)
    plt.plot(time, 180.0 / np.pi * outputs['aerodynamics']['alpha1'][0], label='Angle of Attack')
    plt.plot(time, 180.0 / np.pi * outputs['aerodynamics']['beta1'][0], label='Side-Slip Angle')

    plt.ylabel('[deg]')
    plt.legend()
    plt.hlines([9, -6], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(313)
    plt.plot(time, outputs['local_performance']['tether_force10'][0], label='Tether Force Magnitude')
    plt.ylabel('[N]')
    plt.xlabel('t [s]')
    plt.legend()
    plt.hlines([50, 1800], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.show()


def main():
    options = general_options()
    options = lift_mode_options(options)

    trial = awe.Trial(options, 'Drag_Ampyx_AP2')
    trial.build()
    trial.optimize(final_homotopy_step='initial_guess')
    trial.plot(['states', 'quad'])
    plot_dict_init = deepcopy(trial.visualization.plot_dict)
    trial.optimize(final_homotopy_step='final')  # final_homotopy_step=['initial_guess', 'final'] to control when to stop the homotopy process
    plot_dict = trial.visualization.plot_dict
    
    trial.plot(['states', 'quad'])
    trial.plot(['controls', 'invariants'])
    plot_states(plot_dict)
    return trial, plot_dict_init, plot_dict

if __name__ == "__main__":
    main()


