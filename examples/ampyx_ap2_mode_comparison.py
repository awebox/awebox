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

# TODO: define common_options, then rocking_mode_options, lift_mode_options, drag_mode_options
def common_options():
    options = {}
    fixed_params = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    options['user_options.system_model.kite_dof'] = 3
    options['quality.test_param.t_f_min'] =  1
    options['quality.test_param.z_min'] = -np.inf  # The kite shouldn't go below z=0 but at least we don't get an error

    # indicate desired operation mode
    options['user_options.trajectory.type'] = 'power_cycle'

    # tether parameters
    options['params.tether.cd'] = 1.2
    options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)
    fixed_params['diam_t'] = 2e-3
    options['user_options.tether_drag_model'] = 'multi'
    options['model.tether.aero_elements'] = 5

    # Wind profile
    options['params.wind.z_ref'] = 10  # m
    options['user_options.wind.u_ref'] = 9  # m/s
    options['user_options.wind.model'] = 'power'
    options['params.wind.power_wind.exp_ref'] = 0.15  # Power in the power model

    # Other equality and inequality constraints
    options['model.model_bounds.rotation.include'] = False
    options['model.model_bounds.airspeed.include'] = False
    options['model.model_bounds.acceleration.include'] = False
    options['model.model_bounds.tether_force.include'] = False
    options['params.model_bounds.tether_force_limits'] = np.array([1e0, 7.5e3])
    options['model.model_bounds.tether_stress.include'] = True
    options['model.system_bounds.x.q'] = np.array([-np.inf, -np.inf, 10]), np.array([np.inf, np.inf, np.inf])

    # Initialization
    options['solver.initialization.groundspeed'] = 55  # m/s
    options['solver.initialization.init_clipping'] = False  # Iteratively refine initialization **assuming the trajectory is circular**

    # NLP options
    # By default, direct collocation using Radau scheme with order 4 lagrange polynomials
    options['nlp.n_k'] = 20
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
    options['solver.cost.theta_regularisation.0'] = 1e-8  # Default of 1 barely optimizes the parameters

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = True

    options['user_options.trajectory.fixed_params'] = fixed_params
    return options

def drag_mode_options(options):
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['model.system_bounds.theta.t_f'] = [10, 25]  # more than 25 seconds allows for ill solutions
    return options

def lift_mode_options(options):
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 2
    options['model.system_bounds.theta.t_f'] = [30, 70]
    return options

def rocking_mode_options(options):
    options['user_options.trajectory.system_type'] = 'rocking_mode'
    options['model.system_bounds.theta.t_f'] = [2, 8]

    ## Rocking mode options
    # Parameter values
    # All parameters are fixed by default, and can be optimized if `options['solver.initialization.theta.***]` is set
    # Or `options['solver.initialization.l_t` for 'l_t'.
    # If `options['solver.initialization.theta.***] = None`, the value found in fixed_params is used as a default.
    fixed_params['l_t'] = 50
    fixed_params['arm_length'] = 2
    fixed_params['arm_inertia'] = 2000
    fixed_params['torque_slope'] = 1500

    # Control of the torque of the arm
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = False
    options['model.system_bounds.u.dactive_torque'] = [-np.inf, np.inf]  # By default, dactive_torque is not directly constrained
    options['model.system_bounds.x.active_torque'] = [-np.inf, np.inf]  # This can be used to constrain active_torque
    options['model.arm.zero_avg_active_torque'] = True  # True by default, necessary for symmetry
    options['model.arm.zero_avg_active_power'] = None  # When None: any([torque_slope, arm_inertia] in fixed_params), cf. opts.model_funcs.build_arm_control_options

    # Initialize the trajectory (new lemniscate option)
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.inclination_deg'] = 30
    options['solver.initialization.lemniscate.az_width_deg'] = 60
    options['solver.initialization.lemniscate.el_width_deg'] = 20
    options['solver.initialization.lemniscate.rise_on_sides'] = False

    return options


"""
example 0: Finding the optimal kite trajectory for fixed arm parameters and no arm control
No parametric optimization, no control, lines of 35 m. Optimal control of the kite.
"""
def rocking_mode_example_0(options):
    return options

"""
example 1: example 0 + parametric optimization
"""
def rocking_mode_example_1(options):
    options = example_0(options)
    options['solver.initialization.l_t'] = None
    options['solver.initialization.theta.arm_inertia'] = None
    options['solver.initialization.theta.torque_slope'] = None
    return options

"""
example 2: example 1 + optimal control of the arm with constraints
"""
def rocking_mode_example_2(options):
    example_1(options)
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]
    return options

"""
example 3: example 2 + no constraints on arm control
Note: there are still constraints on the tether tension which indirectly act on the control of the arm

Solution is not sound, energy balance is off
"""
def rocking_mode_example_3(options):
    example_2(options)
    options.pop('model.system_bounds.u.dactive_torque')
    return options

"""
example 1 but the arm inertia and passive torque are entirely replaced by controlled torque
"""
def rocking_mode_example_4(options):
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['user_options.trajectory.fixed_params']['arm_inertia'] = 100
    options['user_options.trajectory.fixed_params']['torque_slope'] = 1
    options['model.system_bounds.x.active_torque'] = [-1000, 1000]
    return options


"""
If initialization is set for any parameter (solver.initialization.l_t or solver.initialization.theta.***),
remove it from user_options.trajectory.fixed_params
Initializing with None, as in `options['solver.initialization.theta.arm_inertia'] = None`, will use the value found in `fixed_params`

**This needs testing**, for 'l_t' and 'arm_inertia':
 - don't set `solver.initialization.***` -> same value in fixed_params and `solver.initialization.***`.
 - set `solver.initialization.***` to None -> popped value from fixed_params and assign it to `solver.initialization.***` instead of None
 - set  `solver.initialization.***` to a value -> popped value in fixed_params, and `solver.initialization.***` stays untouched
"""
def post_process_options_for_parameter_optimization(options):
    fixed_params = options['user_options.trajectory.fixed_params']
    print(options)

    # 1. If any initialization value is set, remove the parameter from fixed_params
    # If the value is None, replace None with the value in fixed_params which serves as a default
    if 'solver.initialization.l_t' in options:
        print('removing l_t')
        popped = fixed_params.pop('l_t', None)
        if options['solver.initialization.l_t'] is None and popped is not None:
            options['solver.initialization.l_t'] = popped

    prefix = 'solver.initialization.theta.'
    for opt in options:
        # Pop value from fixed_params and, if initialized to None, use it as initialization instead
        popped = None
        if opt == 'solver.initialization.l_t':
            popped = fixed_params.pop('l_t', None)
        elif opt.startswith(prefix):
            popped = fixed_params.pop(opt.removeprefix(prefix), None)

        if options[opt] is None and popped is not None:
            options[opt] = popped

    # 2. Put every (label, value) pair in solver.initialization
    for label, value in fixed_params.items():
        if label == 'l_t':
            options['solver.initialization.l_t'] = value
        else:
            options[prefix + label] = value

    options['user_options.trajectory.fixed_params'] = fixed_params

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


def _print_stats(u, label):
    avg = np.mean(u)
    min_, med, max_ = np.quantile(u, [0, 0.5, 1])
    msg = label + f': average={avg:.2f} (min={min_:.2f} , median={med:.2f}, max={max_:.2f}).'
    print_op.base_print(msg, level='info')
    return avg, min_, med, max_

def print_stats(plot_dict):
    z = plot_dict['x']['q10'][2]
    u_wind = awe.opts.model_funcs.get_u_at_altitude(plot_dict['options'], z)
    _print_stats(u_wind, 'Wind speed (m/s)')

    u_kite = np.linalg.norm(np.array(plot_dict['x']['dq10']), axis=0)
    _print_stats(u_kite, 'Kite speed (m/s)')

def main():
    options = common_options()
    options = lift_mode_options(options)

    trial = awe.Trial(options, 'Drag_Ampyx_AP2')
    trial.build()
    plot_dicts = {}
    for final_homotopy_step in ['initial_guess', 'initial', 'fictitious', 'power', 'final']:
        trial.optimize(final_homotopy_step=final_homotopy_step)
        plot_dicts[final_homotopy_step] = trial.visualization.plot_dict
        trial.plot(['states', 'quad'])
        print(f'Final homotopy step: {final_homotopy_step}')
        print_stats(plot_dicts[final_homotopy_step])

    plt.show()
    # plot_states(plot_dict)
    return trial, plot_dicts['final'], plot_dicts

if __name__ == "__main__":
    trial = main()
    trial.plot(['states', 'quad'])
    plt.show()

