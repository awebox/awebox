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

import os
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from copy import deepcopy
import time

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import awebox.tools.print_operations as print_op

# Founf in Trial.print_solution()
THETA_INFO = {
    'diam_t': ('Main tether diameter', 1e3, 'mm'),
    'diam_s': ('Secondary tether diameter', 1e3, 'mm'),
    'l_s': ('Secondary tether length', 1, 'm'),
    'l_t': ('Main tether length', 1, 'm'),
    'l_i': ('Intermediate tether length', 1, 'm'),
    'diam_i': ('Intermediate tether diameter', 1e3, 'mm'),
    'P_max': ('Peak power', 1e-3, 'kW'),
    'ell_radius': ('Ellipse radius', 1, 'm'),
    'ell_elevation': ('Ellipse elevation', 180.0/np.pi, 'deg'),
    'ell_theta': ('Ellipse division angle', 180.0/np.pi, 'deg'), 
    'a': ('Average induction', 1, '-'),
    'arm_length': ('Arm length', 1, 'm'),
    'arm_inertia': ('Arm inertia', 1, 'kg.m^2'),
    'torque_slope': ('Torque slope', 1, 'N.m/(rad/s)'),
}

## Defining the experiments

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
    options['nlp.n_k'] = 80
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)
    options['solver.cost.theta_regularisation.0'] = 1e-8  # Default of 1 barely optimizes the parameters

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False

    options['user_options.trajectory.fixed_params'] = fixed_params
    return options

def drag_mode_options(options):
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['model.system_bounds.theta.t_f'] = [3, 15]  # more than 25 seconds allows for ill solutions
    options['nlp.n_k'] = 20
    return options

def lift_mode_options(options):
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 2
    options['model.system_bounds.theta.t_f'] = [30, 80]
    options['nlp.n_k'] = 80

    return options

def rocking_mode_options(options):
    options['user_options.trajectory.system_type'] = 'rocking_mode'
    options['model.system_bounds.theta.t_f'] = [2, 8]
    options['nlp.n_k'] = 10

    ## Rocking mode options
    # Parameter values
    # All parameters are fixed by default, and can be optimized if `options['solver.initialization.theta.***]` is set
    # Or `options['solver.initialization.l_t` for 'l_t'.
    # If `options['solver.initialization.theta.***] = None`, the value found in fixed_params is used as a default.
    fixed_params = options['user_options.trajectory.fixed_params']
    fixed_params['l_t'] = 50
    fixed_params['arm_length'] = 2
    fixed_params['arm_inertia'] = 2000
    fixed_params['torque_slope'] = 1500
    options['user_options.trajectory.fixed_params'] = fixed_params

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
    options = rocking_mode_example_0(options)
    options['solver.initialization.l_t'] = None
    options['solver.initialization.theta.arm_inertia'] = None
    options['solver.initialization.theta.torque_slope'] = None
    return options

"""
example 2: example 1 + optimal control of the arm with constraints
"""
def rocking_mode_example_2(options):
    rocking_mode_example_1(options)
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]
    return options

"""
example 3: example 2 + no constraints on arm control
Note: there are still constraints on the tether tension which indirectly act on the control of the arm

Solution is not sound, energy balance is off
"""
def rocking_mode_example_3(options):
    rocking_mode_example_2(options)
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

## Running and processing the experiments

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

    # 1. If any initialization value is set, remove the parameter from fixed_params
    # If the value is None, replace None with the value in fixed_params which serves as a default
    if 'solver.initialization.l_t' in options:
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

def save_all_figures(directory, prefix):
    """Saves and closes all open matplotlib figures."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for i, fig in enumerate(figs):
        filename = f"{prefix}_plot_{i}.png"
        path = os.path.join(directory, filename)
        fig.savefig(path)
        print_op.base_print(f"Saved plot to: {path}", level='info')
    plt.close('all')

def get_step_results(trial, step_name):
    """Extracts quantities from the trial object similar to print_solution."""
    opt = trial.optimization
    res = {"Step": step_name}

    # Time period - Cast CasADi DM to float before rounding
    t_f = float(opt.global_outputs_opt['time_period'])
    res["Time period (s)"] = round(t_f, 3)

    # Average Power
    if 'e' in trial.model.integral_outputs.keys():
        e_final = float(opt.integral_outputs_final_si['int_out', -1, 'e'])
    else:
        e_final = float(opt.V_final_si['x', -1, 'e'][-1])
    res["Average power output (kW)"] = round((e_final / t_f) / 1000.0, 4)

    # Theta Parameters
    for theta in trial.model.variables_dict['theta'].keys():
        if theta != 't_f' and theta in THETA_INFO:
            info = THETA_INFO[theta]
            val = float(opt.V_final_si['theta', theta])
            res[f"{info[0]} ({info[2]})"] = round(val * info[1], 4)
    
    return res

def run_task(task_idx, total_tasks, subfolder, prefix, config_func, base_options, base_path):
    header = f" [{task_idx}/{total_tasks}] STARTING TASK: {prefix} "
    print("\n" + "="*80)
    print(header.center(80, "="))
    print("="*80 + "\n")

    start_time = time.time()
    save_dir = os.path.join(base_path, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{prefix}_results.csv")
    
    csv_data = []
    try:
        options = config_func(deepcopy(base_options))
        options = post_process_options_for_parameter_optimization(options)
        trial = awe.Trial(options, prefix)
        trial.build()
        
        steps = ['initial_guess', 'initial', 'fictitious', 'power', 'final']
        for step in steps:
            print_op.base_print(f"Homotopy Step: {step}", level='info')
            trial.optimize(final_homotopy_step=step)
            
            print_stats(trial.visualization.plot_dict)
            csv_data.append(get_step_results(trial, step))

            trial.plot(['states', 'quad'])
            save_all_figures(save_dir, f"{prefix}_step_{step}")

        if csv_data:
            keys = csv_data[0].keys()
            with open(csv_path, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(csv_data)
            print_op.base_print(f"Results saved to {csv_path}", level='info')

        # Timing Calculation
        duration = time.time() - start_time
        
        # Visible Footer
        footer = f" TASK {prefix} COMPLETED IN {duration:.2f}s "
        print("\n" + "-"*80)
        print(footer.center(80, "-"))
        print("-"*80)

    except Exception as e:
        end_time = time.time()
        print_op.base_print(f"CRITICAL ERROR in task {prefix} after {end_time-start_time:.2f}s: {str(e)}", level='error')
        raise(e)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join("outputs", timestamp)
    
    # Define tasks: (Subfolder, Filename Prefix, Function)
    tasks = [
        ("lift", "lift_mode", lambda opt: lift_mode_options(opt)),
        ("drag", "drag_mode", lambda opt: drag_mode_options(opt)),
        ("rocking", "rocking_ex1", lambda opt: rocking_mode_example_1(rocking_mode_options(opt))),
        ("rocking", "rocking_ex2", lambda opt: rocking_mode_example_2(rocking_mode_options(opt))),
    ]
    
    base_options = common_options()
    
    for i, (subfolder, prefix, func) in enumerate(tasks, 1):
        run_task(i, len(tasks), subfolder, prefix, func, base_options, base_path)
        print("\n" * 3)

if __name__ == "__main__":
    main()