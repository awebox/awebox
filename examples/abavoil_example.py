#!/usr/bin/python3
"""
Using Ampyx AP2 model in rocking mode

:author: Antonin Bavoil
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
import awebox.tools.print_operations as print_op

matplotlib.use(DEFAULT_MPL_BACKEND)


def rocking_mode_options():
    ## General options
    # System architecture
    options = {}
    fixed_params = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['user_options.system_model.kite_dof'] = 3  # Only 3DOF converged, possible to warmstart 6DOF with 3DOF ?
    options['model.system_bounds.theta.t_f'] = [1, 10]
    options['quality.test_param.t_f_min'] =  1
    options['quality.test_param.z_min'] = -np.inf  # The kite shouldn't go below z=0 but at least we don't get an error

    # Flight parameters
    options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 40.0*np.pi/180.0])
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 9.0
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -6.0
    omega_bound = 50.0*np.pi/180.0
    options['model.system_bounds.x.omega'] = [np.array(3*[-omega_bound]), np.array(3*[omega_bound])]
    options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])
    options['user_options.induction_model'] = 'not_in_use'  # don't include induction effects

    # tether parameters
    options['params.tether.cd'] = 1.2
    options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)
    fixed_params['diam_t'] = 2e-3
    options['user_options.tether_drag_model'] = 'multi'
    options['model.tether.aero_elements'] = 5

    # Operation mode
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'rocking_mode'

    ## Rocking mode options
    # Parameter values
    # All parameters are fixed by default, and can be optimized if `options['solver.initialization.theta.***]` is set
    # Or `options['solver.initialization.l_t` for 'l_t'.
    # If `options['solver.initialization.theta.***] = None`, the value found in fixed_params is used as a default.
    fixed_params['l_t'] = 50
    fixed_params['arm_length'] = 2
    fixed_params['arm_inertia'] = 2000
    fixed_params['torque_slope'] = 1500

    # Fixed parameter values (not optimized), overwrite initialization values

    # Control of the torque of the arm
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = False
    options['model.system_bounds.u.dactive_torque'] = [-np.inf, np.inf]  # By default, dactive_torque is not directly constrained
    options['model.system_bounds.x.active_torque'] = [-np.inf, np.inf]  # This can be used to constrain active_torque
    options['model.arm.zero_avg_active_torque'] = True  # True by default, necessary for symmetry
    options['model.arm.zero_avg_active_power'] = None  # When None: any([torque_slope, arm_inertia] in fixed_params), cf. opts.model_funcs.build_arm_control_options

    # Other equality and inequality constraints
    options['model.model_bounds.aero_validity.include'] = True
    options['model.model_bounds.rotation.include'] = False
    options['model.model_bounds.airspeed.include'] = False
    options['model.model_bounds.acceleration.include'] = False
    options['model.model_bounds.tether_force.include'] = False
    options['params.model_bounds.tether_force_limits'] = np.array([1e0, 7.5e3])
    options['model.model_bounds.tether_stress.include'] = True
    options['model.system_bounds.x.q'] = np.array([-np.inf, -np.inf, 10]), np.array([np.inf, np.inf, np.inf])

    # Initialize the trajectory (new lemniscate option)
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.lemniscate.az_width_deg'] = 20
    options['solver.initialization.lemniscate.el_width_deg'] = 5
    options['solver.initialization.lemniscate.rise_on_sides'] = False
    options['solver.initialization.groundspeed'] = 50  # m/s
    options['solver.initialization.init_clipping'] = False  # Iteratively refine initialization **assuming the trajectory is circular**

    # Wind profile
    options['params.wind.z_ref'] = 10  # m
    options['user_options.wind.u_ref'] = 9  # m/s
    options['user_options.wind.model'] = 'power'
    options['params.wind.power_wind.exp_ref'] = 0.15  # Power in the power model

    # NLP options
    # By default, direct collocation using Gauss scheme with order 4 lagrange polynomials
    options['nlp.n_k'] = 20  # Number of control intervals
    options['nlp.collocation.u_param'] = 'zoh'  # zero-order-hold (onstant) control over each control interval
    options['solver.linear_solver'] = 'ma57'  # recommended: 'ma57' if HSL installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)

    options['user_options.trajectory.fixed_params'] = fixed_params

    return options

"""
If initialization is set for any parameter (solver.initialization.l_t or solver.initialization.theta.***),
remove it from user_options.trajectory.fixed_params
Initializing with None, as in `options['solver.initialization.theta.arm_inertia'] = None`, will use the value found in `fixed_params`

**This needs testing**, for for 'l_t' and 'arm_inertia':
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

"""
Baseline example
No parametric optimization, no control, lines of 35 m. Optimal control of the kite.

5.2 kW
"""
def example_0(options):
    return options

"""
Baseline example + free arm control

12.7 kW --> High output but weird behavior, sometimes 'hacking' energy ?
"""
def example_1(options):
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    return options

"""
Baseline example + constrained arm control

6.2 kW --> Higher output and nicely behaved
"""
def example_2(options):
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]
    return options

"""
Baseline example + constrained arm control & optimizing all parameters (except arm length)

--> Even higher output
"""
def example_3(options):
    options['solver.initialization.l_t'] = None
    options['solver.initialization.theta.arm_inertia'] = None
    options['solver.initialization.theta.torque_slope'] = None  # Verify that it uses this value instead of the default
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]
    return options

"""
Baseline example but arm is only driven by control, small inertia and 0 torque_slope
+ Constraint on torque
"""
def example_4(options):
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['user_options.trajectory.fixed_params']['arm_inertia'] = 100
    options['user_options.trajectory.fixed_params']['torque_slope'] = 0
    options['model.system_bounds.x.active_torque'] = [-5000, 5000]
    options['model.system_bounds.u.dactive_torque'] = [-10000, 10000]
    return options

"""
Visual test: set torque_slope = 1 and dactive_torque bounds to [-1, 1] and verify that both passive and active power are positive
If every power is integrated into work (I don't think so) verify that the arm received as much energy as it extracted
"""
def test_1(options):
    options['solver.initialization.l_t'] = 200.
    options['initialization.theta.torque_slope'] = 100
    options['initialization.theta.arm_length'] = 2
    options['initialization.theta.arm_inertia'] = 2000
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['model.system_bounds.u.dactive_torque'] = [-100, 100]
    return options

def plot_arm_torques_and_energies(plot_dict):
    arm_outputs = plot_dict['outputs']['arm']
    power_balance = plot_dict['outputs']['power_balance']
    x = plot_dict['x']
    arm_angle = x['arm_angle'][0]
    darm_angle = x['darm_angle'][0]
    dkinetic_energy = -power_balance['P_kin_arm_rot'][0]  # positive when arm accelerates
    tether_power_on_arm = power_balance['P_tether_arm'][0]  # positive when tether accelerates the arm
    generator_power = -power_balance['P_gen_arm'][0]  # positive when generator decelerates the arm

    plt.figure()
    plt.tight_layout()
    n, m, i = 6, 1, 0
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_angle / np.max(abs(arm_angle)), label="arm_angle")
    plt.plot(darm_angle / np.max(abs(darm_angle)), label="darm_angle")
    plt.title("Normalized arm state [1]")
    plt.legend()
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(dkinetic_energy, label="1: dK/dt")
    plt.plot(tether_power_on_arm, label="2: P_tether")
    plt.plot(generator_power, label="3: P_gen")
    plt.plot(dkinetic_energy - tether_power_on_arm + generator_power, label="1-2+3=0")
    plt.title("Power balance of the arm [W]")
    plt.grid()
    plt.legend()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['passive_torque'][0])
    plt.title("Passive torque [Nm]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['passive_power'][0])
    plt.title("Passive power output [W]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['active_torque'][0])
    plt.title("Active torque [Nm]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['active_power'][0])
    plt.title("Active power output [W]")
    plt.grid()

def plot_arm_states(plot_dict):
    arm_outputs = plot_dict['outputs']['arm']
    x = plot_dict['x']
    u = plot_dict['u']

    plt.figure()
    plt.tight_layout()
    n, m, i = 6, 1, 0
    plt.subplot(n, m, (i := i + 1))
    plt.plot(x['arm_angle'][0])
    plt.title("Arm angle [rad]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(x['darm_angle'][0])
    plt.title("d(arm angle)/dt [rad/s]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['tether_tension'][0])
    plt.title("Tether tension [N]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(x['active_torque'][0])
    plt.title("Active torque [Nm]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.plot(arm_outputs['tether_torque_on_arm'][0])
    plt.title("Tether torque [Nm]")
    plt.grid()
    plt.subplot(n, m, (i := i + 1))
    plt.stairs(u['dactive_torque'][0], lw=2)
    plt.title("d(active torque)/dt [Nm/s]")
    plt.grid()


def print_stats(plot_dict):
    def _print_stats(u, label):
        avg = np.mean(u)
        min_, med, max_ = np.quantile(u, [0, 0.5, 1])
        msg = label + f': average={avg:.2f} (min={min_:.2f} , median={med:.2f}, max={max_:.2f}).'
        print_op.base_print(msg, level='info')
        return avg, min_, med, max_

    z = plot_dict['x']['q10'][2]
    u_wind = awe.opts.model_funcs.get_u_at_altitude(plot_dict['options'], z)
    _print_stats(u_wind, 'Wind speed (m/s)')

    u_kite = np.linalg.norm(np.array(plot_dict['x']['dq10']), axis=0)
    _print_stats(u_kite, 'Kite speed (m/s)')

def main():
    # Opti 1: no arm control, find best torque_slope
    # Opti 2: no torque_slope, find best arm control
    # Opti 3: mixed, find best torque_slope and arm control st. avg of active power = 0
    # What about arm length and inertia ?

    # --->
    # Always: avg(active_torque) = 0, because the arm does not perform full rotations. It should avoid suboptimal solutions
    # If optimizing for torque_slope, then avg(active_power) = 0, to avoid trading passive for active torque

    # Add power balance check & fix todos in dynamics.py

    options = rocking_mode_options()
    options = example_0(options)
    options = post_process_options_for_parameter_optimization(options)
    options['nlp.n_k'] = 20
    trial = awe.Trial(options, 'Rocking_arm_Ampyx_AP2')
    trial.build()
    trial.optimize(final_homotopy_step='initial_guess')
    trial.plot(['states', 'quad'])
    plot_dict_init = deepcopy(trial.visualization.plot_dict)
    trial.optimize(final_homotopy_step='final')  # final_homotopy_step=['initial_guess', 'final'] to control when to stop the homotopy process
    plot_dict = trial.visualization.plot_dict

    print_op.base_print("## Stats of solution", level='info')
    print_stats(plot_dict_init)
    print_op.base_print("## Stats of initialization", level='info')
    print_stats(plot_dict)
    trial.plot(['states', 'quad'])
    trial.plot(['controls', 'invariants'])
    plot_arm_torques_and_energies(plot_dict)
    plot_arm_states(plot_dict)
    return trial, plot_dict_init, plot_dict

if __name__ == "__main__":
    trial, plot_dict_init, plot_dict = main()
    xi = plot_dict_init['x']
    x = plot_dict['x']
    plt.show()

# Test: set torque_slope to a small value (eg. 1 Nm/(rad/s)) and verify that
# active_torque and passive_torque are of the same sign most of the time
# Which means that the optimisation process gives a solution that extracts energy when possible instead of burning it
