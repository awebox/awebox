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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
import awebox.tools.print_operations as print_op

matplotlib.use(DEFAULT_MPL_BACKEND)



def rocking_mode_options(overwrite_options={}):
    # indicate desired system architecture
    # here: single kite with 6DOF Ampyx AP2 model
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options = ampyx_ap2_settings.set_ampyx_ap2_settings(options)
    options['user_options.system_model.kite_dof'] = 3
    options['model.system_bounds.theta.t_f'] = [1, 6]
    options['quality.test_param.t_f_min'] =  1

    # indicate desired operation mode
    # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'rocking_mode'
    # options['user_options.trajectory.lift_mode.windings'] = 1  # TODO: rocking mode, make sure that 2+ changes nothing

    # Bounds on tether stress instead of tether force, no bounds on airspeed and rotation
    # Why does this give a better solution?
    options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.tether_force.include'] = False
    options['model.model_bounds.airspeed.include'] = False
    options['model.model_bounds.rotation.include'] = False

    # indicate rocking mode options
    options['params.arm.arm_length'] = 2  # m
    options['params.arm.arm_inertia'] = 2000  # kg m^2
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = False
    options['params.arm.torque_slope'] = 2000  # Nm / (rad/s)
    options['solver.initialization.l_t'] = 200.  # m

    # Test this later since the initialization is less complete than for 'circular'
    # # indicate initialization, cf. new lemniscate options
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.lemniscate.az_width_deg'] = 40
    options['solver.initialization.lemniscate.el_width_deg'] = 10

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
    options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)

    for option_name, option_val in overwrite_options.items():
        if option_val is not None:
            options[option_name] = option_val

    return options

def plot_arm_torques_and_energies(trial):
    plot_dict = trial.visualization.plot_dict
    arm_outputs = plot_dict['outputs']['arm']
    x = plot_dict['x']
    arm_angle = x['arm_angle'][0]
    darm_angle = x['darm_angle'][0]
    dkinetic_energy = -plot_dict['outputs']['power_balance']['P_kin_arm_rot'][0]  # positive when energy is added to the system
    tether_power_on_arm = arm_outputs['tether_torque_on_arm'][0] * darm_angle  # positive when energy is added to the system
    generator_power = arm_outputs['passive_power'][0] + arm_outputs['active_power'][0]  # positive when energy is subtracted from the system

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

def plot_arm_states(trial):
    plot_dict = trial.visualization.plot_dict
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
    plt.plot(arm_outputs['tension'][0])
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

def main():
    override_options = {}
    override_options['params.arm.arm_inertia'] = None
    override_options['params.arm.torque_slope'] = 1450  # This gives an average active power of 0 !
    override_options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    override_options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]

    # Opti 1: no arm control, find best torque_slope
    # Opti 2: no torque_slope, find best arm control
    # Opti 3: mixed, find best torque_slope and arm control st. avg of active power = 0
    # What about arm length and inertia ?

    # Add power balance check & fix todos in dynamics.py

    options = rocking_mode_options(override_options)
    trial = awe.Trial(options, 'Rocking_arm_Ampyx_AP2')
    trial.build()
    trial.optimize()  # final_homotopy_step=['final] to control when to stop
    trial.plot(['states', 'controls', 'invariants', 'quad'])
    plot_arm_torques_and_energies(trial)
    plot_arm_states(trial)
    plt.show()
    return trial

if __name__ == "__main__":
    trial = main()

# Test: set torque_slope to a small value (eg. 1 Nm/(rad/s)) and verify that
# active_torque and passive_torque are of the same sign most of the time
# Which means that the optimisation process gives a solution that extracts energy when possible instead of burning it
