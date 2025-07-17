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

def main():
    override_options = {}
    override_options['params.arm.arm_inertia'] = None
    override_options['params.arm.torque_slope'] = 1000
    override_options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    override_options['model.system_bounds.u.dactive_torque'] = [-1000, 1000]

    options = rocking_mode_options(override_options)
    trial = awe.Trial(options, 'Rocking_arm_Ampyx_AP2')
    trial.build()
    trial.optimize()  # final_homotopy_step=['final] to control when to stop
    trial.plot(['states', 'controls', 'invariants', 'isometric'])
    return trial

if __name__ == "__main__":
    trial = main()


plot_dict = trial.visualization.plot_dict
perf = plot_dict['outputs']['performance']
pb = plot_dict['outputs']['power_balance']
arm = plot_dict['outputs']['arm']
x = plot_dict['x']
xdot = plot_dict['xdot']

plt.figure()
passive_power = arm['passive_power'][0]
active_power = arm['active_power'][0]
plt.plot(passive_power / np.max(passive_power), label="passive")
plt.plot(active_power / np.max(active_power), label="active")
plt.title("normalized power")
plt.legend()

P_kin_arm_rot = pb['P_kin_arm_rot'][0]

# Test: set torque_slope to a small value (eg. 1 Nm/(rad/s)) and verify that
# active_torque and passive_torque are of the same sign most of the time
# Which means that the optimisation process gives a solution that extracts energy when possible instead of inputing it
darm_angle = x['darm_angle'][0]
passive_torque = arm['passive_torque'][0]
active_torque = arm['active_torque'][0]
plt.figure()
plt.plot(passive_torque / np.max(passive_torque), label="passive torque")
plt.plot(active_torque / np.max(active_torque), label="active torque")
plt.legend()
plt.grid()
plt.title("Are active and passive torques of the same sign?")

# Kinetic energy theorem
# (d(1/2 * I * darm_angle**2) / dt = I * darm_angle * ddarm_angle) = tether_power_on_arm - (passive_power + active_power)
# Or equivalently, int(tether_power_on_arm) - int(passive_power) - int(active_power) = I * (darm_angle(tf) - darm_angle(0)) = 0
arm_inertia = plot_dict['options']['params']['arm']['arm_inertia']
arm_angle = x['arm_angle'][0]
ddarm_angle = xdot['ddarm_angle'][0]

dkinetic_energy = arm_inertia * darm_angle * ddarm_angle
passive_power = passive_torque * darm_angle
active_power = active_torque * darm_angle
generator_power = passive_power + active_power

tether_torque_on_arm = arm['tether_torque_on_arm'][0]
tether_power_on_arm = tether_torque_on_arm * darm_angle

zero = generator_power - tether_power_on_arm - dkinetic_energy  # Positive = extracting energy out of the void, negative = losing energy to the void
plt.figure()
plt.plot(generator_power, label="1: power extracted by the generator")
plt.plot(tether_power_on_arm, label="2: power injected by the tethers")
plt.plot(dkinetic_energy, label="3: variation of the kinetic energy")
plt.plot(zero, label="(1) - (2) - (3) = 0 W")
plt.title("Power balance of the arm")
plt.legend()

# Plot 1:
# arm angle&darm_angle | power balance
# active_torque | active power
# passive_torque | passive power

plt.figure()
plt.tight_layout()
plt.subplot(3, 2, 1)
plt.plot(arm_angle / np.max(abs(arm_angle)), label="normalized arm_angle")
plt.plot(darm_angle / np.max(abs(darm_angle)), label="normalized darm_angle")
plt.title("Arm state")
plt.legend()
plt.grid()
plt.subplot(3, 2, 2)
plt.plot(tether_power_on_arm)
plt.title("Tether power on arm")
plt.grid()
plt.subplot(3, 2, 3)
plt.plot(arm['passive_torque'][0])
plt.title("Passive generator torque")
plt.grid()
plt.subplot(3, 2, 4)
plt.plot(arm['passive_power'][0])
plt.title("Passive power output")
plt.grid()
plt.subplot(3, 2, 5)
plt.plot(arm['active_torque'][0])
plt.title("Active generator torque")
plt.grid()
plt.subplot(3, 2, 6)
plt.plot(arm['active_power'][0])
plt.title("Active power output")
plt.grid()

# Plot 2:
# arm angle | darm_angle
# tension | active_torque
# tether_torque_on_arm | dactive_torque

active_torque = x['active_torque'][0]
dactive_torque = xdot['dactive_torque'][0]

plt.figure()
plt.tight_layout()
plt.subplot(3, 2, 1)
plt.plot(arm_angle)
plt.title("Arm angle")
plt.grid()
plt.subplot(3, 2, 2)
plt.plot(darm_angle)
plt.title("Variation of arm angle")
plt.grid()
plt.subplot(3, 2, 3)
plt.plot(arm['tension'][0])
plt.title("Tether tension")
plt.grid()
plt.subplot(3, 2, 4)
plt.plot(active_torque)
plt.title("Active torque")
plt.grid()
plt.subplot(3, 2, 5)
plt.plot(tether_torque_on_arm)
plt.title("Tether torque on arm")
plt.grid()
plt.subplot(3, 2, 6)
plt.plot(dactive_torque)
plt.title("Variation of active torque")
plt.grid()
plt.show()
