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

    # indicate desired operation mode
    # here: lift-mode system with pumping-cycle operation, with a one winding trajectory
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'rocking_mode'
    # options['trajectory.lift_mode.windings'] = 1  # TODO: rocking mode, make sure that 2+ changes nothing

    # indicate rocking mode options
    options['params.arm.arm_length'] = 2  # m
    options['params.arm.arm_inertia'] = 2000  # kg m^2
    options['user_options.trajectory.rocking_mode.enable_arm_control'] = True
    options['params.arm.torque_slope'] = 0  # Nm / (rad/s)
    options['solver.initialization.l_t'] = 30.  # m, Is this how we define tether length ?

    # Test this later since the initialization is less complete than for 'circular'
    # # indicate initialization, cf. new lemniscate options
    # options['solver.initialization.shape'] = 'lemniscate'
    # options['solver.initialization.lemniscate.az_width'] = 40*np.pi/180
    # options['solver.initialization.lemniscate.el_width'] = 10*np.pi/180
    # options['solver.initialization.fix_tether_length'] = True ?

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
    options['nlp.n_k'] = 5   # Change this later, for faster debugging
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'
    options['nlp.cost.beta'] = False # penalize side-slip (can improve convergence)

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False

    for option_name, option_val in overwrite_options.items():
        options[option_name] = option_val

    return options

def main():
    options = rocking_mode_options()
    trial = awe.Trial(options, 'Rocking_arm_Ampyx_AP2')
    trial.build()
    trial.optimize()
    return trial

if __name__ == "__main__":
    trial = main()
