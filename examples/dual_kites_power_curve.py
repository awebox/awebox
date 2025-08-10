#!/usr/bin/python3
"""
Efficient computation of a power curve for a dual-drone system with Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Rachel Leuthold
"""

import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2
import matplotlib.pyplot as plt
import numpy as np

def run(plot_show_block=True, overwrite_options={}):

    # dual kite with point-mass model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
    options = ampyx_ap2.set_ampyx_ap2_settings(options)

    # trajectory should be a single pumping cycle with 1 windings
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1
    options['model.system_bounds.x.l_t'] = [0.0, 1.0e3]
    options['model.system_bounds.theta.t_f'] = [10.0, 20.0]

    # wind model
    options['params.wind.z_ref'] = 10.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # discretization
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
    options['nlp.n_k'] = 25
    options['solver.linear_solver'] = 'ma57'

    for option_name, option_val in overwrite_options.items():
        options[option_name] = option_val

    # set-up sweep options
    sweep_opts = [('user_options.wind.u_ref', np.linspace(5,8,4, endpoint=True))]

    sweep = awe.Sweep(name = 'dual_kites_power_curve', options = options, seed = sweep_opts)
    sweep.build()
    sweep.run(apply_sweeping_warmstart = True)
    sweep.plot(['comp_stats', 'comp_convergence'])

    plt.show(block=plot_show_block)

    return sweep

if __name__ == "__main__":
    sweep = run()
