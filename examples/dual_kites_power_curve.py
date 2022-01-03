#!/usr/bin/python3
"""
Efficient computation of a power curve for a dual-drone system with Ampyx AP2 aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
"""

import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

# dual kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
options = set_ampyx_ap2_settings(options)

# trajectory should be a single pumping cycle with five windings
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 5
options['model.system_bounds.x.l_t'] = [1.0e-2, 1.0e3]

# wind model
options['params.wind.z_ref'] = 10.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# discretization
options['nlp.n_k'] = 60
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'

# set-up sweep options
sweep_opts = [('user_options.wind.u_ref', np.linspace(5,9,5, endpoint=True))]

sweep = awe.Sweep(name = 'dual_kites_power_curve', options = options, seed = sweep_opts)
sweep.build()
sweep.run(apply_sweeping_warmstart = True)
sweep.plot(['comp_stats', 'comp_convergence'])
plt.show()
