#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
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

# single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_ampyx_ap2_settings(options)

# trajectory should be a single pumping cycle with initial number of five windings
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1

# wind model
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# nlp discretization
options['nlp.n_k'] = 80
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'mumps' # ma57 if HSL installed

# optimize trial
trial = awe.Trial(options, 'Ampyx_AP2')
trial.build()
trial.optimize()

# first look
trial.plot(['states', 'controls', 'constraints','quad'])

# plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3

print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

plt.subplots(5, 1, sharex=True)
plt.subplot(511)
plt.plot(time, plot_dict['x']['l_t'][0], label = 'l')
plt.ylabel('[m]')
plt.legend()
plt.grid(True)

plt.subplot(512)
plt.plot(time, plot_dict['x']['dl_t'][0], label = 'v_l')
plt.ylabel('[m/s]')
plt.legend()
plt.hlines([20, -15], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(513)
plt.plot(time, outputs['aerodynamics']['airspeed1'][0], label = 'V_T')
plt.ylabel('[m/s]')
plt.legend()
plt.hlines([10, 32], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(514)
plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['alpha1'][0], label = 'AoA')
plt.plot(time, 180.0/np.pi*outputs['aerodynamics']['beta1'][0], label = 'side-slip')
plt.ylabel('[deg]')
plt.legend()
plt.hlines([9, -6], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)

plt.subplot(515)
plt.plot(time, outputs['local_performance']['tether_force10'][0], label = 'Ft')
plt.ylabel('[kN]')
plt.xlabel('t [s]')
plt.legend()
plt.hlines([50, 1800], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)
plt.show()