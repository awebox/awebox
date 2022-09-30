#!/usr/bin/python3
"""
MPC-based closed loop simulation example for a single 3DOF kite system.
"""

# imports
import awebox as awe
import casadi as ca
import copy
import matplotlib.pyplot as plt

from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel('DEBUG')

# single kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
options['user_options.system_model.kite_dof'] = 3

# trajectory should be a single pumping cycle
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1
options['model.system_bounds.theta.t_f'] =  [5.0, 15.0]
# wind model
options['params.wind.z_ref'] = 10.0
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 5.

# NLP discretization
options['nlp.n_k'] = 40
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57' # if HSL
options['solver.mu_hippo'] = 1e-2

# initialize and optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize()
trial.plot(['isometric'])
plt.show()

# set-up closed-loop simulation
N_mpc = 10 # MPC horizon
N_sim = 200  # closed-loop simulation steps
ts = 0.1 # sampling time

# MPC options
options['mpc.scheme'] = 'radau'
options['mpc.d'] = 4
options['mpc.jit'] = False
options['mpc.cost_type'] = 'tracking'
options['mpc.expand'] = True
options['mpc.linear_solver'] = 'ma57'
options['mpc.max_iter'] = 1000
options['mpc.max_cpu_time'] = 2000
options['mpc.N'] = N_mpc
options['mpc.plot_flag'] = False
options['mpc.ref_interpolator'] = 'spline'
options['mpc.u_param'] = 'zoh'
options['mpc.homotopy_warmstart'] = True
options['mpc.terminal_point_constr'] = False

# simulation options
options['sim.number_of_finite_elements'] = 20 # integrator steps within one sampling time
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# reduce average wind speed
options['sim.sys_params']['wind']['u_ref'] = 1.0*options['sim.sys_params']['wind']['u_ref']

# make simulator
closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', ts, options)
closed_loop_sim.run(N_sim)
closed_loop_sim.plot(['isometric','states'])
plt.show()