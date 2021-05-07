#!/usr/bin/python3
"""
MPC-based closed loop simulation example for a single 3DOF kite system.
"""

# imports
import awebox as awe
import casadi as ca
import pickle
import copy

from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)

# make default options object
options = awe.Options(True)

# single kite with point-mass model
options['user_options']['system_model']['architecture'] = {1:0}
options['user_options']['system_model']['kite_dof'] = 3
options['user_options']['kite_standard'] = awe.ampyx_data.data_dict()

# trajectory should be a single pumping cycle with initial number of five windings
options['user_options']['trajectory']['type'] = 'power_cycle'
options['user_options']['trajectory']['system_type'] = 'lift_mode'
options['user_options']['trajectory']['lift_mode']['windings'] = 1

# don't include induction effects, use simple tether drag
options['user_options']['induction_model'] = 'not_in_use'
options['user_options']['tether_drag_model'] = 'split'
options['model']['tether']['lift_tether_force'] = False
options['model']['aero']['lift_aero_force'] = False

options['nlp']['n_k'] = 40
options['nlp']['collocation']['u_param'] = 'poly'

# initialize and optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
trial.build()
trial.optimize()

# set-up closed-loop simulation
N_mpc = 15 # MPC horizon
N_sim = 1000  # closed-loop simulation steps
ts = 0.1 # sampling time

# MPC options
options['mpc']['scheme'] = 'radau'
options['mpc']['d'] = 4
options['mpc']['jit'] = True
options['mpc']['cost_type'] = 'tracking'
options['mpc']['expand'] = True
options['mpc']['linear_solver'] = 'ma57'
options['mpc']['max_iter'] = 1000
options['mpc']['max_cpu_time'] = 2000
options['mpc']['N'] = N_mpc
options['mpc']['plot_flag'] = False
options['mpc']['ref_interpolator'] = 'spline'
options['mpc']['u_param'] = 'zoh'
options['mpc']['homotopy_warmstart'] = True
options['mpc']['terminal_point_constr'] = False

# simulation options
options['sim']['number_of_finite_elements'] = 20 # integrator steps within one sampling time
options['sim']['sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# reduce average wind speed
options['sim']['sys_params']['wind']['u_ref'] = 0.9*options['sim']['sys_params']['wind']['u_ref']

# make simulator
closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', ts, options)
closed_loop_sim.run(N_sim)
closed_loop_sim.plot(['states','controls','algebraic_variables','constraints','invariants','quad'])
import matplotlib.pyplot as plt
plt.show()
