#!/usr/bin/python3
"""
Airborne actuators with vortex loop and dipole elements code

:author: Jochem De Schutter
"""

import awebox as awe
import awebox.opts.kite_data.three_dof_kite_data as three_dof_kite_data
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import casadi as ca
from os import path
import copy
import sys

def check_convergence(trial):
    return_status = trial.solution_dict['stats']['return_status']
    if return_status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Feasible_Point_Found']:
        return True
    else:
        return False

def run(plot_show_block=True, overwrite_options={}):

    # indicate desired system architecture
    # here: single kite with 6DOF Ampyx AP2 model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}

    # 6DOF Ampyx Ap2 model
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = three_dof_kite_data.data_dict()
    R_ring = 0.53*options['user_options.kite_standard']['geometry']['b_ref']

    # trajectory type: AAA
    options['user_options.trajectory.type'] = 'aaa'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1

    # tether parameters
    options['params.tether.cd'] = 1.2
    options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)

    # tether drag model (more accurate than the Argatov model in Licitra2019)
    options['user_options.tether_drag_model'] = 'multi'
    options['model.tether.aero_elements'] = 5

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.tether_force.include'] = False
    options['model.model_bounds.airspeed.include'] = False
    options['model.model_bounds.acceleration.include'] = False
    options['model.model_bounds.aero_validity.include'] = False
    options['model.model_bounds.rotation.include'] = False
    options['model.model_bounds.anticollision.include'] = True
    options['model.model_bounds.anticollision.safety_factor'] = 2.2 

    # variable bounds
    options['model.system_bounds.x.l_t'] = [10.0, 700.0]  # [m]
    options['model.system_bounds.x.q'] = [np.array([-ca.inf, -ca.inf, 200.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] = [1., 10.]  # [s]
    options['model.system_bounds.z.lambda'] = [0., ca.inf]  # [N/m]
    options['model.system_bounds.x.coeff'] = [np.array([0., -30.0 * np.pi / 180.]), np.array([1., 30.0 * np.pi / 180.])]
    options['model.system_bounds.u.dcoeff'] = [np.array([-5, -0.1]), np.array([5., 0.1])]

    # don't include induction effects
    options['user_options.induction_model'] = 'not_in_use'

    # initialization
    options['solver.initialization.groundspeed'] = 30.
    options['solver.initialization.inclination_deg'] = 30.
    options['solver.initialization.cone_deg'] = 25.
    options['solver.initialization.l_t'] = 600.
    options['solver.initialization.theta.l_s'] = 120.
    options['solver.initialization.theta.diam_t'] = 2e-2
    options['solver.initialization.theta.diam_s'] = 1e-2
    options['solver.cost.theta_regularisation.0'] = 1e-9
    # indicate desired environment
    # here: wind velocity profile according to power-law
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'uniform'
    options['user_options.wind.u_ref'] = 11.979113452507281

    options['user_options.atmosphere'] = 'uniform'
    options['params.atmosphere.rho_ref'] = 1.225

    # vortex ring model
    N = 8
    N_rings = 3 # (M = N * N_rings)
    N_duplicates = 3
    N_far = 4 # window of influence
    d = 4
    vtype = 'rectangle'
    conv_type = 'far'
    R_ring = options['user_options.kite_standard']['geometry']['b_ref']
    options['model.aero.vortex_rings.N_rings'] = N_rings
    options['model.aero.vortex_rings.N_far'] = N_far
    options['params.aero.vortex_rings.R_ring'] = R_ring
    options['model.aero.vortex_rings.N_duplicates'] = N_duplicates
    options['model.aero.vortex_rings.type'] = vtype
    options['model.aero.vortex_rings.convection_type'] = conv_type
    # options['model.aero.vortex_rings.N_elliptic_int'] = 5
    # options['model.aero.vortex_rings.elliptic_method'] = 'power'

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and
    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = N
    options['nlp.collocation.u_param'] = 'zoh'
    options['nlp.collocation.d'] = d
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'
    options['solver.max_iter'] = 2000
    options['solver.ipopt.autoscale'] = False
    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False
    # options['solver.expand_overwrite'] = False
    options['solver.cost.iota.1'] = 1e3 

    options['model.model_bounds.azimuth_elevation.include'] = True
    options['params.model_bounds.azimuth_elevation.bounds'] = np.array([0, np.pi/2, 0])

    options['visualization.cosmetics.plot_eq_constraints'] = False
    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'AAA_3DOF')
    trial.build()
    trial.optimize(intermediate_solve=True)
    intermediate_sol_elev = copy.deepcopy(trial.solution_dict)
    trial.optimize(warmstart_file = intermediate_sol_elev, intermediate_solve=False)
    # trial.plot(['isometric'])
    # plt.show()
    filename = path.join('./',f'AAA_3DOF_N{N}_d{d}_Nrings{N_rings}_Ndup{N_duplicates}_Nfar{N_far}_{vtype}_conv{conv_type}')
    # # write the solution to CSV file, interpolating the collocation solution with given frequency.
    trial.write_to_csv(filename = filename, frequency = 30, rotation_representation='dcm')

    # store optimization data
    import pandas as pd
    df = pd.read_csv(filename + '.csv')
    df['converged'] = check_convergence(trial)
    df['stats_t_wall_total'] = trial.optimization.t_wall['optimization']
    df['stats_t_wall_f_eval'] = trial.optimization.t_f_eval['optimization']
    df['stats_iterations'] = trial.optimization.iterations['optimization']
    df.to_csv(filename + '.csv', index=False)

    return trial

if __name__ == "__main__":
    trial = run()