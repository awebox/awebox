#!/usr/bin/python3
"""
Airborne atmospheric actuator code

:author: Jochem De Schutter
"""

import awebox as awe
import awebox.opts.kite_data.three_dof_kite_data as three_dof_kite_data
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import casadi as ca

def run(plot_show_block=True, overwrite_options={}):

    # indicate desired system architecture
    # here: single kite with 6DOF Ampyx AP2 model
    options = {}
    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}

    # 6DOF Ampyx Ap2 model
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = three_dof_kite_data.data_dict()

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
    # acceleration constraint

    # aircraft-tether anti-collision

    # variable bounds
    options['model.system_bounds.x.l_t'] = [10.0, 700.0]  # [m]
    options['model.system_bounds.x.q'] = [np.array([-ca.inf, -ca.inf, 100.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] = [1., 20.]  # [s]
    options['model.system_bounds.z.lambda'] = [0., ca.inf]  # [N/m]
    options['model.system_bounds.x.coeff'] = [np.array([0., -30.0 * np.pi / 180.]), np.array([1., 30.0 * np.pi / 180.])]

    # don't include induction effects
    options['user_options.induction_model'] = 'not_in_use'

    # initialization
    options['solver.initialization.groundspeed'] = 30.
    options['solver.initialization.inclination_deg'] = 30.
    options['solver.initialization.cone_deg'] = 15.
    options['solver.initialization.l_t'] = 600.
    options['solver.initialization.theta.l_s'] = 70.
    options['solver.initialization.theta.diam_t'] = 1e-2
    options['solver.initialization.theta.diam_s'] = 1e-2
    options['solver.cost.theta_regularisation.0'] = 1e-9
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
    options['nlp.n_k'] = 10
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
    options['solver.linear_solver'] = 'ma57'  # if HSL is installed, otherwise 'mumps'

    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = True


    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'AAA_3DOF')
    trial.build()
    trial.optimize()

    # write the solution to CSV file, interpolating the collocation solution with given frequency.
    # trial.write_to_csv(filename = 'Ampyx_AP2_solution', frequency = 30)

    # draw some of the pre-coded plots for analysis
    trial.plot(['states', 'controls', 'constraints', 'quad'])

    # extract information from the solution for independent plotting or post-processing
    # here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
    plot_dict = trial.visualization.plot_dict
    outputs = plot_dict['outputs']
    time = plot_dict['time_grids']['ip']
    avg_power = plot_dict['power_and_performance']['avg_power']/1e3

    print('======================================')
    print('Average power: {} kW'.format(avg_power))
    print('======================================')

    plt.subplots(5, 1, sharex=True)
    plt.subplot(511)
    plt.plot(time, plot_dict['x']['coeff21'][0], label='CL21')
    plt.plot(time, plot_dict['x']['coeff31'][0], label='CL31')
    plt.ylabel('[-]')
    plt.legend()
    plt.grid(True)

    plt.subplot(512)
    plt.plot(time, plot_dict['x']['coeff21'][1], label='psi21')
    plt.plot(time, plot_dict['x']['coeff31'][1], label='psi31')
    plt.ylabel('[-]')
    plt.legend()
    plt.grid(True)

    plt.subplot(513)
    plt.plot(time, outputs['aerodynamics']['airspeed2'][0], label='Airspeed21')
    plt.plot(time, outputs['aerodynamics']['airspeed3'][0], label='Airspeed31')
    plt.ylabel('[m/s]')
    plt.legend()
    plt.hlines([10, 32], time[0], time[-1], linestyle='--', color='black')
    plt.grid(True)

    plt.subplot(514)
    plt.plot(time, outputs['aerodynamics']['LoverD2'][0], label='CD21')
    plt.plot(time, outputs['aerodynamics']['LoverD3'][0], label='CD31')
    plt.ylabel('[-]')
    plt.legend()
    plt.grid(True)

    plt.subplot(515)
    plt.plot(time, outputs['local_performance']['tether_force10'][0], label='Tether Force Magnitude')
    plt.ylabel('[N]')
    plt.xlabel('t [s]')
    plt.legend()
    plt.grid(True)

    # a block=False argument will automatically close the figures after they've been created
    plt.show()

    return trial

if __name__ == "__main__":
    trial = run()


