import numpy as np
import awebox as awe
import casadi as ca

def set_reference_options(user = 'A'):

    # make default options object
    options = {}

    # 6DOF Ampyx Ap2 model
    options['user_options.system_model.architecture'] = {1:0}   
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()

    # trajectory should be a single pumping cycle with initial number of five windings
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1

    # tether parameters
    options['params.tether.cd'] = 1.2
    options['params.tether.rho'] = 0.0046*4/(np.pi*0.002**2)
    # options['user_options.trajectory.fixed_params'] = {'diam_t': 2e-3}
    options['model.tether.use_wound_tether'] = False # don't model generator inertia
    options['model.tether.control_var'] = 'ddl_t' # tether acceleration control

    # tether drag model (more accurate than the Argatov model in Licitra2019)
    options['user_options.tether_drag_model'] = 'multi' 
    options['model.tether.aero_elements'] = 5

    # wind model
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # don't model generator
    options['model.model_bounds.wound_tether_length.include'] = False

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = True
    options['params.tether.max_stress'] = 3.6e9
    options['params.tether.stress_safety_factor'] = 3
    options['model.model_bounds.tether_force.include'] = False
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1800.0])
    options['params.tether.kappa'] = 1.0
  
    # flight envelope
    options['model.model_bounds.airspeed.include'] = False
    options['params.model_bounds.airspeed_limits'] = np.array([10, 32.0])
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 20.
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -20.
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 9.0
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -6.0

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = False

    # aircraft-tether anticollision
    options['model.model_bounds.rotation.include'] = True
    options['model.model_bounds.rotation.type'] = 'yaw'
    options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 40.0*np.pi/180.0])

    # variable bounds
    # if user == 'A':
    l_t_max = 700.0
    t_f_max = 70.0
    # elif user == 'B':
    #     l_t_max = 400.0
    #     t_f_max = 40.0
    options['model.system_bounds.x.l_t'] =  [10.0, l_t_max] # [m]
    options['model.system_bounds.x.dl_t'] =  [-15.0, 20.0] # [m/s]
    options['model.ground_station.ddl_t_max'] = 2.4 # [m/s^2]
    options['model.system_bounds.x.q'] =  [np.array([-ca.inf, -ca.inf, 100.0]), np.array([ca.inf, ca.inf, ca.inf])]
    options['model.system_bounds.theta.t_f'] =  [20.0, t_f_max] # [s]
    options['model.system_bounds.z.lambda'] =  [0., ca.inf] # [N/m]
    omega_bound = 50.0*np.pi/180.0
    options['model.system_bounds.x.omega'] = [np.array(3*[-omega_bound]), np.array(3*[omega_bound])]
    options['user_options.kite_standard.geometry.delta_max'] = np.array([20., 30., 30.]) * np.pi / 180.
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array([2., 2., 2.])

    # don't include induction effects
    options['user_options.induction_model'] = 'not_in_use'

    # initialization
    options['solver.initialization.groundspeed'] = 19.
    options['solver.initialization.inclination_deg'] = 45.
    options['solver.initialization.l_t'] = 400.0
    options['solver.initialization.cone_deg'] = 15.0
    options['solver.initialization.kite_dcm'] = 'aero_validity'

    # nlp discretization
    options['nlp.n_k'] = 100
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
    options['solver.linear_solver'] = 'ma57'

    # solver options
    options['solver.callback'] = True
    options['solver.callback_step'] = 1
    options['solver.mu_hippo'] = 1e-2
    options['solver.max_cpu_time'] = 1e5
    options['solver.max_iter'] = 1000

    # options['solver.mu_target'] = 1e-4
    options['solver.homotopy_method.gamma'] = 'penalty'
    options['solver.homotopy_method.psi'] = 'penalty'
    options['solver.homotopy_step.gamma'] = 0.1
    options['solver.homotopy_step.psi'] = 1.0

    # homotopy tuning
    options['solver.cost.fictitious.0'] = 1e3
    options['solver.cost.fictitious.1'] = 1e3
    options['solver.cost.gamma.0'] = 0
    options['solver.cost.gamma.1'] = 1e2
    options['solver.cost.psi.0'] = 0
    options['solver.cost.psi.1'] = 1e0
    options['solver.cost.tracking.0'] = 1e-1
    options['solver.cost_factor.power'] = 1e1
    options['solver.max_iter_hippo'] = 100
    options['visualization.cosmetics.plot_ref'] = True

    return options

def set_dual_kite_options(options):

    options['user_options.system_model.architecture'] = {1:0, 2:1, 3:1}
    options['solver.initialization.l_t'] = 640.0
    options['solver.initialization.inclination_deg'] = 25.0
    options['solver.initialization.cone_deg'] = 20.0
    options['solver.initialization.theta.l_s'] = 100.0
    options['solver.initialization.groundspeed'] = 50.0
    options['solver.initialization.psi0_rad'] = 0.0
    options['solver.initialization.theta.diam_s'] = 4e-3/np.sqrt(2)
    options['solver.initialization.theta.diam_t'] = 4e-3
    options['nlp.n_k'] = 100
    options['user_options.trajectory.lift_mode.windings'] = 3
    options['solver.mu_hippo'] = 1e-4

    # options['params.tether.kappa'] = 1e-2

    # options['model.system_bounds.theta.t_f'] =  [51.0, 70.0] # [s]
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
    options['nlp.phase_fix_reelout'] = 0.7
    # options['user_options.induction_model'] = 'actuator'
    # options['solver.cost.u_regularisation.0'] = 1e-1
    # options['solver.cost.beta.0'] = 1e2
    # options['solver.weights.domega'] = 1e9

    # options['model.model_bounds.tether_force.include'] = False
    # options['model.model_bounds.tether_stress.include'] = True
    options['model.model_bounds.anticollision.include'] = True
    options['model.model_bounds.anticollision.safety_factor'] = 4
    options['model.model_bounds.ellipsoidal_flight_region.include'] = False

    return options
