#!/usr/bin/python3
"""
MPC tracking of pumping trajectory for the 6DOF megAWES reference rigid-wing aircraft using external forces.

Aircraft dimensions adapted from:
"Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems",
Dylan Eijkelhof, Roland Schmehl
Renewable Energy, Vol.196, pp. 137-150, 2022.

Aerodynamic model and constraints from BORNE project (Ghent University, UCLouvain, 2024)

:author: Thomas Haas, Ghent University, 2024 (adapted from Jochem De Schutter)

Current issues:
- option 'collocation_nodes' doesn't work with MPC, use 'shooting_nodes' instead.
"""

# imports
import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
import awebox.pmpc as pmpc
import casadi as ca
import numpy as np
import os
import pickle
import copy
import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)
import matplotlib.pyplot as plt
from awebox.opts.kite_data.megawes_settings import set_megawes_path_generation_settings, set_megawes_path_tracking_settings


# ----------------- compile dependencies ----------------- #

# compilation flag
compilation_flag = False

# ----------------- set trajectory options ----------------- #

# indicate desired system architecture
aero_model='VLM'
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_path_generation_settings(aero_model, options)

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # ('single_reelout': positive/null reel-out during generation)
options['user_options.trajectory.lift_mode.windings'] = 1 # number of loops
options['model.system_bounds.theta.t_f'] = [1., 20.] # cycle period [s]

# indicate desired wind environment
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 12.
options['params.wind.z_ref'] = 100.
options['params.wind.log_wind.z0_air'] = 0.0002

# indicate numerical nlp details
options['nlp.n_k'] = 40 # approximately 40 per loop
options['nlp.collocation.u_param'] = 'zoh' # constant control inputs
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'
options['nlp.collocation.ineq_constraints'] = 'shooting_nodes' # ('collocation_nodes': constraints on Radau collocation nodes - Not available in MPC)

# compile subfunctions to speed up construction time
options['nlp.compile_subfunctions'] = True # False

# ----------------- create reference trajectory ----------------- #

# options for path optimization
optimization_options = copy.deepcopy(options)

# optimize MegAWES flight path
trial = awe.Trial(optimization_options, 'MegAWES')
trial.build()
trial.optimize(options_seed=optimization_options)

# reference power
P_ave_ref = trial.visualization.plot_dict['power_and_performance']['avg_power'].full()[0][0]

# ----------------- set tracking options for MPC and integrator ----------------- #

# simulation horizon
t_end = 1*trial.visualization.plot_dict['time_grids']['x'][-1].full().squeeze()

# adjust options for path tracking (incl. aero model)
tracking_options = copy.deepcopy(optimization_options)
tracking_options = set_megawes_path_tracking_settings(aero_model='ALM', options = tracking_options)

# don't use subfunction compilation as it slows down MPC evaluation time
tracking_options['nlp.compile_subfunctions'] = False
if compilation_flag:
    tracking_options['nlp.compile_subfunctions'] = False # incompatibility, to be sure

# feed-in tracking options to trial
trial.options_seed = tracking_options

# ----------------- create MPC controller with tracking-specific options ----------------- #

# set MPC options
ts = 0.1 # sampling time (length of one MPC window)
N_mpc = 20 # MPC horizon (number of MPC windows in prediction horizon)

# simulation options
N_dt = 20 # integrator steps within one sampling time
N_sim = int(round(t_end/ts)) # number of MPC evaluations

# create MPC options
mpc_opts = awe.Options()
mpc_opts['mpc']['N'] = N_mpc
mpc_opts['mpc']['max_iter'] = 1000
mpc_opts['mpc']['max_cpu_time'] = 10.
mpc_opts['mpc']['homotopy_warmstart'] = True
mpc_opts['mpc']['terminal_point_constr'] = False
mpc_opts['mpc']['ip_type'] = 'linear'

if tracking_options['nlp.compile_subfunctions']:
    mpc_opts['mpc']['expand'] = False # incompatible

# MPC weights
nx = 23
nu = 10
nz = 1
weights_x = trial.model.variables_dict['x'](1.)
weights_x['delta10'] = 1e-2
weights_x['l_t'] = 100
weights_x['dl_t'] = 100
Q = weights_x.cat
R = 1e-2*np.ones((nu, 1))
P = weights_x.cat
Z = 1000*np.ones((nz, 1))

# create PMPC object (requires feed-in of tracking options to trial)
mpc = pmpc.Pmpc(mpc_opts['mpc'], ts, trial)

# ----------------- create integrator for dynamics ----------------- #

# specify modified options (Turn ON flag for external forces)
int_opts = copy.deepcopy(tracking_options)
int_opts['model.aero.fictitious_embedding'] = 'substitute'
integrator_options = awe.opts.options.Options()
integrator_options.fill_in_seed(int_opts)

# re-build architecture
architecture = awe.mdl.architecture.Architecture(integrator_options['user_options']['system_model']['architecture'])
integrator_options.build(architecture)

# re-build model
system_model = awe.mdl.model.Model()
system_model.build(integrator_options['model'], architecture)

# get model scaling (alternative: = trial.model.scaling)
scaling = system_model.scaling

# get model DAE
dae = system_model.get_dae()

# make casadi (collocation) integrators from DAE
dae.build_rootfinder() # Build root finder of DAE

# get optimized parameters (theta.keys() = ['diam_t', 't_f'])
theta = system_model.variables_dict['theta'](0.0)
theta['diam_t'] = trial.optimization.V_opt['theta', 'diam_t'] #Scaled!
theta['t_f'] = 1.0

# get numerical parameters (params.keys() = ['theta0', 'phi'])
params = system_model.parameters(0.0)
param_num = integrator_options['model']['params']
for param_type in list(param_num.keys()):
    if isinstance(param_num[param_type], dict):
        for param in list(param_num[param_type].keys()):
            if isinstance(param_num[param_type][param], dict):
                for subparam in list(param_num[param_type][param].keys()):
                    params['theta0', param_type, param, subparam] = param_num[param_type][param][subparam]
            else:
                params['theta0', param_type, param] = param_num[param_type][param]
    else:
        params['theta0', param_type] = param_num[param_type]
params['phi', 'gamma'] = 1

# build parameter structure from DAE (p0.keys() = ['u', 'theta', 'param'])
p0 = dae.p(0.0)
p0['theta'] = theta.cat
p0['param'] = params

# create integrator (integrators: rk4 (4th order Runge-Kutta) or ee1 (1st order explicit Euler))
dt = float(ts/N_dt) # time step length
integrator = awe_integrators.ee1root('F', dae.dae, dae.rootfinder, {'tf': dt, 'number_of_finite_elements': 1})

# ----------------- create CasADI function of integrator ----------------- #

# create symbolic structures for integrators and aerodynamics model
nparam = 150
x0_init = ca.MX.sym('x', nx)
u0_init = ca.MX.sym('u', nu)
z0_init = dae.z(0.0)
p0_init = ca.MX.sym('p', nparam)

# outputs
integrator_outputs = integrator(x0=x0_init, z0=z0_init, p=p0_init)

# evaluate integrator
z0_out = integrator_outputs['zf']
x0_out = integrator_outputs['xf']
q0_out = integrator_outputs['qf']

# Create CasADi function
F_int = ca.Function('F_int', [x0_init, p0_init], [x0_out, z0_out, q0_out], ['x0', 'p0'], ['xf', 'zf', 'qf'])

# ----------------- create CasADi function for external aerodynamics ----------------- #
# F_aero: Returns f_earth and m_body for specified states and controls

# solve for z0 with DAE rootfinder (p0 already created)
z0_rf = dae.z(dae.rootfinder(z0_init, x0_init, p0))

vars0_init = ca.vertcat(
   x0_init,
   z0_rf['xdot'],
   u0_init,
   z0_rf['z'],
   system_model.variables_dict['theta'](0.0)
)

# outputs
outputs = system_model.outputs(system_model.outputs_fun(vars0_init, p0['param']))

# extract forces and moments from outputs
F_ext_evaluated = outputs['aerodynamics', 'f_aero_earth1']
M_ext_evaluated = outputs['aerodynamics', 'm_aero_body1']

# Create CasADi function
F_aero = ca.Function('F_aero', [x0_init, u0_init], [F_ext_evaluated, M_ext_evaluated], ['x0', 'u0'], ['F_ext', 'M_ext'])

# ----------------- build CasADi time functions for MPC ----------------- #
# F_tgrids: Returns time grid of simulation horizon from collocation grid

# time grid in symbolic form
t0 = ca.MX.sym('t0')

# reference interpolation time grid in symbolic form
t_grid = ca.MX.sym('t_grid', mpc.t_grid_coll.shape[0])
t_grid_x = ca.MX.sym('t_grid_x', mpc.t_grid_x_coll.shape[0])
t_grid_u = ca.MX.sym('t_grid_u', mpc.t_grid_u.shape[0])

# time function
F_tgrids = ca.Function('F_tgrids',[t0], [t0 + mpc.t_grid_coll, t0 + mpc.t_grid_x_coll, t0 + mpc.t_grid_u],
                       ['t0'],['tgrid','tgrid_x','tgrid_u'])

# ----------------- build CasADi reference function for MPC ----------------- #
# F_ref: Returns tracked reference on specified time grid

# reference function
ref = mpc.get_reference(t_grid, t_grid_x, t_grid_u)

# reference function
F_ref = ca.Function('F_ref', [t_grid, t_grid_x, t_grid_u], [ref], ['tgrid', 'tgrid_x', 'tgrid_u'],['ref'])

# ----------------- build CasADi helper functions for MPC ----------------- #
# helper_functions: Return initial guess and controls

# shift solution
V = mpc.trial.nlp.V
V_init = [V['theta'], V['phi']]
for k in range(N_mpc-1):
   V_init.append(V['x',k+1])
   if mpc_opts['mpc']['u_param'] == 'zoh':
       V_init.append(V['u', k+1])
       V_init.append(V['xdot', k+1])
       V_init.append(V['z', k+1])
   for j in range(mpc_opts['mpc']['d']):
       V_init.append(V['coll_var', k+1, j, 'x'])
       V_init.append(V['coll_var', k+1, j, 'z'])
       if mpc_opts['mpc']['u_param'] == 'poly':
           V_init.append(V['coll_var', k+1, j, 'u'])

# copy final interval
V_init.append(V['x', N_mpc-1])
if mpc_opts['mpc']['u_param'] == 'zoh':
   V_init.append(V['u', N_mpc-1])
   V_init.append(V['xdot', N_mpc-1])
   V_init.append(V['z', N_mpc-1])
for j in range(mpc_opts['mpc']['d']):
   V_init.append(V['coll_var', N_mpc-1, j, 'x'])
   V_init.append(V['coll_var', N_mpc-1, j, 'z'])
   if mpc_opts['mpc']['u_param'] == 'poly':
       V_init.append(V['coll_var', N_mpc-1, j, 'u'])
V_init.append(V['x',N_mpc])

# shifted solution
V_shifted = ca.vertcat(*V_init)

# first control
if mpc_opts['mpc']['u_param'] == 'poly':
   u0_shifted = ca.mtimes(mpc.trial.nlp.Collocation.quad_weights[np.newaxis,:], ca.horzcat(*V['coll_var',0,:,'u']).T).T
elif mpc_opts['mpc']['u_param'] == 'zoh':
   u0_shifted = V['u',0]
u0_shifted = mpc.trial.model.variables_dict['u'](u0_shifted)

# controls
u_si = []
for name in list(mpc.trial.model.variables_dict['u'].keys()):
   u_si.append(u0_shifted[name]*scaling['u', name])
u_si = ca.vertcat(*u_si)

# helper function
helper_functions = ca.Function('helper_functions',[V], [V_shifted, u_si], ['V'], ['V_shifted', 'u0'])

# ----------------- save and compile dependencies ----------------- #
if compilation_flag:

    # create output folder
    output_folder="./tracking_files/"
    if not os.path.isdir(output_folder):
        os.system('mkdir '+output_folder)

    # save trial outputs (MPC requires attitude in DCM representation)
    trial.write_to_csv(output_folder+'megawes_optimised_path_'+aero_model.lower(), rotation_representation="dcm") 

    # save solver bounds
    for var in list(mpc.solver_bounds.keys()):
        filename = output_folder + var + '_bounds.pckl'
        with open(filename, 'wb') as handle:
            pickle.dump(mpc.solver_bounds[var], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # compile mpc solver
    print("start compilation mpc solver...")
    src_filename = output_folder + 'mpc_solver.c'
    lib_filename = output_folder + 'mpc_solver.so'
    mpc.solver.generate_dependencies('mpc_solver.c')
    os.system("mv ./mpc_solver.c" + " " + src_filename)
    os.system("gcc -fPIC -shared " + src_filename + " -o " + lib_filename)
    print("mpc solver compilation done...")

    # compile dependencies F_int
    src_filename = output_folder + 'F_int.c'
    lib_filename = output_folder + 'F_int.so'
    F_int.generate('F_int.c')
    os.system("mv F_int.c "+ src_filename)
    os.system("gcc -fPIC -shared -O3 "+ src_filename + " -o " + lib_filename)

    # compile dependencies F_aero
    src_filename = output_folder + 'F_aero.c'
    lib_filename = output_folder + 'F_aero.so'
    F_aero.generate('F_aero.c')
    os.system("mv F_aero.c "+ src_filename)
    os.system("gcc -fPIC -shared -O3 "+ src_filename + " -o " + lib_filename)

    # compile dependencies F_tgrids
    src_filename = output_folder + 'F_tgrids.c'
    lib_filename = output_folder + 'F_tgrids.so'
    F_tgrids.generate('F_tgrids.c')
    os.system("mv F_tgrids.c"+" "+src_filename)
    os.system("gcc -fPIC -shared -O3 "+src_filename+" -o "+lib_filename)

    # compile dependencies F_ref
    src_filename = output_folder + 'F_ref.c'
    lib_filename = output_folder + 'F_ref.so'
    F_ref.generate('F_ref.c')
    os.system("mv F_ref.c"+" "+src_filename)
    os.system("gcc -fPIC -shared "+src_filename+" -o "+lib_filename)

    # compile dependencies helper_functions
    src_filename = output_folder + 'helper_functions.c'
    lib_filename = output_folder + 'helper_functions.so'
    helper_functions.generate('helper_functions.c')
    os.system("mv helper_functions.c"+" "+src_filename)
    os.system("gcc -fPIC -shared -O3 "+src_filename+" -o "+lib_filename)

    # create CasADi symbolic variables and dicts
    x0 = system_model.variables_dict['x'](0.0)  # initialize states
    u0 = system_model.variables_dict['u'](0.0)  # initialize controls
    z0 = dae.z(0.0)  # algebraic variables initial guess
    w0 = mpc.w0
    vars0 = system_model.variables(0.0)
    vars0['theta'] = system_model.variables_dict['theta'](0.0)
    
    # gather into dict
    simulation_variables = {'x0':x0, 'u0':u0, 'z0':z0, 'p0':p0, 'w0':w0,
                            'vars0':vars0, 'scaling':scaling.cat.full()}
    
    # save simulation variables
    filename = output_folder + 'simulation_variables.pckl'
    with open(filename, 'wb') as handle:
        pickle.dump(simulation_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------- initialize simulation ----------------- #

# create CasADi symbolic variables and dicts
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess
w0 = mpc.w0
vars0 = system_model.variables(0.0)
vars0['theta'] = system_model.variables_dict['theta'](0.0)
bounds = mpc.solver_bounds

# load trial parameters
t_f = trial.visualization.plot_dict['time_grids']['x'][-1].full().squeeze()

# Scaled initial states
plot_dict = trial.visualization.plot_dict
x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x','q10']
x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x','dq10']
x0['omega10'] = np.array(plot_dict['x']['omega10'])[:, -1] / scaling['x','omega10']
x0['r10'] = np.array(plot_dict['x']['r10'])[:, -1] / scaling['x','r10']
x0['delta10'] = np.array(plot_dict['x']['delta10'])[:, -1] / scaling['x','delta10']
x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x','l_t']
x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x','dl_t']

# Scaled algebraic vars
z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z','lambda10']

# ----------------- run simulation ----------------- #

# initialize simulation
tsim = [0.0]
xsim = [x0.cat.full().squeeze()]
zsim = [z0.cat.full().squeeze()]
usim = []
fsim = []
msim = []
stats = []
N_mpc_fail = 0

# Loop through time steps
N_max_fail = 10 # stop count for failed MPC evaluations
N_steps = int(t_end / dt)
for k in range(N_steps):

    # current time
    current_time = k * dt

    # evaluate MPC
    if (k % N_dt) < 1e-6:

        # ----------------- evaluate mpc step ----------------- #
        print(str(k)+'/'+str(N_steps)+': evaluate MPC step')

        # initial guess
        if k == 0:
            w0 = w0.cat.full().squeeze().tolist()

        # get reference time
        tgrids = F_tgrids(t0 = current_time)
        for grid in list(tgrids.keys()):
            tgrids[grid] = ca.vertcat(*list(map(lambda x: x % t_f, tgrids[grid].full()))).full().squeeze()

        # get reference
        ref = F_ref(tgrid = tgrids['tgrid'], tgrid_x = tgrids['tgrid_x'], tgrid_u = tgrids['tgrid_u'])['ref']
 
        # solve MPC problem
        u_ref = tracking_options['user_options.wind.u_ref']
        sol = mpc.solver(x0=w0, lbx=bounds['lbw'], ubx=bounds['ubw'], lbg=bounds['lbg'], ubg=bounds['ubg'],
                        p=ca.vertcat(x0, ref, u_ref, Q, R, P, Z))

        # MPC stats
        stats.append(mpc.solver.stats())
        if stats[-1]["success"]==False:
            N_mpc_fail += 1

        # MPC outputs
        out = helper_functions(V=sol['x'])

        # write shifted initial guess
        V_shifted = out['V_shifted']
        w0 = V_shifted.full().squeeze().tolist()

        # retrieve new controls
        u0_call = out['u0']

        # fill in controls
        u0['ddelta10'] = u0_call[6:9] / scaling['u', 'ddelta10'] # scaled!
        u0['ddl_t'] = u0_call[-1] / scaling['u', 'ddl_t'] # scaled!

        # message
        print("iteration=" + "{:3d}".format(k + 1) + "/" + str(N_steps) + ", t=" + "{:.4f}".format(current_time) + " > compute MPC step")

    else:
        # message
        print("iteration=" + "{:3d}".format(k + 1) + "/" + str(N_steps) + ", t=" + "{:.4f}".format(current_time))

    # ----------------- evaluate aerodynamics ----------------- #

    # evaluate forces and moments
    aero_out = F_aero(x0=x0, u0=u0)

    # fill in forces and moments
    u0['f_fict10'] = aero_out['F_ext'] / scaling['u', 'f_fict10']  # external force in inertial frame
    u0['m_fict10'] = aero_out['M_ext'] / scaling['u', 'm_fict10']  # external moment in body-fixed frame

    # fill controls and aerodynamics into dae parameters
    p0['u'] = u0

    # ----------------- evaluate system dynamics ----------------- #

    # evaluate dynamics with integrator
    out = F_int(x0=x0, p0=p0)
    z0 = out['zf']
    x0 = out['xf']
    qf = out['qf']

    # Simulation outputs
    tsim.append((k+1) * dt)
    xsim.append(out['xf'].full().squeeze())
    zsim.append(out['zf'].full().squeeze())
    usim.append([u0.cat.full().squeeze()][0])
    fsim.append(aero_out['F_ext'].full().squeeze())
    msim.append(aero_out['M_ext'].full().squeeze())

    if N_mpc_fail == N_max_fail:
        print(str(N_max_fail)+" failed MPC evaluations: Interrupt loop")
        break

# generated power
lam = np.array([z[-1] for z in zsim]) * scaling['z', 'lambda10'].full()[0][0]
l_t = np.array([x[-2] for x in xsim]) * scaling['x', 'l_t'].full()[0][0]
dl_t = np.array([x[-1] for x in xsim]) * scaling['x', 'dl_t'].full()[0][0]
P_inst = lam * l_t * dl_t
P_ave_ext = np.sum(P_inst[1:] * np.array([tsim[i+1]-tsim[i] for i in range(0,len(tsim)-1)]))/tsim[-1]

# end of simulation
print("end of simulation...")

# ----------------- specific plots ----------------- #

# Legend labels
legend_labels = ['reference (VLM, P={:.2f}MW)'.format(1e-6*P_ave_ref), 'ext. MPC (ALM, P={:.2f}MW)'.format(1e-6*P_ave_ext)]

# plot 3D flight path
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(8,8)
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
ax = fig.get_axes()[0]
scaling_q = scaling['x', 'q10',0].full().squeeze()
ax.plot([x[0]*scaling_q for x in xsim], [x[1]*scaling_q for x in xsim], [x[2]*scaling_q for x in xsim])
ax.tick_params(labelsize=12)
ax.set_xlabel(ax.get_xlabel(), fontsize=12)
ax.set_ylabel(ax.get_ylabel(), fontsize=12)
ax.set_zlabel(ax.get_zlabel(), fontsize=12)
ax.set_xlim([0,400])
ax.set_ylim([-200,200])
ax.set_zlim([0,400])
ax.view_init(azim=-70, elev=20)
l = ax.get_lines()
l[0].set_color('b')
l[-1].set_color('g')
ax.get_legend().remove()
ax.legend([l[0], l[-1]], legend_labels, fontsize=12)
fig.suptitle("")
fig.savefig('outputs_megawes_external_forces_tracking_plot_3dpath.png')

# plot power profile
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
ax.plot(trial.visualization.plot_dict['time_grids']['ip'], 1e-6*trial.visualization.plot_dict['outputs']['performance']['p_current'][0])
ax.plot(tsim, 1e-6*P_inst)
l = ax.get_lines()
l[0].set_color('b')
l[-1].set_color('g')
ax.legend(legend_labels, loc=1, fontsize=12)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlabel('t [s]', fontsize=12)
ax.set_ylabel('P [MW]', fontsize=12)
ax.grid()
fig.savefig('outputs_megawes_external_forces_tracking_plot_power.png')

# plot actuators
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
for k in range(3):
    ax[k].plot(trial.visualization.plot_dict['time_grids']['ip'], (180./np.pi)*trial.visualization.plot_dict['x']['delta10'][k])
    ax[k].plot(tsim, (180. / np.pi) * scaling['x','delta10', k].full()[0][0]*np.array([x[18 + k] for x in xsim]))
ax[-1].plot(trial.visualization.plot_dict['time_grids']['ip'], trial.visualization.plot_dict['x']['dl_t'][0], 'b')
ax[-1].plot(tsim, [x[-1]*scaling['x', 'dl_t'].full()[0][0] for x in xsim])
for k in range(4):
    l = ax[k].get_lines()
    l[0].set_color('b')
    l[-1].set_color('g')
ax[0].legend(legend_labels, fontsize=12) # ['tracking mpc', 'built-in mpc', 'reference']
ax[0].plot([0, t_end], [-15, -15], 'k--')
ax[0].plot([0, t_end], [15, 15], 'k--')
for k in range(1,3):
    ax[k].plot([0,t_end], [-7.5,-7.5], 'k--')
    ax[k].plot([0,t_end], [7.5,7.5], 'k--')
ax[-1].plot([0,t_end], [-12,-12], 'k--')
ax[-1].plot([0,t_end], [12,12], 'k--')
for axes, var in zip(ax, ['da','de','dr','dlt']):
    axes.tick_params(axis='both', labelsize=12)
    axes.set_ylabel(var, fontsize=12)
    axes.grid()
ax[-1].set_xlabel('t [s]', fontsize=12)
fig.savefig('outputs_megawes_external_forces_tracking_plot_actuators.png')

# plot control inputs
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
for k in range(3):
    ax[k].step(trial.visualization.plot_dict['time_grids']['ip'], (180./np.pi)*trial.visualization.plot_dict['u']['ddelta10'][k], where='post')
    ax[k].step(tsim[:-1], (180./np.pi)*np.array([u[6+k] for u in usim]), where='post')
ax[-1].step(trial.visualization.plot_dict['time_grids']['ip'], trial.visualization.plot_dict['u']['ddl_t'][0], where='post')
ax[-1].step(tsim[:-1], np.array([u[-1]*scaling['u', 'ddl_t'].full()[0][0] for u in usim]), where='post')
for k in range(4):
    l = ax[k].get_lines()
    l[0].set_color('b')
    l[-1].set_color('g')
ax[0].legend(legend_labels, fontsize=12) # ['tracking mpc', 'built-in mpc', 'reference']
for k in range(3):
    ax[k].plot([0,t_end], [-25,-25], 'k--')
    ax[k].plot([0,t_end], [25,25], 'k--')
ax[-1].plot([0,t_end], [-2.5,-2.5], 'k--')
ax[-1].plot([0,t_end], [2.5,2.5], 'k--')
for axes, var in zip(ax, ['dda','dde','ddr','ddlt']):
    axes.tick_params(axis='both', labelsize=12)
    axes.set_ylabel(var, fontsize=12)
    axes.grid()
ax[-1].set_xlabel('t [s]', fontsize=12)
fig.savefig('outputs_megawes_external_forces_tracking_plot_controls.png')

# ----------------- evaluate MPC performance ----------------- #
def visualize_mpc_perf(stats):
   # Visualize MPC stats
   fig = plt.figure(figsize=(10., 5.))
   ax1 = fig.add_axes([0.12, 0.12, 0.75, 0.75])
   ax2 = ax1.twinx()

   # MPC stats
   eval =  np.arange(1, len(stats) + 1)
   status =  np.array([s['return_status'] for s in stats])
   walltime =  np.array([s['t_proc_total'] for s in stats])
   iterations =  np.array([s['iter_count'] for s in stats])

   # Create masks
   mask1 = status == 'Solve_Succeeded'
   mask2 = status == 'Solved_To_Acceptable_Level'
   mask3 = status == 'Maximum_Iterations_Exceeded'
   mask4 = status == 'Infeasible_Problem_Detected'
   mask5 = status == 'Maximum_CpuTime_Exceeded'
   mask_all = np.array([True] * eval.max())
   mask_list = [mask1, mask2, mask3, mask4, mask5]
   mask_name = ['Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Maximum_Iterations_Exceeded',
                'Infeasible_Problem_Detected', 'Maximum_CpuTime_Exceeded']
   mask_clr = ['tab:green', 'tab:blue', 'tab:purple', 'tab:red', 'tab:orange']

   # Plot
   for mask, clr, name in zip(mask_list, mask_clr, mask_name):
       ax1.bar(eval[mask], iterations[mask], color=clr, label=name)
   ax2.plot(eval, walltime, '-k')  # , markeredgecolor='k', markerfacecolor=clr, label=name)

   # Layout
   ax1.set_title('Performance of MPC evaluations', fontsize=14)
   ax1.set_xlabel('Evaluations', fontsize=14)
   ax1.set_ylabel('Iterations', fontsize=14)
   ax2.set_ylabel('Walltime [s]', fontsize=14)
   ax1.set_xlim([1, eval.max()])
   ax1.legend(loc=2)
   ax1.set_ylim([0,250])
   ax2.set_ylim([0,6])

   return fig

fig = visualize_mpc_perf(stats)
fig.savefig('outputs_megawes_external_forces_tracking_plot_mpc_performance.png')

# ----------------- end ----------------- #
