#!/usr/bin/python3
"""
MPC path-tracking simulation of pumping trajectory for the 6DOF megAWES reference rigid-wing aircraft using external forces.

Aircraft dimensions adapted from:
"Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems",
Dylan Eijkelhof, Roland Schmehl
Renewable Energy, Vol.196, pp. 137-150, 2022.

Aerodynamic model and constraints from BORNE project (Ghent University, UCLouvain, 2024)

:author: Thomas Haas, Ghent University, 2024 (adapted from Jochem De Schutter)

Current issues:
- option 'single_reelout' doesn't work with MPC, use 'simple' instead.
- option 'collocation_nodes' doesn't work with MPC, use 'shooting_nodes' instead.
"""

# imports
import casadi as ca
import numpy as np
import pickle
import csv
import sys
import os
import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
import matplotlib.pyplot as plt
matplotlib.use(DEFAULT_MPL_BACKEND)

# ----------------- import results from AWEbox ----------------- #
def csv2dict(fname):
    '''
    Import CSV outputs from awebox to Python dictionary
    '''

    # read csv file
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)

        # get fieldnames from DictReader object and store in list
        headers = reader.fieldnames

        # store data in columns
        columns = {}
        for row in reader:
            for fieldname in headers:
                val = row.get(fieldname).strip('[]')
                if val == '':
                    val = '0.0'
                columns.setdefault(fieldname, []).append(float(val))

    # add periodicity
    for fieldname in headers:
        columns.setdefault(fieldname, []).insert(0, columns[fieldname][-1])
    columns['time'][0] = 0.0

    return columns

# ----------------- load trajectory settings ----------------- #

# check output folder
output_folder="./tracking_files/"
if not os.path.isdir(output_folder):
    print("output folder doesn't not exist. Terminate simulation...")
    sys.exit()

# load trial results
awes = csv2dict(output_folder+'megawes_optimised_path_vlm.csv')

# load parameters and CasADi symbolic variables
filename = output_folder + 'simulation_variables.pckl'
with open(filename, 'rb') as handle:
    struct = pickle.load(handle)
x0 = struct['x0']
u0 = struct['u0']
z0 = struct['z0']
p0 = struct['p0']
w0 = struct['w0']
vars0 = struct['vars0']
scaling = vars0(struct['scaling'])

# load solver bounds
bounds = {}
for var in ['lbw', 'ubw', 'lbg', 'ubg']:
    filename = output_folder + var + '_bounds.pckl'
    with open(filename, 'rb') as handle:
        bounds[var] = pickle.load(handle)

# ----------------- initialize states/alg. vars ----------------- #

# set mpc options
mpc_opts = {}
mpc_opts['ipopt.linear_solver'] = 'ma57'
mpc_opts['ipopt.max_iter'] = 250
mpc_opts['ipopt.max_cpu_time'] = 10.
mpc_opts['ipopt.print_level'] = 0
mpc_opts['ipopt.sb'] = "yes"
mpc_opts['print_time'] = 0
mpc_opts['record_time'] = 1

# ----------------- load compiled CasADi functions ----------------- #

# Load function objects and solver
F_tgrids = ca.external('F_tgrids', output_folder + 'F_tgrids.so')
F_ref = ca.external('F_ref', output_folder + 'F_ref.so')
F_aero = ca.external('F_aero', output_folder + 'F_aero.so')
F_int = ca.external('F_int', output_folder + 'F_int.so')
helpers = ca.external('helper_functions', output_folder + 'helper_functions.so')
solver = ca.nlpsol('solver', 'ipopt', output_folder + 'mpc_solver.so', mpc_opts)

# ----------------- initialize states and algebraic variabless ----------------- #

# Scaled initial states
x0['q10'] = np.array([awes['x_q10_'+str(i)][0] for i in range(3)]) / scaling['x', 'q10',0]
x0['dq10'] = np.array([awes['x_dq10_'+str(i)][0] for i in range(3)]) / scaling['x', 'dq10',0]
x0['omega10'] = np.array([awes['x_omega10_'+str(i)][0]/ scaling['x', 'omega10', i].full().squeeze() for i in range(3)]) 
x0['r10'] = np.array([awes['x_r10_'+str(i)][0] for i in range(9)]) / scaling['x', 'r10']
x0['delta10'] = np.array([awes['x_delta10_'+str(i)][0]/scaling['x', 'delta10',i].full().squeeze() for i in range(3)])
x0['l_t'] = np.array(awes['x_l_t_0'][0]) / scaling['x', 'l_t']
x0['dl_t'] = np.array(awes['x_dl_t_0'][0]) / scaling['x', 'dl_t']

# Scaled algebraic vars
z0['z'] = np.array(awes['z_lambda10_0'][0]) / scaling['z', 'lambda10']

# ----------------- run simulation ----------------- #

# simulation settings
t_f = awes['time'][-1] # trajectory period
t_end = 1 * t_f # simulation horizon
ts = 0.1 # sampling time
N_dt = 20 # time steps per sampling time
dt = ts/N_dt # time step
N_steps = int(t_end / dt)

# initialize simulation
tsim = [0.0]
xsim = [x0.cat.full().squeeze()]
zsim = [z0.cat.full().squeeze()]
usim = []
fsim = []
msim = []
stats = []

# MPC parameters
nx = 23
nu = 10
weights_x = x0(1.)
weights_x['delta10'] = 1e-2
weights_x['l_t'] = 100
weights_x['dl_t'] = 100
Q = weights_x.cat
R = 1e-2*np.ones((nu, 1))
P = weights_x.cat
Z = 1000*np.ones((1, 1))
u_ref = 12.

# loop through time steps
N_mpc_fail = 0
N_max_fail = 10 # stop count for failed MPC evaluations
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
        sol = solver(x0=w0, lbx=bounds['lbw'], ubx=bounds['ubw'], lbg=bounds['lbg'], ubg=bounds['ubg'],
                       p=ca.vertcat(x0, ref, u_ref, Q, R, P, Z))

        # MPC stats
        stats.append(solver.stats())
        if stats[-1]["success"]==False:
           N_mpc_fail += 1

        # MPC outputs
        out = helpers(V=sol['x'])

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

# power outputs
lam = np.array([z[-1] for z in zsim]) * scaling['z', 'lambda10'].full()[0][0]
l_t = np.array([x[-2] for x in xsim]) * scaling['x', 'l_t'].full()[0][0]
dl_t = np.array([x[-1] for x in xsim]) * scaling['x', 'dl_t'].full()[0][0]
P_inst = lam * l_t * dl_t
P_ave_ext = np.sum(P_inst[1:] * np.array([tsim[i+1]-tsim[i] for i in range(0,len(tsim)-1)]))/t_end

# reference power
time = np.array(awes['time'])
P_inst_ref = np.array(awes['outputs_performance_p_current_0'])
P_ave_ref = np.sum(P_inst_ref[1:]*(time[1:]-time[:-1]))/t_f

# end of simulation
print("end of simulation...")

# ----------------- specific plots ----------------- #

# number of cycles
from math import ceil
N_cycles = ceil(t_end/t_f)
reference_time = np.concatenate([np.array(awes['time']) + i*t_f for i in range(N_cycles)])

# Legend labels
legend_labels = ['ref (AVL, P={:.2f}MW)'.format(1e-6*P_ave_ref), 'MPC (ALM, P={:.2f}MW)'.format(1e-6*P_ave_ext)]

# plot 3D flight path
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
ax.plot(awes['x_q10_0'], awes['x_q10_1'], awes['x_q10_2'])
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
l[-1].set_color('r')
ax.legend([l[0], l[-1]], legend_labels, fontsize=12)
fig.savefig('outputs_megawes_external_forces_simulation_plot_3dpath.png')

# plot power profile
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
ax.plot(reference_time, np.tile(1e-6*np.array(awes['outputs_performance_p_current_0']), N_cycles))
ax.plot(tsim, 1e-6*P_inst)
l = ax.get_lines()
l[0].set_color('b')
l[-1].set_color('r')
ax.legend(legend_labels, loc=1, fontsize=12) # ['tracking mpc', 'built-in mpc', 'reference']
ax.tick_params(axis='both', labelsize=12)
ax.set_xlabel('t [s]', fontsize=12)
ax.set_ylabel('P [MW]', fontsize=12)
ax.grid()
fig.savefig('outputs_megawes_external_forces_simulation_plot_power.png')

# plot actuators
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
for k in range(3):
    ax[k].plot(reference_time, np.tile((180./np.pi)*np.array(awes['x_delta10_'+str(k)]), N_cycles))
    ax[k].plot(tsim, (180./np.pi)*np.array([x[18+k]*scaling['x', 'delta10', k] for x in xsim]).squeeze())
ax[-1].plot(reference_time, np.tile(awes['x_dl_t_0'], N_cycles))
ax[-1].plot(tsim, [x[-1]*scaling['x','dl_t'].full()[0][0] for x in xsim])
for k in range(4):
    l = ax[k].get_lines()
    l[0].set_color('b')
    l[-1].set_color('r')
ax[0].legend(legend_labels, fontsize=12)
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
fig.savefig('outputs_megawes_external_forces_simulation_plot_actuators.png')

# plot control inputs
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95)
for k in range(3):
   ax[k].step(reference_time, np.tile((180. / np.pi) * np.array(awes['u_ddelta10_' + str(k)]), N_cycles), where='post')
   ax[k].step(tsim[:-1], (180./np.pi)*np.array([u[6+k]*scaling['u', 'ddelta10', k] for u in usim]).squeeze(), where='post')
ax[-1].step(reference_time, np.tile(awes['u_ddl_t_0'], N_cycles), where='post')
ax[-1].step(tsim[:-1], np.array([u[-1]*scaling['u', 'ddl_t'].full()[0][0] for u in usim]), where='post')
for k in range(4):
    l = ax[k].get_lines()
    l[0].set_color('b')
    l[-1].set_color('r')
ax[0].legend(legend_labels, fontsize=12)
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
fig.savefig('outputs_megawes_external_forces_simulation_plot_controls.png')

# plot MPC performance
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
fig.savefig('outputs_megawes_external_forces_simulation_plot_mpc_performance.png')
print('end')
# ----------------- end ----------------- #
