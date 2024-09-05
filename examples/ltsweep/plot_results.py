import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import awebox.opts.options as opts
from collections import OrderedDict
import math
# set-up TeX-settings
# mpl.use("pgf")
# import matplotlib.pyplot as plt
# pgf_with_pdflatex = {
#     "font.family": "serif", # use serif/main font for text elements
#     "font.serif": [], # use latex default serif font
#     "text.usetex": True, # use inline math for ticks
#     "pgf.rcfonts": False, # don't setup fonts from rc parameters
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": [
#     # r"\usepackage[utf8x]{inputenc}",
#     # r"\usepackage[T1]{fontenc}",
#     # r"\usepackage{amsmath}",
#     # r"\usepackage{cmbright}",
#     ],
#     "text.latex.preamble":[r'\usepackage{lmodern}']
# }
# plt.rcParams.update(pgf_with_pdflatex)
import matplotlib.pyplot as plt
import pickle
import casadi as ca
from tabulate import tabulate
import plot_airplane
import copy
import csv

figfolder = './'
filefolder = './'
textheight = 7.00137 # inches
textwidth  = 9.46637 # inches

# =======================================
# FIGURE 4: Local solution comparison
# =======================================

name = 'lt1000'
traj_dict = {}
# List of keys
with open(filefolder+name+'.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    lines = list(reader)
    key_list = lines[0]

# Retrieve data
with open(filefolder+name+'.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = list(reader)
    traj_dict[name] = {}
    for key in key_list:
        tmp = np.array([row[key] for row in lines])
        tmp[tmp == ''] = '0'
        if tmp[0][0] == '[':
            x = []
            for k in range(len(tmp)):
                x.append(float(tmp[k][1:-1]))
            x = np.array(x)
        else:    
            x = tmp.astype(float)
        traj_dict[name][key] = x

for key in traj_dict.keys():

    traj = traj_dict[key]

    c_max = 0.0
    dc_max = 0.0
    cr_max = 0.0

    # select trajectory
    homotopy = traj
    ref_state = 'x_q10'
    ref_state_0 = [x for x in homotopy[ref_state+'_0']]
    ref_state_1 = [x for x in homotopy[ref_state+'_1']]
    ref_state_2 = [x for x in homotopy[ref_state+'_2']]
    ref_rot = 'x_r10'

    # consistency conditions
    for kk in range(len(traj['x_q10_0'])):
        q10 = ca.vertcat(traj['x_q10_0'][kk], traj['x_q10_1'][kk],traj['x_q10_2'][kk])
        dq10 = ca.vertcat(traj['x_dq10_0'][kk], traj['x_dq10_1'][kk],traj['x_dq10_2'][kk])
        l_t = traj['x_l_t_0'][kk]
        dl_t = traj['x_dl_t_0'][kk]
        c = np.abs(0.5 * (ca.mtimes(q10.T, q10) - l_t**2))[0][0]
        dc = np.abs(ca.mtimes(q10.T, dq10) - l_t*dl_t)[0][0]

        Rvec10 = ca.vertcat(*[traj['x_r10_'+str(yy)][kk] for yy in range(9)]) 
        R10 = Rvec10.reshape((3,3)).T
        orth10 = np.abs(ca.reshape(ca.mtimes(R10.T, R10) - np.eye(3), 9, 1))
        c_max = max(c_max, c)
        dc_max = max(dc_max, dc)
        cr_max = max(cr_max, np.max(orth10))

    print('c_max = {}'.format(c_max))
    print('dc_max = {}'.format(dc_max))
    print('R10 orthogonality = {}'.format(cr_max))

    # plot power
    fig1 = plt.figure(1)
    t_x_coll = traj['time']
    power = [traj['x_l_t_0'][k]*traj['x_dl_t_0'][k]*traj['z_lambda10_0'][k] for k in range(len(homotopy['z_lambda10_0']))]
    power = [x/1e3 for x in power]
    plt.plot(t_x_coll, power, linewidth = 2)
    plt.plot(t_x_coll, [np.mean(power) for k in range(len(t_x_coll))], 'k--', linewidth = 2)
    plt.plot(t_x_coll, [np.max(power) for k in range(len(t_x_coll))], linestyle = '--', color = 'red', linewidth = 2)
    plt.plot(t_x_coll, [np.min(power) for k in range(len(t_x_coll))], linestyle = '--', color = 'blue', linewidth = 2)

    plt.show()


    # plot trajectory
    fig = plt.figure(2)
    ax = Axes3D(fig)
    fig.add_axes(ax)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.xaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.2
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.2

    ax.view_init(elev=25., azim=40)
    power_list = [homotopy['z_lambda10_0'][kk]*homotopy['x_l_t_0'][kk]*homotopy['x_dl_t_0'][kk] for kk in range(len(ref_state_0))]
    max_power = max(power_list)
    min_power = -min(power_list)

    jump_states = 7
    for kk in range(0, len(ref_state_0), jump_states):
        qkk = np.array([ref_state_0[kk], ref_state_1[kk], ref_state_2[kk]]).squeeze()
        Rkkvec = np.array([homotopy[ref_rot+'_'+str(yy)][kk] for yy in range(9)]) 
        Rkk = Rkkvec.reshape((3,3)).T
        if power_list[kk] >= 0.0:
            color = cm.get_cmap('Greens')((power_list[kk]/max_power))
            color = cm.get_cmap('Greens')(0.8)
        else:
            color = cm.get_cmap('Reds')((-power_list[kk]/min_power))
            color = cm.get_cmap('Reds')(0.8)
        
        qparent = [0, 0, 0]
        plot_airplane.plot_airplane(ax, qkk, qparent, Rkk, color, 26.0)

    xmax = max(ref_state_0)
    ymin = min(ref_state_1)
    ymax = max(ref_state_1)
    zmax = max(ref_state_2)

    # plot trajectory projections
    for kk in range(0, len(ref_state_0), jump_states):
        if power_list[kk] >= 0.0:
            color = cm.get_cmap('Greens')(0.8)
        else:
            color = cm.get_cmap('Reds')(0.8)
        ax.plot(ref_state_0[kk], ref_state_1[kk], 0, color = color, marker = 'o', markersize = 2, alpha = 0.3)
        # ax.plot(0, ref_state_1[kk], ref_state_2[kk], color = color, marker = 'o', markersize = 2, alpha = 0.3)
    
    ax.plot(ref_state_0, ref_state_1, [0]*len(ref_state_0), color = 'black', linewidth = 0.5, alpha = 0.3)
    # ax.plot([0]*len(ref_state_0), ref_state_1, ref_state_2, color = 'black', linewidth = 0.5, alpha = 0.3)
    
    # plot tether projections
    idx_ymin = np.argmin(ref_state_1)
    idx_ymax = np.argmax(ref_state_1)
    # ax.plot([0, 0], [0, ymin], [0, ref_state_2[idx_ymin]], linewidth = 0.1, color = 'black')
    # ax.plot([0, 0], [0, ymax], [0, ref_state_2[idx_ymax]], linewidth = 0.1, color = 'black')
    ax.plot([0, ref_state_0[idx_ymin]], [0, ymin], [0, 0], linewidth = 0.1, color = 'black')
    ax.plot([0, ref_state_0[idx_ymax]], [0, ymax], [0, 0], linewidth = 0.1, color = 'black')

    # ax.set_xlim3d(0, xmax + 20)
    # ax.set_ylim3d(ymin - 20, ymax + 20)
    ax.set_zlim3d(0, zmax + 20)
    # T_opt = homotopy['theta']['t_f_0'][-1].full()[0][0]
    # d_opt = homotopy['theta']['diam_t_0'][-1].full()[0][0]
    # ax.text(50.0, 50.0, 200, r'$\bar P = {{{}}}$ kW \n $T = {{{}}}$ s \n $d_{\mathrm{t}} = {{{}}}$ mm  {{{}}}$'.format(round(p_mean/1e3, 1), round(T_opt, 1), round(d_opt*1e3, 1)), bbox=dict(boxstyle="square", fc = 'white'))
    
    ax.set_xlabel(r'x [m]')
    ax.set_ylabel(r'y [m]')
    ax.set_zlabel(r'z [m]')
    plt.show()
    # plt.savefig(figfolder+'init_rokbustness_xyz_{}_{}.pgf'.format(jdx, suffix),bbox_inches='tight')

