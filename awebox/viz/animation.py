#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
"""
animation routines for awebox trajectories
python-3.5 / casadi 3.0.0
- authors: jochem de schutter, rachel leuthold alu-fr 2018-2020
"""

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import casadi.tools as cas
from . import tools
import awebox.viz.trajectory as trajectory
import numpy as np

def animate_monitor_plot(plot_dict, cosmetics, fig_name, init_colors=bool(False), plot_kites=bool(True)):
    """ Create monitor plot for optimal trajectory.
    """

    # extract trial information
    trial_name = plot_dict['name']

    # set-up figure
    fig = plt.figure()
    fig.clf()

    plt.suptitle(fig_name)

    # define axes
    axes = {}
    axes['ax2'] = []
    axes['ax_xz'] = plt.subplot(2, 2, 1)
    axes['ax_yz'] = plt.subplot(2, 2, 2)
    axes['ax_xy'] = plt.subplot(2, 2, 3)

    plt.ion()

    time_grid = plot_dict['time_grids']['ip']

    # time grid length
    N = time_grid.shape[0]

    # set-up mp4-writer
    total_time = time_grid[-1]
    fps = int(np.ceil(N / total_time))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=trial_name, artist='awebox', comment='monitor_plot')
    writer = FFMpegWriter(fps=fps, metadata=metadata,codec="libx264",bitrate=-1)

    with writer.saving(fig, "./" + trial_name + ".mp4", 100):

        for index in range(N):
            animation_snapshot(axes, plot_dict, index, cosmetics, init_colors, plot_kites)

            # make text plot
            txt_glb, txt_loc = fill_in_dashboard(fig, plot_dict,index)

            # write movie
            writer.grab_frame()

            # remove text plot
            txt_glb.remove()
            txt_loc.remove()

    return None


def animate_snapshot(plot_dict, cosmetics, fig_name, init_colors=bool(False), plot_kites=bool(True)):
    """ Create monitor plot for optimal trajectory.
    """

    # extract trial information
    trial_name = plot_dict['name']

    # set-up figure
    fig = plt.figure()
    fig.clf()

    plt.suptitle(fig_name)

    # define axes
    axes = {}
    axes['ax2'] = []
    axes['ax_xz'] = plt.subplot(2, 2, 1)
    axes['ax_yz'] = plt.subplot(2, 2, 2)
    axes['ax_xy'] = plt.subplot(2, 2, 3)

    index = cosmetics['animation']['snapshot_index']
    animation_snapshot(axes, plot_dict, index, cosmetics, init_colors, plot_kites)

    # make text plot
    txt_glb, txt_loc = fill_in_dashboard(fig, plot_dict, index)

    return None


def animation_snapshot(axes, plot_dict, index, cosmetics, init_colors=bool(False), plot_kites=bool(True)):

    dims = ['xy', 'xz', 'yz']

    # figure limits
    q_limits = tools.get_q_limits(plot_dict, cosmetics)

    architecture = plot_dict['architecture']
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    for dim in dims:
        ax = 'ax_' + dim
        axes[ax].clear()

        # plot system
        trajectory.plot_trajectory_instant(axes[ax], axes['ax2'], plot_dict, index, cosmetics, dim, init_colors=init_colors, plot_kites=plot_kites)

    # plot trajectories
    counter = 0
    alph = cosmetics['trajectory']['alpha']
    for n in kite_nodes:
        if init_colors == True:
            local_color = 'k'
        elif init_colors == False:
            local_color = cosmetics['trajectory']['colors'][counter]
        else:
            local_color = init_colors

        parent = parent_map[n]
        vertically_stacked_kite_locations = cas.horzcat(plot_dict['xd']['q' + str(n) + str(parent)][0],
                                                        plot_dict['xd']['q' + str(n) + str(parent)][1],
                                                        plot_dict['xd']['q' + str(n) + str(parent)][2])

        for dim in dims:
            ax = 'ax_' + dim
            tools.make_side_plot(axes[ax], vertically_stacked_kite_locations, dim, local_color, alpha=alph)

        counter += 1

    # change axes limits
    for dim in dims:
        ax = 'ax_' + dim

        xdim = dim[0]
        ydim = dim[1]

        axes[ax].set_xlim(q_limits[xdim])
        axes[ax].set_ylim(q_limits[ydim])
        axes[ax].set_aspect('equal', adjustable='box')

        axes[ax].set_xlabel(xdim + ' [m]')
        axes[ax].set_ylabel(ydim + ' [m]')

    # flip x-axis to get "upstream" view
    axes['ax_yz'].invert_xaxis()

    # move axes out of way of three-view
    axes['ax_yz'].yaxis.set_label_position("right")
    axes['ax_yz'].yaxis.tick_right()

    axes['ax_yz'].xaxis.set_label_position("top")
    axes['ax_yz'].xaxis.tick_top()

    axes['ax_xz'].xaxis.set_label_position("top")
    axes['ax_xz'].xaxis.tick_top()

    plt.tight_layout(w_pad=-11, h_pad=1, rect=[0, 0, 1, 0.95]) #pad=2.5)

    return None

def fill_in_dashboard(fig, plot_dict,index):

    global_string = ''

    # GLOBAL INFORMATION
    # immediate power output
    power = (plot_dict['outputs']['performance']['p_current'][0][index]*1e-3).round(1)
    global_string += 'P   = ' + str(power) + ' kW\n'

    # immediate tether forces
    for name in list(plot_dict['outputs']['local_performance'].keys()):
        if 'tether_force' in name:
            num = name[12:]
            tether_force = (plot_dict['outputs']['local_performance'][name][0][index]*1e-3).round(1)
            global_string += 'Ft' + num + ' = ' + str(tether_force) + ' kN\n'

    # tether speed
    dl_t = plot_dict['xd']['dl_t'][0][index].toarray().round(1)[0][0]
    global_string += 'dlt = ' + str(dl_t) + ' m/s\n'


    # LOCAL INFORMATION
    local_string = 'kite 1:\n'
    kite = str(plot_dict['architecture'].kite_nodes[0]) # todo: set where?
    if plot_dict['options']['model']['kite_dof'] == 6:
        # angle of attack
        alpha = plot_dict['outputs']['aerodynamics']['alpha_deg'+kite][0][index].round(1)
        local_string += 'alpha = ' + str(alpha) + ' deg\n'

        # side-slip
        beta = plot_dict['outputs']['aerodynamics']['beta_deg'+kite][0][index].round(1)
        local_string += 'beta  = ' + str(beta)  + ' deg\n'

    # airspeed
    va =  plot_dict['outputs']['aerodynamics']['airspeed'+kite][0][index].round(1)
    local_string += 'va     = ' + str(va) + ' m/s\n'

    textbox_global = plt.gcf().text(0.55, 0.1, global_string)
    textbox_local = plt.gcf().text(0.75, 0.1, local_string)

    return textbox_global, textbox_local
