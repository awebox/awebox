#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
- authors: jochem de schutter, rachel leuthold alu-fr 2018
"""

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import casadi.tools as cas
from . import tools
import numpy as np

def animate_monitor_plot(plot_dict, cosmetics, fig_name, init_colors=bool(False), plot_kites=bool(True)):
    """ Create monitor plot for optimal trajectory.
    """

    # extract trial information
    trial_name = plot_dict['name']
    architecture = plot_dict['architecture']
    parent_map = architecture.parent_map
    kite_nodes = architecture.kite_nodes

    # set-up figure
    fig = plt.figure()
    fig.clf()

    plt.suptitle(fig_name)

    #ax_iso = plt.subplot(2, 2, 1, projection='3d')
    ax_xz = plt.subplot(2, 2, 1)
    ax_yz = plt.subplot(2, 2, 2)
    ax_xy = plt.subplot(2, 2, 3)
    ax2 = []

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

    # figure limits
    limqx = [1e5, 0.0]
    limqy = [1e5,-1e5]
    limqz = [1e5, 0.0]

    for name in list(plot_dict['xd'].keys()):
        if name[0] == 'q':
            limqx[0] = np.min(cas.vertcat(limqx[0], np.min(plot_dict['xd'][name][0])))
            limqx[1] = np.max(cas.vertcat(limqx[1], np.max(plot_dict['xd'][name][0])))

            limqy[0] = np.min(cas.vertcat(limqy[0], np.min(plot_dict['xd'][name][1])))
            limqy[1] = np.max(cas.vertcat(limqy[1], np.max(plot_dict['xd'][name][1])))

            limqz[0] = np.min(cas.vertcat(limqz[0], np.min(plot_dict['xd'][name][2])))
            limqz[1] = np.max(cas.vertcat(limqz[1], np.max(plot_dict['xd'][name][2])))

    # get margins
    margin = cosmetics['trajectory']['margin']
    lmargin = 1.0 - margin
    umargin = 1.0 + margin

    # add margins to limits
    limqx[0] = limqx[0]*lmargin
    limqx[1] = limqx[1]*umargin
    limqz[0] = limqz[0]*lmargin
    limqz[1] = limqz[1]*umargin
    if limqy[0] > 0.0:
        limqy[0] = lmargin*limqy[0]
    else:
        limqy[0] = umargin*limqy[0]
    if limqy[1] < 0.0:
        limqy[1] = lmargin*limqy[1]
    else:
        limqy[1] = umargin*limqy[1]

    with writer.saving(fig, "./" + trial_name + ".mp4", 100):

        counter = 0
        for i in range(N):

            #ax_iso.clear()
            ax_xy.clear()
            ax_xz.clear()
            ax_yz.clear()

            # plot system
            #tools.plot_trajectory_instant(ax_iso, ax2, plot_dict, q_values, r_values, i, cosmetics, 'isometric', init_colors, plot_kites)
            tools.plot_trajectory_instant(ax_xy, ax2, plot_dict, i, cosmetics, 'xy', init_colors=init_colors, plot_kites=plot_kites)
            tools.plot_trajectory_instant(ax_xz, ax2, plot_dict, i, cosmetics, 'xz', init_colors=init_colors, plot_kites=plot_kites)
            tools.plot_trajectory_instant(ax_yz, ax2, plot_dict, i, cosmetics, 'yz', init_colors=init_colors, plot_kites=plot_kites)

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
                vertically_stacked_kite_locations = cas.horzcat(plot_dict['xd']['q'+str(n)+str(parent)][0],
                                                            plot_dict['xd']['q'+str(n)+str(parent)][1],
                                                            plot_dict['xd']['q'+str(n)+str(parent)][2])

                #tools.make_side_plot(ax_iso, vertically_stacked_kite_locations, 'isometric', local_color, alpha=alph)
                tools.make_side_plot(ax_xy, vertically_stacked_kite_locations, 'xy', local_color, alpha=alph)
                tools.make_side_plot(ax_xz, vertically_stacked_kite_locations, 'xz', local_color, alpha=alph)
                tools.make_side_plot(ax_yz, vertically_stacked_kite_locations, 'yz', local_color, alpha=alph)

                counter += 1

            # adjust limits
            #ax_iso.set_xlim(limqx)
            ax_xy.set_xlim(limqx)
            ax_xz.set_xlim(limqx)
            ax_yz.set_xlim(limqy)

            #ax_iso.set_ylim(limqy)
            ax_xy.set_ylim(limqy)
            ax_xz.set_ylim(limqz)
            ax_yz.set_ylim(limqz)

            #ax_iso.set_zlim(limqz)

            # set axis equal
            #ax_iso.set_aspect('equal', adjustable='box')
            ax_xy.set_aspect('equal', adjustable='box')
            ax_xz.set_aspect('equal', adjustable='box')
            ax_yz.set_aspect('equal', adjustable='box')

            # set labels
            # ax_iso.set_xlabel('x - [m]')
            # ax_iso.set_ylabel('y - [m]')
            # ax_iso.set_zlabel('z - [m]')

            ax_xy.set_xlabel('x - [m]')
            ax_xy.set_ylabel('y - [m]')

            ax_xz.set_xlabel('x - [m]')
            ax_xz.set_ylabel('z - [m]')

            ax_yz.set_xlabel('y - [m]')
            ax_yz.set_ylabel('z - [m]')

            # flip x-axis to get "upstream" view
            ax_yz.invert_xaxis()

            # make text plot
            txt_glb, txt_loc = fill_in_dashboard(fig, plot_dict,i)

            # write movie
            writer.grab_frame()

            # remove text plot
            txt_glb.remove()
            txt_loc.remove()

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
    dl_t = plot_dict['xd']['dl_t'][0][index].toarray().round(1)
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
    va =  plot_dict['outputs']['aerodynamics']['speed'+kite][0][index].round(1)
    local_string += 'va     = ' + str(va) + ' m/s\n'

    textbox_global = plt.gcf().text(0.55, 0.1, global_string)
    textbox_local = plt.gcf().text(0.75, 0.1, local_string)

    return textbox_global, textbox_local
