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


import matplotlib
from awebox.viz.plot_configuration import DEFAULT_MPL_BACKEND
matplotlib.use(DEFAULT_MPL_BACKEND)
import matplotlib.pyplot as plt

import awebox.viz.tools as tools
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import MaxNLocator
import numpy as np
import awebox.viz.tools as tools
import awebox.viz.wake as wake
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import casadi.tools as cas


import matplotlib.animation as manimation

def plot_trajectory(plot_dict, cosmetics, fig_name, side, init_colors=False, label = []):

    fig = plt.figure()

    if side == 'isometric':
        ax = plt.subplot(1, 1, 1, projection='3d')
    elif side == 'quad':
        ax_xy = plt.subplot(221)
        ax_xz = plt.subplot(222)
        ax_yz = plt.subplot(223)
        ax_iso = plt.subplot(224, projection='3d')
    else:
        ax = plt.subplot(1, 1, 1)
        plt.axis('equal')

    if side == 'quad':
        tools.plot_trajectory_contents(ax_xy, plot_dict, cosmetics, 'xy', init_colors, label=label)
        tools.plot_trajectory_contents(ax_xz, plot_dict, cosmetics, 'xz', init_colors, label=label)
        tools.plot_trajectory_contents(ax_yz, plot_dict, cosmetics, 'yz', init_colors, label=label)
        tools.plot_trajectory_contents(ax_iso, plot_dict, cosmetics, 'isometric', init_colors, label=label)
    else:
        tools.plot_trajectory_contents(ax, plot_dict, cosmetics, side, init_colors, label=label)

    if side == 'isometric':
        ax.set_xlabel('\n x [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel('\n y [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_zlabel('z [m]', **cosmetics['trajectory']['axisfont'])

        ax.zaxis.set_major_locator(MaxNLocator(4))

        ax.xaxis._axinfo['label']['space_factor'] = 2.8
        ax.yaxis._axinfo['label']['space_factor'] = 2.8
        ax.zaxis._axinfo['label']['space_factor'] = 2.8

    elif side == 'quad':
        ax_xy.xaxis.tick_top()
        ax_xy.xaxis.set_label_position('top')
        ax_xy.set_xlabel(r'x [m]')
        ax_xy.set_ylabel(r'y [m]')
        xlim = ax_xy.get_xlim()
        ax_xy.set_xlim([0, xlim[1]])

        ax_xz.xaxis.tick_top()
        ax_xz.yaxis.tick_right()
        ax_xz.xaxis.set_label_position('top')
        ax_xz.yaxis.set_label_position('right')
        ax_xz.set_xlabel(r'x [m]')
        ax_xz.set_ylabel(r'z [m]')
        xlim = ax_xz.get_xlim()
        ax_xz.set_xlim([0, xlim[1]])

        ax_yz.set_xlabel(r'y [m]')
        ax_yz.set_ylabel(r'z [m]')
        ylim = ax_yz.get_ylim()
        ax_yz.set_ylim([0, ylim[1]])

        ax_iso.set_xlabel(r'x [m]')
        ax_iso.set_ylabel(r'y [m]')
        ax_iso.set_zlabel(r'z [m]')
        xlim = ax_iso.get_xlim()
        zlim = ax_iso.get_zlim()
        ax_iso.set_xlim([0, xlim[1]])
        ax_iso.set_zlim([0, zlim[1]])
    else:
        ax.set_xlabel(side[0] + ' [m]', **cosmetics['trajectory']['axisfont'])
        ax.set_ylabel(side[1] + ' [m]', **cosmetics['trajectory']['axisfont'])

    if side != 'quad':
        ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])

        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.legend(loc = 'upper right')
    plt.suptitle(fig_name)

    # set equal aspect ratio for a trajectory plots
    if side not in ['isometric', 'quad']:
        for ax in fig.axes[:-1]:
            ax.set_aspect('equal')
    if side == 'quad':
        for ax in fig.axes[:-1]:
            ax.set_aspect('equal')

def plot_trajectory_against_wind_velocity(solution_dict, cosmetics, fig_num, reload_dict):

    # read in inputs
    wind = solution_dict['wind']
    maxlim = reload_dict['maxlim']

    plt.ion()

    fig = plt.figure(fig_num)
    fig.clf()

    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twiny()

    h = list(range(1, int(maxlim)))
    ax2.plot([float(wind.get_velocity(zz)[0])
              for zz in h], h, color='b')

    tools.plot_trajectory_contents(ax, solution_dict, cosmetics, 'xz', reload_dict, bool(False), bool(False))

    ax.set_xlim([0, maxlim])
    ax.set_ylim([0, maxlim])
    # ax2.set_xlim([0, 30])

    ax.set_xlabel('x [m]', **cosmetics['trajectory']['axisfont'])
    ax.set_ylabel('z [m]', **cosmetics['trajectory']['axisfont'])
    ax2.set_xlabel('x component of wind velocity [m/s]', color='b', **cosmetics['trajectory']['axisfont'])
    ax2.tick_params('x', colors='b')

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    ax2.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))

    ax.grid()

    plt.show()

def plot_trajectory_against_wind_shear(solution_dict, cosmetics, fig_num, reload_dict):

    # read in inputs
    wind = solution_dict['wind']
    maxlim = reload_dict['maxlim']

    plt.ion()

    fig = plt.figure(fig_num)
    fig.clf()

    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twiny()

    h = list(range(1, int(maxlim)))
    ax2.plot([float(wind.get_velocity(zz)[1]) for zz in h], h, color='b')

    tools.plot_trajectory_contents(ax, solution_dict, cosmetics, 'yz', reload_dict, bool(False), bool(False))

    ax.set_xlim([- maxlim/2., maxlim/2.])
    ax.set_ylim([0, maxlim])
    # ax2.set_xlim([-10., 10.])
    #
    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    ax2.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])

    ax.set_xlabel('y [m]', **cosmetics['trajectory']['axisfont'])
    ax.set_ylabel('z [m]', **cosmetics['trajectory']['axisfont'])
    ax2.set_xlabel('y component of wind velocity [m/s]', color='b', **cosmetics['trajectory']['axisfont'])
    ax2.tick_params('x', colors='b')

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))

    ax.grid()

    plt.show()

def plot_trajectory_against_wind_power(solution_dict, cosmetics, fig_num, reload_dict):

    # read in inputs
    atmos = solution_dict['atmos']
    maxlim = reload_dict['maxlim']

    plt.ion()

    fig = plt.figure(fig_num)
    fig.clf()

    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twiny()

    h = list(range(1, int(maxlim)))
    ax2.plot([float(atmos.get_density(zz)) for zz in h], h, color='b')

    tools.plot_trajectory_contents(ax, solution_dict, cosmetics, 'xz', reload_dict, bool(False), bool(False))

    ax.set_xlim([0, maxlim])
    ax.set_ylim([0, maxlim])
    # ax2.set_xlim([0, 800])

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])
    ax2.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])

    ax.set_xlabel('x [m]', **cosmetics['trajectory']['axisfont'])
    ax.set_ylabel('z [m]', **cosmetics['trajectory']['axisfont'])
    ax2.set_xlabel('wind power density [W/m^2]', color='b', **cosmetics['trajectory']['axisfont'])
    ax2.tick_params('x', colors='b')

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))

    ax.grid()

    plt.show()

def plot_trajectory_along_elevation(solution_dict, cosmetics, fig_num):

    # read in inputs
    output = solution_dict['outputs']

    fig = plt.figure(fig_num)

    ax = plt.subplot(1, 1, 1, projection='3d')
    tools.plot_trajectory_contents(ax, solution_dict, cosmetics, 'isometric')

    angle = np.mean(np.array(output['outputs',:,:,'performance','elevation'])) * 180. / np.pi

    ax.view_init(0., angle)

    ax.xaxis.set_major_locator(MaxNLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.zaxis.set_major_locator(MaxNLocator(4))

    ax.set_xlabel('\n x [m]', **cosmetics['trajectory']['axisfont'])
    ax.set_ylabel('\n y [m]', **cosmetics['trajectory']['axisfont'])
    ax.set_zlabel('z [m]', **cosmetics['trajectory']['axisfont'])

    ax.tick_params(labelsize=cosmetics['trajectory']['ylabelsize'])

    ax.xaxis._axinfo['label']['space_factor'] = 2.8
    ax.yaxis._axinfo['label']['space_factor'] = 2.8
    ax.zaxis._axinfo['label']['space_factor'] = 2.8

    plt.draw()


def plot_trajectory_instant(ax, plot_dict, index, cosmetics, side, init_colors=bool(False), plot_kites=bool(True)):

    tools.plot_all_tethers(ax, side, plot_dict, index=index)

    if cosmetics['trajectory']['kite_bodies'] and plot_kites:
        tools.draw_all_kites(ax, plot_dict, index, cosmetics, side, init_colors)

    if cosmetics['trajectory']['kite_aero_dcm']:
        tools.draw_kite_aero_dcm(ax, side, plot_dict, cosmetics, index)
    if cosmetics['trajectory']['trajectory_rotation_dcm']:
        tools.draw_trajectory_rotation_dcm(ax, side, plot_dict, cosmetics, index)

    if cosmetics['trajectory']['wake_nodes']:
        wake.draw_wake_nodes(ax, side, plot_dict, cosmetics, index)

    if cosmetics['trajectory']['actuator']:
        wake.draw_actuator(ax, side, plot_dict, cosmetics, index)

    ax.get_figure().canvas.draw()

    return None
