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
import casadi.tools as cas
import awebox.tools.struct_operations as struct_op
import awebox.viz.tools as viz_tools
from awebox.logger.logger import Logger as awelogger


def get_splines(variables, xi_dict, initial_or_terminal):

    # set cosmetics
    cosmetics = {}
    cosmetics['plot_coll'] = True
    cosmetics['interpolation'] = {}
    cosmetics['interpolation']['type'] = 'poly'
    cosmetics['interpolation']['N'] = 100

    # get plot_dict
    plot_dict = xi_dict['plot_dict_pickle_' + initial_or_terminal]

    # set interpolant options
    interpolant_options = {}

    # initialize spline list
    spline_list = []

    # interpolate data
    plot_dict = viz_tools.interpolate_data(plot_dict, cosmetics)

    # merge xd values
    for variable in struct_op.subkeys(variables, 'xd'):
        for j in range(variables['xd', variable].shape[0]):
            xd_values = plot_dict['xd'][variable][j]
            xd_values = xd_values.full().reshape(xd_values.shape[0],).tolist()
            time_grid = plot_dict['time_grids']['ip']
            theta_grid = [t / time_grid[-1] for t in time_grid]
            awelogger.logger.info('Approximating ' + variable + '_' + str(j) + '...')
            if all(v == 0 for v in xd_values):
                raise ValueError('Cannot approximate constant 0 function with spline!')
            spline = cas.interpolant(variable + '_' + str(j), 'bspline', [theta_grid], xd_values, interpolant_options)
            spline_list.append(spline)

    # build dict
    spline_dict = {}
    for spline in spline_list:
        spline_dict[spline.name()] = spline

    return spline_dict
