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
'''
file to generate the tether drag coefficient vs. reynolds number relationship, for a long cylinder
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2018
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op

import matplotlib.pyplot as plt

def get_tether_cd_fun(model_options, parameters):

    reynolds = cas.SX.sym('reynolds')

    smoothing = model_options['tether']['reynolds_smoothing']
    if model_options['tether']['cd_model'] == 'polyfit':
        drag_coeff = get_interpolation(reynolds, smoothing)

    elif model_options['tether']['cd_model'] == 'piecewise':
        drag_coeff = get_roshko_unitstep(reynolds, smoothing)

    elif model_options['tether']['cd_model'] == 'constant':
        drag_coeff = parameters['theta0','tether','cd']

    else:
        raise ValueError('invalid tether drag coefficient model selected: %s',model_options['tether']['cd_model'])

    tether_cd_fun = cas.Function('tether_cd_fun', [reynolds], [drag_coeff])

    return tether_cd_fun

def plot_cd_vs_reynolds(num_fig, model_options):

    log_reynolds_list = np.arange(101.) * 7. / 100.
    cd_list_unitstep = []
    cd_list_poly = []
    cd_list_default = []
    reynolds_list = []

    smoothing = model_options['tether']['reynolds_smoothing']

    for log_reynolds in log_reynolds_list:
        reynolds = 10.**log_reynolds
        reynolds_list = cas.vertcat(reynolds_list, reynolds)

        cd_list_unitstep = cas.vertcat(cd_list_unitstep, get_roshko_unitstep(reynolds, smoothing))
        cd_list_poly = cas.vertcat(cd_list_poly, get_interpolation(reynolds, smoothing))
        cd_list_default = cas.vertcat(cd_list_default, model_options['tether']['cd'])

    cd_list_unitstep = np.array(cd_list_unitstep)
    cd_list_poly = np.array(cd_list_poly)
    cd_list_default = np.array(cd_list_default)
    reynolds_list = np.array(reynolds_list)

    [achenbach_re, achenbach_cd] = get_achenbach_datapoints()

    plt.figure(num_fig)
    plt.loglog(reynolds_list, cd_list_unitstep, 'r', label='roshko unitstep of linear fits')
    plt.loglog(reynolds_list, cd_list_poly, 'g', label='polyfit of roshko')
    plt.loglog(achenbach_re, achenbach_cd, 'b*', label='achenbach datapoints')
    plt.loglog(reynolds_list, cd_list_default, 'k--', label='constant')
    plt.legend()
    plt.show()

def get_roshko_unitstep(reynolds, eps):

    log_reynolds = np.log10(reynolds)

    # high reynolds polynomials fit from experimental + Wieselsberger + Delany/Sorenson data:
    # corresponds to air pressures between 1 and 4 atm.
    # Roshko, A.(1961).Experiments on the flow past a circular cylinder at very high Reynolds number.
    # Journal of Fluid Mechanics 10(3), 345 - 356.
    # doi:10.1017 / S0022112061000950

    # low reynolds number relationship suggested by
    # http://scienceworld.wolfram.com/physics/CylinderDrag.html

    lbreyn = 0.

    # for log Re < 2
    cd_stokes = 100./reynolds
    ubreyn = 2.
    # R_squared_stokes = no fit performed
    h_stokes = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_stokes

    # for 2 < log Re < 4
    cd_laminar = 1.
    lbreyn = ubreyn
    ubreyn = 4.
    # R_squared_laminar = no fit performed
    h_laminar = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_laminar

    # for 4 < log Re < 4.3
    cd_lamsep = 1.02198077356237e-5 * reynolds + 1.01141242
    lbreyn = ubreyn
    ubreyn = 4.3
    # R_squared_lamsep = 0.99
    h_lamsep = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_lamsep

    # for 4.3 < log Re < 5.26
    cd_level = -1.03659206648679e-7 * reynolds + 1.2046901692
    lbreyn = ubreyn
    ubreyn = 5.26
    # R_squared_level = 0.55
    h_level = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_level

    # for 5.26 < log Re < 5.74
    cd_transition = -3.28441892597317e-6 * reynolds + 1.8415437577
    lbreyn = ubreyn
    ubreyn = 5.74
    # R_squared_transition = 0.94
    h_transition = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_transition

    # for 5.74 < log Re < 7
    cd_turbsep = 7.10799367510221e-8 * reynolds + 0.2824178662
    lbreyn = ubreyn
    ubreyn = 7.
    # R_squared_turbsep = 0.84
    h_turbsep = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_turbsep

    # for 7 < log Re
    cd_final = 0.8
    lbreyn = ubreyn
    ubreyn = 10.
    # R_squared_final = no fit performed
    h_final = vect_op.step_in_out(log_reynolds, lbreyn, ubreyn, eps) * cd_final

    drag_coeff = h_stokes + h_laminar + h_lamsep + h_level + h_transition + h_turbsep + h_final

    return drag_coeff

def get_achenbach_datapoints():

    # data 39376 < Re < 4815676, from fig 9, pg. 635 of
    # Achenbach, E. (1968). Distribution of local pressure and skin friction around a circular cylinder in cross-flow up to Re = 5e6
    # Journal of Fluid Mechanics, 34(4), 625-639. doi:10.1017/S0022112068002120
    # digitized by Mikko Folkersma (TU Delft, 2017)

    achenbach_re = np.array([39376.9444845985,
                            59180.3220925593,
                            98042.1284988708,
                            130950.661845122,
                            177279.465034671,
                            189171.336716672,
                            208523.702821741,
                            219382.685340574,
                            250474.492764236,
                            251176.102207476,
                            272662.507546918,
                            293391.667767627,
                            297121.039888131,
                            329418.920869394,
                            356054.506766152,
                            430668.007087765,
                            442913.779559282,
                            474051.811029239,
                            563061.279073674,
                            639260.260850478,
                            744111.718331051,
                            856111.913874845,
                            975544.366401701,
                            1284502.83554852,
                            1521822.48808959,
                            1545281.10611444,
                            1808117.79265064,
                            2165145.64882329,
                            2946433.47747528,
                            3600828.01647156,
                            4815676.41745568])

    achenbach_cd = np.array([1.2699208416,
                            1.2699208416,
                            1.1926111128,
                            1.0424288576,
                            0.9132048434,
                            0.7512979227,
                            0.8889507915,
                            0.7822542222,
                            0.6378155003,
                            0.7597718936,
                            0.6822150289,
                            0.6003319778,
                            0.4136881663,
                            0.540261156,
                            0.6003319778,
                            0.5318440363,
                            0.6030313516,
                            0.7280702551,
                            0.6566890304,
                            0.6003319778,
                            0.6776394186,
                            0.6335376772,
                            0.6307017467,
                            0.5976446873,
                            0.6992581883,
                            0.581771682,
                            0.6112025174,
                            0.7597718936,
                            0.7805014403,
                            0.7614781229,
                            0.7346324652])

    return achenbach_re, achenbach_cd

def make_poly_points(smoothing):

    number_points = 200.
    log_reynolds_list = 2. +  np.arange(number_points + 1.) * 5. / number_points
    cd_list = []
    reynolds_list = []

    for log_reynolds in log_reynolds_list:
        reynolds = 10.**log_reynolds
        reynolds_list = cas.vertcat(reynolds_list, reynolds)

        cd_list = cas.vertcat(cd_list, get_roshko_unitstep(reynolds, smoothing))

    cd_list = np.array(cd_list)
    reynolds_list = np.array(reynolds_list)

    return reynolds_list, cd_list

def get_interpolation(reynolds, smoothing):

    [reynolds_array, cd_array] = make_poly_points(smoothing)

    log_re = np.log10(reynolds_array)

    cd_list = cd_array.T.tolist()[0]
    log_re_list = log_re.T.tolist()[0]

    max_dim = 10

    poly = np.polyfit(log_re_list, cd_list, max_dim)

    estimate = cas.polyval(poly, np.log10(reynolds))

    return estimate
