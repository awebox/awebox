#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
object-oriented-vortex-filament-and-cylinder operations
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold 2021
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import copy
import pdb

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class VortexObjectStructure:
    def __init__(self):
        pass

    def compute_the_scaled_haas_error(self, radius, l_hat, u_infty):

        # reference points digitized from haas 2017
        data02 = [(0, -1.13084, -0.28134), (0, -0.89466, -0.55261), (0, -0.79354, -0.41195), (0, -0.79127, 0.02826),
                  (0, -0.04449, 1.01208), (0, 0.22119, 0.83066), (0, 0.71947, -0.96407), (0, 0.52187, -0.66125),
                  (0, 0.45227, 1.09891), (0, 0.92121, -0.46219), (0, 0.95625, -0.62566), (0, 0.19546, 1.12782)]
        data00 = [(0, -0.69372, 1.24284), (0, -0.5894, -0.17062), (0, -0.83795, -1.18138), (0, 0.27077, 1.35428),
                  (0, -1.20982, -0.71437), (0, -0.10682, 0.54558), (0, -0.3789, -0.3959), (0, 0.5081, -0.34622),
                  (0, 0.23767, 0.61635), (0, 0.57386, -0.09835), (0, 1.10956, -0.867), (0, 1.4683, -0.06319)]
        datan005 = [(0, -1.27365, 0.33177), (0, -1.21638, 0.65872), (0, -0.64295, 0.15234), (0, -0.50651, 0.31282),
                    (0, -0.01271, -1.39519), (0, 0.03669, -0.60921), (0, 0.20264, -0.63812), (0, 0.38143, -1.27208),
                    (0, 0.49584, 0.49075), (0, 0.52938, 0.32778), (0, 0.96982, 0.94579), (0, 1.3061, 0.48278)]

        data = {-0.05: datan005, 0.0: data00, 0.2: data02}

        total_squared_error = 0.
        baseline_squared_error = 0.
        for aa_haas in data.keys():
            coord_list = data[aa_haas]
            for scaled_x_obs_tuple in coord_list:

                baseline_squared_error += aa_haas ** 2.

                scaled_x_obs = []
                for dim in range(3):
                    scaled_x_obs = cas.vertcat(scaled_x_obs, scaled_x_obs_tuple[dim])
                x_obs = scaled_x_obs * radius

                u_ind_computed = self.evaluate_total_biot_savart_induction(x_obs)
                aa_computed = float(vect_op.dot(u_ind_computed, -1. * l_hat) / u_infty)
                diff = aa_haas - aa_computed
                total_squared_error += diff ** 2.

        scaled_squared_error = total_squared_error / baseline_squared_error
        return scaled_squared_error
