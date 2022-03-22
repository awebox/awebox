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
- authors: rachel leuthold 2021-2022
'''
import copy
import pdb

import casadi.tools as cas
import matplotlib.pyplot as plt
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.cylinder as vortex_cylinder

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')


class TangentialCylinder(vortex_cylinder.Cylinder):
    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('tangential_cylinder')

    def draw(self, ax, side, variables_scaled, parameters, cosmetics):
        evaluated = self.evaluate_info(variables_scaled, parameters)
        unpacked = self.unpack_info(evaluated)

        x_center = unpacked['x_center']
        l_hat = unpacked['l_hat']
        r_cyl = unpacked['radius']
        l_start = unpacked['l_start']

        a_hat, b_hat, _ = biot_savart.get_cylinder_axes(unpacked)

        s_start = l_start
        s_end = s_start + cosmetics['trajectory']['cylinder_s_length']

        n_theta = cosmetics['trajectory']['cylinder_n_theta']
        n_s = cosmetics['trajectory']['cylinder_n_s']

        for sdx in range(n_s):

            s = s_start + (s_end - s_start) / float(n_s) * float(sdx)

            for tdx in range(n_theta):
                theta_start = 2. * np.pi / float(n_theta) * float(tdx)
                theta_end = 2. * np.pi / float(n_theta) * (tdx + 1.)

                x_start = x_center + l_hat * s + r_cyl * np.sin(theta_start) * a_hat + r_cyl * np.cos(theta_start) * b_hat
                x_end = x_center + l_hat * s + r_cyl * np.sin(theta_end) * a_hat + r_cyl * np.cos(theta_end) * b_hat
                super().basic_draw(ax, side, unpacked['strength'], x_start, x_end, cosmetics)

        return None

def get_test_tangential_cylinder():
    cyl_data = vortex_cylinder.get_test_cylinder().info_dict
    cyl = TangentialCylinder(cyl_data)
    return cyl