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
import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.element as vortex_element

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger

import matplotlib
matplotlib.use('TkAgg')

class SemiInfiniteCylinder(vortex_element.Element):
    def __init__(self, info_dict):
        super().__init__(info_dict)
        self.set_element_type('semi_infinite_cylinder')
        self.define_info_order()
        packed_info = self.pack_info()
        self.set_info(packed_info)

    def define_info_order(self):
        order = {0: ('x_center', 3),
                 1: ('l_hat', 3),
                 2: ('radius', 1),
                 3: ('l_start', 1),
                 4: ('epsilon', 1),
                 5: ('strength', 1)
                 }
        self.set_info_order(order)
        return None

def construct_test_object():
    x_center = np.array([0., 0., 0.])
    radius = 1.
    l_start = 0.
    l_hat = vect_op.xhat_np()
    epsilon = 0.
    strength = 1.
    unpacked = {'x_center': x_center,
                'l_hat': l_hat,
                'radius': radius,
                'l_start': l_start,
                'epsilon': epsilon,
                'strength': strength
                }

    cyl = SemiInfiniteCylinder(unpacked)
    return cyl
2
def test():
    cyl = construct_test_object()
    cyl.test_basic_criteria(expected_object_type='semi_infinite_cylinder')
    return None

test()