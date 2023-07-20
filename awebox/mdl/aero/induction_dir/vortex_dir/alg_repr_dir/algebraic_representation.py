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
vortex model of awebox aerodynamics
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-21
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import pdb

import casadi.tools as cas
import numpy as np

import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools

import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.structure as alg_structure
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.bound_wake as alg_bound_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.near_wake as alg_near_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.far_wake as alg_far_wake
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.fixing as alg_fixing
import awebox.mdl.aero.induction_dir.vortex_dir.alg_repr_dir.initialization as alg_initialization

import awebox.mdl.aero.induction_dir.vortex_dir.vortex_objects_dir.wake as obj_wake

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.print_operations as print_op

from awebox.logger.logger import Logger as awelogger


def build(options, architecture, wind, variables_si, parameters):

    vortex_tools.check_positive_vortex_wake_nodes(options)

    bound_wake = alg_bound_wake.build(options, architecture, variables_si, parameters)
    near_wake = alg_near_wake.build(options, architecture, variables_si, parameters)
    far_wake = alg_far_wake.build(options, architecture, wind, variables_si, parameters)

    wake = obj_wake.Wake()
    wake.set_substructure(bound_wake)
    wake.set_substructure(near_wake)
    wake.set_substructure(far_wake)

    biot_savart_residual_assembly = options['induction']['vortex_biot_savart_residual_assembly']
    wake.define_biot_savart_induction_residual_functions(biot_savart_residual_assembly)

    return wake


def get_ocp_constraints(nlp_options, V, Outputs, Integral_outputs, model, time_grids):
    return alg_fixing.get_constraint(nlp_options, V, Outputs, Integral_outputs, model, time_grids)


def get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model):
    return alg_initialization.get_initialization(nlp_options, V_init_si, p_fix_num, nlp, model)


def test_drawing(test_includes_visualization):

    if test_includes_visualization:

        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_vortex_ring_test_object()
        wake = build(options, architecture, wind, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        wake.draw(ax, 'isometric')

        options, architecture, wind, var_struct, param_struct, variables_dict, variables_si, parameters = alg_structure.construct_straight_flight_test_object()
        wake = build(options, architecture, wind, variables_si, parameters)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        wake.draw(ax, 'isometric')

        plt.show()

    return None

def test(test_includes_visualization=False):
    alg_bound_wake.test(test_includes_visualization)
    alg_near_wake.test(test_includes_visualization)
    alg_far_wake.test(test_includes_visualization)
    test_drawing(test_includes_visualization)

    return None

# test()