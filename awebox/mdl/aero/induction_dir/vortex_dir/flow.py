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
flow functions for the vortex based model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
"""

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow
import awebox.mdl.aero.induction_dir.vortex_dir.biot_savart as biot_savart
import awebox.mdl.aero.induction_dir.vortex_dir.tools as vortex_tools
from awebox.logger.logger import Logger as awelogger
import awebox.mdl.aero.induction_dir.general_dir.geom as general_geom
import awebox.mdl.aero.induction_dir.actuator_dir.flow as actuator_flow
import awebox.mdl.aero.induction_dir.vortex_dir.filament_list as vortex_filament_list
import awebox.tools.print_operations as print_op

import awebox.tools.vector_operations as vect_op

import casadi.tools as cas

#
def get_induced_velocity_at_kite(options, filament_list, variables, architecture, kite_obs, n_hat=None):
    x_obs = variables['xd']['q' + str(kite_obs) + str(architecture.parent_map[kite_obs])]
    u_ind = get_induced_velocity_at_observer(options, filament_list, x_obs, n_hat=n_hat)
    return u_ind


def get_induced_velocity_at_observer(options, filament_list, x_obs, n_hat=None):

    filament_list = vortex_filament_list.append_observer_to_list(filament_list, x_obs)

    include_normal_info = (n_hat is not None)
    if include_normal_info:
        filament_list = vortex_filament_list.append_normal_to_list(filament_list, n_hat)

    u_ind = make_symbolic_filament_and_sum(options, filament_list, include_normal_info)
    return u_ind



def get_induction_factor_at_kite(options, filament_list, wind, variables, parameters, architecture, kite_obs, n_hat=vect_op.xhat()):

    x_obs = variables['xd']['q' + str(kite_obs) + str(architecture.parent_map[kite_obs])]

    parent = architecture.parent_map[kite_obs]
    u_zero_vec = actuator_flow.get_uzero_vec(options, wind, parent, variables, parameters, architecture)
    u_zero = vect_op.smooth_norm(u_zero_vec)

    a_calc = get_induction_factor_at_observer(options, filament_list, x_obs, u_zero, n_hat=n_hat)

    return a_calc


def get_induction_factor_at_observer(options, filament_list, x_obs, u_zero, n_hat=vect_op.xhat()):
    u_ind = get_induced_velocity_at_observer(options, filament_list, x_obs, n_hat=n_hat)
    a_calc = -1. * u_ind / u_zero
    return a_calc



def make_symbolic_filament_and_sum(options, filament_list, include_normal_info=False):

    r_core = options['induction']['vortex_core_radius']

    # define the symbolic function
    n_symbolics = filament_list.shape[0]

    u_ind = cas.DM.zeros((3, 1))
    if n_symbolics > 0:
        seg_data_sym = cas.SX.sym('seg_data_sym', (n_symbolics, 1))

        if include_normal_info:
            filament_sym = biot_savart.filament_normal(seg_data_sym, r_core=r_core)
        else:
            filament_sym = biot_savart.filament(seg_data_sym, r_core=r_core)

        filament_fun = cas.Function('filament_fun', [seg_data_sym], [filament_sym])

        # evaluate the symbolic function
        u_ind = vortex_tools.evaluate_symbolic_on_segments_and_sum(filament_fun, filament_list)

    return u_ind

def test(test_list):

    options = {}
    options['induction'] = {}
    options['induction']['vortex_core_radius'] = 0.

    x_obs = 0.5 * vect_op.xhat_np()

    u_ind = get_induced_velocity_at_observer(options, test_list, x_obs)

    xhat_component = cas.mtimes(u_ind.T, vect_op.xhat())
    if not (xhat_component == 0):
        message = 'induced velocity at observer does not work as expected. ' \
                  'test u_ind component in plane of QSVR (along xhat) is ' + str(xhat_component)
        awelogger.logger.error(message)
        raise Exception(message)

    yhat_component = cas.mtimes(u_ind.T, vect_op.yhat())
    if not (yhat_component == 0):
        message = 'induced velocity at observer does not work as expected. ' \
                  'test u_ind component in plane of QSVR (along yhat) is ' + str(yhat_component)
        awelogger.logger.error(message)
        raise Exception(message)

    zhat_component = cas.mtimes(u_ind.T, vect_op.zhat())
    sign_along_zhat = vect_op.sign(zhat_component)
    sign_comparison = (sign_along_zhat - (-1))**2.
    if not (sign_comparison < 1.e-8):
        message = 'induced velocity at observer does not work as expected. ' \
                  'sign of test u_ind component out-of-plane of QSVR (projected on zhat) is ' + str(sign_along_zhat)
        awelogger.logger.error(message)
        raise Exception(message)

    calculated_norm = vect_op.norm(u_ind)
    expected_norm = 0.752133
    norm_comparison = (calculated_norm - expected_norm)**2.
    if not (norm_comparison < 1.e-8):
        message = 'induced velocity at observer does not work as expected. ' \
                  'squared difference of norm of test u_ind vector is ' + str(norm_comparison)
        awelogger.logger.error(message)
        raise Exception(message)

    return None