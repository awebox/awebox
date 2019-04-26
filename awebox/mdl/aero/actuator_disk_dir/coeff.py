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
'''
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-19
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

from . import geom as geom
from . import flow as flow


def get_ct_var(model_options, variables, parent):

    var_type = geom.get_var_type(model_options)
    ct_var = variables[var_type]['ct' + str(parent)]
    return ct_var

def get_dct_var(variables, parent):

    dct_var = variables['xd']['dct' + str(parent)]
    return dct_var




def get_actuator_force(outputs, parent, architecture):

    children = architecture.kites_map[parent]

    total_force_aero = np.zeros((3, 1))
    for kite in children:
        aero_force = outputs['aerodynamics']['f_aero' + str(kite)]

        total_force_aero = total_force_aero + aero_force

    return total_force_aero

def get_actuator_thrust(model_options, variables, outputs, parent, architecture):

    total_force_aero = get_actuator_force(outputs, parent, architecture)
    normal = geom.get_nhat_var(variables, parent)
    thrust = cas.mtimes(total_force_aero.T, normal)

    return thrust


def get_thrust_residual(model_options, atmos, wind, variables, parameters, outputs, parent, architecture):

    thrust_val = get_actuator_thrust(model_options, variables, outputs, parent, architecture)

    area_var = geom.get_area_var(model_options, variables, parent, parameters)
    qapp_var = flow.get_qapp_var(atmos, wind, variables, parent)

    ct_var = get_ct_var(model_options, variables, parent)

    resi_unscaled = thrust_val - ct_var * area_var * qapp_var

    thrust_ref = get_thrust_ref(model_options, atmos, wind, parameters)

    resi_scaled = resi_unscaled / thrust_ref
    return resi_scaled


def get_ct_ref(model_options):
    a_ref = flow.get_a_ref(model_options)
    ct_ref = 4. * a_ref * (1. - a_ref)

    return ct_ref

def get_thrust_ref(model_options, atmos, wind, parameters):

    qapp_ref = flow.get_qapp_ref(atmos, wind)
    area_ref = geom.get_area_ref(model_options, parameters)
    ct_ref = get_ct_ref(model_options)

    thrust_ref = ct_ref * qapp_ref * area_ref

    scaling = model_options['aero']['actuator']['scaling']
    reference = scaling * thrust_ref

    return reference
