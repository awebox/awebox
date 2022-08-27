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
various structural tools for the vortex model
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2019-2021
'''
import pdb

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger

import awebox.mdl.aero.induction_dir.general_dir.flow as general_flow

import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op
import awebox.mdl.architecture as archi


def get_option_from_possible_dicts(options, name, actuator_or_vortex):

    specific_name = 'vortex_' + name

    has_induction_toplevel = 'induction' in options.keys()
    name_in_induction_keys = has_induction_toplevel and (name in options['induction'].keys())
    specific_name_in_induction_keys = has_induction_toplevel and (specific_name in options['induction'].keys())

    has_aero_toplevel = 'aero' in options.keys()
    has_specific_midlevel = has_aero_toplevel and (actuator_or_vortex in options['aero'].keys())
    name_in_aero_specific_keys = has_specific_midlevel and (name in options['aero'][actuator_or_vortex].keys())

    has_induction_midlevel = has_aero_toplevel and ('induction' in options['aero'].keys())
    name_in_aero_induction_keys = has_induction_midlevel and (name in options['aero']['induction'].keys())

    has_model_toplevel = 'model' in options.keys()
    has_aero_midlevel = has_model_toplevel and ('aero' in options['model'].keys())
    has_specific_lowlevel = has_aero_midlevel and (actuator_or_vortex in options['model']['aero'].keys())
    name_in_model_aero_specific_keys = has_specific_lowlevel and (name in options['model']['aero'][actuator_or_vortex].keys())

    if specific_name_in_induction_keys:
        value = options['induction'][specific_name]
    elif name_in_induction_keys:
        value = options['induction'][name]
    elif name_in_aero_specific_keys:
        value = options['aero'][actuator_or_vortex][name]
    elif name_in_aero_induction_keys:
        value = options['aero']['induction'][name]
    elif name_in_model_aero_specific_keys:
        value = options['model']['aero'][actuator_or_vortex][name]
    else:
        message = 'no available information about the desired option (' + name + ') found.'
        awelogger.logger.error(message)
        raise Exception(message)

    return value