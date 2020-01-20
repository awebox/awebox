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
kite aerodynamics model of an awe system
takes states and inputs and creates aerodynamic forces and moments
dependent on the position of the kite.
_aerodynamic coefficients are assumptions.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2017-2019
'''

import awebox.mdl.aero.induction_dir.induction as induction
import awebox.mdl.aero.indicators as indicators

from . import three_dof_kite

from . import six_dof_kite

def get_forces_and_moments(options, atmos, wind, variables, outputs, parameters, architecture):

    if int(options['kite_dof']) == 3:
        outputs = three_dof_kite.get_outputs(options, atmos, wind, variables, outputs, parameters, architecture)
    elif int(options['kite_dof']) == 6:
        outputs = six_dof_kite.get_outputs(options, atmos, wind, variables, outputs, parameters, architecture)
    else:
        raise ValueError('failure: unsupported kite_dof chosen in options: %i',options['kite_dof'])

    outputs = indicators.get_performance_outputs(options, atmos, wind, variables, outputs, parameters, architecture)

    if not (options['induction_model'] == 'not_in_use'):
        outputs = induction.collect_outputs(options, atmos, wind, variables, outputs, parameters, architecture)

    return outputs

def get_wingtip_position(kite, options, model, variables, parameters, ext_int):
    if int(options['kite_dof']) == 3:
        wingtip_pos = three_dof_kite.get_wingtip_position(kite, options, model, variables, parameters, ext_int)
    elif int(options['kite_dof']) == 6:
        wingtip_pos = six_dof_kite.get_wingtip_position(kite, model, variables, parameters, ext_int)
    else:
        raise ValueError('failure: unsupported kite_dof chosen in options: %i',options['kite_dof'])

    return wingtip_pos