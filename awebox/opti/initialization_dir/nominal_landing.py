#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
initialization functions specific to the nominal landing scenario
_python _version 2.7 / casadi-3.4.5
- _author: rachel leuthold, jochem de schutter, thilo bronnenmeyer (alu-fr, 2017 - 20)
'''

import numpy as np
import casadi.tools as cas
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.opti.initialization_dir.induction as induction_init
import awebox.opti.initialization_dir.tools as tools_init
import awebox.opti.initialization_dir.general_landing as general_landing

def get_normalized_time_param_dict(ntp_dict, formulation):
    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    n0 = len(V_0['coll_var', :, :, 'xd'])
    d0 = len(V_0['coll_var', 0, :, 'xd']) - 1
    n_min = min_dl_t_arg / (d0 + 1)
    d_min = min_dl_t_arg - min_dl_t_arg / (d0 + 1) * (d0 + 1)

    ntp_dict['n0'] = n0
    ntp_dict['n_min'] = n_min
    ntp_dict['d_min'] = d_min

    return ntp_dict

def set_normalized_time_params(formulation, V_init):
    V_0 = formulation.xi_dict['V_pickle_initial']
    min_dl_t_arg = np.argmin(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']))
    xi_0_init = min_dl_t_arg / float(np.array(V_0['coll_var', :, :, 'xd', 'dl_t']).size)
    xi_f_init = 0.0
    V_init['xi', 'xi_0'] = xi_0_init
    V_init['xi', 'xi_f'] = xi_f_init

    return V_init

def guess_final_time(init_options, formulation, ntp_dict):
    tf_guess = general_landing.guess_final_time(init_options, formulation, ntp_dict)
    return tf_guess

def guess_values_at_time(t, init_options, model, formulation, tf_guess, ntp_dict):
    ret = general_landing(t, init_options, model, formulation, tf_guess, ntp_dict)
    return ret