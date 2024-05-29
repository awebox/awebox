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
performance helping file,
finds various total-trajectory performance metrics requiring knowlege of V
_python-3.5 / casadi-3.4.5
- authors: rachel leuthold, jochem de schutter alu-fr 2017-18
'''


import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op
import awebox.mdl.aero.induction_dir.vortex_dir.vortex as vortex


def collect_global_outputs(nlp_options, Outputs, Outputs_structured, Integral_outputs, Integral_outputs_fun, model, V, P):

    global_outputs = {}
    global_outputs = include_time_period(nlp_options, V, global_outputs)

    if 'Outputs_structured' != None and ('vortex' in model.outputs.keys()):
        global_outputs = vortex.compute_global_performance(global_outputs, Outputs_structured, model.architecture)

    [outputs_struct, outputs_dict] = make_output_structure(global_outputs)

    return outputs_struct, outputs_dict


def include_time_period(nlp_options, V, outputs):

    if 'time_period' not in list(outputs.keys()):
        outputs['time_period'] = {}

    time_period = find_time_period(nlp_options, V)

    outputs['time_period']['val'] = time_period

    return outputs


def make_output_structure(outputs):
    outputs_vec = []
    full_list = []

    outputs_dict = {}

    for output_type in list(outputs.keys()):

        local_list = []
        for name in list(outputs[output_type].keys()):
            # prepare empty entry list to generate substruct
            local_list += [cas.entry(name, shape=outputs[output_type][name].shape)]

            # generate vector with outputs - SX expressions
            outputs_vec = cas.vertcat(outputs_vec, outputs[output_type][name])

        # generate dict with sub-structs
        outputs_dict[output_type] = cas.struct_symMX(local_list)
        # prepare list with sub-structs to generate struct
        full_list += [cas.entry(output_type, struct=outputs_dict[output_type])]

        # generate "empty" structure
    out_struct = cas.struct_symMX(full_list)
    # generate structure with SX expressions
    outputs_struct = out_struct(outputs_vec)

    return [outputs_struct, outputs_dict]


def find_time_spent_in_reelout(nlp_numerics_options, V):

    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_zeroth = V['theta', 't_f', 0] * round(nk * phase_fix_reel_out) / nk
    return time_period_zeroth

def find_time_spent_in_reelin(nlp_numerics_options, V):
    nk = nlp_numerics_options['n_k']
    phase_fix_reel_out = nlp_numerics_options['phase_fix_reelout']
    time_period_first = V['theta', 't_f', 1] * (nk - round(nk * phase_fix_reel_out)) / nk
    return time_period_first

def find_time_period(nlp_numerics_options, V):

    if nlp_numerics_options['phase_fix'] == 'single_reelout':
        reelout_time = find_time_spent_in_reelout(nlp_numerics_options, V)
        reelin_time = find_time_spent_in_reelin(nlp_numerics_options, V)

        time_period = (reelout_time + reelin_time)
    else:
        time_period = V['theta', 't_f']

    return time_period

