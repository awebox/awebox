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
model (time-independent) constraint handling
_python-3.5 / casadi-3.4.5
- authors: jochem de schutter, rachel leuthold alu-fr 2017-20
'''

import casadi.tools as cas
import awebox.tools.constraint_operations as cstr_op
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op


class MdlConstraintList(cstr_op.ConstraintList):
    def __init__(self):
        super().__init__(list_name='model_constraints_list')

    def get_structure(self, cstr_type):

        cstr_list = self.get_list(cstr_type)

        entry_list = []
        for cstr in cstr_list:
            joined_name = cstr.name
            local = cas.entry(joined_name, shape=cstr.expr.shape)
            entry_list.append(local)

        return cas.struct_symSX(entry_list)

    def get_dict(self):
        dict = {}
        dict['equality'] = self.get_structure('eq')(self.get_expression_list('eq'))
        dict['inequality'] = self.get_structure('ineq')(self.get_expression_list('ineq'))
        return dict