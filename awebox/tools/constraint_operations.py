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
model (time-independent) constraint handling
_python-3.5 / casadi-3.4.5
- authors: jochem de schutter, rachel leuthold alu-fr 2017-20
'''

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op

import pdb

class Constraint:
    def __init__(self, expr=None, cstr_type=None, name=None, include=True, scale=1.):

        self.__expr = None
        self.__cstr_type = None
        self.__name = None
        self.__include = False
        self.__scale = 1.

        expr_is_expected = self.is_expr_as_expected(expr)
        if expr_is_expected:
            self.__expr = expr
        else:
            message = 'unexpected constraint expression: ' + str(expr) + ', for constraint ' + name
            awelogger.logger.warning(message)

        cstr_type_is_expected = self.is_cstr_type_as_expected(cstr_type)
        if cstr_type_is_expected:
            self.__cstr_type = cstr_type
        else:
            message = 'unexpected constraint type: ' + str(cstr_type) + ', for constraint ' + name
            awelogger.logger.warning(message)

        name_is_expected = self.is_name_as_expected(name)
        if name_is_expected:
            self.__name = name
        else:
            message = 'unexpected constraint name: ' + name
            awelogger.logger.warning(message)

        scale_is_expected = self.is_scale_expected(scale)
        if scale_is_expected:
            self.__scale = scale
        else:
            message = 'unexpected constraint scale: ' + str(scale) + ', for constraint ' + name
            awelogger.logger.warning(message)

        if expr_is_expected and cstr_type_is_expected and name_is_expected and isinstance(include, bool):
            self.__include = include

    def is_constraint_complete(self):
        return (self.__expr is not None) and (self.__name is not None) and (self.__cstr_type is not None)

    def is_scale_expected(self, scale):

        if isinstance(scale, int):
            scale = float(scale)
        float_scale_is_expected = isinstance(scale, float)

        casadi_scale_is_expected = isinstance(scale, cas.DM)

        return casadi_scale_is_expected or float_scale_is_expected

    def is_expr_as_expected(self, expr):

        if (isinstance(expr, list) and not expr):
            return False

        rows_as_expected = (expr.shape[0] > 0)
        cols_as_expected = (expr.shape[1] == 1)

        type_as_expected = isinstance(expr, cas.SX) or isinstance(expr, cas.DM) or isinstance(expr, cas.MX)
        type_is_not_constant = not expr.is_constant()
        return (rows_as_expected and cols_as_expected and type_as_expected and type_is_not_constant)
        
    def is_cstr_type_as_expected(self, cstr_type):
        type_is_string = isinstance(cstr_type, str)
        return type_is_string and (cstr_type == 'eq' or cstr_type == 'ineq')

    def is_name_as_expected(self, name):
        name_is_string = isinstance(name, str)
        return name_is_string

    def is_equality(self):
        return self.__cstr_type == 'eq'

    def is_inequality(self):
        return self.__cstr_type == 'ineq'

    def get_function(self, variables, parameters):

        output = self.__expr
        return cas.Function('cstr_fun', [variables, parameters], [output])

    @property
    def expr(self):
        return self.__expr

    @expr.setter
    def expr(self, value):
        awelogger.logger.warning('Cannot set expr object.')
        
    @property
    def cstr_type(self):
        return self.__cstr_type

    @cstr_type.setter
    def cstr_type(self, value):
        awelogger.logger.warning('Cannot set cstr_type object.')

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        awelogger.logger.warning('Cannot set name object.')

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        awelogger.logger.warning('Cannot set scale object.')

    @property
    def include(self):
        return self.__include

    @include.setter
    def include(self, value):
        awelogger.logger.warning('Cannot set include object.')


class ConstraintList:
    def __init__(self):
        self.__eq_list = []
        self.__ineq_list = []
        self.__all_list = []

    def append(self, cstr):

        if isinstance(cstr, Constraint):

            if cstr.is_equality():
                self.append_eq(cstr)
                self.append_all(cstr)
            elif cstr.is_inequality():
                self.append_ineq(cstr)
                self.append_all(cstr)
            else:
                message = 'tried to append constraint (' + cstr.name + ') of unexpected cstr_type. append ignored.'
                awelogger.logger.warning(message)

        elif isinstance(cstr, ConstraintList):
            # preserve order during import
            for local_cstr in cstr.all_list:
                self.append(local_cstr)

        elif isinstance(cstr, list) and not cstr:
            message = 'tried to append empty list as constraint. append ignored.'
            awelogger.logger.warning(message)

        else:
            message = 'tried to append constraint of unexpected object type. append ignored.'
            awelogger.logger.warning(message)

        return None

    def append_ineq(self, cstr):
        is_complete = cstr.is_constraint_complete()
        is_inequality = cstr.is_inequality()
        is_new_constraint = self.does_new_constraint_have_new_name(cstr, 'ineq')

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_inequality:
            message = 'tried to append a constraint (' + cstr.name + ') that is not an inequality to ineq_list. append ignored.'
            awelogger.logger.warning(message)
        elif is_complete and is_new_constraint and is_inequality:
            self.__ineq_list += [cstr]
        else:
            message = 'something went wrong when appending constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)

        return None

    def append_eq(self, cstr):
        is_complete = cstr.is_constraint_complete()
        is_equality = cstr.is_equality()
        is_new_constraint = self.does_new_constraint_have_new_name(cstr, 'eq')

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_equality:
            message = 'tried to append a constraint (' + cstr.name + ') that is not an equality to eq_list. append ignored.'
            awelogger.logger.warning(message)
        elif is_complete and is_new_constraint and is_equality:
            self.__eq_list += [cstr]
        else:
            message = 'something went wrong when appending constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)

        return None

    def append_all(self, cstr):
        is_complete = cstr.is_constraint_complete()
        is_new_constraint = self.does_new_constraint_have_new_name(cstr, 'all')

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif is_complete and is_new_constraint:
            self.__all_list += [cstr]
        else:
            message = 'something went wrong when appending constraint (' + cstr.name + '). append ignored.'
            awelogger.logger.warning(message)

        return None

    def does_new_constraint_have_new_name(self, cstr, cstr_type):
        return not (cstr.name in self.get_name_list(cstr_type))

    def get_list(self, cstr_type):
        if cstr_type == 'eq':
            list = self.__eq_list
        elif cstr_type == 'ineq':
            list = self.__ineq_list
        elif cstr_type == 'all':
            list = self.__all_list
        else:
            message = 'unexpected model constraint type'
            awelogger.logger.error(message)
            raise Exception(message)

        return list

    def get_expression_list(self, cstr_type):

        cstr_list = self.get_list(cstr_type)

        expr_list = []
        for cstr in cstr_list:
            if cstr.include:
                local_expr = cstr.expr / cstr.scale
                expr_list = cas.vertcat(expr_list, local_expr)

        return expr_list

    def get_name_list(self, cstr_type):

        list = self.get_list(cstr_type)

        name_list = []
        for cstr in list:
            if cstr.include:
                local_expr = cstr.expr
                number = local_expr.shape[0]

                local_name = cstr.name
                name_list += number * [local_name]

        return name_list

    def get_function(self, options, relevant_variables, relevant_parameters, cstr_type):

        expr_list = self.get_expression_list(cstr_type)

        # constraints function options
        if options['jit_code_gen']['include']:
            opts = {'jit': True, 'compiler': options['jit_code_gen']['compiler']}
        else:
            opts = {}

        # create function
        return cas.Function('cstr_fun', [relevant_variables, relevant_parameters], [expr_list], opts)

    def get_dict(self):

        dict = {}
        type_dict = {'equality': self.get_list('eq'), 'inequality': self.get_list('ineq')}

        for cstr_type in type_dict.keys():
            cstr_list = type_dict[cstr_type]

            if cstr_type not in dict.keys():
                dict[cstr_type] = {}

            for cstr in cstr_list:
                local_name, local_id = struct_op.split_name_and_node_identifier(cstr.name)

                if local_name not in dict[cstr_type].keys():
                    dict[cstr_type][local_name] = []

                dict[cstr_type][local_name] = cas.vertcat(dict[cstr_type][local_name], cstr.expr)

        return dict

    @property
    def eq_list(self):
        return self.__eq_list

    @eq_list.setter
    def eq_list(self, value):
        awelogger.logger.warning('Cannot set eq_list object.')

    @property
    def ineq_list(self):
        return self.__ineq_list

    @ineq_list.setter
    def ineq_list(self, value):
        awelogger.logger.warning('Cannot set ineq_list object.')

    @property
    def all_list(self):
        return self.__all_list

    @all_list.setter
    def all_list(self, value):
        awelogger.logger.warning('Cannot set all_list object.')

