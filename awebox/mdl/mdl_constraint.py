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
- author: rachel leuthold, alu-fr 2020
'''

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import pdb

class MdlConstraint:
    def __init__(self, expr=None, cstr_type=None, name=None, include=True, ref=1.):

        self.__expr = None
        self.__cstr_type = None
        self.__name = None
        self.__include = False
        self.__ref = 1.

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

        ref_is_expected = self.is_ref_expected(ref)
        if ref_is_expected:
            self.__ref = ref
        else:
            message = 'unexpected constraint reference: ' + str(ref) + ', for constraint ' + name
            awelogger.logger.warning(message)

        if expr_is_expected and cstr_type_is_expected and name_is_expected and isinstance(include, bool):
            self.__include = include

    def is_constraint_complete(self):
        return (self.__expr is not None) and (self.__name is not None) and (self.__cstr_type is not None)

    def is_ref_expected(self, ref):
        if isinstance(ref, int):
            ref = float(ref)

        is_casadi_ref = (isinstance(ref, cas.DM) or isinstance(ref, cas.SX) or isinstance(ref, cas.MX))
        casadi_ref_is_expected = is_casadi_ref
        float_ref_is_expected = isinstance(ref, float)

        return casadi_ref_is_expected or float_ref_is_expected

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
    def ref(self):
        return self.__ref

    @ref.setter
    def ref(self, value):
        awelogger.logger.warning('Cannot set ref object.')

    @property
    def include(self):
        return self.__include

    @include.setter
    def include(self, value):
        awelogger.logger.warning('Cannot set include object.')


class MdlConstraintList:
    def __init__(self):
        self.__eq_list = []
        self.__ineq_list = []

    def append(self, mdl_constraint):

        if isinstance(mdl_constraint, MdlConstraint):

            if mdl_constraint.is_equality():
                self.append_eq(mdl_constraint)
            elif mdl_constraint.is_inequality():
                self.append_ineq(mdl_constraint)
            else:
                message = 'tried to append constraint (' + mdl_constraint.name + ') of unexpected cstr_type. append ignored.'
                awelogger.logger.warning(message)

        elif isinstance(mdl_constraint, MdlConstraintList):
            for cstr in mdl_constraint.eq_list:
                self.append(cstr)
            for cstr in mdl_constraint.ineq_list:
                self.append(cstr)

        elif isinstance(mdl_constraint, list) and not mdl_constraint:
            message = 'tried to append empty list as constraint. append ignored.'
            awelogger.logger.warning(message)

        else:
            message = 'tried to append constraint of unexpected object type. append ignored.'
            awelogger.logger.warning(message)

        return None
        
    def append_ineq(self, mdl_constraint):
        is_complete = mdl_constraint.is_constraint_complete()
        is_inequality = mdl_constraint.is_inequality()
        is_new_constraint = self.does_new_constraint_have_new_name(mdl_constraint)

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_inequality:
            message = 'tried to append a constraint (' + mdl_constraint.name + ') that is not an inequality to ineq_list. append ignored.'
            awelogger.logger.warning(message)
        elif is_complete and is_new_constraint and is_inequality:
            self.__ineq_list += [mdl_constraint]
        else:
            message = 'something went wrong when appending constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)

        return None
        
    def append_eq(self, mdl_constraint):
        is_complete = mdl_constraint.is_constraint_complete()
        is_equality = mdl_constraint.is_equality()
        is_new_constraint = self.does_new_constraint_have_new_name(mdl_constraint)

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)
        elif not is_equality:
            message = 'tried to append a constraint (' + mdl_constraint.name + ') that is not an equality to eq_list. append ignored.'
            awelogger.logger.warning(message)
        elif is_complete and is_new_constraint and is_equality:
            self.__eq_list += [mdl_constraint]
        else:
            message = 'something went wrong when appending constraint (' + mdl_constraint.name + '). append ignored.'
            awelogger.logger.warning(message)

        return None

    def does_new_constraint_have_new_name(self, mdl_constraint):
        not_in_eqs_names = not (mdl_constraint.name in self.get_name_list('eq'))
        not_in_ineqs_names = not (mdl_constraint.name in self.get_name_list('ineq'))
        return (not_in_eqs_names and not_in_ineqs_names)

    def get_list(self, cstr_type):
        if cstr_type == 'eq':
            list = self.__eq_list
        elif cstr_type == 'ineq':
            list = self.__ineq_list
        else: 
            message = 'unexpected model constraint type'
            awelogger.logger.error(message)
            raise Exception(message)
        
        return list
        
    def get_expression_list(self, cstr_type):
        
        cstr_list = self.get_list(cstr_type)

        expr_list = []
        for mdl_cstr in cstr_list:
            if mdl_cstr.include:
                local_expr = mdl_cstr.expr / mdl_cstr.ref
                expr_list = cas.vertcat(expr_list, local_expr)
        
        return expr_list

    def get_name_list(self, cstr_type):

        list = self.get_list(cstr_type)

        name_list = []
        for mdl_cstr in list:
            if mdl_cstr.include:

                local_expr = mdl_cstr.expr
                number = local_expr.shape[0]

                local_name = mdl_cstr.name
                name_list += number * [local_name]

        return name_list

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



def test():

    rdx = 0
    results = {}

    # can we make a MdlConstraint?
    var = cas.SX.sym('var')
    expr = cas.vertcat(var**2. - 2., 8. * var)
    cstr_type = 'eq'
    name = 'cstr1'
    cstr1 = MdlConstraint(expr=expr, name=name, cstr_type=cstr_type, include=True, ref=1.)

    # is the length of that constraint as expected?
    results[rdx] = (cstr1.expr.shape == (2, 1))
    rdx += 1

    # can we make a MdlConstraintList?
    cstr_list = MdlConstraintList()

    # are the lengths of the eq_list and ineq_list both zero?
    results[rdx] = (len(cstr_list.eq_list) == 0) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we add non-empty constraints to cstr_list?
    expr2 = var + 4.
    cstr_type2 = 'eq'
    name2 = 'cstr2'
    cstr2 = MdlConstraint(expr=expr2, name=name2, cstr_type=cstr_type2, include=True, ref=1.)

    cstr_list.append(cstr1)
    cstr_list.append(cstr2)

    # does the list record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # is the number of expressions in the equality constraints == 3?
    results[rdx] = (cstr_list.get_expression_list('eq').shape == (3, 1))
    rdx += 1

    # can we add an empty list to the cstr_list?
    cstr_list.append([])

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make an incomplete constraint?
    expr3 = []
    cstr_type3 = 'eq'
    name3 = 'cstr3'
    cstr3 = MdlConstraint(expr=expr3, name=name3, cstr_type=cstr_type3, include=True, ref=1.)

    # can we add the incomplete constraint to the list?
    cstr_list.append(cstr3)

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make an empty list?
    cstr_list_empty = MdlConstraintList()

    # can we add the empty list to the existing list?
    cstr_list.append(cstr_list_empty)

    # does the list still record two equality constraints and 0 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 0)
    rdx += 1

    # can we make a non-empty list, and append it to the main list?
    cstr_list_nonempty = MdlConstraintList()
    expr4 = var + 8.
    cstr_type4 = 'ineq'
    name4 = 'cstr4'
    cstr4 = MdlConstraint(expr=expr4, name=name4, cstr_type=cstr_type4, include=True, ref=1.)
    cstr_list_nonempty.append(cstr4)
    cstr_list.append(cstr_list_nonempty)

    # does the list now record two equality constraints and 1 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 1)
    rdx += 1

    # can we make a constraint with a duplicate name, and append it to the main list?
    expr5 = cas.sin(var) + 8.
    cstr_type5 = 'eq'
    name5 = 'cstr4'
    cstr5 = MdlConstraint(expr=expr5, name=name5, cstr_type=cstr_type5, include=True, ref=1.)
    cstr_list.append(cstr5)

    # does the list still record two equality constraints and 1 inequality constraints?
    results[rdx] = (len(cstr_list.eq_list) == 2) and (len(cstr_list.ineq_list) == 1)
    rdx += 1

    # get the 'dynamics'
    dynamics = cstr_list.get_expression_list('eq')


    ##############
    # summarize results
    ##############

    print(results)
    for res_value in results.values():
        assert(res_value)

    return None

test()