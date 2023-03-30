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
import pdb

import casadi.tools as cas
from awebox.logger.logger import Logger as awelogger
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op


class Constraint:
    def __init__(self, expr=None, cstr_type=None, name=None):

        self.__expr = None
        self.__cstr_type = None
        self.__name = None

        expr_is_expected = self.is_expr_as_expected(expr)
        if expr_is_expected:
            self.__expr = expr
        else:
            message = 'unexpected constraint expression: ' + str(expr) + ', for constraint ' + repr(name)
            awelogger.logger.warning(message)

        cstr_type_is_expected = self.is_cstr_type_as_expected(cstr_type)
        if cstr_type_is_expected:
            self.__cstr_type = cstr_type
        else:
            message = 'unexpected constraint type: ' + str(cstr_type) + ', for constraint ' + repr(name)
            awelogger.logger.warning(message)

        name_is_expected = self.is_name_as_expected(name)
        if name_is_expected:
            self.__name = name
        else:
            message = 'unexpected constraint name: ' + repr(name)
            awelogger.logger.warning(message)

    def is_constraint_complete(self):
        return (self.__expr is not None) and (self.__name is not None) and (self.__cstr_type is not None)

    def is_expr_as_expected(self, expr):

        if (isinstance(expr, list) and not expr):
            return False
        if not hasattr(expr, 'shape'):
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

    def get_lb(self):
        if self.is_inequality():
            return -1 * cas.inf * cas.DM.ones(self.expr.shape)
        elif self.is_equality():
            return cas.DM.zeros(self.expr.shape)
        else:
            message = 'unable to get lower bound for constraint ' + self.name + '. unexpected constraint type: ' + str(self.cstr_type) + '.'
            awelogger.logger.warning(message)
        return None

    def get_ub(self):
        if self.is_inequality() or self.is_equality():
            return cas.DM.zeros(self.expr.shape)
        else:
            message = 'unable to get lower bound for constraint ' + self.name + '. unexpected constraint type: ' + str(self.cstr_type) + '.'
            awelogger.logger.warning(message)
        return None

    def scale(self, scaling):
        self.expr = scaling * self.expr
        return None

    @property
    def expr(self):
        return self.__expr

    @expr.setter
    def expr(self, value):
        self.__expr = value
        
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




class ConstraintList:
    def __init__(self, list_name=None):
        self.__eq_list = []
        self.__ineq_list = []
        self.__all_list = []

        if list_name is None:
            list_name = 'constraint list'

        self.__list_name = list_name

    def append(self, cstr):

        if isinstance(cstr, Constraint):

            # appending to all should happen before appending to specific list,
            # based on how the "is duplicate" check is done.

            if cstr.is_equality():
                self.append_all(cstr)
                self.append_eq(cstr)

            elif cstr.is_inequality():
                self.append_all(cstr)
                self.append_ineq(cstr)

            else:
                message = 'tried to append constraint (' + cstr.name + ') of unexpected cstr_type. append ignored.'
                awelogger.logger.warning(message)

        elif isinstance(cstr, ConstraintList):
            passes = self.check_compatibility_before_appending_constraint_list(cstr)
            if passes:
                # preserve order during import
                for local_cstr in cstr.all_list:
                    self.append(local_cstr)

        elif isinstance(cstr, list) and not cstr:
            message = 'tried to append empty list as constraint. append ignored.'
            awelogger.logger.warning(message)

        elif isinstance(cstr, list):
            message = 'tried to append a list of constraints, without using a ConstraintList object. append ignored.'
            awelogger.logger.warning(message)

        else:
            message = 'tried to append constraint of unexpected object type. append ignored.'
            awelogger.logger.warning(message)

        return None

    def check_compatibility_before_appending_constraint_list(self, cstr):
        passes = True

        is_compatible = self.is_new_constraint_compatible(cstr)
        if not is_compatible:
            message = 'tried to append an over-specialized or wrongly-specialized constraint list to an under-specialized constraint list. append ignored.'
            awelogger.logger.warning(message)
            passes = False

        return passes


    def check_completeness_newness_and_type_before_appending_individual_constraint(self, cstr, cstr_type):
        passes = True

        is_complete = cstr.is_constraint_complete()
        is_new_constraint = self.does_new_constraint_have_new_name(cstr)

        if cstr_type == 'eq':
            is_correct_type = cstr.is_equality()
        elif cstr_type == 'ineq':
            is_correct_type = cstr.is_inequality()
        elif cstr_type == 'all':
            is_correct_type = cstr.is_equality() or cstr.is_inequality()

        if not is_complete:
            message = 'tried to append an incomplete constraint (' + repr(cstr.name) + ') to ' + self.__list_name + '. append ignored.'
            awelogger.logger.warning(message)
            passes = False
        elif not is_new_constraint:
            message = 'tried to append a duplicate constraint (' + repr(cstr.name) + ') to ' + self.__list_name + '. append ignored.'
            awelogger.logger.warning(message)
            passes = False
        elif not is_correct_type:
            message = 'tried to append a constraint (' + repr(cstr.name) + ') that is not of type ' + cstr_type + ' to appropriate list of ' + self.__list_name + '. append ignored.'
            awelogger.logger.warning(message)
            passes = False

        return passes



    def append_ineq(self, cstr):
        passes = self.check_completeness_newness_and_type_before_appending_individual_constraint(cstr, 'ineq')
        if passes:
            self.__ineq_list += [cstr]
        return None

    def append_eq(self, cstr):
        passes = self.check_completeness_newness_and_type_before_appending_individual_constraint(cstr, 'eq')
        if passes:
            self.__eq_list += [cstr]
        return None

    def append_all(self, cstr):

        passes = self.check_completeness_newness_and_type_before_appending_individual_constraint(cstr, 'all')
        if passes:
            self.__all_list += [cstr]
        return None

    def is_new_constraint_compatible(self, cstr):
        # want to be able to add:
        # (x) Constraint to ConstraintList
        # (x) ConstraintList to ConstraintList
        # but not:
        # (x) MdlConstraintList to ConstraintList
        # (x) OcpConstraintList to ConstraintList

        if type(self) == type(cstr):
            return True
        elif isinstance(cstr, MdlConstraintList) or isinstance(cstr, OcpConstraintList):
            return False
        elif isinstance(cstr, Constraint) or isinstance(cstr, ConstraintList):
            return True
        else:
            return False

    def does_new_constraint_have_new_name(self, cstr):
        return not (cstr.name in self.get_name_list(cstr.cstr_type))

    def get_list(self, cstr_type):
        if cstr_type == 'eq':
            list = self.__eq_list
        elif cstr_type == 'ineq':
            list = self.__ineq_list
        elif cstr_type == 'all':
            list = self.__all_list
        else:
            message = 'unexpected model constraint type: ' + repr(cstr_type)
            print_op.log_and_raise_error(message)

        return list

    def get_expression_list(self, cstr_type):

        cstr_list = self.get_list(cstr_type)

        expr_list = []
        for cstr in cstr_list:
            local_expr = cstr.expr
            expr_list = cas.vertcat(expr_list, local_expr)

        return expr_list

    def get_lb(self, cstr_type):
        cstr_list = self.get_list(cstr_type)
        lb_list = []
        for cstr in cstr_list:
            local_lb = cstr.get_lb()
            lb_list = cas.vertcat(lb_list, local_lb)
        return lb_list

    def get_ub(self, cstr_type):
        cstr_list = self.get_list(cstr_type)
        ub_list = []
        for cstr in cstr_list:
            local_ub = cstr.get_ub()
            ub_list = cas.vertcat(ub_list, local_ub)
        return ub_list

    def get_name_list(self, cstr_type):

        list = self.get_list(cstr_type)

        name_list = []
        for cstr in list:
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

    def get_constraint_by_name(self, name):
        for cstr in self.__all_list:
            if cstr.name == name:
                return cstr

        message = 'no constraint found with searched name. returning None object'
        awelogger.logger.warning(message)

        return None

    def scale(self, scaling):

        cstr_type_list = ['eq', 'ineq', 'all']
        for cstr_type in cstr_type_list:

            list = self.get_list(cstr_type)
            for cstr in list:
                cstr.scale(scaling)

        return None

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

    @property
    def list_name(self):
        return self.__list_name

    @list_name.setter
    def list_name(self, value):
        awelogger.logger.warning('Cannot set list_name object.')

class MdlConstraintList(ConstraintList):
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

    def is_new_constraint_compatible(self, cstr):
        # want to be able to add:
        # (x) Constraint to ConstraintList
        # (x) and ConstraintList to MdlConstraintList
        # (x) and MdlConstraintList to MdlConstraintList
        # but not:
        # (x) OcpConstraintList to MdlConstraintList

        if type(self) == type(cstr):
            return True
        elif isinstance(cstr, OcpConstraintList):
            return False
        elif isinstance(cstr, Constraint) or isinstance(cstr, ConstraintList):
            return True
        else:
            return False



class OcpConstraintList(ConstraintList):

    def __init__(self):
        super().__init__(list_name='ocp_constraints_list')

    def is_new_constraint_compatible(self, cstr):
        # want to be able to add:
        # (x) Constraint to OcpConstraintList
        # (x) and ConstraintList to OcpConstraintList
        # (x) and OcpConstraintList to OcpConstraintList
        # but not:
        # or MdlConstraintList to OcpConstraintList

        if type(self) == type(cstr):
            return True
        elif isinstance(cstr, MdlConstraintList):
            return False
        elif isinstance(cstr, Constraint) or isinstance(cstr, ConstraintList):
            return True
        else:
            return False

def make_test_constraint(val=2., cstr_type='eq'):
    var = cas.SX.sym('var')
    expr = var - val
    name = 'test_' + cstr_type + '_' + str(val)
    cstr = Constraint(
        expr=expr,
        cstr_type=cstr_type,
        name=name
    )
    return cstr

def make_test_constraint_list(addition=0.):

    cstr_list = ConstraintList()

    val = 0. + addition
    eq_cstr_0 = make_test_constraint(val=val, cstr_type='eq')
    cstr_list.append(eq_cstr_0)

    val += 1.
    eq_cstr_1 = make_test_constraint(val=val, cstr_type='eq')
    cstr_list.append(eq_cstr_1)

    val += 1.
    ineq_cstr_0 = make_test_constraint(val=val, cstr_type='ineq')
    cstr_list.append(ineq_cstr_0)

    return cstr_list


def make_test_ocp_constraint_list(addition=0.):
    ocp_cstr_list = OcpConstraintList()
    cstr_list = make_test_constraint_list(addition)
    ocp_cstr_list.append(cstr_list)
    return ocp_cstr_list

def make_test_mdl_constraint_list(addition=0.):
    mdl_cstr_list = MdlConstraintList()
    cstr_list = make_test_constraint_list(addition)
    mdl_cstr_list.append(cstr_list)
    return mdl_cstr_list


def test_that_constraint_list_appending_works_for_distinct_individual_constraints():

    cstr_list = make_test_constraint_list(addition=0.)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending distinct individual constraints.'
        print_op.log_and_raise_error(message)


def test_that_constraint_list_appending_works_for_distinct_constraint_lists():
    cstr_list = make_test_constraint_list(addition=0.)

    cstr_list2 = make_test_constraint_list(addition=5.)
    cstr_list.append(cstr_list2)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 2 * len(cstr_list2.all_list))
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2 * len(cstr_list2.eq_list))
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 2 * len(cstr_list2.ineq_list))

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending two distinct constraint lists'
        print_op.log_and_raise_error(message)


def test_that_constraint_list_appending_works_for_constraint_list_onto_ocp_constraint_list():
    cstr_list = make_test_ocp_constraint_list(addition=0.)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending constraint list onto ocp constraint list'
        print_op.log_and_raise_error(message)

def test_that_constraint_list_appending_works_for_constraint_list_onto_mdl_constraint_list():
    cstr_list = make_test_mdl_constraint_list(addition=0.)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending constraint list onto mdl constraint list'
        print_op.log_and_raise_error(message)

def test_that_constraint_list_ignores_append_if_specialized_constraint_lists_are_added_to_regular_constraint_list():

    cstr_list = make_test_constraint_list(addition=0.)
    ocp_cstr_list = make_test_ocp_constraint_list(addition=5.)
    mdl_cstr_list = make_test_mdl_constraint_list(addition=10.)

    cstr_list.append(ocp_cstr_list)
    cstr_list.append(mdl_cstr_list)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending specialized constraint list onto regular constraint list'
        print_op.log_and_raise_error(message)


def test_that_constraint_list_ignores_append_if_mdl_constraint_lists_is_added_to_ocp_constraint_list():
    ocp_cstr_list = make_test_ocp_constraint_list(addition=5.)
    mdl_cstr_list = make_test_mdl_constraint_list(addition=10.)

    ocp_cstr_list.append(mdl_cstr_list)

    correct_number_of_constraints_total = (len(ocp_cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(ocp_cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(ocp_cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending mdl constraint list onto ocp constraint list'
        print_op.log_and_raise_error(message)


def test_that_constraint_list_ignores_append_if_ocp_constraint_lists_is_added_to_mdl_constraint_list():
    ocp_cstr_list = make_test_ocp_constraint_list(addition=5.)
    mdl_cstr_list = make_test_mdl_constraint_list(addition=10.)

    mdl_cstr_list.append(ocp_cstr_list)

    correct_number_of_constraints_total = (len(mdl_cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(mdl_cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(mdl_cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending ocp constraint list onto mdl constraint list'
        print_op.log_and_raise_error(message)

def test_that_constraint_list_ignores_append_for_incomplete_constraints():
    cstr_list = ConstraintList()

    var = cas.SX.sym('var')
    val = 2.
    expr = var - val
    cstr_type = 'eq'
    name = 'test_' + cstr_type + '_' + str(val)

    expressionless_cstr = Constraint(
        expr=None,
        cstr_type=cstr_type,
        name=name
    )
    typeless_cstr = Constraint(
        expr=expr,
        cstr_type=None,
        name=name
    )
    nameless_cstr = Constraint(
        expr=expr,
        cstr_type=cstr_type,
        name=None
    )

    cstr_list.append(expressionless_cstr)
    cstr_list.append(typeless_cstr)
    cstr_list.append(nameless_cstr)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 0)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 0)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 0)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending incomplete constraints'
        print_op.log_and_raise_error(message)


def test_that_constraint_list_ignores_append_for_repeated_constraints():
    cstr_list = make_test_constraint_list(addition=0.)

    cstr_list2 = make_test_constraint_list(addition=0.)
    cstr_list.append(cstr_list2)

    correct_number_of_constraints_total = (len(cstr_list.all_list) == 3)
    correct_number_of_equality_constraints = (len(cstr_list.eq_list) == 2)
    correct_number_of_inequality_constraints = (len(cstr_list.ineq_list) == 1)

    criteria = correct_number_of_constraints_total and correct_number_of_equality_constraints and correct_number_of_inequality_constraints
    if not criteria:
        message = 'something went wrong when appending repeated individual constraints.'
        print_op.log_and_raise_error(message)


def test_constraint_list():
    test_that_constraint_list_appending_works_for_distinct_individual_constraints()
    test_that_constraint_list_appending_works_for_distinct_constraint_lists()
    test_that_constraint_list_appending_works_for_constraint_list_onto_mdl_constraint_list()
    test_that_constraint_list_appending_works_for_constraint_list_onto_ocp_constraint_list()

    test_that_constraint_list_ignores_append_if_mdl_constraint_lists_is_added_to_ocp_constraint_list()
    test_that_constraint_list_ignores_append_if_ocp_constraint_lists_is_added_to_mdl_constraint_list()
    test_that_constraint_list_ignores_append_if_specialized_constraint_lists_are_added_to_regular_constraint_list()
    test_that_constraint_list_ignores_append_for_incomplete_constraints()
    test_that_constraint_list_ignores_append_for_repeated_constraints()

def test():
    test_constraint_list()