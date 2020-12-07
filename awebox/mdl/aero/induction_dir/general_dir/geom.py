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
geometry values needed for general induction modelling
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-20
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.mdl.aero.induction_dir.tools_dir.unit_normal as unit_normal



def get_rot_matr_var(variables, parent):
    rot_cols = variables['xl']['rot_matr' + str(parent)]
    rot_matr = cas.reshape(rot_cols, (3, 3))

    return rot_matr

def get_n_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    n_hat = rot_matr[:, 0]
    return n_hat

def get_y_rotor_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    y_hat = rot_matr[:, 1]
    return y_hat

def get_z_rotor_hat_var(variables, parent):
    rot_matr = get_rot_matr_var(variables, parent)
    y_hat = rot_matr[:, 2]
    return y_hat

def get_z_vec_length_var(variables, parent):
    len_var = variables['xl']['z_vec_length' + str(parent)]
    return len_var



def get_rot_matr_ortho_residual(model_options, parent, variables, parameters, architecture):
    # rotation matrix is in SO3 = 6 constraints
    rot_matr_var = get_rot_matr_var(variables, parent)
    ortho_matr = cas.mtimes(rot_matr_var.T, rot_matr_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    return f_ortho

def get_rot_matr_n_along_normal_residual(model_options, parent, variables, parameters, architecture):
    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, parameters, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = unit_normal.get_n_vec_length_var(variables, parent)

    n_diff = n_vec_val - n_hat_var * n_vec_length_var

    n_vec_length_ref = unit_normal.get_n_vec_length_ref(variables, parent)
    f_n_vec = n_diff / n_vec_length_ref

    return f_n_vec

def get_rot_matr_n_along_tether_residual(model_options, parent, variables, parameters, architecture):
    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = unit_normal.get_n_vec_default(model_options, parent, variables, parameters, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = unit_normal.get_n_vec_length_var(variables, parent)

    n_diff = n_vec_val - n_hat_var * n_vec_length_var

    n_vec_length_ref = unit_normal.get_n_vec_length_ref(variables, parent)
    f_n_vec = n_diff / n_vec_length_ref

    return f_n_vec


def get_rot_matr_residual(model_options, parent, variables, parameters, architecture):

    # total number of variables = 10 (9 from rot_matr, 1 lengths)
    f_ortho = get_rot_matr_ortho_residual(model_options, parent, variables, parameters, architecture)
    f_n_vec = get_rot_matr_n_along_normal_residual(model_options, parent, variables, parameters, architecture)
    #
    # join the constraints
    f_combi = cas.vertcat(f_ortho, f_n_vec)

    return f_combi



def get_n_vec_val(model_options, parent, variables, parameters, architecture):
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, parameters, architecture)
    return n_vec_val

