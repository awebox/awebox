#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2021 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
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
actuator_disk model of awebox aerodynamics
sets up the axial-induction actuator disk equation
currently for untilted rotor with no tcf.
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2017-21
- edit: jochem de schutter, alu-fr 2019
'''

import casadi.tools as cas
import numpy as np

import awebox.tools.vector_operations as vect_op
import awebox.tools.constraint_operations as cstr_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.print_operations as print_op

import awebox.mdl.aero.geometry_dir.geometry as geom
import awebox.mdl.aero.geometry_dir.unit_normal as unit_normal
import awebox.mdl.aero.induction_dir.general_dir.tools as general_tools
import awebox.viz.tools as viz_tools
import mpl_toolkits.mplot3d.art3d as art3d
# from matplotlib.patches import Annulus
from matplotlib.patches import Circle, PathPatch

# switches

def get_mu_radial_ratio(variables, kite, parent):
    varrho_var = get_varrho_var(variables, kite, parent)
    mu = varrho_var / (varrho_var + 0.5)

    return mu
# variables

def get_actuator_velocity_var(variables_si, parent):
    var_type = 'z'
    var_name = 'act_dq' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var

def get_actuator_position_var(variables_si, parent):
    var_type = 'z'
    var_name = 'act_q' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_area_var(variables_si, parent):
    var_type = 'z'
    var_name = 'area' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var

def get_bar_varrho_var(variables_si, parent):
    var_type = 'z'
    var_name = 'bar_varrho' + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var

def get_varrho_var(variables_si, kite, parent):
    var_type = 'z'
    var_name = 'varrho' + str(kite) + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var

def get_psi_var(variables_si, kite, parent):
    var_type = 'z'
    var_name = 'psi' + str(kite) + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_cospsi_var(variables_si, kite, parent):
    var_type = 'z'
    var_name = 'cospsi' + str(kite) + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_sinpsi_var(variables_si, kite, parent):
    var_type = 'z'
    var_name = 'sinpsi' + str(kite) + str(parent)
    var = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    return var


def get_n_vec_length_var(variables, parent):
    len_var = variables['z']['n_vec_length' + str(parent)]
    return len_var


# references

def get_tstar_ref(parameters, wind):
    b_ref = parameters['theta0', 'geometry', 'b_ref']
    uinfty_ref = wind.get_speed_ref()
    tstar = b_ref / uinfty_ref
    return tstar


def get_varrho_ref(model_options):
    varrho_ref = model_options['aero']['actuator']['varrho_ref']
    return varrho_ref

def get_area_ref(model_options, parameters):
    b_ref = parameters['theta0','geometry','b_ref']
    varrho_ref = get_varrho_ref(model_options)
    r_ref = varrho_ref * b_ref
    area_ref = 2. * np.pi * r_ref * b_ref
    return area_ref

# residuals

def get_center_cstr(model_options, parent, variables_si, architecture, scaling):

    cstr_list = cstr_op.ConstraintList()

    q_center_var = get_actuator_position_var(variables_si, parent)
    q_center_val = geom.get_center_position(model_options, parent, variables_si, architecture)

    pos_resi_unscaled = q_center_var - q_center_val
    pos_resi_scaled = struct_op.var_si_to_scaled('z', 'act_q' + str(parent), pos_resi_unscaled, scaling)

    name = 'actuator_center_position_' + str(parent)
    pos_cstr = cstr_op.Constraint(expr=pos_resi_scaled,
                              name=name,
                              cstr_type='eq')
    cstr_list.append(pos_cstr)

    dq_center_var = get_actuator_velocity_var(variables_si, parent)
    dq_center_val = geom.get_center_velocity(model_options, parent, variables_si, architecture)

    vel_resi_unscaled = dq_center_var - dq_center_val
    vel_resi_scaled = struct_op.var_si_to_scaled('z', 'act_dq' + str(parent), vel_resi_unscaled, scaling)

    name = 'actuator_center_velocity_' + str(parent)
    vel_cstr = cstr_op.Constraint(expr=vel_resi_scaled,
                              name=name,
                              cstr_type='eq')
    cstr_list.append(vel_cstr)

    return cstr_list


def get_area_cstr(parent, variables_si, parameters, scaling):

    area_var = get_area_var(variables_si, parent)
    area_val = get_actuator_area(parent, variables_si, parameters)

    resi_unscaled = area_var - area_val
    resi_scaled = struct_op.var_si_to_scaled('z', 'area' + str(parent), resi_unscaled, scaling)

    name = 'actuator_area_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')

    return cstr


def check_that_actuator_center_is_above_minimum_altitude(init_options, variables_si, parent):
    q_center_var = get_actuator_position_var(variables_si, parent)
    min_altitude = init_options['min_altitude']
    if q_center_var[2] < min_altitude:
        message = 'proposed actuator center is below minimum flight altitude'
        print_op.log_and_raise_error(message)
    return None


def get_bar_varrho_cstr(parent, variables, architecture, scaling):

    bar_varrho_val = get_bar_varrho_val(variables, parent, architecture)
    bar_varrho_var = get_bar_varrho_var(variables, parent)

    resi_si = bar_varrho_var - bar_varrho_val
    resi_scaled = struct_op.var_si_to_scaled('z', 'bar_varrho' + str(parent), resi_si, scaling)

    name = 'actuator_bar_varrho_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')
    return cstr


def get_varrho_and_psi_cstr(model_options, kite, variables, parameters, architecture):

    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    #
    # rvec = radius((zhat') * cos(psi) + (-yhat') * sin(psi))
    # dot(rvec, zhat') = radius * cos(psi)
    # dot(rvec, yhat') = - radius * sin(psi)

    parent = architecture.parent_map[kite]
    b_ref = parameters['theta0', 'geometry', 'b_ref']

    q_kite = variables['x']['q' + str(kite) + str(parent)]
    q_center = get_actuator_position_var(variables, parent)
    vec_from_center_to_kite = q_kite - q_center

    y_rotor_hat_var = get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = get_z_rotor_hat_var(variables, parent)

    y_rotor_comp = cas.mtimes(vec_from_center_to_kite.T, y_rotor_hat_var)
    z_rotor_comp = cas.mtimes(vec_from_center_to_kite.T, z_rotor_hat_var)

    psi_var = get_psi_var(variables, kite, parent)
    cospsi_var = get_cospsi_var(variables, kite, parent)
    sinpsi_var = get_sinpsi_var(variables, kite, parent)

    f_sin = cas.sin(psi_var) - sinpsi_var
    f_cos = cas.cos(psi_var) - cospsi_var

    varrho_var = get_varrho_var(variables, kite, parent)
    radius = varrho_var * b_ref

    varrho_ref = get_varrho_ref(model_options)
    radius_ref = b_ref * varrho_ref

    f_cos_proj = (radius * cospsi_var - z_rotor_comp) / radius_ref
    f_sin_proj = (radius * sinpsi_var + y_rotor_comp) / radius_ref

    resi_combi = cas.vertcat(f_cos, f_sin, f_cos_proj, f_sin_proj)

    name = 'actuator_varrho_and_psi_' + str(kite)
    cstr = cstr_op.Constraint(expr=resi_combi,
                              name=name,
                              cstr_type='eq')
    return cstr


# processing

def get_actuator_area(parent, variables, parameters):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    bar_varrho_var = get_bar_varrho_var(variables, parent)

    radius = bar_varrho_var * b_ref
    annulus_area = 2. * np.pi * b_ref * radius

    area = annulus_area

    return area

def get_kite_radial_vector(kite, variables, architecture):

    parent = architecture.parent_map[kite]

    y_rotor_hat_var = get_y_rotor_hat_var(variables, parent)
    z_rotor_hat_var = get_z_rotor_hat_var(variables, parent)

    cospsi_var = get_cospsi_var(variables, kite, parent)
    sinpsi_var = get_sinpsi_var(variables, kite, parent)

    rhat = parametric_rhat(z_rotor_hat_var, y_rotor_hat_var, cospsi_var, sinpsi_var)
    return rhat


def parametric_rhat(z_rotor_hat, y_rotor_hat, cospsi, sinpsi):
    # for positive yaw(turns around +zhat, normal towards +yhat):
    #     rhat = zhat * cos(psi) - yhat * sin(psi)
    rhat = z_rotor_hat * cospsi - y_rotor_hat * sinpsi

    return rhat

def get_kite_radius(kite, variables, architecture, parameters):

    b_ref = parameters['theta0', 'geometry', 'b_ref']
    parent = architecture.parent_map[kite]
    varrho_var = get_varrho_var(variables, kite, parent)

    radius = varrho_var * b_ref

    return radius


def get_average_radius(variables, parent, architecture, parameters):
    kite_children = architecture.get_kite_children(parent)
    number_kite_children = len(kite_children)

    total_radius = 0.
    for kite in kite_children:
        total_radius += get_kite_radius(kite, variables, architecture, parameters)

    average_radius = total_radius / float(number_kite_children)

    return average_radius


def get_bar_varrho_val(variables, parent, architecture):
    kite_children = architecture.get_kite_children(parent)

    number_kite_children = len(kite_children)

    sum_varrho = 0.
    for kite in kite_children:
        varrho_kite = get_varrho_var(variables, kite, parent)
        sum_varrho = sum_varrho + varrho_kite

    bar_varrho_val = sum_varrho / float(number_kite_children)
    return bar_varrho_val


def approximate_tip_radius(model_options, variables, kite, architecture, tip, parameters):

    b_ref = parameters['theta0','geometry','b_ref']
    half_span_proj = b_ref / 2.
    parent = architecture.parent_map[kite]

    radial_vector = get_kite_radial_vector(kite, variables, architecture)

    if int(model_options['kite_dof']) == 6:

        r_column = variables['x']['r' + str(kite) + str(parent)]
        r = cas.reshape(r_column, (3, 3))
        ehat2 = r[:, 1]  # spanwise, from pe to ne

        ehat2_proj_radial = vect_op.smooth_abs(cas.mtimes(radial_vector.T, ehat2))

        half_span_proj = b_ref * ehat2_proj_radial / 2.

    radius = get_kite_radius(kite, variables, architecture, parameters)

    tip_radius = radius
    if ('int' in tip) or (tip == 0):
        tip_radius = tip_radius - half_span_proj
    elif ('ext' in tip) or (tip == 1):
        tip_radius = tip_radius + half_span_proj
    else:
        message = 'invalid tip designated'
        print_op.log_and_raise_error(message)

    return tip_radius

def get_average_exterior_radius(model_options, variables, parent, parameters, architecture):

    children = architecture.kites_map[parent]
    number_children = float(len(children))

    average_radius = 0.
    for kite in children:
        radius = approximate_tip_radius(model_options, variables, kite, architecture, 'ext', parameters)

        average_radius = average_radius + radius / number_children

    return average_radius


def get_act_dcm_var(variables_si, parent):
    var_type = 'z'
    var_name = 'act_dcm' + str(parent)
    rot_cols = struct_op.get_variable_from_model_or_reconstruction(variables_si, var_type, var_name)
    act_dcm = cas.reshape(rot_cols, (3, 3))
    return act_dcm

def get_n_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    n_hat = act_dcm[:, 0]
    return n_hat

def get_y_rotor_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    y_hat = act_dcm[:, 1]
    return y_hat

def get_z_rotor_hat_var(variables, parent):
    act_dcm = get_act_dcm_var(variables, parent)
    y_hat = act_dcm[:, 2]
    return y_hat



def get_act_dcm_ortho_cstr(parent, variables):
    # rotation matrix is in SO3 = 6 constraints
    act_dcm_var = get_act_dcm_var(variables, parent)
    ortho_matr = cas.mtimes(act_dcm_var.T, act_dcm_var) - np.eye(3)
    f_ortho = vect_op.upper_triangular_inclusive(ortho_matr)

    name = 'actuator_geom_dcm_ortho_' + str(parent)
    cstr = cstr_op.Constraint(expr=f_ortho,
                              name=name,
                              cstr_type='eq')

    return cstr

def get_act_dcm_n_along_normal_cstr(model_options, parent, variables, architecture, scaling):

    # n_hat * length equals normal direction = 3 constraints
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, architecture)
    n_hat_var = get_n_hat_var(variables, parent)
    n_vec_length_var = get_n_vec_length_var(variables, parent)

    resi_si = n_vec_val - n_hat_var * n_vec_length_var

    scale = scaling['z', 'n_vec_length' + str(parent)]
    resi_scaled = resi_si / scale

    name = 'actuator_nhat_' + str(parent)
    cstr = cstr_op.Constraint(expr=resi_scaled,
                              name=name,
                              cstr_type='eq')

    return cstr


def get_n_vec_val(model_options, parent, variables, architecture, scaling):
    n_vec_val = unit_normal.get_n_vec(model_options, parent, variables, architecture)
    return n_vec_val

def draw_actuator_geometry(ax, side, plot_dict, cosmetics, index):
    draw_dot_at_actuator_center(ax, side, plot_dict, cosmetics, index)
    draw_radial_vectors_from_center_to_kites(ax, side, plot_dict, cosmetics, index)
    # draw_average_radius(ax, side, plot_dict, cosmetics, index)
    draw_psi_angles(ax, side, plot_dict, cosmetics, index)
    draw_actuator_dcm(ax, side, plot_dict, cosmetics, index)
    draw_actuator_annulus(ax, side, plot_dict, cosmetics, index)
    return None


def draw_actuator_annulus(ax, side, plot_dict, cosmetics, index):

    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
    n_theta = cosmetics['trajectory']['actuator_n_theta']

    architecture = plot_dict['architecture']
    for parent in architecture.layer_nodes:
        bar_varrho = get_bar_varrho_var(variables_si, parent)
        mu_start = (bar_varrho - 0.5) / (bar_varrho + 0.5)
        mu_end = 1.

        for psi_val in np.linspace(0., 2. * np.pi, n_theta):
            draw_radial_segment_around_actuator_center(ax, side, plot_dict, cosmetics, index, parent, mu_start, mu_end,
                                                       psi_val, color='grey', alpha=0.3)
    return None


def draw_actuator_dcm(ax, side, plot_dict, cosmetics, index):
    # b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']
    dcm_colors = cosmetics['trajectory']['dcm_colors']
    # visibility_scaling = b_ref

    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)

    architecture = plot_dict['architecture']
    for parent in architecture.layer_nodes:
        n_hat = get_n_hat_var(variables_si, parent)
        rotor_y_hat = get_y_rotor_hat_var(variables_si, parent)
        rotor_z_hat = get_z_rotor_hat_var(variables_si, parent)

        avg_radius = plot_dict['outputs']['actuator']['avg_radius' + str(parent)][0][index]
        visibility_scaling = avg_radius

        ehat_dict = {'x': n_hat,
                     'y': rotor_y_hat,
                     'z': rotor_z_hat}

        x_start = get_actuator_position_var(variables_si, parent)

        for vec_name, vec_ehat in ehat_dict.items():
            x_end = x_start + visibility_scaling * vec_ehat

            color = dcm_colors[vec_name]
            viz_tools.basic_draw(ax, side, color=color, x_start=x_start, x_end=x_end, linestyle=':')

    return None


def draw_average_radius(ax, side, plot_dict, cosmetics, index):
    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)

    architecture = plot_dict['architecture']
    for parent in architecture.layer_nodes:

        y_rotor_hat = get_y_rotor_hat_var(variables_si, parent)
        z_rotor_hat = get_z_rotor_hat_var(variables_si, parent)
        psi = 0.
        rhat = parametric_rhat(z_rotor_hat, y_rotor_hat, np.cos(psi), np.sin(psi))

        avg_radius = plot_dict['outputs']['actuator']['avg_radius' + str(parent)][0][index]

        local_center = get_actuator_position_var(variables_si, parent)

        x_start = local_center
        x_end = local_center + avg_radius * rhat

        color = 'k'
        viz_tools.basic_draw(ax, side, color=color, x_start=x_start, x_end=x_end)

    return None

def draw_psi_angles(ax, side, plot_dict, cosmetics, index):
    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)

    architecture = plot_dict['architecture']
    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        bar_varrho = get_bar_varrho_var(variables_si, parent)
        kite_index = architecture.kite_nodes.index(kite)
        number_of_kites = architecture.number_of_kites
        distinguishability_factor = float(kite_index + 1) / float(number_of_kites + 1)
        avg_midspan_mu_val = bar_varrho / (bar_varrho + 0.5)
        mu_val = avg_midspan_mu_val * distinguishability_factor

        psi_start = 0.
        psi_end = np.mod(float(get_psi_var(variables_si, kite, parent)), 2. * np.pi)

        draw_arc_around_actuator_center(ax, side, plot_dict, cosmetics, index, parent, mu_val, psi_start=psi_start,
                                        psi_end=psi_end, kite=kite, color=None, linestyle=':')

    return None


def draw_radial_segment_around_actuator_center(ax, side, plot_dict, cosmetics, index, parent, mu_start, mu_end, psi_val, kite=None, color=None, linestyle='-', alpha=1.):
    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']

    architecture = plot_dict['architecture']
    if parent in architecture.layer_nodes:
        y_rotor_hat = get_y_rotor_hat_var(variables_si, parent)
        z_rotor_hat = get_z_rotor_hat_var(variables_si, parent)

        x_center = get_actuator_position_var(variables_si, parent)

        bar_varrho = get_bar_varrho_var(variables_si, parent)
        radius_start = mu_start * (b_ref * (bar_varrho + 0.5))
        radius_end = mu_end * (b_ref * (bar_varrho + 0.5))

        cospsi = np.cos(psi_val)
        sinpsi = np.sin(psi_val)
        rhat = parametric_rhat(z_rotor_hat, y_rotor_hat, cospsi, sinpsi)

        x_start = x_center + radius_start * rhat
        x_end = x_center + radius_end * rhat

        if (color is None) and (kite in architecture.kite_nodes):
            kite_index = architecture.kite_nodes.index(kite)
            color = cosmetics['trajectory']['colors'][kite_index]

        viz_tools.basic_draw(ax, side, color=color, x_start=x_start, x_end=x_end, linestyle=linestyle, alpha=alpha)

    return None



def draw_arc_around_actuator_center(ax, side, plot_dict, cosmetics, index, parent, mu_val, psi_start=0., psi_end=2.*np.pi, kite=None, color=None, linestyle=':'):
    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']

    architecture = plot_dict['architecture']
    if parent in architecture.layer_nodes:
        y_rotor_hat = get_y_rotor_hat_var(variables_si, parent)
        z_rotor_hat = get_z_rotor_hat_var(variables_si, parent)

        x_center = get_actuator_position_var(variables_si, parent)

        bar_varrho = get_bar_varrho_var(variables_si, parent)
        drawing_radius = mu_val * (b_ref * (bar_varrho + 0.5))

        n_theta = cosmetics['trajectory']['actuator_n_theta']
        rads_per_step = 2. * np.pi / float(n_theta)
        delta_psi = rads_per_step

        local_psi = psi_start
        data = []
        while local_psi < psi_end:
            cospsi = np.cos(local_psi)
            sinpsi = np.sin(local_psi)
            rhat_start = parametric_rhat(z_rotor_hat, y_rotor_hat, cospsi, sinpsi)
            x_local = x_center + drawing_radius * rhat_start
            data = cas.horzcat(data, x_local)
            local_psi += delta_psi

        if (color is None) and (kite in architecture.kite_nodes):
            kite_index = architecture.kite_nodes.index(kite)
            color = cosmetics['trajectory']['colors'][kite_index]

        if hasattr(data, 'shape') and (len(data.shape) == 2):
            viz_tools.basic_draw(ax, side, color=color, data=data, linestyle=linestyle)

    return None


def draw_radial_vectors_from_center_to_kites(ax, side, plot_dict, cosmetics, index):

    variables_si = viz_tools.assemble_variable_slice_from_interpolated_data(plot_dict, index)
    b_ref = plot_dict['options']['model']['params']['geometry']['b_ref']

    architecture = plot_dict['architecture']
    for kite in architecture.kite_nodes:
        parent = architecture.parent_map[kite]

        bar_varrho = get_bar_varrho_var(variables_si, parent)
        radius = plot_dict['outputs']['actuator']['radius' + str(kite)][0][index]
        mu_start = 0.
        mu_end = radius / (b_ref * (bar_varrho + 0.5))

        psi_val = get_psi_var(variables_si, kite, parent)
        color = cosmetics['trajectory']['colors'][architecture.kite_nodes.index(kite)]

        draw_radial_segment_around_actuator_center(ax, side, plot_dict, cosmetics, index, parent, mu_start, mu_end,
                                                   psi_val, color=color)

    return None


def draw_dot_at_actuator_center(ax, side, plot_dict, cosmetics, index):

    architecture = plot_dict['architecture']
    for parent in architecture.layer_nodes:

        x_start = []
        for dim in range(3):
            local = plot_dict['outputs']['actuator']['center' + str(parent)][dim][index]
            x_start = cas.vertcat(x_start, local)

        color = 'k'
        viz_tools.basic_draw(ax, side, x_start=x_start, x_end=x_start, color=color, marker='o')

    return None
