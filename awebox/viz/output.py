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
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import awebox.viz.tools as tools

import casadi.tools as cas

def plot_outputs(plot_dict, cosmetics, fig_name, output_path, fig_num = None):

    time_grid_ip = plot_dict['time_grids']['ip']
    outputs = plot_dict['outputs']

    if cosmetics['plot_ref']:
        ref_time_grid_ip = plot_dict['time_grids']['ref']['ip']
        ref_outputs = plot_dict['ref']['outputs']

    output_key_list = output_path.split(':')
    if len(output_key_list) == 1:
        output = outputs[output_key_list[0]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]]
    elif len(output_key_list) == 2:
        output = outputs[output_key_list[0]][output_key_list[1]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]][output_key_list[1]]
    elif len(output_key_list) == 3:
        output = outputs[output_key_list[0]][output_key_list[1]][output_key_list[2]]
        if cosmetics['plot_ref']:
            ref_output = ref_outputs[output_key_list[0]][output_key_list[1]][output_key_list[2]]
    else:
        raise ValueError('Error: Wrong recursion depth (' + str(len(output_key_list)) + ') for output plots!' + str(output_key_list))
    recursive_output_plot(output, fig_name, time_grid_ip, fig_num)
    if cosmetics['plot_ref']:
        recursive_output_plot(ref_output, fig_name, ref_time_grid_ip,  plt.gcf().number , linestyle = '--')

    return None

def recursive_output_plot(outputs, fig_name, time_grid_ip, fig_num = None, linestyle = '-'):

    try:
        for key in list(outputs.keys()):
            recursive_output_plot(outputs[key], key, time_grid_ip, fig_num, linestyle = linestyle)
    except:
        if fig_num is None:
            fig = plt.figure()
            fig.clf()
        else:
            fig = plt.figure(fig_num)

        plt.plot(time_grid_ip, outputs[0], linestyle = linestyle)
        plt.title(fig_name)

def plot_induction_factor_vs_tether_reel(solution_dict, cosmetics, reload_dict, fig_num):

    V_plot = solution_dict['V_final']
    variables_dict = solution_dict['variables_dict']
    scaling = solution_dict['scaling']
    steadyness = solution_dict['options']['model']['aero']['actuator']['steadyness']

    if steadyness == 'steady':
        var_keys = list(variables_dict['xl'].keys())
    elif steadyness == 'unsteady':
        var_keys = list(variables_dict['xd'].keys())

    a_keys = set()
    for var in var_keys:
        if var[0] == 'a':
            a_keys.add(var)

    a_keys = list(a_keys)
    num_a = len(a_keys)

    # get tether reel values
    dlt_vals = tools.merge_xd_values(V_plot, 'dl_t',0, reload_dict, cosmetics)[0]

    plt.figure(fig_num).clear()
    fig, axes = plt.subplots(nrows=num_a, ncols=1, sharex='all', num=fig_num)

    for adx in range(num_a):
        if steadyness == 'steady':
            a_vals = tools.merge_xa_values(V_plot,'xl', a_keys[adx],0,reload_dict,cosmetics)[0]

        elif steadyness == 'unsteady':
            a_vals = tools.merge_xd_values(V_plot, a_keys[adx],0,reload_dict,cosmetics)[0]

        if num_a == 1:
            axes.plot(dlt_vals, a_vals, 'ko:')
            axes.set_ylabel(a_keys[adx] + ' [-]')
            axes.set_xlabel('reel-out speed [m/s]')
            axes.set_title('induction factor vs tether reel-out speed')
        else:
            axes[adx].plot(dlt_vals, a_vals, 'ko:')
            axes[adx].set_ylabel(a_keys[adx] + ' [-]')
            axes[adx].set_xlabel('reel-out speed [m/s]')
            if adx == 0:
                axes[adx].set_title('induction factor vs tether reel-out speed')

def plot_reduced_frequency(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    V_plot = solution_dict['V_final']
    options = solution_dict['options']
    architecture = solution_dict['architecture']

    kite_nodes = options['model']['architecture'].kite_nodes
    parent_map = options['model']['architecture'].parent_map
    kite_dof = options['user_options']['system_model']['kite_dof']
    n_k = options['nlp']['n_k']
    d = options['nlp']['collocation']['d']

    plt.figure(fig_num).clear()
    # fig, axes = plt.subplots(nrows=len(kite_nodes), ncols=1, sharex='all', num=fig_num)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex='all', num=fig_num)

    width = 8
    text_loc = 1e4
    text_height = 0.

    max_fstar_def = -999.
    f_min_all = 999.
    counter = 0.

    control_surfaces = []
    if int(kite_dof) == 6:
        for kite in kite_nodes:
            parent = parent_map[kite]

            delta_name ='delta' + str(kite) + str(parent)
            control_surfaces += [delta_name]

        for control in control_surfaces:
            for dim in range(3):

                fstar_control = float(outputs['final', 'control_freq', control + '_' + str(dim)])
                fstar_control_min = 0.9 * fstar_control
                fstar_control_max = 1.1 * fstar_control

                axes.semilogx([fstar_control_min, fstar_control_max], [counter, counter], 'g', linewidth = width)
                axes.text(text_loc, counter+ text_height, 'f* ' + control + ' (' + str(dim) + ') [Hz]')
                counter += 1.

                max_fstar_def = np.max([max_fstar_def, fstar_control])
                f_min_all = np.min([f_min_all, fstar_control_min])


    fstar_traj = float(1. / V_plot['theta', 't_f'])

    fstar_traj_min = 0.9 * fstar_traj
    fstar_traj_max = 1.1 * fstar_traj

    max_fstar_def = np.max([max_fstar_def, fstar_traj])
    f_min_all = np.min([f_min_all, fstar_traj_min])

    axes.semilogx([fstar_traj_min, fstar_traj_max], [counter, counter], 'r', linewidth=width)
    axes.text(text_loc, counter + 0.1, 'f* traj [Hz]')
    counter += 1


    for kite in kite_nodes:
        windings = np.round(float(outputs['final', 'winding', 'winding' + str(kite)]))
        fstar_loop = float(windings / V_plot['theta', 't_f'])

        fstar_loop_min = 0.9 * fstar_loop
        fstar_loop_max = 1.1 * fstar_loop

        max_fstar_def = np.max([max_fstar_def, fstar_loop])
        f_min_all = np.min([f_min_all, fstar_loop_min])

        axes.semilogx([fstar_loop_min, fstar_loop_max], [counter, counter], 'r', linewidth = width)
        axes.text(text_loc, counter+ text_height, 'f* loop ' + str(kite) + ' [Hz]')
        counter += 1

    for kite in kite_nodes:
        fstar_aero = []
        for kdx in range(n_k):
            for ddx in range(d):
                local_fstar_aero = outputs['coll_outputs', kdx, ddx, 'aerodynamics', 'fstar_aero' + str(kite)]
                fstar_aero = cas.vertcat(fstar_aero, local_fstar_aero)

        fstar_aero_max = np.max(np.array(fstar_aero))
        fstar_aero_min = np.min(np.array(fstar_aero))

        f_min_all = np.min([f_min_all, fstar_aero_min])

        axes.semilogx([fstar_aero_min, fstar_aero_max], [counter, counter], 'b', linewidth = width)
        axes.text(text_loc, counter+ text_height, 'f* kite ' + str(kite) + ' [Hz]')
        counter += 1.

    layer_parents = architecture.layer_nodes
    for parent in layer_parents:
        center_x = []
        u_app_c = []

        for kdx in range(n_k):
            center_x = cas.vertcat(center_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 0]))
            u_app_c = cas.vertcat(u_app_c, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'u_app' + str(parent), 0]))
        center_x = np.array(center_x)
        u_app_c = np.abs(np.array(u_app_c))
        delta_center_x = np.max(center_x) - np.min(center_x)

        fstar_act_min = np.min(u_app_c) / delta_center_x
        fstar_act_max = np.max(u_app_c) / delta_center_x

        axes.semilogx([fstar_act_min, fstar_act_max], [counter, counter], 'b', linewidth = width)
        axes.text(text_loc, counter + 0.1, 'f* actuator ' + str(parent) + ' [Hz]')
        counter += 1.

        f_min_all = np.min([f_min_all, fstar_act_min])

    plt.axhline(y = counter, color='k', linestyle='-')

    counter += 1

    fkite_max = max_fstar_def / fstar_aero_min
    fkite_min = max_fstar_def / fstar_aero_max

    axes.semilogx([fkite_min, fkite_max], [counter, counter], 'k', linewidth = width)
    axes.text(text_loc, counter+ text_height, 'f kite [-]')
    counter += 1.

    f_min_all = np.min([f_min_all, fkite_min])

    fact_max = max_fstar_def / fstar_act_min
    fact_min = max_fstar_def / fstar_act_max

    axes.semilogx([fact_min, fact_max], [counter, counter], 'k', linewidth = width)
    axes.text(text_loc, counter+ text_height, 'f actuator [-]')
    counter += 1.

    f_min_all = np.min([f_min_all, fact_min])

    axes.set_ylim([-1., counter])
    axes.set_xlim([f_min_all / 10., text_loc * 1e3])

    plt.axvline(x=1., color='k', linestyle='--')

    plt.title('reduced frequency components')
    plt.yticks([], [])

    plt.show()

def plot_energy_over_time(solution_dict, cosmetics, fig_num, reload_dict):
    outputs = solution_dict['outputs']
    options = solution_dict['options']
    architecture = solution_dict['architecture']

    tgrid_coll = np.array(reload_dict['tgrid_coll'])
    potential_energy = {}
    kinetic_energy = {}

    elements = ['groundstation']
    number_of_nodes = architecture.number_of_nodes
    parent_map = architecture.parent_map
    for node in range(1, number_of_nodes):
        elements += ['q' + str(node) + str(parent_map[node])]

    for elem in elements:
        if elem not in list(potential_energy.keys()):
            potential_energy[elem] = []

        if elem not in list(kinetic_energy.keys()):
            kinetic_energy[elem] = []

    n_k = options['nlp']['n_k']
    d_k = options['nlp']['collocation']['d']
    for n in range(n_k):
        for d in range(d_k):
            for elem in elements:
                potential_energy[elem] += [outputs['coll_outputs', n, d, 'e_potential', elem]]
                kinetic_energy[elem] += [outputs['coll_outputs', n, d, 'e_kinetic', elem]]

    fig, axes = plt.subplots(nrows=(len(elements)+1), ncols=1, sharex='all', num=fig_num)

    axes_counter = 0

    e_kin_total = 0.0
    e_pot_total = 0.0

    for elem in elements:
        e_kin = np.array(kinetic_energy[elem])
        e_pot = np.array(potential_energy[elem])

        ax = axes[axes_counter]
        ax.set_title('energy for ' + elem)
        # ax.set_xlabel('t [s]')
        ax.set_ylabel('e [J]')

        ax.stackplot(tgrid_coll.flatten(), e_kin.flatten(), e_pot.flatten(), labels=["e_kin", "e_pot"])
        ax.legend(loc = 'upper right')

        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.yaxis.set_major_locator(MaxNLocator(3))

        axes_counter += 1
        e_kin_total += e_kin
        e_pot_total += e_pot

    ax = axes[axes_counter]
    ax.set_title('energy for whole system')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('e [J]')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_locator(MaxNLocator(3))

    ax.stackplot(tgrid_coll.flatten(), e_kin_total.flatten(), e_pot_total.flatten(), labels=['e_kin','e_pot'])
    ax.legend(loc = 'upper right')

    plt.tight_layout(w_pad=1.)

    plt.show()

def plot_loyd_comparison(solution_dict, cosmetics, fig_num, reload_dict):

    # read in inputs
    outputs = solution_dict['outputs']
    options = solution_dict['options']
    tgrid_coll = reload_dict['tgrid_coll']

    fig = plt.figure(fig_num)

    plt.ion()

    number_of_rows = 3 + len(options['model']['architecture'].kite_nodes)

    phf_loyd_total = np.array(outputs['coll_outputs',:,:,'performance','phf_loyd_total'])
    phf = np.array(outputs['coll_outputs',:,:,'performance','phf'])

    loyd_factor = np.array(outputs['coll_outputs',:,:,'performance','loyd_factor'])
    loyd_factor_comparison = np.ones(loyd_factor.shape)

    freelout = np.array(outputs['coll_outputs',:,:,'performance','freelout'])
    freelout_loyd = 1./3. * np.ones(freelout.shape)

    ax1 = plt.subplot(number_of_rows, 1, 1)
    ax1.plot(tgrid_coll, cas.vertcat(*phf_loyd_total), 'b--')
    ax1.plot(tgrid_coll, cas.vertcat(*phf), 'b')
    ax1.grid('on')
    ax1.set_ylabel('zeta')
    ax1.yaxis.set_major_locator(MaxNLocator(4))

    ax2 = plt.subplot(number_of_rows, 1, 2)
    ax2.plot(tgrid_coll, cas.vertcat(*loyd_factor_comparison), 'b--')
    ax2.plot(tgrid_coll, cas.vertcat(*loyd_factor), 'b')
    ax2.grid('on')
    ax2.set_ylabel('eta')
    ax2.yaxis.set_major_locator(MaxNLocator(4))

    ax3 = plt.subplot(number_of_rows, 1, 3)
    ax3.plot(tgrid_coll, cas.vertcat(*freelout_loyd), 'b--')
    ax3.plot(tgrid_coll, cas.vertcat(*freelout), 'b')
    ax3.grid('on')
    ax3.set_ylabel('f')
    ax3.yaxis.set_major_locator(MaxNLocator(4))

    for n in options['model']['architecture'].kite_nodes:

        speed_ratio = np.array(outputs['coll_outputs',:,:,'local_performance','speed_ratio' + str(n)])
        speed_ratio_loyd = np.array(outputs['coll_outputs',:,:,'local_performance','speed_ratio_loyd' + str(n)])

        axn = plt.subplot(number_of_rows, 1, 4 + options['model']['architecture'].kite_nodes.index(n))
        axn.plot(tgrid_coll, cas.vertcat(*speed_ratio_loyd), 'b--')
        axn.plot(tgrid_coll, cas.vertcat(*speed_ratio), 'b')
        axn.grid('on')
        axn.set_ylabel('upsilon ' + str(n))
        axn.yaxis.set_major_locator(MaxNLocator(4))

def plot_aero_forces(solution_dict, cosmetics, fig_num, reload_dict):

    # read in input
    options = solution_dict['options']
    outputs = solution_dict['outputs']

    fig = plt.figure(fig_num)

    selected_outputs = [('aerodynamics','f_aero')]
    dimensions = 3

    plot_table_r = 4
    plot_table_c = int(len(selected_outputs) * dimensions / plot_table_r) + \
        1 * (not np.mod(len(selected_outputs) * dimensions, plot_table_r) == 0)

    pdu = 1
    for output_pair in selected_outputs:
        output_type = output_pair[0]
        output_name = output_pair[1]
        for dim in range(dimensions):

            tools.plot_output_block(plot_table_r, plot_table_c, options, outputs, plt, fig, pdu, output_type, output_name, cosmetics, reload_dict, dim)

            pdu = pdu + 1

    fig.canvas.draw()

# def plot_output(solution_dict, cosmetics, fig_num, reload_dict): #todo: fix output plot!
#
#     # read in input
#     options = solution_dict['options']
#     outputs = solution_dict['outputs']
#
#     fig = plt.figure(fig_num)
#     selected_outputs = [('aerodynamics','alpha_deg'), ('aerodynamics','beta_deg'), ('aerodynamics','CA'), ('aerodynamics','CY'), ('aerodynamics','CN'), ('aerodynamics','CD'), ('aerodynamics','CS'), ('aerodynamics','CL'), ('aerodynamics','reynolds'), ('aerodynamics','mach'), ('aerodynamics','speed')]
#
#     plot_table_r = 4
#     plot_table_c = int(len(selected_outputs) / plot_table_r) + \
#         1 * (not np.mod(len(selected_outputs), plot_table_r) == 0)
#
#     pdu = 1
#     for name in selected_outputs:
#         tools.plot_output_block(plot_table_r, plot_table_c, options, outputs, plt, fig, pdu, name, cosmetics, reload_dict, dim)
#
#         pdu = pdu + 1
#
#     fig.canvas.draw()

def plot_actuator_center_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig = plt.figure(fig_num)

    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 2), (2, 0))
    ax3 = plt.subplot2grid((4, 2), (3, 0), sharex=ax2)
    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax5 = plt.subplot2grid((4, 2), (3, 1), sharex=ax4)

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    layer_parents = architecture.layer_nodes

    for parent in layer_parents:
        tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

        center_x = []
        center_z = []
        v_x = []
        v_z = []

        for kdx in range(n_k):

            center_x = cas.vertcat(center_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 0]))
            center_z = cas.vertcat(center_z, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'center' + str(parent), 2]))

            v_x = cas.vertcat(v_x, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'velocity' + str(parent), 0]))
            v_z = cas.vertcat(v_z, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'velocity' + str(parent), 2]))

        center_x = np.array(center_x)
        center_z = np.array(center_z)

        v_x = np.array(v_x)
        v_z = np.array(v_z)

        avg_radius = reload_dict['avg_radius' + str(parent)]

        ax1.plot(center_x / avg_radius, center_z / avg_radius)
        ax2.plot(tgrid_coll, center_x / avg_radius)
        ax3.plot(tgrid_coll, center_z / avg_radius)
        ax4.plot(tgrid_coll, v_x / reload_dict['u_hub' + str(parent)])
        ax5.plot(tgrid_coll, v_z / reload_dict['u_hub' + str(parent)])

    ax1.axis('equal')

    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.xaxis.set_major_locator(MaxNLocator(4))

    ax1.set_xlabel('x_c / bar R [-]')
    ax1.set_ylabel('z_c / bar R [-]')

    ax3.set_xlabel('t u_infty / bar R [-]')
    ax2.set_ylabel('x_c / bar R [-]')
    ax3.set_ylabel('z_c / bar R [-]')
    ax2.grid(True)
    ax3.grid(True)

    ax5.set_xlabel('t u_infty / bar R [-]')
    ax4.set_ylabel('u_c / u_infty [-]')
    ax5.set_ylabel('w_c / u_infty [-]')
    ax4.grid(True)
    ax5.grid(True)

    ax2.yaxis.set_major_locator(MaxNLocator(4))
    ax2.xaxis.set_major_locator(MaxNLocator(4))
    ax3.yaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_major_locator(MaxNLocator(4))
    ax4.yaxis.set_major_locator(MaxNLocator(4))
    ax4.xaxis.set_major_locator(MaxNLocator(4))
    ax5.yaxis.set_major_locator(MaxNLocator(4))
    ax5.xaxis.set_major_locator(MaxNLocator(4))

    plt.tight_layout()

    plt.show()

def plot_actuator_area_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig = plt.figure(fig_num)

    layer_parents = architecture.layer_nodes

    for parent in layer_parents:
        tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

        area = []

        for kdx in range(n_k):
            area = cas.vertcat(area, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'area' + str(parent)]))

        area = np.array(area)

        avg_radius = reload_dict['avg_radius' + str(parent)]
        avg_area = np.pi * avg_radius**2.

        plt.plot(tgrid_coll, area / avg_area)

    plt.xlabel('t u_infty / bar R [-]')
    plt.ylabel('A / (pi bar R^2) [-]')

    plt.show()

def plot_actuator_thrust_coeff_in_aerotime(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', num=fig_num)

    layer_parents = architecture.layer_nodes

    for parent in layer_parents:
        tgrid_coll = reload_dict['tgrid_xa_aerotime' + str(parent)]

        thrust = []
        thrust1_coeff = []
        thrust2_area_coeff = []
        thrust3_coeff = []

        for kdx in range(n_k):

            thrust = cas.vertcat(thrust, cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust' + str(parent)]))

            thrust1_coeff = cas.vertcat(thrust1_coeff,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust1_coeff' + str(parent)]))
            thrust2_area_coeff = cas.vertcat(thrust2_area_coeff,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust2_area_coeff' + str(parent)]))
            thrust3_coeff = cas.vertcat(thrust3_coeff,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'actuator', 'thrust3_coeff' + str(parent)]))

        avg_radius = reload_dict['avg_radius' + str(parent)]
        avg_area = np.pi * avg_radius**2.

        thrust = np.array(thrust)

        # T / (1/2 rho u_infty^2 A)
        thrust1_coeff = np.array(thrust1_coeff)

        # T / (1/2 rho u_infty^2 Abar)
        thrust2_coeff = np.array(thrust2_area_coeff) / float(avg_area)

        # 4 a (cos gamma - a)
        thrust3_coeff = np.array(thrust3_coeff)

        axes[0].plot(tgrid_coll, thrust)
        axes[1].plot(tgrid_coll, thrust1_coeff)
        axes[2].plot(tgrid_coll, thrust2_coeff)
        axes[3].plot(tgrid_coll, thrust3_coeff)

    axes[-1].set_xlabel('t u_infty / bar R [-]')

    axes[0].set_ylabel('T [N]')
    axes[1].set_ylabel('CT_1 [-]')
    axes[2].set_ylabel('CT_2 [-]')
    axes[3].set_ylabel('CT_3 [-]')

    axes[0].set_title('actuator thrust and thrust coefficients')

    for adx in range(len(axes)):
        axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axes[adx].yaxis.set_major_locator(MaxNLocator(3))

    plt.tight_layout(w_pad=1.)

    plt.show()

def plot_dimensionless_aero_indictors(solution_dict, cosmetics, fig_num, reload_dict):

    outputs = solution_dict['outputs']
    architecture = solution_dict['architecture']
    options = solution_dict['options']

    n_k = options['nlp']['n_k']

    fig = plt.figure(fig_num)

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', num=fig_num)

    kite_nodes = architecture.kite_nodes

    tgrid_coll = reload_dict['tgrid_coll']
    for kite in kite_nodes:

        alpha_deg = []
        beta_deg = []
        reynolds = []
        mach = []

        for kdx in range(n_k):

            alpha_deg = cas.vertcat(alpha_deg,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'aerodynamics', 'alpha_deg' + str(kite)]))  #todo: find new names of alpha_deg and beta_deg!
            beta_deg = cas.vertcat(beta_deg,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'aerodynamics', 'beta_deg' + str(kite)]))
            reynolds = cas.vertcat(reynolds,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'aerodynamics', 'reynolds' + str(kite)]))
            mach = cas.vertcat(mach,
                                    cas.vertcat(*outputs['coll_outputs', kdx, :, 'aerodynamics', 'mach' + str(kite)]))

        alpha_deg = np.array(alpha_deg)
        beta_deg = np.array(beta_deg)
        reynolds = np.array(reynolds)
        mach = np.array(mach)

        local_color = cosmetics['trajectory']['colors'][kite_nodes.index(kite)]

        axes[0].plot(tgrid_coll, alpha_deg, color=local_color)
        axes[1].plot(tgrid_coll, beta_deg, color=local_color)
        axes[2].plot(tgrid_coll, reynolds, color=local_color)
        axes[3].plot(tgrid_coll, mach, color=local_color)

    axes[-1].set_xlabel('t [s]')

    axes[0].set_ylabel('alpha [deg]')
    axes[1].set_ylabel('beta [deg]')
    axes[2].set_ylabel('reynolds [-]')
    axes[3].set_ylabel('mach [-]')

    for adx in range(len(axes)):
        axes[adx].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        axes[adx].yaxis.set_major_locator(MaxNLocator(3))

    plt.tight_layout(w_pad=1.)

    axes[0].set_title('kite flight parameters')

    plt.show()


def plot_constraints(plot_dict, cosmetics, fig_num, constr_type):

    outputs = plot_dict['outputs']
    constraints = plot_dict['constraints_dict'][constr_type]
    n_constr = len(list(constraints.keys())) # number of constraints
    fig, axes = plt.subplots(nrows=n_constr, ncols=1, sharex='all')

    counter = 0
    for constr_name in list(constraints.keys()):

        # plot all constraints of similar type
        for name in list(plot_dict['outputs'][constr_name].keys()):
            for idx in range(constraints[constr_name, name].shape[0]):

                # exract data
                output_vals = plot_dict['outputs'][constr_name][name][idx]
                tgrid = plot_dict['time_grids']['ip']

                # add labels
                if constraints[constr_name, name].shape[0] == 1:
                    label = name
                else:
                    label = name+'_'+str(idx)

                # plot data with label
                p = axes[counter].plot(tgrid, output_vals, label = label)

                if cosmetics['plot_ref']:
                    ref_output_vals = plot_dict['ref']['outputs'][constr_name][name][idx]
                    ref_tgrid = plot_dict['time_grids']['ref']['ip']
                    axes[counter].plot(ref_tgrid, ref_output_vals, linestyle = '--',color = p[-1].get_color())

        axes[counter].plot(tgrid, np.zeros(tgrid.shape),'k--')
        axes[counter].set_ylabel(constr_name)
        axes[counter].set_xlabel('time [s]')
        axes[counter].set_xlim([tgrid[0], tgrid[-1]])
        axes[counter].legend(loc = 'upper right')

        if counter == 0:
            axes[counter].set_title(constr_type + ' constraints')
        counter += 1


    plt.suptitle(fig_num)
