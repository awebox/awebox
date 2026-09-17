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
tether aerodynamics model of an awe system
takes states, finds approximate total force and moment for a tether element
finds equivalent forces corresponding to the total force and moment.
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, jochem de schutter alu-fr 2020
- edited: rachel leuthold, 2026
'''

import casadi.tools as cas
import numpy as np
from sympy.codegen.ast import continue_

from awebox.logger.logger import Logger as awelogger

import awebox.mdl.wind as wind_mod
import awebox.mdl.atmosphere as atmos_mod

import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

import awebox.mdl.aero.tether_dir.reynolds as tether_reynolds
import awebox.mdl.aero.tether_dir.coefficients as tether_coefficients
import awebox.mdl.aero.tether_dir.element as tether_element
import awebox.mdl.aero.tether_dir.segment as tether_segment
from awebox.mdl.architecture import Architecture


class Tether(print_op.PrintableObject):
    def __init__(self, model_options, parameters, wind, atmos, options_object=None):
        super().__init__(options_object=options_object, name='tether')

        self.add_to_applied_params_dict('model.tether.control_var', model_options['tether']['control_var'])

        tether_constraint_includes = model_options['model_bounds']['tether']['tether_constraint_includes']
        if tether_constraint_includes['stress']:
            self.add_to_applied_params_dict('params.tether.max_stress', parameters['theta0', 'tether', 'max_stress'])

        model_tether_drag = model_options['tether']['tether_drag']['model_type'] != 'not_in_use'
        #todo: only add these things if model_tether_drag is true
        self.set_density(parameters)
        self.set_reynolds_number_function(atmos, parameters)
        self.set_drag_coefficient_function(model_options, parameters)
        self.set_element_drag_function(parameters, wind, atmos)
        self.set_segment_drag_function(model_options, parameters, atmos, wind)
        self.set_drag_force_distribution_function(model_options, parameters)

    def set_density(self, parameters):
        self.__density = parameters['theta0', 'tether', 'rho']

    def set_reynolds_number_function(self, atmos, parameters):
        q_local = cas.SX.sym('q_local', (3, 1))
        ua_local = cas.SX.sym('ua_local', (3, 1))
        diameter = cas.SX.sym('diameter')
        reynolds = tether_reynolds.get_reynolds_number(atmos, q_local=q_local, ua_local=ua_local, diam=diameter)
        self.__reynolds_fun = cas.Function('reynolds_fun', [q_local, ua_local, diameter, parameters], [reynolds])
        return None

    def set_drag_coefficient_function(self, model_options, parameters):
        reynolds = cas.SX.sym('reynolds')
        cd_fun, info_to_add_to_applied_params_dict = tether_coefficients.get_tether_cd_fun(model_options, parameters)
        self.add_dict_to_applied_params_dict(info_to_add_to_applied_params_dict)
        cd_found = cd_fun(reynolds, parameters)
        self.__cd_fun = cas.Function('cd_fun', [reynolds, parameters], [cd_found])
        return None

    def add_dict_to_applied_params_dict(self, info_to_add_to_applied_params_dict):
        for param_address, param_value in info_to_add_to_applied_params_dict.items():
            self.add_to_applied_params_dict(param_address, param_value)
        return None

    def set_element_drag_function(self, parameters, wind, atmos):
        self.__element_drag_fun, info_to_add_to_applied_params_dict = tether_element.get_element_drag_fun(wind, atmos, parameters, cd_tether_fun=self.__cd_fun, reynolds_fun=self.__reynolds_fun)
        self.add_dict_to_applied_params_dict(info_to_add_to_applied_params_dict)
        return None

    def set_segment_drag_function(self, model_options, parameters, atmos, wind):
        self.__segment_drag_fun, info_to_add_to_applied_params_dict = tether_segment.get_segment_drag_fun(model_options, parameters, atmos, wind, self.__element_drag_fun, self.__reynolds_fun, self.__cd_fun)
        self.add_dict_to_applied_params_dict(info_to_add_to_applied_params_dict)
        return None

    def set_drag_force_distribution_function(self, model_options, parameters):
        self.__drag_distribution_fun, info_to_add_to_applied_params_dict = tether_segment.get_drag_force_distribution_fun(model_options, parameters, segment_drag_fun=self.__segment_drag_fun)
        self.add_dict_to_applied_params_dict(info_to_add_to_applied_params_dict)
        return None

    def calculate_distribute_drag_forces_on_nodes(self, upper_node, model_options, variables, parameters, architecture, ehat_1=None, ehat_3=None, alpha=None, kite_dynamic_pressure=None, air_velocity=None):
        q_top, q_bottom, dq_top, dq_bottom = tether_element.get_upper_and_lower_pos_and_vel(variables, upper_node,
                                                                                            architecture)
        diam = tether_element.get_element_diameter(variables, upper_node, architecture)
        segment_info_dict = {'q_upper': q_top,
                             'q_lower': q_bottom,
                             'dq_upper': dq_top,
                             'dq_lower': dq_bottom,
                             'diameter': diam,
                             'ehat_1': ehat_1,
                             'ehat_3': ehat_3,
                             'alpha': alpha,
                             'kite_dynamic_pressure': kite_dynamic_pressure,
                             'air_velocity': air_velocity}
        kite_only = model_options['tether']['tether_drag']['model_type'] == 'kite_only'
        segment_info_columnized = tether_element.columnize_element_info(unpacked_as_dict=segment_info_dict, kite_only=kite_only)
        columnized_node_forces = self.__drag_distribution_fun(segment_info_columnized, parameters)
        node_forces_dict = tether_segment.unpack_node_force_column(columnized_node_forces)
        return node_forces_dict

    def plot_upper_node_force(self, num_fig, model_options, V_opt, p_fix_num, parameters, atmos, wind, projection=vect_op.xhat_np()):
        params_at_time = struct_op.get_parameters_at_time(V_opt, p_fix_num, parameters)
        tether_segment.plot_upper_node_force(num_fig, model_options, parameters, params_at_time, self.__element_drag_fun, atmos, wind, projection=projection)
        return None

    @property
    def density(self):
        return self.__density

    @density.setter
    def density(self, value):
        awelogger.logger.warning('Cannot set density object.')

    @property
    def drag_distribution_fun(self):
        return self.__drag_distribution_fun

    @drag_distribution_fun.setter
    def drag_distribution_fun(self, value):
        awelogger.logger.warning('Cannot set drag_distribution_fun object.')

    @property
    def segment_drag_fun(self):
        return self.__segment_drag_fun

    @segment_drag_fun.setter
    def segment_drag_fun(self, value):
        awelogger.logger.warning('Cannot set segment_drag_fun object.')

    @property
    def element_drag_fun(self):
        return self.__element_drag_fun

    @element_drag_fun.setter
    def element_drag_fun(self, value):
        awelogger.logger.warning('Cannot set element_drag_fun object.')

    @property
    def cd_fun(self):
        return self.__cd_fun

    @cd_fun.setter
    def cd_fun(self, value):
        awelogger.logger.warning('Cannot set cd_fun object.')

    @property
    def reynolds_fun(self):
        return self.__reynolds_fun

    @reynolds_fun.setter
    def reynolds_fun(self, value):
        awelogger.logger.warning('Cannot set reynolds_fun object.')

def construct_test_objects(drag_model_type='not_in_use', cd_model='constant', wind_model='uniform', aero_elements=5, rho_ref=1.225, mu_ref=1.802e-5, reynolds=1e5, cd=1.2, u_ref=10., atmosphere_heightsdata=None, atmosphere_featuresdata=None):

    # https://web.mit.edu/shear7/papers/VANDIVER-HOLLER-KIM@OMAE.pdf

    options = {'model': {'atmosphere': {'model': 'uniform'},
                         'wind': {'model': wind_model, 'u_ref': u_ref, 'atmosphere_heightsdata': atmosphere_heightsdata, 'atmosphere_featuresdata': atmosphere_featuresdata, 'log_wind': {'z0_air': 0.01}},
                         'tether': {'cd_model': cd_model, 'aero_elements': aero_elements, 'reynolds_smoothing': 1e-8, 'tether_drag':{'model_type': drag_model_type}}},
               'user_options': {'atmosphere': 'uniform', 'wind': u_ref},
               'params': {'atmosphere': {'rho_ref': rho_ref, 'mu_ref': mu_ref, 'g': 9.81},
                          'tether': {'cd': cd},
                          'wind': {'u_ref': u_ref}
                          }}
    parameters_dict = {}

    # extract parametric options
    parametric_options = options['params']
    parameters_dict['theta0'] = struct_op.generate_nested_dict_struct(parametric_options)
    parameters = cas.struct_symSX([
        cas.entry('theta0', struct=parameters_dict['theta0'])
    ])
    param_list = []
    for param_type_name, local_param_dict in parametric_options.items():
        param_list += [local_val for local_val in local_param_dict.values()]
    p_fix_num = parameters(param_list)

    atmos_obj = atmos_mod.Atmosphere(options['model']['atmosphere'], parameters)
    wind_obj = wind_mod.Wind(options['model']['wind'], parameters)

    q_local = cas.SX.sym('q_local', (3, 1))
    ua_local = cas.SX.sym('ua_local', (3, 1))
    diameter = cas.SX.sym('diameter')
    reynolds_fun_constant = cas.Function('reynolds_fun_constant', [q_local, ua_local, diameter, parameters], [reynolds])

    reynolds = tether_reynolds.get_reynolds_number(atmos_obj, q_local=q_local, ua_local=ua_local, diam=diameter)
    reynolds_fun_variable = cas.Function('reynolds_fun_variable', [q_local, ua_local, diameter, parameters], [reynolds])

    cd_fun, _ = tether_coefficients.get_tether_cd_fun(options['model'], parameters)

    element_drag_fun, _ = tether_element.get_element_drag_fun(wind_obj, atmos_obj,
                                                          parameters,
                                                          cd_tether_fun=cd_fun,
                                                          reynolds_fun=reynolds_fun_constant)
    segment_drag_fun, _ = tether_segment.get_segment_drag_fun(options['model'],
                                                              parameters,
                                                              atmos_obj, wind_obj,
                                                              element_drag_fun,
                                                              reynolds_fun_variable,
                                                              cd_fun)
    drag_distribution_fun, _ = tether_segment.get_drag_force_distribution_fun(options['model'], parameters, segment_drag_fun=segment_drag_fun)

    return parameters, p_fix_num, atmos_obj, wind_obj, element_drag_fun, segment_drag_fun, drag_distribution_fun

def test_segment_integration_simple(drag_model_type='multi', thresh=1e-3):
    # Consider a segment, where ua varies linearly between 0 m/s at the lower node and 8 m/s at the upper
    # node. The average speed is then 4 m/s. With one element, our total drag will be proportional to
    # (4m/s)^2 = 16 m^2 /s^2 . With two elements, our total drag will be proportional to (1/2) ((2m/s)2 +(6m/s)2 ) = 20 m^2/s^2
    # ; with four, 21 m^2 /s^2 . In fact, the ”true” value would be proportional to the integral \int_0^1 ua(zeta)^2 \dd \zeta, or 21.33 m^2/s^2

    u_ref = 0.
    rho_ref = 1.
    mu_ref = 1.

    integrated_drag = {1:{'expected': 16.}, 2:{'expected': 20.}, 4:{'expected': 21.}, 100:{'expected': 21.33}}
    for aero_elements in integrated_drag.keys():

        parameters, p_fix_num, atmos_obj, wind_obj, _, _, drag_distribution_fun = construct_test_objects(drag_model_type=drag_model_type,
                                                                                                         cd_model='constant',
                                                                                                         wind_model='uniform',
                                                                                                         u_ref=u_ref,
                                                                                                         mu_ref=mu_ref,
                                                                                                         rho_ref=rho_ref,
                                                                                                         aero_elements=aero_elements)

        q_upper = 100. * vect_op.zhat_np()
        q_lower = 0. * vect_op.zhat_np()
        dq_upper = 8. * vect_op.xhat_np()
        dq_lower = 0. * vect_op.xhat_np()
        diameter = 0.01

        air_velocity = dq_upper
        kite_dynamic_pressure = 0.5 * rho_ref * cas.mtimes(dq_upper.T, dq_upper)
        ehat_1 = vect_op.xhat_np()
        ehat_3 = vect_op.zhat_np()
        alpha = 0.

        inputs = tether_element.columnize_element_info(q_upper=q_upper,
                                                       q_lower=q_lower,
                                                       dq_upper=dq_upper,
                                                       dq_lower=dq_lower,
                                                       diameter=diameter,
                                                       ehat_1=ehat_1,
                                                       ehat_3=ehat_3,
                                                       alpha=alpha,
                                                       kite_dynamic_pressure=kite_dynamic_pressure,
                                                       air_velocity=air_velocity,
                                                       unpacked_as_dict=None,
                                                       kite_only=drag_model_type=='kite_only')
        columnized_node_forces = drag_distribution_fun(inputs, p_fix_num)
        node_forces_dict = tether_segment.unpack_node_force_column(columnized_node_forces)
        total_drag_vec = node_forces_dict['upper'] + node_forces_dict['lower']

        integrated_drag[aero_elements]['found'] = cas.mtimes(total_drag_vec.T, vect_op.xhat_np())

    for aero_elements in (set(integrated_drag.keys()) - set([1])):
        prop_found = integrated_drag[aero_elements]['found'] / integrated_drag[1]['found']
        prop_expected = integrated_drag[aero_elements]['expected'] / integrated_drag[1]['expected']
        integrated_drag[aero_elements]['ratio_found'] = prop_found
        integrated_drag[aero_elements]['ratio_expected'] = prop_expected

        error = float((prop_expected - prop_found) / prop_expected)
        integrated_drag[aero_elements]['error'] = error

        if float(prop_found) == 0.:
            message = 'linear apparent velocity drag integration test is predicting zero drag with model ' + drag_model_type + ' '
            message += 'at ' + str(aero_elements) + ' aero_elements. '
            for name in ['ratio_found', 'ratio_expected', 'error']:
                message += name + " is " + str(integrated_drag[aero_elements][name]) + '. '
            print_op.log_and_raise_error(message)

        if np.abs(error) > thresh:
            message = 'linear apparent velocity drag integration test gives a different ratio than expected numerically, with model ' + drag_model_type  + ' '
            message += 'at ' + str(aero_elements) + ' aero_elements. '
            for name in ['ratio_found', 'ratio_expected', 'error']:
                message += name + " is " + str(integrated_drag[aero_elements][name]) + '. '
            print_op.log_and_raise_error(message)

    return None


def test_segment_integration_varying(aero_elements=100, cd_model='polyfit', drag_model_type='multi', thresh=0.3):
    # https://web.mit.edu/shear7/papers/VANDIVER-HOLLER-KIM@OMAE.pdf
    # https://asmedigitalcollection.asme.org/energyresources/article-abstract/108/1/77/427541/Vortex-Induced-Vibration-and-Drag-Coefficients-of
    # Kim1986
    # at ~35 g/kg salinity and 0d C (https://en.wikipedia.org/wiki/Arctic_Ocean)
    mu_ref = 1.906e-3 #kg/m/s page 5, see note on bottom right of page for why e-3 and not e3, https://web.mit.edu/seawater/2017_MIT_Seawater_Property_Tables_r2b_2023c.pdf
    rho_ref = 1028.0 #kg/m^3, page 24 of seawater properties table
    # # # test version
    # atmosphere_heightsdata = np.array([0, 100, 200, 300, 400, 500, 600])
    # atmosphere_featuresdata = np.array([0, 0.179307744, 0.207885792, 0.236466888, 0.265044936, 0.293626032, 0.32220408])

    measured_average_drag_coeff = 1.56
    corresponding_rigid_cylinder_drag_coeff = 1.0

    # export every 20 pixels
    atmosphere_heightsdata = np.array([0, 14.078712, 16.754856, 19.431, 22.110192, 24.786336, 26.621232, 28.456128, 30.287976, 32.119824, 32.637984, 33.11652, 33.595056, 34.073592, 34.454592, 34.485072, 34.515552, 35.3568, 36.83508, 42.681144, 46.0248, 47.347632, 48.667416, 49.990248, 51.31308, 52.632864, 53.617368, 54.486048, 55.357776, 56.229504, 57.098184, 57.969912, 58.838592, 61.252608, 64.065912, 66.882264, 70.725792, 77.504544, 90.31224, 97.685352, 109.968792, 124.013976, 138.071352, 144.060672, 148.782024, 152.851104, 155.38704, 156.319728, 158.502096, 161.434272, 171.172632, 184.141872, 194.151504, 204.776832, 215.377776, 223.058736, 232.300272, 245.534688, 258.168648, 268.729968, 280.306272, 293.18712, 307.369464, 321.222624, 334.603344, 347.404944, 361.486704, 375.31548, 390.232392, 398.794224, 409.852368, 421.648128, 433.142136, 447.754248, 454.219056, 462.073752, 472.040712, 485.43972, 499.853712, 515.962392, 532.031448, 548.027352, 563.322216, 578.117208, 594.079584])
    atmosphere_featuresdata = np.array([0, 0.179307744, 0.236466888, 0.293626032, 0.350785176, 0.40794432, 0.465548472, 0.523152624, 0.580759824, 0.638367024, 0.69636132, 0.754367808, 0.812377344, 0.870383832, 1.044509976, 0.986451672, 0.928393368, 1.102376256, 1.160126712, 1.109856048, 1.05340404, 0.995458512, 0.937509936, 0.879564408, 0.82161888, 0.763673352, 0.705666864, 0.647642088, 0.589617312, 0.531592536, 0.47356776, 0.415542984, 0.35751516, 0.300005496, 0.242626896, 0.185251344, 0.12867132, 0.076568808, 0.06544056, 0.1148334, 0.109222032, 0.089059512, 0.118929912, 0.171233592, 0.226658424, 0.282735528, 0.284097984, 0.226082352, 0.168505632, 0.111194088, 0.066114168, 0.071944992, 0.117823488, 0.102897432, 0.05878068, 0.0983742, 0.145971768, 0.143353536, 0.138766296, 0.09325356, 0.098169984, 0.07629144, 0.07045452, 0.100565712, 0.132965952, 0.145840704, 0.114141504, 0.08135112, 0.068881752, 0.109648752, 0.123919488, 0.082558128, 0.081012792, 0.099117912, 0.151095456, 0.149019768, 0.102147624, 0.095210376, 0.09633204, 0.096749616, 0.111395256, 0.126744984, 0.13420344, 0.106887264, 0.092214192])

    parameters, p_fix_num, atmos_obj, wind_obj, _, _, drag_distribution_fun = construct_test_objects(drag_model_type=drag_model_type,
                                                                                                     cd_model=cd_model,
                                                                                                     cd=corresponding_rigid_cylinder_drag_coeff,
                                                                                                     wind_model='datafile',
                                                                                                     mu_ref=mu_ref,
                                                                                                     rho_ref=rho_ref,
                                                                                                     atmosphere_heightsdata=atmosphere_heightsdata,
                                                                                                     atmosphere_featuresdata=atmosphere_featuresdata,
                                                                                                     aero_elements=aero_elements)

    approx_max_deflection_ft = 45.
    cable_length_ft = 950.
    pythagorean_depth_ft = (cable_length_ft**2. - approx_max_deflection_ft**2.)**0.5
    ft_to_m = 0.3048
    q_upper = ft_to_m * (approx_max_deflection_ft * vect_op.xhat_np() + pythagorean_depth_ft * vect_op.zhat_np())

    q_upper = q_upper
    q_lower = 0. * vect_op.zhat_np()
    dq_upper = 0. * vect_op.xhat_np()
    dq_lower = 0. * vect_op.xhat_np()
    inch_to_ft = (1./12.)
    diameter = 0.162 * inch_to_ft * ft_to_m

    air_velocity = wind_obj.get_velocity(q_upper[2])
    kite_dynamic_pressure = 0.5 * atmos_obj.get_density(q_upper[2]) * cas.mtimes(air_velocity.T, air_velocity)
    ehat_1 = vect_op.xhat_np()
    ehat_3 = vect_op.zhat_np()
    alpha = 20. * np.pi/180.

    inputs = tether_element.columnize_element_info(q_upper=q_upper,
                                                   q_lower=q_lower,
                                                   dq_upper=dq_upper,
                                                   dq_lower=dq_lower,
                                                   diameter=diameter,
                                                   ehat_1=ehat_1,
                                                   ehat_3=ehat_3,
                                                   alpha=alpha,
                                                   kite_dynamic_pressure=kite_dynamic_pressure,
                                                   air_velocity=air_velocity,
                                                   unpacked_as_dict=None,
                                                   kite_only=drag_model_type=='kite_only')
    columnized_node_forces = drag_distribution_fun(inputs, p_fix_num)
    node_forces_dict = tether_segment.unpack_node_force_column(columnized_node_forces)
    total_drag_vec = node_forces_dict['upper'] + node_forces_dict['lower']

    maximum_current_ftps = 1.0
    max_current_mps = maximum_current_ftps * ft_to_m
    normalization_value = 0.5 * rho_ref * max_current_mps**2. * diameter * (cable_length_ft * ft_to_m)
    integrated_average_drag_coeff = cas.mtimes(total_drag_vec.T, wind_obj.get_wind_direction()) / normalization_value

    error = float( (integrated_average_drag_coeff - measured_average_drag_coeff) / measured_average_drag_coeff )
    if np.abs(error) > thresh:
        message = 'segment varying-drag integration test with model ' + drag_model_type + ' gives a wildly different average drag coefficient than reported in Kim/Vandiver/Holler (1986). '
        message += 'error at ' + str(aero_elements) + ' aero_elements is ' + str(error) + "."
        print_op.log_and_raise_error(message)

    return integrated_average_drag_coeff, measured_average_drag_coeff, corresponding_rigid_cylinder_drag_coeff, atmos_obj, wind_obj, atmosphere_heightsdata, atmosphere_featuresdata, inputs, p_fix_num

def make_plots_for_integration_test():

    import os
    import matplotlib
    if "MPLBACKEND" not in os.environ:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    end = 9
    aero_elements_list = 2**np.linspace(0, end, end+1)
    for cd_model in ['piecewise', 'polyfit', 'constant']:
        for drag_model_type in ['multi']: #'equivalent', 'equivalent_buggy',
            integrated_drag_list = []
            for aero_elements in aero_elements_list:
                integrated_average_drag_coeff, measured_average_drag_coeff, corresponding_rigid_cylinder_drag_coeff, atmos_obj, wind_obj, atmosphere_heightsdata, atmosphere_featuresdata, inputs, p_fix_num = test_segment_integration_varying(aero_elements=int(aero_elements), thresh=500., cd_model=cd_model, drag_model_type=drag_model_type)
                integrated_drag_list += [float(integrated_average_drag_coeff)]
            print(integrated_drag_list)
            plt.semilogx(np.array(aero_elements_list), np.array(integrated_drag_list), 'o', label='awebox integration (' + drag_model_type + ', ' + cd_model + ')')

    ax.set_xlim(ax.get_xlim())
    plt.semilogx(ax.get_xlim(), 2 * [measured_average_drag_coeff], '--',
                 label='measurement reported in 1986 paper')
    plt.semilogx(ax.get_xlim(), 2 * [corresponding_rigid_cylinder_drag_coeff], '--',
                 label='rigid cylinder drag coefficient')
    plt.xlabel('number of aero elements [-]')
    plt.ylabel('average drag coefficient over tether')
    plt.title('Kim et al. (1986) tether drag integration test')
    legend = ax.legend(loc='upper right')

    wind_obj.plot_velocity_profile(z_min=atmosphere_heightsdata[0], z_max=atmosphere_heightsdata[-1])

    inputs_dict = tether_element.unpack_element_info_column(inputs, kite_only=False)
    reynolds_list = []
    tether_zz_list = []
    windspeed_list = []

    for phi in np.linspace(0., 1., 1000)[1:-1]:
        q_local = inputs_dict['q_lower'] + phi * (inputs_dict['q_upper'] - inputs_dict['q_lower'])
        tether_zz_list += [q_local[2]]
        dq_local = inputs_dict['dq_lower'] + phi * (inputs_dict['dq_upper'] - inputs_dict['dq_lower'])
        wind_velocity = wind_obj.get_velocity(q_local[2])
        windspeed_list += [wind_velocity[0]]
        ua_local = wind_velocity - dq_local
        reynolds = tether_reynolds.get_reynolds_number(q_local=q_local, ua_local=ua_local, diam=inputs_dict['diameter'], mu_infty=p_fix_num['theta0', 'atmosphere', 'mu_ref'], rho_infty=p_fix_num['theta0', 'atmosphere', 'rho_ref'])
        reynolds_list += [reynolds]

    windspeed_list = np.array(windspeed_list).flatten()
    tether_zz_list = np.array(tether_zz_list).flatten()
    reynolds_list = np.array(reynolds_list).flatten()
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].plot(windspeed_list, tether_zz_list, label='flow speed')
    ax[0].plot(atmosphere_featuresdata, atmosphere_heightsdata, 'r*', label='Kim et al. (1986)')
    ax[0].set_xlabel('flow speed [m/s]')
    ax[0].set_ylim(tether_zz_list[0], tether_zz_list[-1])
    ax[0].legend(loc='upper right')
    ax[1].plot(reynolds_list, tether_zz_list, label='Reynolds number')
    ax[1].set_xlabel('Reynolds number [-]')
    ax[0].set_ylabel("'altitude' below sea-level along tether [m]")
    plt.suptitle('wind speed and Reynolds number for the Kim et al. (1986) test')

    plt.show()

def test():
    tether_reynolds.test()
    tether_coefficients.test()
    test_segment_integration_simple()
    test_segment_integration_varying()

if __name__ == "__main__":
    # test()
    # make_plots_for_integration_test()
    pass