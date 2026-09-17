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
wind model for the awebox
_python-3.5 / casadi-3.4.5
- author: jochem de schutter, rachel leuthold, alu-fr 2018-25
'''

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.lagr_interpol as lagr_interpol
import matplotlib.pyplot as plt


class Wind(print_op.PrintableObject):
    def __init__(self, wind_model_options, params, suppress_type_incompatibility_warning=False, options_object=None):
        super().__init__(options_object=options_object, name='wind')
        self.__options = wind_model_options
        self.__params = params #NOTE: where do those parameters come from?

        self.__xwind_data = None
        self.__ywind_data = None
        if self.__options['model'] == 'datafile':
            self.find_u_polynomial_from_datafile()
            # self.find_p_polynomial_from_datafile(params) # pressure is set as constant for now

        self.__type_incompatibility_warning_already_given = False
        self.__suppress_type_incompatibility_warning = suppress_type_incompatibility_warning

        self.add_to_applied_params_dict('user_options.wind.model', wind_model_options['model'])

    def get_local_param_value(self, zz, external_parameters, param_name, second_name=None):
        if isinstance(zz, cas.SX):
            if external_parameters is None:
                params = self.__params.prefix['theta0', 'wind']
            else:
                params = external_parameters.prefix['theta0', 'wind']

            if second_name is None:
                return params[param_name]
            else:
                return params[param_name, second_name]
        else:
            if not self.__type_incompatibility_warning_already_given and not self.__suppress_type_incompatibility_warning:
                warn_about_importing_from_options()
                self.__type_incompatibility_warning_already_given = True

            if second_name is None:
                return self.__options[param_name]
            else:
                return self.__options[param_name][second_name]


    def get_velocity(self, zz, external_parameters=None):

        options = self.__options
        model = options['model']

        u_hat = self.get_wind_direction()
        if model == 'log_wind':

            u_ref = self.get_local_param_value(zz, external_parameters, 'u_ref')
            z_ref = self.get_local_param_value(zz, external_parameters, 'z_ref')
            z0_air = self.get_local_param_value(zz, external_parameters, 'log_wind', second_name='z0_air')

            self.add_to_applied_params_dict('user_options.wind.u_ref', u_ref)
            self.add_to_applied_params_dict('params.wind.z_ref', z_ref)
            self.add_to_applied_params_dict('params.wind.log_wind.z0_air', z0_air)
            u_val = get_log_law_speed(u_ref, z_ref, z0_air, zz)
            u = u_val * u_hat

        elif model == 'power':

            u_ref = self.get_local_param_value(zz, external_parameters, 'u_ref')
            z_ref = self.get_local_param_value(zz, external_parameters, 'z_ref')
            exp_ref = self.get_local_param_value(zz, external_parameters, 'power_wind', second_name='exp_ref')

            self.add_to_applied_params_dict('user_options.wind.u_ref', u_ref)
            self.add_to_applied_params_dict('params.wind.z_ref', z_ref)
            self.add_to_applied_params_dict('params.wind.power_wind.exp_ref', exp_ref)
            u_val = get_power_law_speed(u_ref, z_ref, exp_ref, zz)
            u = u_val * u_hat

        elif model == 'uniform':
            u_ref = self.get_local_param_value(zz, external_parameters, 'u_ref')
            self.add_to_applied_params_dict('user_options.wind.u_ref', u_ref)
            u_val = get_uniform_speed(u_ref)
            u = u_val * u_hat

        elif model == 'datafile':
            u = self.get_velocity_from_datafile(zz)

        else:
            raise ValueError('unsupported atmospheric option chosen: %s', model)

        return u


    def plot_velocity_profile(self, z_min=None, z_max=800.):

        import os
        import matplotlib
        if "MPLBACKEND" not in os.environ:
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        if z_min is None:
            z0_air = self.__options['log_wind']['z0_air']
            z_min = z0_air
        z_vals = list(np.arange(z_min, z_max, 1).flatten())
        u_vals = list(np.array([self.get_velocity(zz)[0] for zz in z_vals]).flatten())

        fig = plt.figure()
        if self.xwind_data is not None:
            plt.plot(self.xwind_data, self.heights, 'o')
        plt.plot(list(u_vals), list(z_vals))

        plt.xlabel('wind speed [m/s]')
        plt.ylabel('altitude [m]')
        plt.title('wind speed profile')
        return None

    def get_wind_direction(self):
        return vect_op.xhat_dm()

    def get_speed_ref(self, from_parameters=True):
        if from_parameters:
            params = self.__params.prefix['theta0','wind']
            u_ref = params['u_ref']
        else:
            u_ref = self.__options['u_ref']
            if not self.__type_incompatibility_warning_already_given:
                warn_about_importing_from_options()
                self.__type_incompatibility_warning_already_given = True


        return u_ref

    def find_u_polynomial_from_datafile(self, use_buffer=True, order=12):
        """_data description:
        a function to create x- and y-direction wind-speed-vs-altitude polynomials, from a datafile that gives
        data at some number of lowest different pressure levels at multiple time instants - (eg, from 3h resolution
        data over the year 2016 in goeteborg).

        heightsdata:    heights corresponding to the pressure levels at that time point
        featuresdata:   the wind speed and angle corresponding to the heightsdata. later in the code
                        this will be converter to x and y wind component. in the code x is the main wind direction.

        a note on input format: it seems likely (??) that this function is modeled after the implementation in Malz2019,
        in which case the input data format would (??) be according to some MERRA-2 standard, but M2I3NVASM and
        M2I3NPASM both seem to use x- and y- direction wind speeds (cartesian components u and v) rather
        than the w- and a- cylindrical coordinates that this function is using, so: there is some pre-processing
        step missing. if you use this function (successfully or unsucessfully) please contact the awebox developers, to
        share your experiences.
        """
        options = self.__options
        heightsdata  = options['atmosphere_heightsdata']
        featuresdata = options['atmosphere_featuresdata']

        k = 0 # evaluates at the first time-stamp from the wind-data datafiles: the awebox does
        # not presently allow time-varying wind polynomials

        if vect_op.is_numeric_columnar(heightsdata):
            self.heights = np.array(heightsdata)
        else:
            self.heights = np.array(heightsdata[:, k])

        featuresdata_axes = 0
        try:
            test = featuresdata[0, 0, 0]
            featuresdata_axes = 3
        except:
            try:
                test = featuresdata[0, 0]
                featuresdata_axes = 2
            except:
                try:
                    test = featuresdata[0]
                    featuresdata_axes = 1
                except:
                    message = 'featuresdata for wind polynomial does not have a recognizable data-stape'
                    print_op.log_and_raise_error(message)

        # create x and y wind component
        if featuresdata_axes == 1:
            xwind = np.array(featuresdata)
            ywind = 0. * xwind
        elif featuresdata_axes == 2:
            xwind = [w * np.abs(np.cos(-a)) for w, a in featuresdata[:, :]]
            ywind = [w * np.sin(-a) for w, a in featuresdata[:, :]]
            xwind = np.array(xwind, dtype=float)
            ywind = np.array(ywind, dtype=float)
        else:
            xwind = [w * np.abs(np.cos(-a)) for w, a in featuresdata[:, k, :]]
            ywind = [w * np.sin(-a) for w, a in featuresdata[:, k, :]]
            xwind = np.array(xwind, dtype=float)
            ywind = np.array(ywind, dtype=float)

        self.__xwind_data = xwind
        self.__ywind_data = ywind

        if use_buffer: # try to damp out any interpolation oscillations near the edge of the domain.
            buffer = 1 #int(np.floor(float(len(self.heights))/5.))
            delta_h = self.heights[-1] - self.heights[-2]
            heights_buffer = [self.heights[-1] + delta_h * (float(bdx) + 1.) for bdx in range(buffer)]
            heights = np.array(cas.vertcat(self.heights, heights_buffer))
            xwind = np.array(cas.vertcat(xwind, buffer*[xwind[-1]]))
            ywind = np.array(cas.vertcat(ywind, buffer*[ywind[-1]]))
        else:
            heights = self.heights

        # do the interpolation
        zz_sym = cas.SX.sym('zz_sym')

        # we had lagrange polynomial fit before, but it worked really badly with the vandiver/holler/kim verification test.
        # L_funx, taux_opt = lagr_interpol.smooth_lagrange_poly(heights, xwind)
        # L_funy, tauy_opt = lagr_interpol.smooth_lagrange_poly(heights, ywind)
        # self.__Lagr_x_fun = cas.Function('Lagr_x_fun', [zz_sym], [L_funx(zz_sym, taux_opt)])
        # self.__Lagr_y_fun = cas.Function('Lagr_y_fun', [zz_sym], [L_funy(zz_sym, tauy_opt)])
        # self.taux_opt = taux_opt
        # self.tauy_opt = tauy_opt

        unit_steps_x = vect_op.interpolate_by_unit_stepping(heights, xwind, zz_sym, 1e-5)
        unit_steps_y = vect_op.interpolate_by_unit_stepping(heights, ywind, zz_sym, 1e-5)
        self.__Lagr_x_fun = cas.Function('Lagr_x_fun', [zz_sym], [unit_steps_x])
        self.__Lagr_y_fun = cas.Function('Lagr_y_fun', [zz_sym], [unit_steps_y])

        def quick_sanity_check_on_interpolation(thresh=1.0):
            for idx in range(2, len(self.heights)):
                comparison = {'x': {'found': self.Lagr_x_fun(heights[idx]), 'expected': xwind[idx]},
                              'y': {'found': self.Lagr_y_fun(heights[idx]), 'expected': ywind[idx]}
                              }
                for dir, test_dict in comparison.items():
                    diff = (test_dict['found'] - test_dict['expected'])
                    if np.abs(diff) > thresh:
                        message = 'datafile wind speed interpolation in direction (' + dir + ') at altitude ' + str(self.heights[idx]) + ' does not work as expected. '
                        for pair_name, pair_val in test_dict.items():
                            message += pair_name + ": " + str(pair_val) + ". "
                        print_op.log_and_raise_error(message)
        quick_sanity_check_on_interpolation()

    @property
    def Lagr_x_fun(self):
        return self.__Lagr_x_fun

    @Lagr_x_fun.setter
    def Lagr_x_fun(self, value):
        awelogger.logger.warning('Cannot set Lagr_x_fun object.')

    @property
    def Lagr_y_fun(self):
        return self.__Lagr_y_fun

    @Lagr_y_fun.setter
    def Lagr_y_fun(self, value):
        awelogger.logger.warning('Cannot set Lagr_y_fun object.')

    @property
    def xwind_data(self):
        return self.__xwind_data

    @xwind_data.setter
    def xwind_data(self, value):
        awelogger.logger.warning('Cannot set xwind_data object.')

    @property
    def ywind_data(self):
        return self.__ywind_data

    @ywind_data.setter
    def ywind_data(self, value):
        awelogger.logger.warning('Cannot set ywind_data object.')


    def find_p_polynomial_from_datafile(self):
        options = self.__options
        pressures = np.array(
            cas.vertcat(
                895.0000,
                910.0000,
                925.0000,
                940.0000,
                955.0000,
                970.0000,
                985.0000)) * 1e2

        heightsdata = np.load(options['atmosphere_heightsdata'])

        k = options['atmosphere_dataseries']

        heights = np.array(heightsdata[:-3, k])

        # pressures = np.array(cas.vertcat(pressures, params['p_ref']))
        # heights = np.array(cas.vertcat(heights, params['z_ref']))

        self.lp_fun, taup_opt = lagr_interpol.smooth_lagrange_poly(heights, pressures)
        self.p_polynomials = [self.lp_fun, taup_opt]

    def get_velocity_from_datafile(self, zz):
        # compute the x,y,z components
        x_component = self.__Lagr_x_fun(zz)
        y_component = self.__Lagr_y_fun(zz)
        z_component = 0.
        u_wind = cas.vertcat(x_component, y_component, z_component)
        return u_wind

    # customize the object state to hopefully allow trial.save to pickle everything
    def __getstate__(self):
        state = self.__dict__.copy()
        # CasADi/struct/SWIG-heavy object; reconstruct separately if needed
        state.pop("_Wind__params", None)
        # parent PrintableObject may hold a non-picklable options object
        state.pop("_PrintableObject__options_object", None)
        state.pop("_PrintableObject__applied_parameters_dict", None)
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_Wind__params"] = None
        self.__dict__["_PrintableObject__options_object"] = None
        self.__dict__["_PrintableObject__applied_parameters_dict"] = {}


    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
        awelogger.logger.warning('Cannot set options object.')


def warn_about_importing_from_options():
    message = 'to prevent casadi type incompatibility, wind parameters are imported ' \
              'directly from options. this may interfere with expected operation, especially in sweeps.'
    awelogger.logger.warning(message)
    return None


def get_z_cropped(zz, epsilon=1e-4):
    # approximates the maximum of (zz vs. 0)
    z_cropped = vect_op.smooth_abs(zz, epsilon=epsilon)
    return z_cropped

def get_log_law_speed(u_ref, z_ref, z0_air, zz):
    z_cropped = get_z_cropped(zz)

    # mathematically: it doesn't make a difference what the base of
    # these logarithms is, as long as they have the same base.
    # but, the values will be smaller in base 10 (since we're describing
    # altitude differences), which makes convergence nicer.
    # u = u_ref * np.log10(zz / z0_air) / np.log10(z_ref / z0_air)
    u = u_ref * cas.log10(z_cropped / z0_air) / cas.log10(z_ref / z0_air)
    return u

def get_power_law_speed(u_ref, z_ref, exp_ref, zz):
    z_cropped = get_z_cropped(zz)
    # u = u_ref * (zz / z_ref) ** exp_ref
    u = u_ref * (z_cropped / z_ref) ** exp_ref
    return u

def get_uniform_speed(u_ref):
    u = u_ref
    return u

def get_speed(model, u_ref, z_ref, z0_air, exp_ref, zz):
    if model == 'log_wind':
        return get_log_law_speed(u_ref, z_ref, z0_air, zz)
    elif model == 'power':
        return get_power_law_speed(u_ref, z_ref, exp_ref, zz)
    elif model == 'uniform':
        return get_uniform_speed(u_ref)
    elif model == 'datafile':
        message = 'the mdl.wind external get_speed function is not currently set-up to allow wind velocity profile importing from datafile.'
        print_op.log_and_raise_error(message)
    else:
        raise ValueError('unsupported atmospheric option chosen: %s', model)
    return None
