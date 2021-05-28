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
- author: jochem de schutter, rachel leuthold, alu-fr 2018-20
'''

import casadi.tools as cas
import numpy as np
from awebox.logger.logger import Logger as awelogger
import awebox.tools.vector_operations as vect_op
import awebox.tools.lagr_interpol as lagr_interpol


class Wind:
    def __init__(self, wind_model_options, params):
        self.__options = wind_model_options
        self.__params = params #NOTE: where do those parameters come from?
        if self.__options['model'] == 'datafile':
            self.find_u_polynomial_from_datafile()
            # self.find_p_polynomial_from_datafile(params) # pressure is set as constant for now

        self.__type_incompatibility_warning_already_given = False

    def get_velocity(self, zz):
        params = self.__params.prefix['theta0','wind']
        options = self.__options

        model = options['model']

        xhat = vect_op.xhat_np()

        if isinstance(zz, cas.SX):
            u_ref = params['u_ref']
            z_ref = params['z_ref']
            z0_air = params['log_wind', 'z0_air']
            exp_ref = params['power_wind', 'exp_ref']
        else:
            u_ref = options['u_ref']
            z_ref = options['z_ref']
            z0_air = options['log_wind']['z0_air']
            exp_ref = options['power_wind']['exp_ref']

            if not self.__type_incompatibility_warning_already_given:
                message = 'to prevent casadi type incompatibility, wind parameters are imported ' \
                          'directly from options. this may interfere with expected operation, especially in sweeps.'
                awelogger.logger.warning(message)
                self.__type_incompatibility_warning_already_given = True

        if model in ['log_wind', 'power', 'uniform']:
            u_val = get_speed(model, u_ref, z_ref, z0_air, exp_ref, zz)
            u = u_val * xhat

        elif model == 'datafile':
            u = self.get_velocity_from_datafile(zz)

        else:
            raise ValueError('unsupported atmospheric option chosen: %s', model)

        return u


    def get_velocity_ref(self):
        params = self.__params.prefix['theta0','wind']
        u_ref = params['u_ref']

        return u_ref

    def find_u_polynomial_from_datafile(self):
        """_data description:
        data given at 10 lowest different pressure levels (see text file) at 2928 time instants.
        3h resolution over the year 2016 in goeteborg

        winddata:       north and east wind component
        heightsdata:    heights corresponding to the pressure levels at that time point
        featuresdata:   north and east wind component converted to main wind direction
                        and angle derivation. later in the code this is converter to
                        x and y wind component. in the code x is the main wind direction.
        """
        options = self.__options
        heightsdata  = options['atmosphere_heightsdata']
        featuresdata = options['atmosphere_featuresdata']

        # k = options['atmosphere_dataseries'] ## NOTE: What's that??? the number of pressure points?

        # create x and y wind component
        k = 0 # example for time stamp 1
        xwind = [w * np.abs(np.cos(-a)) for w, a in featuresdata[:, k, :]]
        ywind = [w * np.sin(-a) for w, a in featuresdata[:, k, :]]
        xwind = np.array(xwind, dtype=float)
        ywind = np.array(ywind, dtype=float)
        self.heights = heightsdata[:, k]

        # create the function of the lagrange polynomial for x and y wind
        # and give out the polynomial parameters, respectively.
        _, taux_opt = lagr_interpol.smooth_lagrange_poly(self.heights, xwind)
        _, tauy_opt = lagr_interpol.smooth_lagrange_poly(self.heights, ywind)

        self.taux_opt = taux_opt
        self.tauy_opt = tauy_opt

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

        # generate the lagrange polynomial with the 'optimized' poly. parameters
        Lagr_x_fun = lagr_interpol.lagrange_poly(self.heights, self.taux_opt)
        Lagr_y_fun = lagr_interpol.lagrange_poly(self.heights, self.tauy_opt)
        # compute the x,y,z components
        x_component = Lagr_x_fun(zz)
        y_component = Lagr_y_fun(zz)
        z_component = 0.

        u_wind = cas.vertcat(x_component, y_component, z_component)
        return u_wind

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
        awelogger.logger.warning('Cannot set options object.')


def get_speed(model, u_ref, z_ref, z0_air, exp_ref, zz):

    # approximates the maximum of (zz vs. 0)
    epsilon = 1.
    z_cropped = vect_op.smooth_abs(zz, epsilon=epsilon)

    if model == 'log_wind':

        # mathematically: it doesn't make a difference what the base of
        # these logarithms is, as long as they have the same base.
        # but, the values will be smaller in base 10 (since we're describing
        # altitude differences), which makes convergence nicer.
        # u = u_ref * np.log10(zz / z0_air) / np.log10(z_ref / z0_air)
        u = u_ref * cas.log10(z_cropped / z0_air) / cas.log10(z_ref / z0_air)

    elif model == 'power':
        # u = u_ref * (zz / z_ref) ** exp_ref
        u = u_ref * (z_cropped / z_ref) ** exp_ref

    elif model == 'uniform':
        u = u_ref

    else:
        raise ValueError('unsupported atmospheric option chosen: %s', model)

    return u
