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
atmospheric model for the a_w_ebox
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: jochem de schutter, rachel leuthold, a_l_u-_f_r 2017
'''
from awebox.logger.logger import Logger as awelogger
import casadi.tools as cas
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op

class Atmosphere(print_op.PrintableObject):
    def __init__(self, options, params, options_object=None):
        super().__init__(options_object=options_object, name='atmosphere')
        # if options['model'] == 'datafile':
            # self.find_u_polynomial_from_datafile(params)
            # self.find_p_polynomial_from_datafile(params)
        self.__options = options
        self.__params = params

        self.add_to_applied_params_dict('model.atmosphere.model', options['model'])
        self.add_to_applied_params_dict('params.atmosphere.g', params['theta0', 'atmosphere', 'g'])

    def get_temperature(self, zz):
        params = self.__params.prefix['theta0','atmosphere']
        options = self.__options

        self.add_to_applied_params_dict('params.atmosphere.t_ref', params['t_ref'])

        if options['model'] == 'isa':
            t = params['t_ref'] - params['gamma_air'] * zz
            self.add_to_applied_params_dict('params.atmosphere.gamma_air', params['gamma_air'])

        elif options['model'] == 'windshear':
            t = params['t_ref'] - params['gamma_air'] * zz
            self.add_to_applied_params_dict('params.atmosphere.gamma_air', params['gamma_air'])

        elif options['model'] == 'log_wind':
            t = params['t_ref'] * cas.DM.ones((1, 1))
        elif options['model'] == 'uniform':
            t = params['t_ref']
        elif options['model'] == 'datafile':
            t = params['t_ref'] - params['gamma_air'] * zz
            self.add_to_applied_params_dict('params.atmosphere.gamma_air', params['gamma_air'])
        else:
            raise ValueError('failure: unsupported atmospheric option chosen: %s', options['model'])
        return t

    def get_density(self, zz):
        params = self.__params.prefix['theta0','atmosphere']
        options = self.__options
        if options['model'] == 'isa':
            t = self.get_temperature(zz)
            rho = params['rho_ref'] * (t / params['t_ref']) ** (
                params['g'] / params['gamma_air'] / params['r'] - 1.0)
            self.add_to_applied_params_dict('params.atmosphere.rho_ref', params['rho_ref'])
            self.add_to_applied_params_dict('params.atmosphere.t_ref', params['t_ref'])
            self.add_to_applied_params_dict('params.atmosphere.g', params['g'])
            self.add_to_applied_params_dict('params.atmosphere.gamma_air', params['gamma_air'])
            self.add_to_applied_params_dict('params.atmosphere.r', params['r'])

        elif options['model'] == 'log_wind':
            rho = params['rho_ref'] * cas.DM.ones((1, 1))
            self.add_to_applied_params_dict('params.atmosphere.rho_ref', params['rho_ref'])
        elif options['model'] == 'uniform':
            rho = params['rho_ref']
            self.add_to_applied_params_dict('params.atmosphere.rho_ref', params['rho_ref'])
        elif options['model'] == 'datafile':
            rho = self.get_pressure(zz) / params['r'] / self.get_temperature(zz)
            self.add_to_applied_params_dict('params.atmosphere.r', params['r'])
        else:
            raise ValueError('failure: unsupported atmospheric option chosen: %s', options['model'])
        return rho

    def get_density_ref(self):
        params = self.__params.prefix['theta0', 'atmosphere']
        rho = params['rho_ref']
        self.add_to_applied_params_dict('params.atmosphere.rho_ref', params['rho_ref'])
        return rho

    def get_pressure(self, zz):
        params = self.__params.prefix['theta0','atmosphere']
        options = self.__options
        if options['model'] == 'isa':
            p = self.get_density(zz) * \
                params['r'] * self.get_temperature(zz)
            self.add_to_applied_params_dict('params.atmosphere.r', params['r'])

        elif options['model'] == 'log_wind':
            p = params['p_ref'] * cas.DM.ones((1, 1))
            self.add_to_applied_params_dict('params.atmosphere.p_ref', params['p_ref'])
        elif options['model'] == 'uniform':
            p = params['p_ref']
            self.add_to_applied_params_dict('params.atmosphere.p_ref', params['p_ref'])
        elif options['model'] == 'datafile':
            p = params['p_ref'] * cas.DM.ones((1, 1)) # constant value for now, could be computed with the files..
            self.add_to_applied_params_dict('params.atmosphere.p_ref', params['p_ref'])
        else:
            raise ValueError('failure: unsupported atmospheric option chosen: %s', options['model'])
        return p

    def get_viscosity(self, zz):
        params = self.__params.prefix['theta0','atmosphere']
        options = self.__options
        if options['model'] == 'isa':
            mu = params['mu_ref'] * (params['t_ref'] + params['c_sutherland']) / (self.get_temperature(zz) +
                 params['c_sutherland']) * (self.get_temperature(zz) / params['t_ref']) ** (3.0 / 2.0)
            self.add_to_applied_params_dict('params.atmosphere.mu_ref', params['mu_ref'])
            self.add_to_applied_params_dict('params.atmosphere.t_ref', params['t_ref'])
            self.add_to_applied_params_dict('params.atmosphere.c_sutherland', params['c_sutherland'])
        elif options['model'] == 'log_wind': #todo: 'log wind' should not be the name of an option here.
            mu = params['mu_ref']
            self.add_to_applied_params_dict('params.atmosphere.mu_ref', params['mu_ref'])
        elif options['model'] == 'uniform':
            mu = params['mu_ref']
            self.add_to_applied_params_dict('params.atmosphere.mu_ref', params['mu_ref'])
        elif options['model'] == 'datafile':
            mu = params['mu_ref'] * (params['t_ref'] + params['c_sutherland']) / (self.get_temperature(zz) +
                 params['c_sutherland']) * (self.get_temperature(zz) / params['t_ref']) ** (3.0 / 2.0)
            self.add_to_applied_params_dict('params.atmosphere.mu_ref', params['mu_ref'])
            self.add_to_applied_params_dict('params.atmosphere.t_ref', params['t_ref'])
            self.add_to_applied_params_dict('params.atmosphere.c_sutherland', params['c_sutherland'])
        else:
            raise ValueError('failure: unsupported atmospheric option chosen: %s', options['model'])
        return mu

    def get_speed_of_sound(self, zz):
        params = self.__params.prefix['theta0','atmosphere']
        a = (params['gamma'] * params['r'] * self.get_temperature(zz)) ** 0.5
        self.add_to_applied_params_dict('params.atmosphere.gamma', params['gamma'])
        self.add_to_applied_params_dict('params.atmosphere.r', params['r'])
        return a
