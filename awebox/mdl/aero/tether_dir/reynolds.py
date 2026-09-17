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
reynolds number of a tether element
_python-3.5 / casadi-3.4.5
- edited: rachel leuthold, jochem de schutter alu-fr 2017-20
'''

import awebox.tools.vector_operations as vect_op
import casadi.tools as cas
import awebox.tools.print_operations as print_op
from awebox.mdl.wind import Wind


def get_reynolds_number(atmos=None, diam=None, q_upper=None, q_lower=None, q_local=None, ua_local=None, wind=None, dq_local=None, rho_infty=None, mu_infty=None):

    if diam is None:
        message = 'not enough diameter information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    def get_zz():
        if q_local is not None:
            q_average = q_local
        elif (q_upper is not None) and (q_lower is not None):
            q_average = (q_upper + q_lower) / 2.
        else:
            message = 'not enough position information to determine the reynolds number function'
            print_op.log_and_raise_error(message)
        zz = q_average[2]
        return zz

    if (rho_infty is not None):
        pass
    elif (rho_infty is None) and (atmos is not None):
        zz = get_zz()
        rho_infty = atmos.get_density(zz)
    else:
        message = 'not enough density information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    if mu_infty is not None:
        pass
    elif (mu_infty is None) and (atmos is not None):
        zz = get_zz()
        mu_infty = atmos.get_viscosity(zz)
    else:
        message = 'not enough viscosity information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    if ua_local is not None:
        vec_ua = ua_local
    elif (wind is not None) and (dq_local is not None):
        vec_u_infty = wind.get_velocity(zz)
        vec_ua = vec_u_infty - dq_local
    else:
        message = 'not enough velocity information to determine the reynolds number function'
        print_op.log_and_raise_error(message)

    norm_ua = cas.mtimes(vec_ua.T, vec_ua) ** 0.5
    reynolds = rho_infty * norm_ua * diam / mu_infty

    return reynolds

def test(thresh=0.2):
    # https://dragonfly.tam.cornell.edu/teaching/mae5230-tritton-chap3.pdf
    # page 22
    tests = {}

    # low reynolds number test
    ua_local = 10e-3 * vect_op.zhat_np() # 10 mm/s
    diam = 10e-3 # 10mm
    # https://en.wikipedia.org/wiki/Glycerol
    rho_infty = 1264.02 #20C
    kinematic_viscosity = 1.18e-3 #20C
    mu_infty = kinematic_viscosity * rho_infty #https://en.wikipedia.org/wiki/Viscosity
    expected = 1e-1
    found = get_reynolds_number(ua_local=ua_local, diam=diam, rho_infty=rho_infty, mu_infty=mu_infty)
    tests['low'] = {'found': found, 'expected': expected}

    # high reynolds number test
    ua_local = 50. * vect_op.zhat_np() #m/s
    diam = 0.3 #m
    # https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm
    rho_infty = 1.225 #at 15C
    mu_infty = 1.802e-5
    expected = 1e6
    found = get_reynolds_number(ua_local=ua_local, diam=diam, rho_infty=rho_infty, mu_infty=mu_infty)
    tests['high'] = {'found': found, 'expected': expected}

    for test_name, test_dict in tests.items():
        error = (test_dict['expected'] - test_dict['found']) / test_dict['expected']
        test_dict['error'] = error
        criteria = (error * error)**0.5 < thresh
        if not criteria:
            message = test_name + ' reynolds number test did not work as expected. '
            for val_name, val_val in test_dict.items():
                message += val_name + ': ' + str(val_val) + ', '
            message = message[:-2]
            print_op.log_and_raise_error(message)


if __name__ == "__main__":
    test()