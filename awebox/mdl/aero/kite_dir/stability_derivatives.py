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
stability_derivatives aerodynamics modelling file
calculates stability derivatives based on orientation, angular velocity, and control surface deflection
_python-3.5 / casadi-3.4.5
- author: elena malz, chalmers 2016
- edited: rachel leuthold, alu-fr 2017-18
'''

import casadi.tools as cas

import awebox.mdl.aero.kite_dir.frames as frames
import awebox.tools.vector_operations as vect_op
from awebox.logger.logger import Logger as awelogger

def stability_derivatives(options, alpha, beta, airspeed, omega, delta, parameters):

    frames.test_conversions()

    force_frame = options['aero']['stab_derivs']['force_frame']
    moment_frame = options['aero']['stab_derivs']['moment_frame']

    inputs = collect_inputs(alpha, beta, airspeed, omega, delta, parameters, force_frame)
    coeffs = collect_contributions(parameters, inputs)

    # concatenate
    CF = distribute_force_coeffs(coeffs, force_frame)
    CM = distribute_moment_coeffs(coeffs, moment_frame)

    force_coeff_info = {'coeffs': CF, 'frame':force_frame}
    moment_coeff_info = {'coeffs': CM, 'frame':moment_frame}

    return force_coeff_info, moment_coeff_info


def check_associated_coeffs_defined_for_frame(associated_coeffs, frame, type=''):
    if not frame in associated_coeffs.keys():
        message = 'desired ' + type + ' frame ' + frame + ' is not in the list of ' \
            + 'expected frames: ' + repr(associated_coeffs.keys())
        awelogger.logger.error(message)

    return None

def check_associated_coeffs_exist_in_coeff_data(coeffs, associated_coeffs, frame):

    coeffs_associated_to_frame = associated_coeffs[frame]
    expected_in_coeffs = []
    for dim in range(3):
        coeffs_data_contains_expected_dimension_coefficient = coeffs_associated_to_frame[dim] in coeffs.keys()
        expected_in_coeffs += [coeffs_data_contains_expected_dimension_coefficient]

    if not all(expected_in_coeffs):
        message = 'the associated coefficients (' + repr(coeffs_associated_to_frame) + ') for the reference frame ' \
            + frame + ' do not all exist in the computed stability derivative data.'
        awelogger.logger.error(message)

    return None

def distribute_arbitrary_coeffs(coeffs, associated_coeffs, frame):

    coeffs_associated_to_frame = associated_coeffs[frame]
    check_associated_coeffs_exist_in_coeff_data(coeffs, associated_coeffs, frame)

    distributed = []
    for dim in range(3):
        distributed = cas.vertcat(distributed, coeffs[coeffs_associated_to_frame[dim]])

    return distributed


def distribute_force_coeffs(coeffs, frame):
    associated_coeffs = get_associated_force_coeffs()
    check_associated_coeffs_defined_for_frame(associated_coeffs, frame, type='force')
    distributed = distribute_arbitrary_coeffs(coeffs, associated_coeffs, frame)
    return distributed

def distribute_moment_coeffs(coeffs, frame):
    associated_coeffs = get_associated_moment_coeffs()
    check_associated_coeffs_defined_for_frame(associated_coeffs, frame, type='moment')
    distributed = distribute_arbitrary_coeffs(coeffs, associated_coeffs, frame)
    return distributed

def get_associated_moment_coeffs():
    associated_coeffs = {
        'control': ['Cl', 'Cm', 'Cn']
    }
    return associated_coeffs

def get_associated_force_coeffs():
    associated_coeffs = {
        'control': ['CX', 'CY', 'CZ'],
        'earth': ['Cx', 'Cy', 'Cz'],
        'body': ['CA', 'CY', 'CN'],
        'wind': ['CD', 'CS', 'CL']
    }
    return associated_coeffs

def list_all_possible_coeffs():

    list = []
    for associated_coeffs in [get_associated_force_coeffs(), get_associated_moment_coeffs()]:
        for frame in associated_coeffs.keys():
            list += associated_coeffs[frame]

    return list

def list_all_possible_inputs():
    list = ['0', 'alpha', 'beta', 'p', 'q', 'r', 'deltaa', 'deltae', 'deltar']
    return list


def collect_inputs(alpha, beta, airspeed, omega, delta, parameters, named_frame):

    # delta:
    # aileron left-right [right teu+, rad], ... positive delta a -> negative roll
    # elevator [ted+, rad],                 ... positive delta e -> negative pitch
    # rudder [tel+, rad])                   ... positive delta r -> positive yaw
    deltaa = delta[0]
    deltae = delta[1]
    deltar = delta[2]

    p, q, r = get_p_q_r(airspeed, omega, parameters, named_frame)

    inputs = {}
    inputs['0'] = cas.DM(1.)
    inputs['alpha'] = alpha
    inputs['beta'] = beta
    inputs['p'] = p
    inputs['q'] = q
    inputs['r'] = r
    inputs['deltaa'] = deltaa
    inputs['deltae'] = deltae
    inputs['deltar'] = deltar

    return inputs


def collect_contributions(parameters, inputs):

    stab_derivs = extract_derivs_from_parameters(parameters)

    coeffs = {}
    for deriv_name in stab_derivs.keys():

        if not deriv_name == 'frame':
            coeffs[deriv_name] = 0.

            for input_name in stab_derivs[deriv_name].keys():

                deriv_stack = stab_derivs[deriv_name][input_name]
                deriv_length = deriv_stack.shape[0]

                if not input_name in inputs.keys():
                    message = 'desired stability derivative input ' + input_name + ' is not recognized. ' \
                        + 'The following inputs are defined: ' + repr(inputs.keys())
                    awelogger.logger.error(message)

                input_val = inputs[input_name]
                alpha_val = inputs['alpha']
                input_stack = []
                for ldx in range(deriv_length):
                    input_stack = cas.vertcat(input_stack, input_val * alpha_val**ldx )

                contrib_from_input = cas.mtimes(deriv_stack.T, input_stack)
                coeffs[deriv_name] += contrib_from_input

    return coeffs

def get_p_q_r(airspeed, omega, parameters, named_frame):

    # omega is defined in "body" reference frame.
    if named_frame == 'control':
        omega = frames.from_body_to_control(omega)

    # p -> roll rate, about ehat1
    # q -> pitch rate, about ehat2
    # r -> yaw rate, about ehat3

    # pqr - damping: in radians
    # notice that the norm is independent of frame, iff frames are orthonormal
    omega_hat = omega / (2. * airspeed)

    b_ref = parameters['theta0','geometry','b_ref']
    c_ref = parameters['theta0','geometry','c_ref']

    omega_hat[0] *= b_ref  # pb/2|ua|
    omega_hat[1] *= c_ref  # qc/2|ua|
    omega_hat[2] *= b_ref  # rb/2|ua|

    # roll, pitch, yaw
    p = omega_hat[0]
    q = omega_hat[1]
    r = omega_hat[2]

    return p, q, r

def extract_derivs_from_parameters(parameters):

    stab_derivs = {}

    all_possible_coeffs = list_all_possible_coeffs()
    all_possible_inputs = list_all_possible_inputs()

    for coeff_name in all_possible_coeffs:

        for input_name in all_possible_inputs:

            local_parameter_label = '[theta0,aero,' + coeff_name + ',' + input_name + ',0]'
            if local_parameter_label in parameters.labels():
                vals = parameters['theta0', 'aero', coeff_name, input_name]

                if not coeff_name in stab_derivs.keys():
                    stab_derivs[coeff_name] = {}

                stab_derivs[coeff_name][input_name] = vals

    return stab_derivs