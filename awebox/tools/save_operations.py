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
"""
Stores helper functions for saving data
-author: Thilo Bronnenmeyer, kiteswarms, 2019
-edit: Rachel Leuthold, ALU-FR, 2020
"""

import pickle

def save(data, file_name, file_type):

    file_pi = open(file_name + '.' + file_type, 'wb')
    pickle.dump(data, file_pi)
    file_pi.close()

    return None

def extract_solution_dict_from_file(file):

    solution_dict = None

    if type(file) == str:

        if not (file[-5:] == '.dict'):
            file = file + '.dict'

        try:
            filehandler = open(file, 'rb')
            saved_dict = pickle.load(filehandler)
            solution_dict = saved_dict['solution_dict']
        except:
            raise ValueError('Specified file cannot be imported. Please check whether correct filename was given.')

    elif type(file) == dict:
        saved_dict = file
        if 'solution_dict' in saved_dict.keys():
            solution_dict = saved_dict['solution_dict']
        elif 'options' in saved_dict.keys():
            solution_dict = saved_dict
        else:
            raise ValueError('Specified file dictionary does not correspond to expected format.')

    else:
        # assuming that the file is already a "trial"
        try:
            solution_dict = file.generate_solution_dict()
        except:
            raise ValueError('Specified file format is not yet accepted.')

    return solution_dict
