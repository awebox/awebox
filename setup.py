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

from setuptools import setup, find_packages

import sys
print(sys.version_info)

if sys.version_info < (3,8):
    sys.exit('Python version 3.8 or later required. Exiting.')

setup(name='awebox',
   version='0.1.0',
   python_requires='>=3.8',
   description='Modeling and optimal control of sinlge- and multi-kite systems',
   url='https://github.com/awebox/awebox',
   author='Jochem De Schutter, Rachel Leuthold, Thilo Bronnenmeyer, Elena Malz, Sebastien Gros, Moritz Diehl',
   author_email='jochem.de.schutter@imtek.de',
   license='LGPLv3.0',
   packages = find_packages(),
   include_package_data = True,
   # setup_requires=['setuptools_scm'],
   # use_scm_version={
   #   "fallback_version": "0.1-local",
   #   "root": "../..",
   #   "relative_to": __file__
   # },
   install_requires=[
        'casadi==3.5.5',
        'cycler==0.11.0',
        'fonttools==4.34.4',
        'kiwisolver==1.4.4',
        'matplotlib==3.5.2',
        'numpy==1.23.1',
        'packaging==21.3',
        'Pillow==9.2.0',
        'pyparsing==3.0.9',
        'python-dateutil==2.8.2',
        'scipy==1.9.0rc3',
        'six==1.16.0',
        'tk==0.1.0',
	'tabulate==0.8.10'
   ],
)
