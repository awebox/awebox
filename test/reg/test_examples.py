#!/usr/bin/python3
"""Test that the examples work.

@author: Rachel Leuthold, ALU-FR 2024
"""

import os
import awebox.tools.print_operations as print_op
import matplotlib.pyplot as plt
import subprocess
import glob

def test_examples():

    current_directory = os.getcwd()
    test_reg_index = current_directory.find("test/reg")
    base_directory = current_directory[:test_reg_index]
    example_directory = base_directory + 'examples/'

    python_files = glob.glob(example_directory + "*.py")
    for filename in python_files:

        print_op.warn_about_temporary_functionality_alteration()
        # todo: @Jochem, I don't know how your mpc interpolation stuff works, and I don't want to break anything.
        #  so, that's just temporarily not included.
        if not 'mpc_closed_loop' in filename:

            with open(filename) as file:

                action_message = 'running >> python3 ' + filename
                print_op.base_print(action_message)

                # use subprocess instead of exec, even though it's slower,
                # because otherwise, you won't be able to automatically close the plots
                p = subprocess.run(["python3", filename])

                if p.returncode != 0:
                    error_message = "something went wrong when " + action_message
                    error_message += ". return code = " + str(p.returncode)
                    print_op.log_and_raise_error(error_message)

                plt.close('all')


# test_examples()