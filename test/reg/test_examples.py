#!/usr/bin/python3
"""Test that the examples work.

@author: Rachel Leuthold, ALU-FR 2024
"""

import os
import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import matplotlib.pyplot as plt
import importlib.util

def get_example_directory():
    current_directory = os.getcwd()
    test_reg_index = current_directory.find("test/reg")
    base_directory = current_directory[:test_reg_index]
    example_directory = base_directory + 'examples/'
    return example_directory

def get_local_module(module_name):
    example_directory = get_example_directory()
    filename = example_directory + module_name + '.py'

    module_spec = importlib.util.spec_from_file_location(module_name, filename)
    local_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(local_module)
    return local_module

def run_example_test(module_name, threshold=0.2):

    local_module = get_local_module(module_name)
    overwrite_options = {'quality.raise_exception': True}

    if hasattr(local_module, 'run'):
        local_output = local_module.run(plot_show_block=False, overwrite_options=overwrite_options)

    if hasattr(local_module, 'make_comparison'):
        comparison_dict = local_module.make_comparison(local_output)
        for comparison_name in comparison_dict.keys():
            expected = comparison_dict[comparison_name]['expected']
            found = comparison_dict[comparison_name]['found']
            error = (expected - found) / vect_op.smooth_norm(expected)

            if vect_op.norm(error) > threshold:
                message = 'something went wrong with the ' + module_name + ' example;\n'
                message += ' error for ' + comparison_name + ' is ' + print_op.repr_g(error)
                print_op.log_and_raise_error(message)

        plt.close('all')
    return

def test_dual_kites_power_curve_example(threshold=0.2):
    module_name = 'dual_kites_power_curve'
    run_example_test(module_name, threshold)

def test_ampyx_ap2_trajectory_example(threshold=0.2):
    module_name = 'ampyx_ap2_trajectory'
    run_example_test(module_name, threshold)

def test_mpc_closed_loop_example(threshold=0.2):
    #todo: Jochem
    return None


if __name__ == "__main__":
    test_ampyx_ap2_trajectory_example()
    test_dual_kites_power_curve_example()
    test_mpc_closed_loop_example()
