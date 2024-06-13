#!/usr/bin/python3
"""Test that the examples work.

@author: Rachel Leuthold, ALU-FR 2024
"""

import os

import awebox.tools.print_operations as print_op
import awebox.tools.vector_operations as vect_op
import matplotlib.pyplot as plt
import glob
import importlib.util

def test_examples(threshold=0.2):

    current_directory = os.getcwd()
    test_reg_index = current_directory.find("test/reg")
    base_directory = current_directory[:test_reg_index]
    example_directory = base_directory + 'examples/'
    extension = '.py'

    python_files = glob.glob(example_directory + "*" + extension)
    for filename in python_files:

        module_name = filename[len(example_directory):-len(extension)]
        module_spec = importlib.util.spec_from_file_location(module_name, filename)
        local_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(local_module)

        local_output = None
        if hasattr(local_module, 'run'):
            local_output = local_module.run(plot_show_block=False, quality_raise_exception=True)

        if (local_output is not None) and hasattr(local_module, 'make_comparison'):
            comparison_dict = local_module.make_comparison(local_output)
            for comparison_name in comparison_dict.keys():
                expected = comparison_dict[comparison_name]['expected']
                found = comparison_dict[comparison_name]['found']
                error = (expected - found) / vect_op.smooth_norm(expected)

                if vect_op.norm(error) > threshold:
                    message = 'something went wrong with the example at ' + filename + ';'
                    message += ' error for ' + comparison_name + ' is ' + print_op.repr_g(error)
                    print_op.log_and_raise_error(message)

                plt.close('all')


if __name__ == "__main__":
    test_examples()