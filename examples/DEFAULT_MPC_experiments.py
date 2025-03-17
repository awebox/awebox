import sys
sys.path.extend(['/Users/jakobharzer/MyDrive/Uni/Research/awebox']) # so we can also run this in the console, pycharm does this automatically

from examples.DEFAULT_experiment import run_DEFAULT_MPC_experiment

# N_list = [1,2,3,4,5,6,7]
N_list = [8,9,10]

for N in N_list:
    export_dict = run_DEFAULT_MPC_experiment(N)