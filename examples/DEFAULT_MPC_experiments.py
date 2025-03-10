import sys
sys.path.extend(['/Users/jakobharzer/MyDrive/Uni/Research/awebox']) # so we can also run this in the console, pycharm does this automatically

from examples.SAM_MPC_experiment import run_SAM_MPC_experiment

# d = 3 # manually 'parellize' over d
# N_list = [3,4,5,6]

# d = 4 # manually 'parellize' over d
# N_list = [5,10,15,20]

d = 3 # manually 'parellize' over d
# N_list = [5,10]
N_list = [15,20]

for N in N_list:
    export_dict = run_SAM_MPC_experiment(d, N, param_regulization=1E-2)