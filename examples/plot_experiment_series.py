from ftplib import all_errors

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import alpha
import matplotlib.ticker as mtick


# %% Latexify the plots
def latexify():
    import matplotlib
    params_MPL_Tex = {
        'text.usetex': True,
        'font.family': 'serif',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    matplotlib.rcParams.update(params_MPL_Tex)


latexify()

class ExperimentInfo:
    def __init__(self, filepath: str):

        print(f'Extracting Experiment Information from: {filepath}')

        data = np.load(filepath, allow_pickle=True)

        self.data_SAM = data['SAM'].item()
        self.data_REC = data['REC'].item()

        self.data_MPC = None
        WITH_MPC = 'MPC' in data.keys()
        if WITH_MPC:
            self.data_MPC = data['MPC'].item()

        self.d = self.data_SAM['d']
        self.N = self.data_SAM['N']
        self.param_regulization = self.data_SAM['regularizationValue']

        # extract solver information
        self.N_var = self.data_SAM['solver_stats']['N_var']
        self.N_eq = self.data_SAM['solver_stats']['N_eq']
        self.t_wall = self.data_SAM['solver_stats']['t_wall']['optimization']
        self.iterations = self.data_SAM['solver_stats']['N_iter']['optimization']
        self.t_iter = self.t_wall/self.iterations

        # extract power optimality
        self.J_SAM = self.data_SAM['x']['e'][0][-1]/self.data_SAM['time'][-1]
        self.J_REC = self.data_REC['x']['e'][0][-1]/self.data_REC['time'][-1]
        if self.data_MPC is not None:
            self.J_MPC = self.data_MPC['x']['e'][0][-1]/self.data_MPC['time'][-1]

class DefaultExperimentInfo:
    def __init__(self, filepath: str):

        print(f'Extracting Experiment Information from: {filepath}')

        data = np.load(filepath, allow_pickle=True)

        self.data_DEFAULT = data['DEFAULT'].item()
        self.data_MPC = None
        WITH_MPC = 'MPC' in data.keys()
        if WITH_MPC:
            self.data_MPC = data['MPC'].item()

        self.N = self.data_DEFAULT['N']

        # extract solver information
        self.N_var = self.data_DEFAULT['solver_stats']['N_var']
        self.N_eq = self.data_DEFAULT['solver_stats']['N_eq']
        self.t_wall = self.data_DEFAULT['solver_stats']['t_wall']['optimization']
        self.iterations = self.data_DEFAULT['solver_stats']['N_iter']['optimization']
        self.t_iter = self.t_wall/self.iterations

        # extract power optimality
        self.J_DEFAULT = self.data_DEFAULT['x']['e'][0][-1]/self.data_DEFAULT['time'][-1]
        if self.data_MPC is not None:
            self.J_MPC = self.data_MPC['x']['e'][0][-1]/self.data_MPC['time'][-1]

# %% Load series of experiements:
import os

all_experiments = []
for file in os.listdir('_export/toPlot'):
    if file.endswith('.npz'):
        all_experiments.append(ExperimentInfo(f'_export/toPlot/{file}'))

default_experiments = []
for file in os.listdir('_export/toPlot_default'):
    if file.endswith('.npz'):
        default_experiments.append(DefaultExperimentInfo(f'_export/toPlot_default/{file}'))


# group the experiments by d
experiments_by_d = {}
for d in [3,4,5,6]:
    d_exp = [exp for exp in all_experiments if exp.d == d]

    # sort the list of experiments by N
    d_exp.sort(key=lambda x: x.N)

    experiments_by_d[d] = d_exp

# sort the default list by N
default_experiments.sort(key=lambda x: x.N)


# %% Print a comparison:
# on t_iter, N, N_var, iterations
exp_comp_1 = experiments_by_d[6][1]
exp_comp_2 = default_experiments[-2]

print(f'Comparison of d=6 and full problem N=6')
print(f'N: {exp_comp_1.N} vs. {exp_comp_2.N}')
print(f't_iter: {exp_comp_1.t_iter} vs. {exp_comp_2.t_iter}')
print(f'N_var: {exp_comp_1.N_var} vs. {exp_comp_2.N_var}')
print(f'iterations: {exp_comp_1.iterations} vs. {exp_comp_2.iterations}')
print(f't_wall: {exp_comp_1.t_wall} vs. {exp_comp_2.t_wall}')


# %% Plot Series

# fig, axes = plt.subplot_mosaic("AA;AA;AA;CC;CC", figsize=(4.5, 3.5))
# fig, axes = plt.subplot_mosaic("AA;AA;AA;AA;BB;BB;CC;CC;DD;DD", figsize=(5.5, 5.5))
fig, axes = plt.subplot_mosaic("AA;AA;AA;BB;BB;CC;CC", figsize=(4.5,4))
plt.sca(axes['A'])


# plot default results
J_DEF_list = np.array([exp.J_DEFAULT for exp in default_experiments])
N_DEF_list = np.array([exp.N for exp in default_experiments])
J_DEF_MPC_list = np.array([exp.J_MPC for exp in default_experiments])

# for N, J_def, J_mpc in zip(N_DEF_list, J_DEF_list, J_DEF_MPC_list):
#     plt.plot([N, N], [J_def / 1000, J_mpc / 1000], f'C2.-', alpha=0.3)
# plt.plot([],[],f'C2.',label=f'Full Problem', alpha=0.3)
plt.plot(N_DEF_list, J_DEF_list/1000, f'r^-', markersize=3, label=f'Full Problem')

plt.sca(axes['B'])
error = (J_DEF_list - J_DEF_MPC_list)/J_DEF_MPC_list
plt.plot(N_DEF_list, error*100, f'r^-', markersize=3, label=f'Full Problem')


# plot sam results
ds_to_plot = [4,5,6]
for index,d in enumerate(ds_to_plot):
    experiments = experiments_by_d[d]
    J_SAM_list = np.array([exp.J_SAM for exp in experiments])
    J_REC_list = np.array([exp.J_REC for exp in experiments])
    J_MPC_list = np.array([exp.J_MPC for exp in experiments])
    N_list = np.array([exp.N for exp in experiments])

    plt.sca(axes['A'])
    # for N, J_sam, J_mpc in zip(N_list, J_SAM_list, J_MPC_list):
    #     plt.plot([N, N], [J_sam/1000, J_mpc/1000], f'C{index}.-', alpha=0.3)
    plt.plot([],[],f'C{index}.-',label=f'd={d}', alpha=1)
    plt.plot(N_list, J_SAM_list/1000, f'C{index}.-', alpha=1)
    # plot with a star marker
    # plt.plot(N_list, J_MPC_list/1000,f'C{index}*-', label=f'SAM, Sim.')

    plt.sca(axes['B'])
    error = (J_SAM_list - J_MPC_list)/J_MPC_list
    plt.plot(N_list, error*100, f'C{index}.-', alpha=1)
    plt.plot([],[],f'C{index}.-',label=f'd={d}', alpha=1)





plt.sca(axes['A'])
# plt.ylim([0, np.max(J_SAM_list/1000)*1.1])
plt.xticks(np.arange(-20,50,5))
plt.xlim([0, np.max(N_list)*1.05])
# plt.xlabel('N')
plt.ylabel('P [kW]')
plt.grid(alpha=0.25)
plt.legend(ncol=2,loc='upper right')

plt.sca(axes['B'])
plt.ylim([0, 14])
plt.xticks(np.arange(-20,50,5))
plt.xlim([0, np.max(N_list)*1.05])
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.ylabel('Rel. Error in \%')
plt.grid(alpha=0.25)
plt.legend(ncol=len(ds_to_plot)+1)



plt.sca(axes['C'])

# plot default results
N_DEF_list = np.array([exp.N for exp in default_experiments])
# t_wall_DEF_list = np.array([exp.t_wall for exp in default_experiments])
# plt.plot(N_DEF_list, t_wall_DEF_list, f'r^-', markersize=3,label=f'Full Problem')
t_iter_DEF_list = np.array([exp.t_iter for exp in default_experiments])
plt.plot(N_DEF_list, t_iter_DEF_list, f'r^-', markersize=3,label=f'Full Problem')


for index,d in enumerate(ds_to_plot):
    experiments = experiments_by_d[d]
    N_list = np.array([exp.N for exp in experiments])
    N_eq_list = np.array([exp.N_eq for exp in experiments])
    t_iter_list = np.array([exp.t_iter for exp in experiments])
    plt.plot(N_list, t_iter_list, f'C{index}.-', label=f'd={d}')
    #
    # t_wall_list = np.array([exp.t_wall for exp in experiments])
    # plt.plot(N_list, t_wall_list, f'C{index}.-', label=f'd={d}')


plt.xticks(np.arange(-20,50,5))
# plt.ylim([0, np.max(t_iter_list)*1.7])
plt.xlim([0, np.max(N_list)*1.05])
plt.yscale('log')

# plt.xlabel('N')
plt.ylabel('$t_\mathrm{iter}$ [s]')
# plt.ylabel('$t_\mathrm{wall}$ [s]')
plt.grid(alpha=0.25)
plt.legend(ncol=2,loc='upper right')

# third subplot: Number of variables

# plt.sca(axes['D'])

# plot default results
# N_DEF_list = np.array([exp.N for exp in default_experiments])
# N_var_def = np.array([exp.N_var for exp in default_experiments])
# plt.plot(N_DEF_list, N_var_def/1000, f'r^-', markersize=3, label=f'Full Problem')
#
#
# for index,d in enumerate(ds_to_plot):
#     experiments = experiments_by_d[d]
#     N_list = np.array([exp.N for exp in experiments])
#     N_var_list = np.array([exp.N_var for exp in experiments])
#     plt.plot(N_list, N_var_list/1000, f'C{index}.-', label=f'd={d}')
#
# plt.xticks(np.arange(-20,50,5))
# plt.ylim([0, np.max(N_var_list/1000)*1.05])
# plt.xlim([0, np.max(N_list)*1.05])
#
# plt.ylabel('No. of Vars. [k]')


plt.xlabel(r'Number of Subcycles $N$')
# plt.legend(ncol=len(ds_to_plot)+1,loc = 'lower right')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('figures/experiment_series.pdf')
plt.show()

