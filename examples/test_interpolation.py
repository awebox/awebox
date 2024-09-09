import csv
import numpy as np

with open('Ampyx_AP2.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = list(reader)
        key_list = lines[0]

# Retrieve data
with open('Ampyx_AP2.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = list(reader)
    traj = {}
    for key in key_list:
        tmp = np.array([row[key] for row in lines])
        tmp[tmp == ''] = '0'
        x = tmp.astype(np.float64)
        traj[key] = x

with open('ampyx_ap2_trajectory.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lines = list(reader)
        key_list = lines[0]

# Retrieve data
with open('ampyx_ap2_trajectory.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    lines = list(reader)
    traj_old = {}
    for key in key_list:
        tmp = np.array([row[key] for row in lines])
        tmp[tmp == ''] = '0'
        x = tmp.astype(np.float64)
        traj_old[key] = x

time = traj['time']
time_old = traj_old['time']

import matplotlib.pyplot as plt
test = [traj['outputs_xdot_from_var_dl_t_0'][k] / traj['x_dl_t_0'][k] for k in range(len(traj['outputs_xdot_from_var_dl_t_0']))]
plt.plot(time, test)
plt.plot(time, traj['outputs_xdot_from_var_dl_t_0'], label = 'Tether Force Magnitude (new)')
plt.plot(time, traj['x_dl_t_0'], label = 'Tether Force Magnitude (old)')
plt.ylabel('[N]')
plt.xlabel('t [s]')
plt.legend()
# plt.hlines([50, 1800], time[0], time[-1], linestyle='--', color = 'black')
plt.grid(True)
plt.show()