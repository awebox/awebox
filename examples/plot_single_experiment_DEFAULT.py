import numpy as np
import matplotlib.pyplot as plt

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

# %% Load Data
filepath = '_export/20250317_1655_DEFAULT_N4.npz'
data = np.load(filepath,allow_pickle=True)

data_DEF = data['DEFAULT'].item()

WITH_MPC = 'MPC' in data.keys()
if WITH_MPC:
    data_MPC = data['MPC'].item()


# %% 3D PLOT
import mpl_toolkits.mplot3d as a3
import matplotlib

_raw_vertices = np.array([[-1.2, 0, -0.4, 0],
                          [0, -1, 0, 1],
                          [0, 0, 0, 0]])
_raw_vertices = _raw_vertices - np.mean(_raw_vertices, axis=1).reshape((3, 1))


def drawKite(pos, rot, wingspan, color='C0', alpha=1):
    rot = np.reshape(rot, (3, 3)).T

    vtx = _raw_vertices * wingspan / 2  # -np.array([[0.5], [0], [0]]) * sizeKite

    vtx = rot @ vtx + pos
    tri = a3.art3d.Poly3DCollection([vtx.T])
    tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
    tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
    # tri.set_alpha(alpha)
    # tri.set_edgealpha(alpha)
    plt.gca().add_collection3d(tri)


q10_DEF = data_DEF['x']['q10']

q10_opt = data_DEF['x']['q10']

if WITH_MPC:
    q10_MPC = data_MPC['x']['q10']


plt.figure(figsize=(5.5, 4.5))
ax = plt.axes(projection='3d')


ax.plot3D(q10_DEF[0], q10_DEF[1], q10_DEF[2], 'C0-', alpha=0.2)

final_index = q10_MPC[0].size//4
section_to_plot = slice(0,final_index )
ax.plot3D(q10_MPC[0][section_to_plot], q10_MPC[1][section_to_plot], q10_MPC[2][section_to_plot], 'C1-', alpha=1)

# plot a kite at the end of the section
r10 = np.vstack([data_MPC['x']['r10'][i][final_index] for i in range(data_MPC['x']['r10'].__len__())])
q10 = np.vstack([data_MPC['x']['q10'][i][final_index] for i in range(data_MPC['x']['q10'].__len__())])
drawKite(q10,
         r10, 30, color='k', alpha=1)
# draw a straight tether to the origin
ax.plot3D([0, float(q10[0])], [0, float(q10[1])], [0, float(q10[2])], 'k-', alpha=0.5,linewidth=1)


# draw a 'ground station' (black block) at the origin
ax.plot3D(0,0,0, 'ks', alpha=1,linewidth=1)



# set bounds for nice view
q10_REC_all = np.vstack([q10_DEF[0],q10_DEF[1],q10_DEF[2]])
meanpos = np.mean(q10_REC_all, axis=1) + np.array([0, -30, 30])

bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))/1.9

# ticks on the axis in 100m steps
ax.set_xticks(np.arange(-1000, 1000, 100))
ax.set_yticks(np.arange(-1000, 1000, 100))
ax.set_zticks(np.arange(-1000, 1000, 100))

ax.set_xlim3d(0,meanpos[0] + bblenght)
ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
ax.set_zlim3d(0, meanpos[2] + bblenght)
ax.set_box_aspect([1,1,1])

pos_wind_arrow = np.array([meanpos[0] - bblenght / 2, meanpos[1] - bblenght, meanpos[2] + bblenght*0.7])
ax.quiver(pos_wind_arrow[0],pos_wind_arrow[1],pos_wind_arrow[2], 1, 0, 0, length=70, color='k')
ax.text(pos_wind_arrow[0],pos_wind_arrow[1],pos_wind_arrow[2], "Wind", 'x', color='k', size=10)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()

# plt.axis('off')
ax.view_init(elev=5., azim=142)

# plt.legend()
plt.tight_layout()
plt.savefig(f'figures/3DReelout_example.pdf')
plt.show()


# %% Plot the states
plt.figure(figsize=(10, 10))
plot_states = ['q10', 'dq10', 'l_t', 'dl_t','e']
for index, state_name in enumerate(plot_states):
    plt.subplot(3, 2, index + 1)
    state_traj = np.vstack([data_DEF['x'][state_name][i] for i in range(data_DEF['x'][state_name].__len__())]).T
    plt.plot(data_DEF['time'],
                    state_traj,  '-')
    plt.gca().set_prop_cycle(None)  # reset color cycle

    plt.plot([], [], label=state_name)

    plt.gca().set_prop_cycle(None)  # reset color cycle

    # add phase switches
    # for region in regions_indeces:
    #     plt.axvline(x=time_grid_SAM_x[region[0]],color='k',linestyle='--')
    #     plt.axvline(x=time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]),color='k',linestyle='--')
    # plt.axvline(x=time_grid_SAM_x[regions_indeces[-1][-1]],color='k',linestyle='--')

    #
    # for region_indeces in regions_indeces[1:-1]:
    #     plt.axvline(x=time_grid_SAM_x[region_indeces[0]],color='b',linestyle='--')

    plt.xlabel('time [s]')

    plt.legend()
plt.tight_layout()
plt.show()