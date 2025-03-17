import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from awebox.ocp.collocation import Collocation


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
# filepath = '_export/AWE_SAM_N10_d4_nicePlotting.npz'
# filepath = '_export/toPlot/20250311_1537_AWE_SAM_N10_d4.npz'
# filepath = '_export/toPlot/20250314_0924_AWE_SAM_N3_d4.npz'
# filepath = '_export/toPlot/20250314_0948_AWE_SAM_N5_d4.npz'
filepath = '_export/toPlot/20250315_1507_AWE_SAM_N4_d4.npz'
# filepath = '_export/20250315_1300_AWE_SAM_N5_d4.npz'
data = np.load(filepath,allow_pickle=True)

data_SAM = data['SAM'].item()
data_REC = data['REC'].item()

WITH_MPC = 'MPC' in data.keys()
if WITH_MPC:
    data_MPC = data['MPC'].item()

d = data_SAM['d']
N = data_SAM['N']


# %% Compute and compare the produced powers
power_SAM = data_SAM['x']['e'][0][-1]/data_SAM['time'][-1]
power_REC = data_REC['x']['e'][0][-1]/data_REC['time'][-1]
if WITH_MPC:
    power_MPC = data_MPC['x']['e'][0][-1]/data_MPC['time'][-1]

print(f'Power produced by SAM: {power_SAM/1000} kW')
print(f'Power produced by REC: {power_REC/1000} kW')
if WITH_MPC:
    print(f'Power produced by MPC: {power_MPC/1000} kW')


# %% draw kite function
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


def drawPlane(pos, rot, wingspan, color='C0', alpha=1, twoDimensions=False):
    rot = np.reshape(rot, (3, 3)).T@np.diag([-1,1,1])

    normalization = 32.0 / (float(wingspan)) # size of triangles is 32, normalize with half of the wingspan
    y_offset = -16
    x_offset = -3.5 / 2
    z_offset = 0

    x = {} # dictionary of x positions of the vertices
    y = {} # dictionary of y positions of the vertices
    z = {} # dictionary of z positions of the vertices

    x['fuselage'] = [0, 3.5, 8, 9, 9.5, 9.75, 9.5, 9, 8, 3.5, 0, 2, -6, -8, -9, -8, -6, -2]
    y['fuselage'] = [15, 15, 15, 15.25, 15.75, 16, 16.25, 16.75, 17, 17, 17, 17, 16.75, 16.5, 16, 15.5, 15.25, 15]
    z['fuselage'] = [-0.01] * 18

    x['wing'] = [0, 2, 3, 3.5, 3.5, 3, 2, 0]
    y['wing'] = [0, 0, 8, 13, 19, 24, 32, 32]
    z['wing'] = [0, 0, 0, 0, 0, 0, 0, 0]

    x['elev'] = [-8, -8, -9, -9]
    y['elev'] = [13.5, 18.5, 18.5, 13.5]
    z['elev'] = [-0.01] * 4



    for part in x.keys():
        x[part] = [(xx + x_offset) / normalization for xx in x[part]]
        y[part] = [(yy + y_offset) / normalization for yy in y[part]]
        z[part] = [(zz + z_offset) / normalization for zz in z[part]]

        # for kk in range(len(x[part])):
        #     vec = np.array([x[part][kk], y[part][kk], z[part][kk]])
        #     print(f'Vec Shape: {vec.shape}')
        #     vec_rot = np.matmul(rot, vec)
        #     print(f'Vec_rot Shape: {vec_rot.shape}')
        #
        #     x[part][kk] = vec_rot[0] + posKite[0]
        #     y[part][kk] = vec_rot[1] + posKite[1]
        #     z[part][kk] = vec_rot[2] + posKite[2]
        # verts = np.array([list(zip(x[part], y[part], z[part]))])


        verts = np.vstack([x[part],y[part],z[part]])

        # print(f"verts:{verts.shape}")
        # print(f'rot:{rot.shape}')

        verts_rot = rot@verts + pos
        # verts_rot = verts + posKite
        # print(verts_rot.shape)
        if part != 'wing':
            zorder = -1
        else:
            zorder = 0

        if not twoDimensions:
            # plot in three dimensions
            tri = a3.art3d.Poly3DCollection([verts_rot.T],linewidth=0.1, zorder=zorder)
            tri.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
            tri.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))

            # bbox.set_alpha(alpha)
            plt.gca().add_collection3d(tri)
        else:
            #plot in two dimensions
            polygon = Polygon(verts_rot[1:3,:].T, facecolor=color)
            polygon.set_color(matplotlib.colors.to_rgba(color, alpha - 0.1))
            polygon.set_edgecolor(matplotlib.colors.to_rgba(color, alpha))
            plt.gca().add_patch(polygon)

# %% 3D PLOT
import mpl_toolkits.mplot3d as a3
import matplotlib

q10_REC = data_REC['x']['q10']

q10_opt = data_SAM['x']['q10']
ip_regions_SAM = data_SAM['regions']

if WITH_MPC:
    q10_MPC = data_MPC['x']['q10']

Q10_SAM = data_SAM['X']['q10']
time_X = data_SAM['time_X']

for figure_type in ['SAM', 'REC', 'MPC']:
    plt.figure(figsize=(5.5, 4.5))
    ax = plt.axes(projection='3d')

    if figure_type == 'SAM':

        # reel in
        ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == d)],
                  q10_opt[1][np.where(ip_regions_SAM == d)],
                  q10_opt[2][np.where(ip_regions_SAM == d)]
                  , '-', color='C0',
                  alpha=1, markersize=3)

        # average
        ax.plot3D(Q10_SAM[0], Q10_SAM[1], Q10_SAM[2], 'C1-', alpha=1)
        ax.plot3D(Q10_SAM[0][0], Q10_SAM[1][0], Q10_SAM[2][0], 'C1.', alpha=1)
        ax.plot3D(Q10_SAM[0][-1], Q10_SAM[1][-1], Q10_SAM[2][-1], 'C1.', alpha=1)

        for region_index in np.arange(0, data_SAM['d'] + 1):
            color = 'C0' if region_index == data_SAM['d'] else 'C2'

            ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == region_index)],
                      q10_opt[1][np.where(ip_regions_SAM == region_index)],
                      q10_opt[2][np.where(ip_regions_SAM == region_index)]
                      , '-', color=color,
                          alpha=1, markersize=3)

    if figure_type == 'REC':
        ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C0-', alpha=1)

    if figure_type == 'MPC':
        ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C0-', alpha=0.2)

        final_index = q10_MPC[0].size//4 + 20
        section_to_plot = slice(0,final_index )
        ax.plot3D(q10_MPC[0][section_to_plot], q10_MPC[1][section_to_plot], q10_MPC[2][section_to_plot], 'r-', alpha=0.75)

        # section mpc:
        section_to_plot_mpc = slice(final_index, final_index + 30)
        ax.plot3D(q10_MPC[0][section_to_plot_mpc], q10_MPC[1][section_to_plot_mpc], q10_MPC[2][section_to_plot_mpc], 'r--', alpha=0.75)

        # plot a kite at the end of the section
        r10 = np.vstack([data_MPC['x']['r10'][i][final_index] for i in range(data_MPC['x']['r10'].__len__())])
        q10 = np.vstack([data_MPC['x']['q10'][i][final_index] for i in range(data_MPC['x']['q10'].__len__())])
        # drawKite(q10,
                 # r10, 30, color='k', alpha=1)
        drawPlane(q10,r10,wingspan=50,color='k')
        # draw a straight tether to the origin
        ax.plot3D([0, float(q10[0])], [0, float(q10[1])], [0, float(q10[2])], 'k-', alpha=0.5,linewidth=1)


    # set bounds for nice view
    q10_REC_all = np.vstack([q10_REC[0],q10_REC[1],q10_REC[2]])
    meanpos = np.mean(q10_REC_all, axis=1) + np.array([0, -50, 30])

    bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))/2.3

    # ticks on the axis in 100m steps
    ax.set_xticks(np.arange(-1000, 1000, 100))
    ax.set_yticks(np.arange(-1000, 1000, 100))
    ax.set_zticks(np.arange(-1000, 1000, 100))

    ax.set_xlim3d(meanpos[0] - bblenght, meanpos[0] + bblenght)
    ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
    ax.set_zlim3d(meanpos[2] - bblenght, meanpos[2] + bblenght)
    ax.set_box_aspect([1, 1, 1])

    pos_wind_arrow = np.array([meanpos[0] - bblenght / 2, meanpos[1] - bblenght, meanpos[2] + bblenght*0.7])
    ax.quiver(pos_wind_arrow[0],pos_wind_arrow[1],pos_wind_arrow[2], 1, 0, 0, length=100, color='k')
    ax.text(pos_wind_arrow[0],pos_wind_arrow[1],pos_wind_arrow[2], "Wind", 'x', color='k', size=12)

    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_zlabel(r'$z$ in m')

    # ax.legend()

    # plt.axis('off')
    ax.view_init(elev=3., azim=142)

    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/3DReelout_{figure_type}.pdf')
    plt.show()


# %% 3D INTRO PLOT: SAM
import mpl_toolkits.mplot3d as a3
import matplotlib

q10_REC = data_REC['x']['q10']
q10_opt = data_SAM['x']['q10']
ip_regions_SAM = data_SAM['regions']

if WITH_MPC:
    q10_MPC = data_MPC['x']['q10']

Q10_SAM = data_SAM['X']['q10']
time_X = data_SAM['time_X']

plt.figure(figsize=(9, 5.5))
ax = plt.axes(projection='3d')


# reel in
ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == d)],
          q10_opt[1][np.where(ip_regions_SAM == d)],
          q10_opt[2][np.where(ip_regions_SAM == d)]
          , '-', color='C0',
          alpha=1, markersize=3)

# average
ax.plot3D(Q10_SAM[0], Q10_SAM[1], Q10_SAM[2], 'C1-', alpha=1)
ax.plot3D(Q10_SAM[0][0], Q10_SAM[1][0], Q10_SAM[2][0], 'C1.', alpha=1)
ax.plot3D(Q10_SAM[0][-1], Q10_SAM[1][-1], Q10_SAM[2][-1], 'C1.', alpha=1)

for region_index in [1,2,d]:
    color = 'C0' if region_index == data_SAM['d'] else 'C0'

    ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == region_index)],
              q10_opt[1][np.where(ip_regions_SAM == region_index)],
              q10_opt[2][np.where(ip_regions_SAM == region_index)]
              , '-', color=color,
                  alpha=1, markersize=3)
    ax.plot3D(q10_opt[0][np.where(ip_regions_SAM == region_index)][[0,-1]],
              q10_opt[1][np.where(ip_regions_SAM == region_index)][[0,-1]],
              q10_opt[2][np.where(ip_regions_SAM == region_index)][[0,-1]], '.', color=color,
                  alpha=1, markersize=3)

ax.plot3D(q10_REC[0], q10_REC[1], q10_REC[2], 'C0--', alpha=0.3)

# final_index = q10_MPC[0].size//10
final_index = 20
section_to_plot = slice(0,final_index )
# ax.plot3D(q10_MPC[0][section_to_plot], q10_MPC[1][section_to_plot], q10_MPC[2][section_to_plot], 'r-', alpha=1)

# section mpc:
section_to_plot_mpc = slice(final_index, final_index + 30)
# ax.plot3D(q10_MPC[0][section_to_plot_mpc], q10_MPC[1][section_to_plot_mpc], q10_MPC[2][section_to_plot_mpc], 'r--', alpha=1)

# plot a kite at the end of the section
r10 = np.vstack([data_MPC['x']['r10'][i][final_index] for i in range(data_MPC['x']['r10'].__len__())])
q10 = np.vstack([data_MPC['x']['q10'][i][final_index] for i in range(data_MPC['x']['q10'].__len__())])
drawKite(q10,
         r10, 40, color='k', alpha=1)
# draw a straight tether to the origin
ax.plot3D([0, float(q10[0])], [0, float(q10[1])], [0, float(q10[2])], 'k-', alpha=0.5,linewidth=1)


# set bounds for nice view
q10_REC_all = np.vstack([q10_REC[0],q10_REC[1],q10_REC[2]])
meanpos = np.mean(q10_REC_all, axis=1) + np.array([0, 30, 30])

bblenght = np.max(np.abs(q10_REC_all - meanpos.reshape(3, 1)))/1.8

# plot a ground station at the origin
ax.plot3D(0,0,0,'ks',markersize=4)

# ticks on the axis in 100m steps
ax.set_xticks(np.arange(-1000, 1000, 100))
ax.set_yticks(np.arange(-1000, 1000, 100))
ax.set_zticks(np.arange(-1000, 1000, 100))

ax.set_xlim3d(0,2*bblenght)
ax.set_ylim3d(meanpos[1] - bblenght, meanpos[1] + bblenght)
ax.set_zlim3d(0, 2*bblenght)
ax.set_box_aspect([1, 1, 1])

pos_wind_arrow = np.array([meanpos[0] - 1.3*bblenght, meanpos[1] - bblenght, meanpos[2] + bblenght*0.4])
ax.quiver(pos_wind_arrow[0],pos_wind_arrow[1],pos_wind_arrow[2], 1, 0, 0, length=100, color='k')
ax.text(pos_wind_arrow[0]+50,pos_wind_arrow[1],pos_wind_arrow[2], "Wind", 'x', color='k', size=12)

ax.set_xlabel(r'$x$ in m')
ax.set_ylabel(r'$y$ in m')
ax.set_zlabel(r'$z$ in m')

# ax.legend()

# plt.axis('off')
ax.view_init(elev=14., azim=131)

# plt.legend()
plt.tight_layout()
plt.savefig(f'figures/3DReelout_Introduction.pdf')
plt.show()

# # %% Plot the states
# plt.figure(figsize=(10, 10))
# plot_states = ['q10', 'dq10', 'l_t', 'dl_t','e']
# for index, state_name in enumerate(plot_states):
#     plt.subplot(3, 2, index + 1)
#     state_traj = np.vstack([data_SAM['x'][state_name][i] for i in range(data_SAM['x'][state_name].__len__())]).T
#
#     for region_index in range(d+1):
#         plt.plot(data_SAM['time'][np.where(ip_regions_SAM == region_index)],
#                     state_traj[np.where(ip_regions_SAM == region_index)],  '-')
#         plt.gca().set_prop_cycle(None)  # reset color cycle
#
#     plt.plot([], [], label=state_name)
#
#     plt.gca().set_prop_cycle(None)  # reset color cycle
#
#     state_recon = np.vstack([data_REC['x'][state_name][i] for i in range(data_REC['x'][state_name].__len__())]).T
#     plt.plot(data_REC['time'], state_recon, label=state_name + '_recon', linestyle='--')
#
#     state_mpc = np.vstack([data_MPC['x'][state_name][i] for i in range(data_MPC['x'][state_name].__len__())]).T
#     plt.plot(data_MPC['time'], state_mpc, label=state_name + '_MPC', linestyle='dotted')
#
#     # add phase switches
#     # for region in regions_indeces:
#     #     plt.axvline(x=time_grid_SAM_x[region[0]],color='k',linestyle='--')
#     #     plt.axvline(x=time_grid_SAM_x[region[-1]]+(time_grid_SAM_x[region[-1]]-time_grid_SAM_x[region[-2]]),color='k',linestyle='--')
#     # plt.axvline(x=time_grid_SAM_x[regions_indeces[-1][-1]],color='k',linestyle='--')
#
#     #
#     # for region_indeces in regions_indeces[1:-1]:
#     #     plt.axvline(x=time_grid_SAM_x[region_indeces[0]],color='b',linestyle='--')
#
#     plt.xlabel('time [s]')
#
#     plt.legend()
# plt.tight_layout()
# plt.show()

# %% plot only a subset of the states for paper

# get the approximate collocation points
import casadi as ca
coll_points_tau = np.array(ca.collocation_points(d,'legendre'))
coll_points_t = coll_points_tau*data_SAM['time_X'][-1]
# find the index in the time grid that is closest to the collocation points
coll_points_index = np.array([np.argmin(np.abs(data_SAM['time_X'] - t)) for t in coll_points_t])

# plt.figure(figsize=(4.5, 2.5))
plt.figure(figsize=(10, 2))
plot_states = ['dq10']
for index, state_name in enumerate(plot_states):
    plt.subplot(1, 1, index + 1)
    state_traj = np.vstack([data_SAM['x'][state_name][i] for i in range(data_SAM['x'][state_name].__len__())]).T
    state_recon = np.vstack([data_REC['x'][state_name][i] for i in range(data_REC['x'][state_name].__len__())]).T
    state_X = np.vstack([data_SAM['X'][state_name][i] for i in range(data_SAM['X'][state_name].__len__())]).T

    # phase switch
    t_switch = time_X[-1]*N
    t_end = data_SAM['time'][-1]
    plt.axvline(x=0,color='k',linestyle='-',alpha=0.5)
    plt.axvline(x=t_switch,color='k',linestyle='-',alpha=0.5)
    plt.axvline(x=t_end,color='k',linestyle='-',alpha=0.5)

    if state_name == 'dq10':

        state_traj = data_SAM['x'][state_name][0]
        state_X = data_SAM['X'][state_name][0]
        state_recon = data_REC['x'][state_name][0]

    # plot the average state poly
    plt.plot(time_X*N, state_X, 'C1-')
    plt.plot(time_X[-1]*N, state_X[-1], 'C1.')
    plt.plot(time_X[0]*N, state_X[0], 'C1.')
    plt.plot(time_X[coll_points_index]*N,state_X[coll_points_index],'C1.')
    plt.plot([],[],'C1-',label=r'Macro-Integration')

    for region_index in range(d+1):
        time_micro = data_SAM['time'][np.where(ip_regions_SAM == region_index)]
        state_micro = state_traj[np.where(ip_regions_SAM == region_index)]
        plt.plot(time_micro,state_micro,'C0-' if region_index == d else 'C2-')



        if region_index < d:
            # point at start and end
            plt.plot(time_micro[0],state_micro[0],'C2.')
            plt.plot(time_micro[-1],state_micro[-1],'C2.')

        plt.gca().set_prop_cycle(None)  # reset color cycle
    plt.plot([], [],'C2.-', label='Micro-Integration')




    plt.gca().set_prop_cycle(None)  # reset color cycle

    plt.plot(data_REC['time'], state_recon, label='Reconstruction', linestyle='--',alpha=0.3)



    plt.xlabel('t [s]')
    plt.ylabel('$\dot{q}_x$ [m/s]')
    plt.grid(alpha=0.25)
    plt.legend(loc='upper right')
    plt.ylim([-30,32])
    # plt.ylim([-22,32])

# fancy annotations
# pos_y = -25
pos_y = -20
plt.gca().annotate(
    'Reel-Out',
    xy=(t_switch/2, pos_y),  # Position of the text (x, y)  # Position above for text
    ha="center",
    va="center",
    fontsize=10,
)
plt.gca().annotate('',xy=(0, pos_y),xytext=(t_switch/3, pos_y),
                   arrowprops=dict(arrowstyle='->', lw=1,color='k',shrinkA=0,shrinkB=0))
plt.gca().annotate('',xy=(t_switch, pos_y),xytext=(2*t_switch/3, pos_y),
                   arrowprops=dict(arrowstyle='->', lw=1,color='k',shrinkA=0,shrinkB=0))

plt.gca().annotate(
    'Reel-In',
    xy=((t_switch+t_end)/2, pos_y),  # Position of the text (x, y)  # Position above for text
    ha="center",
    va="center",
    fontsize=10,
)
delta_end = t_end- t_switch
plt.gca().annotate('',xy=(t_switch, pos_y),xytext=(t_switch + delta_end/3, pos_y),
                   arrowprops=dict(arrowstyle='->', lw=1,color='k',shrinkA=0,shrinkB=0))
plt.gca().annotate('',xy=(t_end, pos_y),xytext=(t_switch + 2*delta_end/3, pos_y),
                   arrowprops=dict(arrowstyle='->', lw=1,color='k',shrinkA=0,shrinkB=0))



plt.tight_layout()
plt.savefig('figures/velocity.pdf')
plt.show()