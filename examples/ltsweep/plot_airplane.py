
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from numpy.core.numeric import _zeros_like_dispatcher

def plot_airplane(ax, q, qparent, R, color, span, alpha = 1.0, plot_tether = True):

    normalization = 32.0/(span)
    y_offset = -16
    x_offset = -3.5/2
    z_offset = 0

    y = OrderedDict()
    y['wing'] = [0,0,8,13,19,24,32,32]
    y['fuselage'] = [15,15,15,15.25,15.75,16,16.25,16.75,17,17,17,17,16.75,16.5,16,15.5,15.25,15]
    y['elev'] = [13.5, 18.5, 18.5, 13.5]

    x = OrderedDict()
    x['fuselage'] = [0,3.5,8,9,9.5,9.75,9.5,9,8,3.5,0,2,-6,-8,-9,-8,-6,-2]
    x['elev'] = [-8, -8, -9, -9]
    x['wing'] = [0,2,3,3.5, 3.5,3,2,0]

    z = OrderedDict()
    z['wing'] = [0,0,0,0,0,0,0,0]
    z['fuselage'] = [-0.01]*18
    z['elev'] = [-0.01]*4

    for part in x.keys():
        x[part] = [-(xx+x_offset)/normalization for xx in x[part]]
        y[part] = [(yy+y_offset)/normalization for yy in y[part]]
        z[part] = [(zz+z_offset)/normalization for zz in z[part]]

        for kk in range(len(x[part])):
            vec = np.array([x[part][kk], y[part][kk], z[part][kk]])
            vec_rot = np.matmul(R, vec)

            x[part][kk] = vec_rot[0] + q[0]
            y[part][kk] = vec_rot[1] + q[1]
            z[part][kk] = vec_rot[2] + q[2]
        verts = [list(zip(x[part],y[part],z[part]))]
        if part != 'wing':
            zorder = -1
        else:
            zorder = 0
        bbox = Poly3DCollection(verts, edgecolor = 'black', alpha = alpha, linewidth = 0.1, zorder = zorder)
        bbox.set_facecolor(color)
        # bbox.set_alpha(alpha)
        ax.add_collection3d(bbox)
        if plot_tether:
            ax.plot([qparent[0], q[0]], [qparent[1], q[1]], [qparent[2], q[2]], alpha = 0.2, color = color, linewidth = 0.1)

def plot_airplane2D(ax, q, qparent, R, color, span):

    normalization = 32.0/(span)
    y_offset = -16
    x_offset = -3.5/2
    z_offset = 0

    x = OrderedDict()
    x['wing'] = [0,2,3,3.5, 3.5,3,2,0]
    x['fuselage'] = [0,3.5,8,9,9.5,9.75,9.5,9,8,3.5,0,2,-6,-8,-9,-8,-6,-2]
    x['elev'] = [-8, -8, -9, -9]

    y = OrderedDict()
    y['wing'] = [0,0,8,13,19,24,32,32]
    y['fuselage'] = [15,15,15,15.25,15.75,16,16.25,16.75,17,17,17,17,16.75,16.5,16,15.5,15.25,15]
    y['elev'] = [13.5, 18.5, 18.5, 13.5]

    z = OrderedDict()
    z['wing'] = [0,0,0,0,0,0,0,0]
    z['fuselage'] = [-0.01]*18
    z['elev'] = [-0.01]*4

    for part in x.keys():
        x[part] = [-(xx+x_offset)/normalization for xx in x[part]]
        y[part] = [(yy+y_offset)/normalization for yy in y[part]]
        z[part] = [(zz+z_offset)/normalization for zz in z[part]]

        for kk in range(len(x[part])):
            vec = np.array([x[part][kk], y[part][kk], z[part][kk]])
            vec_rot = np.matmul(R, vec)

            x[part][kk] = vec_rot[0] + q[0]
            y[part][kk] = vec_rot[1] + q[1]
            z[part][kk] = vec_rot[2] + q[2]
        verts = np.column_stack((y[part],z[part]))
        bbox = plt.Polygon(verts, edgecolor = 'black', linewidth = 0.5, facecolor = color)
        ax.add_patch(bbox)
        ax.plot([qparent[1], q[1]], [qparent[2], q[2]], color = color, linewidth = 0.1)
