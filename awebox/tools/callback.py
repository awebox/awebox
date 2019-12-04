#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2019 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2019 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
import casadi.tools as cas
import numpy as np

import matplotlib.pyplot as plt
import awebox.tools.struct_operations as struct_op

class awebox_callback(cas.Callback):
    def __init__(self, name, model, nlp, options, V, P, nx, ng, np, opts={}):

        if options['callback']:
            cas.Callback.__init__(self)

            self.nx = nx
            self.ng = ng
            self.np = np

            self.V_callback = V
            self.p_fix_num = P
            self.model = model
            self.nlp = nlp

            plt.figure(1)
            plt.subplot(111)

            self.x_sols = []
            self.y_sols = []

            # Initialize internal objects
            self.construct(name, opts)

    def get_n_in(self): return cas.nlpsol_n_out()
    def get_n_out(self): return 1
    def get_name_in(self, i): return cas.nlpsol_out(i)
    def get_name_out(self, i): return "ret"

    def get_sparsity_in(self, i):
        n = cas.nlpsol_out(i)
        if n=='f':
            return cas.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return cas.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return cas.Sparsity.dense(self.ng)
        elif n in ('p', 'lam_p'):
            return cas.Sparsity.dense(self.np)
        else:
            return cas.Sparsity(0,0)

    def eval(self, arg):
        print('TEST')
        darg = {}
        for (i,s) in enumerate(cas.nlpsol_out()): darg[s] = arg[i]
        sol = darg['x']
        V = self.V_callback(sol)
        model = self.model
        #plot_trajectory(V)
        plot_states(V, model)
        plot_controls(V, model)
        plot_algebraic_vars(V,model)
        plot_invariants(V,self.p_fix_num(0.),self.nlp)
        plt.show(block=True)
        # time.sleep(1)
        # plt.close('all')

        return [0]

def plot_trajectory(V):

    xvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q21', 0]).full().flatten()
    yvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q21', 1]).full().flatten()
    zvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q21', 2]).full().flatten()
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xvals, yvals, zs=zvals)
    xvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q31', 0]).full().flatten()
    yvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q31', 1]).full().flatten()
    zvals = struct_op.coll_slice_to_vec(V['xd', :, :, 'q31', 2]).full().flatten()
    ax.plot(xvals, yvals, zs=zvals)

    return None

def plot_states(V, model):
    counter = 0
    plt.figure(1)
    for state in struct_op.subkeys(model.variables,'xd'):
        counter += 1
        plt.subplot(4,4,counter)
        for state_dim in range(V['xd',0,0,state].shape[0]):
            plt.plot(cas.vertcat(*np.array(V['xd',:,:,state,state_dim])))
            plt.title(state)

    plt.subplots_adjust(wspace=0.3, hspace=0.6)

def plot_algebraic_vars(V, model):
    counter = 0
    plt.figure(4)
    for state in struct_op.subkeys(model.variables,'xa'):
        counter += 1
        plt.subplot(2,2,counter)
        for state_dim in range(V['xa',0,0,state].shape[0]):
            plt.plot(cas.vertcat(*np.array(V['xa',:,:,state,state_dim])))
            plt.title(state)

    plt.subplots_adjust(wspace=0.3, hspace=0.6)

def plot_invariants(V, P, nlp):

    [nlp_outputs, nlp_output_fun] = nlp.output_components

    outputs_opt = nlp_outputs(nlp_output_fun(V,P))
    first_tether_indeces = np.array(outputs_opt.f['outputs', 0, 0, 'tether_length'])
    number_of_constraints = first_tether_indeces.shape[0]

    number_tethers = int(np.ceil(np.float(number_of_constraints) / 2))

    plt.figure(3).clear()

    fig, axes = plt.subplots(nrows=number_of_constraints, ncols=1, sharex='all', num=3)

    for idx in range(number_of_constraints):

        cstr_name = outputs_opt.getCanonicalIndex(first_tether_indeces[idx])[-2]
        cstr_vec = np.abs(np.array(struct_op.coll_slice_to_vec(outputs_opt['outputs', :, :, 'tether_length', cstr_name])))

        # axes[idx].semilogy(tgrid_xa, cstr_vec)
        axes[idx].plot(cstr_vec)
        axes[idx].set_ylabel(cstr_name)

        if idx == 0:
            axes[idx].set_title('tether length constraints')
    return None

def plot_controls(V, model):
    counter = 0
    plt.figure(2)
    for state in struct_op.subkeys(model.variables,'u'):
        counter += 1
        plt.subplot(4,3,counter)
        for state_dim in range(V['u',0,state].shape[0]):
            plt.plot(cas.vertcat(*np.array(V['u',:,state,state_dim])))
            plt.title(state)

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
