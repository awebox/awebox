#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
import pickle
import collections
import awebox.tools.struct_operations as struct_op
import copy

class awebox_callback(cas.Callback):
    def __init__(self, name, model, nlp, options, V, P, nx, ng, np, record_states = True, opts={}):

        cas.Callback.__init__(self)

        self.nx = nx
        self.ng = ng
        self.np = np

        self.V = V
        self.P = P
        self.model = model
        self.nlp = nlp
        [self.Out, self.Out_fun] = nlp.output_components

        self.record_states = record_states
        self.options = options
        self.__init_dicts()

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
        darg = {}
        for (i,s) in enumerate(cas.nlpsol_out()): darg[s] = arg[i]
        sol = darg['x']
        lam_x = darg['lam_x']
        lam_g = darg['lam_g']

        # retrieve and reverse-scale solution
        V_opt = self.V(sol)
        V_si = struct_op.scaled_to_si(V_opt, self.model.scaling)
        P = self.P

        for phi in list(self.phi_dict.keys()):
          self.phi_dict[phi].append(V_opt['phi', phi])
        
        if self.record_states:

            # interpolate solution
            n_points = self.options['n_points_callback']
            time_grids = {}
            for grid in self.nlp.time_grids:
                time_grids[grid] = self.nlp.time_grids[grid](V_opt['theta', 't_f'])

            collocation_interpolator = self.nlp.Collocation.build_interpolator(self.nlp.options, V_si)
            time_grids['ip'] = struct_op.build_time_grid_for_interpolation(time_grids, n_points)
            V_interpolated, V_vector_series_interpolated =  struct_op.interpolate_V(
                time_grids, self.model.variables_dict, 'zoh', V_si,
                collocation_interpolator=collocation_interpolator,
                timegrid_label='ip')
            outputs = struct_op.interpolate_outputs(
                V_vector_series_interpolated, V_si, P,
                self.model.variables_dict, self.model.parameters, self.model.scaling.cat,
                self.model.outputs_fun, self.model.outputs_dict
                )

            # store interpolated solution in dicts
            for x in list(self.model.variables_dict['x'].keys()):
                for dim in range(self.model.variables_dict['x'][x].shape[0]):
                    self.x_dict[x+'_'+str(dim)].append(V_interpolated['x'][x][dim])
                
            for u in list(self.model.variables_dict['u'].keys()):
                for dim in range(self.model.variables_dict['u'][u].shape[0]):
                    self.u_dict[u+'_'+str(dim)].append(V_interpolated['u'][u][dim])

            for z in list(self.model.variables_dict['z'].keys()):
                for dim in range(self.model.variables_dict['z'][z].shape[0]):
                    self.z_dict[z+'_'+str(dim)].append(V_interpolated['z'][z][dim])

            for theta in list(self.model.variables_dict['theta'].keys()):
                for dim in range(self.model.variables_dict['theta'][theta].shape[0]):
                    self.theta_dict[theta+'_'+str(dim)].append(V_si['theta', theta, dim])

            for t in list(self.t_dict.keys()):
                self.t_dict[t].append(self.nlp.time_grids[t](V_opt['theta', 't_f']))

            # record power output
            energy = self.nlp.integral_output_components[1](V_opt, P)
            self.avg_power.append(energy[-1]/self.t_dict['x'][-1][-1])
            self.p_current.append(outputs['performance']['p_current'][0])

            # record multipliers and constraints
            for x in list(self.model.variables_dict['x'].keys()):

                for dim in range(self.model.variables_dict['x'][x].shape[0]):
                    self.lam_x_dict[x+'_'+str(dim)].append(self.extract_x_vals(self.V(lam_x), x, dim))

                self.lam_g += [lam_g]
                self.g_dict += [self.nlp.g_fun(V_opt,P)]

        # record cost function values
        for cost in list(self.cost_dict.keys()):
            self.cost_dict[cost].append(self.nlp.cost_components[0][cost+'_fun'](V_opt,P))

        return [0]

    def extract_x_vals(self, V, name, dim):
      x_vals = []
      for k in range(self.nlp.n_k+1):
          # add interval values
          x_vals.append(V['x',k,name,dim])
          if k < self.nlp.n_k:
            # add node values
            x_vals += V['coll_var',k, :, 'x', name,dim]
      return x_vals

    def __init_dicts(self):
      
      phi_dict = collections.OrderedDict()
      for phi in self.model.parameters_dict['phi'].keys():
        phi_dict[phi] = []
      
      x_dict = collections.OrderedDict()
      for x in self.model.variables_dict['x'].keys():
        for dim in range(self.model.variables_dict['x'][x].shape[0]):
          x_dict[x+'_'+str(dim)] = []

      u_dict = collections.OrderedDict()
      for u in self.model.variables_dict['u'].keys():
        for dim in range(self.model.variables_dict['u'][u].shape[0]):
          u_dict[u+'_'+str(dim)] = []

      z_dict = collections.OrderedDict()
      for z in self.model.variables_dict['z'].keys():
        for dim in range(self.model.variables_dict['z'][z].shape[0]):
          z_dict[z+'_'+str(dim)] = []

      theta_dict = collections.OrderedDict()
      for th in self.model.variables_dict['theta'].keys():
        for dim in range(self.model.variables_dict['theta'][th].shape[0]):
          theta_dict[th+'_'+str(dim)] = []

      t_dict = collections.OrderedDict()
      for t in self.nlp.time_grids.keys():
        t_dict[t] = []

      cost_dict = collections.OrderedDict()
      for cost in self.nlp.cost_components[1].keys():
          cost_dict[cost] = []

      lam_x_dict = copy.deepcopy(x_dict)

      self.phi_dict = phi_dict
      self.x_dict = x_dict
      self.u_dict = u_dict
      self.z_dict = z_dict
      self.theta_dict = theta_dict
      self.cost_dict = cost_dict
      self.t_dict = t_dict
      self.lam_x_dict = lam_x_dict
      self.g_dict = []
      self.lam_g = []
      self.avg_power = []
      self.p_current = []

      return None

    def reset(self):
      self.__init_dicts()
      return None

    def update_P(self, P):
      self.P = P
      return None