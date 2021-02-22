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

class awebox_callback(cas.Callback):
    def __init__(self, name, model, nlp, options, V, P, nx, ng, np, opts={}):

        cas.Callback.__init__(self)

        self.nx = nx
        self.ng = ng
        self.np = np

        self.V = V
        self.P = P
        self.model = model
        self.nlp = nlp
        [self.Out, self.Out_fun] = nlp.output_components

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
        V = self.V(sol)
        P = self.P

        for phi in list(self.phi_dict.keys()):
          self.phi_dict[phi].append(V['phi', phi])
        
        for x in list(self.model.variables_dict['xd'].keys()):
          for dim in range(self.model.variables_dict['xd'][x].shape[0]):
            self.x_dict[x+'_'+str(dim)].append(self.extract_x_vals(V, x, dim))
        
        for u in list(self.model.variables_dict['u'].keys()):
          for dim in range(self.model.variables_dict['u'][u].shape[0]):
            self.u_dict[u+'_'+str(dim)].append(self.extract_u_vals(V, u, dim))

        for theta in list(self.model.variables_dict['theta'].keys()):
          for dim in range(self.model.variables_dict['theta'][theta].shape[0]):
            self.theta_dict[theta+'_'+str(dim)].append(V['theta',theta, dim])

        for t in list(self.t_dict.keys()):
          self.t_dict[t].append(self.nlp.time_grids[t](V['theta','t_f']))

        energy = self.nlp.integral_output_components[1](V, P)
        self.avg_power.append(energy[-1]/self.t_dict['x'][-1][-1])
        # Out = self.Out(self.Out_fun(V, self.P))
        return [0]

    def extract_x_vals(self, V, name, dim):
      x_vals = []
      for k in range(self.nlp.n_k+1):
          # add interval values
          x_vals.append(V['xd',k,name,dim])
          if k < self.nlp.n_k:
            # add node values
            x_vals += V['coll_var',k, :, 'xd', name,dim]
      return x_vals
    
    def extract_u_vals(self, V, name, dim):
      u_vals = []
      for k in range(self.nlp.n_k):
        if 'u' in V.keys():
          u_vals.append(V['u',k,name,dim])
        else:
          u_vals += V['coll_var',k, :, 'u', name, dim]
      return u_vals
  
    def __init_dicts(self):
      
      phi_dict = collections.OrderedDict()
      for phi in self.model.parameters_dict['phi'].keys():
        phi_dict[phi] = []
      
      x_dict = collections.OrderedDict()
      for x in self.model.variables_dict['xd'].keys():
        for dim in range(self.model.variables_dict['xd'][x].shape[0]):
          x_dict[x+'_'+str(dim)] = []

      u_dict = collections.OrderedDict()
      for u in self.model.variables_dict['u'].keys():
        for dim in range(self.model.variables_dict['u'][u].shape[0]):
          u_dict[u+'_'+str(dim)] = []

      theta_dict = collections.OrderedDict()
      for th in self.model.variables_dict['theta'].keys():
        for dim in range(self.model.variables_dict['theta'][th].shape[0]):
          theta_dict[th+'_'+str(dim)] = []

      t_dict = collections.OrderedDict()
      for t in self.nlp.time_grids.keys():
        t_dict[t] = []

      self.phi_dict = phi_dict
      self.x_dict = x_dict
      self.u_dict = u_dict
      self.theta_dict = theta_dict
      self.t_dict = t_dict
      self.avg_power = []
      
      return None

    def reset(self):
      self.__init_dicts()
      return None

    def update_P(self, P):
      self.P = P
      return None