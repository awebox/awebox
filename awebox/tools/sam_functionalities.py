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
'''
discretization code (direct collocation or multiple shooting)
creates nlp variables and outputs, and gets discretized constraints
python-3.5 / casadi-3.4.5
- authors: elena malz 2016
           rachel leuthold, jochem de schutter alu-fr 2017-21
'''
from typing import List

import casadi as ca
import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.tools.constraint_operations as cstr_op
from awebox.mdl.model import Model
from awebox.ocp.collocation import Collocation


class CollocationIRK:
    c: np.ndarray = None
    A = None
    b = None
    d = None

    def getButcher(self) -> (np.ndarray,np.ndarray,np.ndarray):
        """Returns the Butcher Tableau (c,A,b) of the Integrator"""
        if self.A is None:
            raise NotImplementedError
        return self.c, self.A, self.b

    def __init__(self, collPoints: np.ndarray):
        """ A simple implementation of an IRK collocation scheme, based on the collocation points provided
        The instance then provides the IRK matrices A, b, c and a nice casadi-function builder to evaluate the polynomials.

        This class is NOT THE DIRECT TRANSCRIPTION METHOD 'Collocation' as implemented in awebox.ocp.collocation.
        # todo: move this somewhere or replace with existing collocation functions
        # todo: currently this is being used in Jakobs SAM implementation

        :param collPoints: the collocation points to be used, must be a 1D numpy array
        """

        assert collPoints.ndim == 1
        assert np.all(np.unique(collPoints, return_counts=True)[1] <= 1), 'CollPoints have to be distinct!'
        assert np.all(collPoints <= 1) and np.all(0 <= collPoints), 'CollPoints must be between 0 and 1'
        self.d = collPoints.shape[-1]


        from numpy.polynomial import Polynomial

        # create list of polynomials
        self._ls = []
        for j in range(self.d):
            l = Polynomial([1])
            for r in range(self.d):
                if r != j:
                    l *= Polynomial([-collPoints[r], 1]) / (collPoints[j] - collPoints[r])
            self._ls.append(l)

        self.c = collPoints
        self.b = np.array([l.integ(1)(1) for l in self._ls])
        self.A = np.array([[l.integ(1)(ci) for l in self._ls] for ci in self.c])

    @property
    def polynomials(self) -> list:
        """A list of the numpy polynomials that correspond to the lagrange polynomials"""
        return self._ls

    @property
    def polynomials_int(self) -> list:
        """A list of the numpy polynomials that correspond to the integrated lagrange polynomials"""
        return [l.integ(1) for l in self._ls]

    def getPolyEvalFunction(self, shape: tuple, includeZero: bool = False, includeOne: bool = False, fixedValues: list = None) -> cas.Function:
        """
        Generates a casadi function that evaluates the polynomial at a given point t of the form

        p(t) = F(t, [x0], x1, ..., xd, [x1])

        where t is a scalar in [0,1] and x0, ..., xd are the collocation points of some shape (can be vector or matrix).

        If fixed values for the nodes x0, ..., xd are provided, the function will be of the form

        p(t) = F(t)

        :param shape: the shape of the collocation nodes, can be matrices or vectors
        :param includeZero: if true, the collocation point at time 0 is included
        :param fixedValues: a list of fixed values for the nodes, if provided, the function will be of the form x(t) = F(t)
        """
        assert not(includeOne and includeZero), 'either includeOne or includeZero can be true, not both!'

        # append zero if needed
        if includeZero:
            collPoints = cas.DM(np.concatenate([[0],self.c]))
            d = self.d + 1
        elif includeOne:
            collPoints = cas.DM(np.concatenate([self.c,[1]]))
            d = self.d + 1
        else:
            collPoints = cas.DM(self.c)
            d = self.d

        nx = shape[0]*shape[1]
        t = cas.SX.sym('t')

        if fixedValues is None:
            # create symbolic variables for the nodes
            Xs = []
            for i in np.arange((0 if includeZero else 1),self.d+1):
                Xs.append(cas.SX.sym(f'x{i}', shape))
        else:
            assert len(fixedValues) == d, f"The number of fixed values ({len(fixedValues)}) must be equal to the number of collocation points ({d})!"
            assert all([v.shape == shape for v in fixedValues]), "The shape of the fixed values must be equal to the shape of the collocation points!"
            assert all([type(v) == cas.DM for v in fixedValues]), "The fixed values must be of type casadi.DM!"
            Xs = fixedValues

        # reshape input variables into a matrix of shape (nx, d)
        p_vals = cas.horzcat(*[X.reshape((nx, 1)) for X in Xs])

        # create list of polynomials
        _ls = []
        for j in range(d):
            l = 1
            for r in range(d):
                if r != j:
                    l *= (t -collPoints[r]) / (collPoints[j] - collPoints[r])
            _ls.append(l)

        # evaluate polynomials
        sum = cas.DM.zeros((nx, 1))
        for i in range(d):
            sum += p_vals[:, i] * _ls[i]

        # reshape the result into the original shape
        result = cas.reshape(sum, shape)

        if fixedValues is None:
            return cas.Function('polyEval', [t] + Xs, [result])
        else:
            return cas.Function('polyEval', [t], [result])

def reconstruct_full_from_SAM(nlpoptions: dict, V_opt_scaled: ca.tools.struct, output_vals_opt: ca.DM) -> tuple:
    """
    Reconstruct the full trajectory from the SAM discretization with micro- and macro-integrations.
    This works by interpolating the polynomials of the algebraic variables (that are the micro-integrations)
    of the DAE for the average dynamics. The time grid is reconstructed by interpolating the time-scaling parameters.

    This functions returns a structure `V` (that has the same structure as the regular variables if no averaging is used)
    that contains the full trajectory, the time grid and the output values. It has `nk_total` integration intervals, with `d_micro` collocation nodes each.

    :param nlpoptions: the nlp options, e.g. trial.options['nlp']
    :param V_opt_scaled: the optimal variables from the SAM discretization
    :param output_vals_opt: the optimal output values from the SAM discretization
    :return: (V_recon_scaled, time_grid_recon, output_recon)
    """
    assert {'x', 'u', 'z', 'coll_var', 'theta'}.issubset(V_opt_scaled.keys())
    d_micro = nlpoptions['collocation']['d']


    assert output_vals_opt.shape[1] == (d_micro + 1) * (nlpoptions['n_k'])
    n_outputs = output_vals_opt.shape[0]

    d_SAM = nlpoptions['SAM']['d']
    N_SAM = nlpoptions['SAM']['N']
    macroIntegrator = CollocationIRK(np.array(ca.collocation_points(d_SAM, nlpoptions['SAM']['MaInt_type'])))
    regions_indeces = struct_op.calculate_SAM_regions(nlpoptions)

    t_f_opt = V_opt_scaled['theta', 't_f']
    assert t_f_opt.shape[0] == d_SAM + 1

    from casadi.tools import struct_symMX, entry
    N_regions = d_SAM + 1
    assert len(regions_indeces) == N_regions
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    n_k_total = regions_deltans[-1] + N_SAM * regions_deltans[0]

    # create structure and initialize all values to zero
    V_reconstruct = struct_symMX([entry('x', struct=V_opt_scaled.getStruct('x'), repeat=[n_k_total + 1]),
                                  entry('u', struct=V_opt_scaled.getStruct('u'), repeat=[n_k_total]),
                                  entry('z', struct=V_opt_scaled.getStruct('z'), repeat=[n_k_total]),
                                  entry('coll_var', struct=V_opt_scaled.getStruct('coll_var'), repeat=[n_k_total, d_micro]),
                                  entry('theta', struct=V_opt_scaled.getStruct('theta')),
                                  entry('phi', struct=V_opt_scaled.getStruct('phi'))
                                  ])(0)

    # sort the micro-integration variables, controls and outputs into the regions for easier access
    zs_micro = []
    us_micro = []
    zs_micro_coll = []
    outputs_micro = []
    for i in range(d_SAM):
        # interpolate the micro-collocation polynomial
        z_micro = []
        u_micro = []
        z_micro_coll = []
        output_micro = []
        for j in regions_indeces[i]:
            z_micro.append(V_opt_scaled['x', j])  # start point of the collocation interval
            z_micro_coll.append(V_opt_scaled['coll_var', j, :])  # the collocation points]]
            u_micro.append(V_opt_scaled['u', j])  # the control
            output_micro.append(output_vals_opt[:, j * (d_micro + 1): (j+1) * (d_micro + 1)])  # the output values
        zs_micro.append(z_micro)
        zs_micro_coll.append(z_micro_coll)
        us_micro.append(u_micro)
        outputs_micro.append(output_micro)

    # functions to interpolate the state and collocation nodes
    z_interpol_f = macroIntegrator.getPolyEvalFunction(shape=zs_micro[0][0].shape, includeZero=False)
    u_interpol_f = macroIntegrator.getPolyEvalFunction(shape=us_micro[0][0].shape, includeZero=False)
    z_interpol_f_coll = macroIntegrator.getPolyEvalFunction(shape=zs_micro_coll[0][0][0].shape, includeZero=False)
    output_interpol_f = macroIntegrator.getPolyEvalFunction(shape=(n_outputs, 1), includeZero=False)

    # for the reconstructed trajectory, build the time grid
    strobos_eval = np.arange(N_SAM) + {'BD': 1, 'CD': 0.5, 'FD': 0.0}[nlpoptions['SAM']['ADAtype']]
    strobos_eval = strobos_eval * 1 / N_SAM

    j = 0

    outputs_reconstructed = []
    # 1. fill reelout values
    for n in range(N_SAM):
        n_micro_ints_per_cycle = regions_deltans[0]  # these are the same for every cycle
        for j_local in range(n_micro_ints_per_cycle):
            # evaluate the reconstruction polynomials
            V_reconstruct['x', j] = z_interpol_f(strobos_eval[n], *[zs_micro[k][j_local] for k in range(d_SAM)])
            V_reconstruct['u', j] = u_interpol_f(strobos_eval[n], *[us_micro[k][j_local] for k in range(d_SAM)])
            outputs_reconstructed.append(output_interpol_f(strobos_eval[n], *[outputs_micro[k][j_local][:,0] for k in range(d_SAM)]))

            for i in range(d_micro):
                V_reconstruct['coll_var', j, i] = z_interpol_f_coll(strobos_eval[n],
                                                                    *[zs_micro_coll[k][j_local][i] for k in
                                                                      range(d_SAM)])
                outputs_reconstructed.append(
                    output_interpol_f(strobos_eval[n], *[outputs_micro[k][j_local][:,i+1] for k in range(d_SAM)]))

            j = j + 1

    # 2. fill reelin values
    for j_local in regions_indeces[-1]:
        V_reconstruct['x', j] = V_opt_scaled['x', j_local]
        V_reconstruct['u', j] = V_opt_scaled['u', j_local]
        V_reconstruct['z', j] = V_opt_scaled['z', j_local]
        V_reconstruct['coll_var', j] = V_opt_scaled['coll_var', j_local]
        outputs_reconstructed.append(output_vals_opt[:, j_local * (d_micro + 1): (j_local + 1) * (d_micro + 1)])
        j = j + 1

    # last value
    assert j == n_k_total
    V_reconstruct['x', -1] = V_opt_scaled['x', -1]

    # other variables
    V_reconstruct['theta'] = V_opt_scaled['theta']
    V_reconstruct['phi'] = V_opt_scaled['phi']

    # reconstruct the time grid
    time_grid_reconstruction = construct_time_grids_SAM_reconstruction(nlpoptions)
    time_grid_recon_eval = {}
    for key in time_grid_reconstruction.keys():
        time_grid_recon_eval[key] = time_grid_reconstruction[key](t_f_opt)

    # do some sanity checks
    assert time_grid_recon_eval['x'].shape[0] == n_k_total + 1
    assert time_grid_recon_eval['u'].shape[0] == n_k_total
    assert time_grid_recon_eval['x_coll'].shape[0] == n_k_total * (d_micro + 1)
    assert V_reconstruct['x'].__len__() == n_k_total + 1
    assert V_reconstruct['u'].__len__() == n_k_total
    assert V_reconstruct['coll_var'].__len__() == n_k_total

    return V_reconstruct, time_grid_recon_eval, ca.horzcat(*outputs_reconstructed)

def construct_time_grids_SAM_reconstruction(nlp_options) -> dict:
    """
    Construct the time grids for the RECONSTRUCTED trajectory after the SAM discretization.
    Returns a dictionary of casadi functions for
        - discrete states ('x'),
        - controls ('u'),
        - collocation nodes ('coll'),
        - state and collocation nodes ('x_coll')

    :param nlp_options:
    :return: {'x': cas.Function, 'u': cas.Function, 'coll': cas.Function, 'x_coll': cas.Function}
    """


    # assert nlp_options['SAM']['use']
    assert nlp_options['discretization'] == 'direct_collocation'

    nk = nlp_options['n_k']
    d = nlp_options['collocation']['d']
    d_SAM = nlp_options['SAM']['d']
    N_SAM = nlp_options['SAM']['N']
    scheme_micro = nlp_options['collocation']['scheme']
    tau_root_micro = ca.vertcat(ca.collocation_points(d, scheme_micro))
    N_regions = d_SAM + 1
    regions_indeces = struct_op.calculate_SAM_regions(nlp_options)
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    assert len(regions_indeces) == N_regions
    t_f_sym = ca.SX.sym('t_f_sym', (N_regions,1))
    T_regions = t_f_sym / nk * regions_deltans  # the duration of each discretization region

    tx = []
    tu = []
    txcoll = []
    tcoll = []

    # function to evaluate the reconstructed time
    # quadrature over the region durations to reconstruct the physical time
    macroIntegrator = CollocationIRK(np.array(ca.collocation_points(d_SAM, nlp_options['SAM']['MaInt_type'])))
    tau_SX = ca.SX.sym('tau', 1)
    t_recon_SX = 0
    for i in range(d_SAM):  # iterate over the collocation nodes
        t_recon_SX += T_regions[i] * macroIntegrator.polynomials_int[i](tau_SX / N_SAM) * N_SAM
    t_recon_f = ca.Function('t_SAM', [tau_SX, t_f_sym], [t_recon_SX])

    # 1. fill reelout values
    tau_tgrid_x = np.linspace(0, N_SAM, regions_deltans[0] * N_SAM, endpoint=False)
    duration_interval_tau = 1 / (regions_deltans[0])
    tau_tgrid_coll = []
    for tau_x in tau_tgrid_x:
        tau_tgrid_coll += [tau_x + tau_root_micro.full() * duration_interval_tau]
    tau_tgrid_coll = np.vstack(tau_tgrid_coll).flatten()

    tau_tgrid_xcoll = []
    for tau_x in tau_tgrid_x:
        tau_tgrid_xcoll += [tau_x]
        tau_tgrid_xcoll += [tau_x + tau_root_micro.full() * duration_interval_tau]
    tau_tgrid_xcoll = np.vstack(tau_tgrid_xcoll).flatten()

    tx.append(t_recon_f.map(tau_tgrid_x.size)(tau_tgrid_x, t_f_sym).T)
    tu.append(t_recon_f.map(tau_tgrid_x.size)(tau_tgrid_x, t_f_sym).T)
    txcoll.append(t_recon_f.map(tau_tgrid_xcoll.size)(tau_tgrid_xcoll, t_f_sym).T)
    tcoll.append(t_recon_f.map(tau_tgrid_coll.size)(tau_tgrid_coll, t_f_sym).T)
    t_local = t_recon_f(N_SAM, t_f_sym)

    # 2. fill reelin values
    for j_local in regions_indeces[-1]:
        # speed of time of the specific interval
        duration_interval = T_regions[-1] / regions_deltans[-1]

        tx.append(t_local)
        tu.append(t_local)
        tcoll.append(t_local + tau_root_micro * duration_interval)
        txcoll.append(t_local)
        txcoll.append(t_local + tau_root_micro * duration_interval)
        # update running variables
        t_local = t_local + duration_interval

    # add last node
    tx.append(t_local)

    time_grid_reconstruction = {'x': ca.Function('time_grid_x', [t_f_sym], [ca.vertcat(*tx)]),
                                'u': ca.Function('time_grid_u', [t_f_sym], [ca.vertcat(*tu)]),
                                'x_coll': ca.Function('time_grid_coll', [t_f_sym], [ca.vertcat(*txcoll)]),
                                'coll': ca.Function('time_grid_coll', [t_f_sym], [ca.vertcat(*tcoll)])}

    return time_grid_reconstruction

def modify_constraints_for_SAM(nlp_options: dict,
                               collocation: Collocation,
                               V: cas.struct,
                               Xdot: cas.struct,
                               model: Model,
                               ocp_cstr_list: cstr_op.OcpConstraintList) -> tuple:
    """
    Modify the constraints of the  build NLP, to now incoporate the SAM discretization. This

    a) cuts open the discretized trajectory to reuse the parts for the micro-integrations and the reel-in phase.
    b) stitches the micro-integrations together with the macro-integrations
    c) adds constraints to enforce the invariants of the SAM discretization

    :param nlp_options: the nlp options, e.g. trial.options['nlp']
    :param collocation: the (micro)-collocation instance of the awebox
    :param V: the variables struct of the nlp
    :param Xdot: the derivatives struct of the nlp
    :param model: trial.model
    :param ocp_cstr_list: the already existing constraints
    :return: a tuple (SAM_cstrs_entry_list, SAM_cstrs_list) with the new constraints
    """

    SAM_cstrs_list = cstr_op.OcpConstraintList()  # create an empty list
    SAM_cstrs_entry_list = []
    N_SAM = nlp_options['SAM']['N']
    d_SAM = nlp_options['SAM']['d']
    # macro-integrator
    macroIntegrator = CollocationIRK(np.array(cas.collocation_points(d_SAM, nlp_options['SAM']['MaInt_type'])))
    c_macro, A_macro, b_macro = macroIntegrator.c, macroIntegrator.A, macroIntegrator.b
    assert d_SAM == c_macro.size
    tf_regions_indices = struct_op.calculate_SAM_regions(nlp_options)
    SAM_regions_indeces = tf_regions_indices[:-1]  # we are not intersted the last region (reelin)
    # build evaluation functions for the invariants c(x), dc(x), orthonormality
    invariant_names_to_constrain = [key for key in model.outputs_dict['invariants'].keys() if
                                    key.startswith(tuple(['c', 'dc', 'orthonormality']))]
    g_inv_SX_SCALED = (
        ca.vertcat(*model.outputs(model.outputs_fun(model.variables, model.parameters))[
            'invariants', invariant_names_to_constrain])
    )
    g_fun = ca.Function('g', [model.variables['x'], model.variables['theta']], [g_inv_SX_SCALED])
    g_jac_x_SCALED_fun = ca.Function('inv_jac_x', [model.variables['x'], model.variables['theta']],
                                     [ca.jacobian(g_inv_SX_SCALED, model.variables['x'])])
    # iterate the SAM micro-integrations
    for i in range(d_SAM):
        n_first = SAM_regions_indeces[i][0]  # first interval index of the region
        n_last = SAM_regions_indeces[i][-1]  # last interval index of the region

        # 1A. XMINUS: connect x_minus with start of the micro integration
        xminus = model.variables_dict['x'](V['x_micro_minus', i])
        micro_connect_xminus = cstr_op.Constraint(expr=xminus.cat - V['x', n_first],
                                                  name=f'micro_connect_xminus_{i}',
                                                  cstr_type='eq')
        SAM_cstrs_list.append(micro_connect_xminus)
        SAM_cstrs_entry_list.append(cas.entry(f'micro_connect_xminus_{i}', shape=xminus.shape))

        # 1B. enforce invartiants for startpoint
        # get the thetas at the startpoint
        theta_xminus = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, n_first)['theta']
        expr_inv = g_fun(xminus.cat, theta_xminus)
        invariants_start_cycle_cstr_i = cstr_op.Constraint(expr=expr_inv,
                                                           name=f'invariants_start_cycle_cstr_{i}',
                                                           cstr_type='eq')
        SAM_cstrs_list.append(invariants_start_cycle_cstr_i)
        SAM_cstrs_entry_list.append(cas.entry(f'invariants_start_cycle_cstr_{i}', shape=expr_inv.shape))

        # 2. XPLUS: replace the continutiy constraint for the last collocation interval of the region
        xplus = model.variables_dict['x'](V['x_micro_plus', i])
        ocp_cstr_list.get_constraint_by_name(f'continuity_{n_last}').expr = xplus.cat - model.variables_dict['x'](
            collocation.get_continuity_expression(V, n_last)).cat

        # 3. SAM dynamics approximation - vcoll
        ada_vcoll_cstr = cstr_op.Constraint(expr=(xplus.cat - xminus.cat) * N_SAM - V['v_macro_coll', i],
                                            name=f'ada_vcoll_cstr_{i}',
                                            cstr_type='eq')
        SAM_cstrs_list.append(ada_vcoll_cstr)
        SAM_cstrs_entry_list.append(cas.entry(f'ada_vcoll_cstr_{i}', shape=xminus.shape))

        # 5. Connect to Macro integration point
        ada_type = nlp_options['SAM']['ADAtype']
        assert ada_type in ['FD', 'BD', 'CD'], 'only FD, BD, CD are supported'
        ada_coeffs = {'FD': [1, -1, 0], 'BD': [0, -1, 1], 'CD': [1, -2, 1]}[ada_type]
        lam_SAM_i = V['lam_SAM', i]

        expr_connect = (ada_coeffs[0] * V['x_micro_minus', i]
                        + ada_coeffs[1] * V['x_macro_coll', i]
                        + ada_coeffs[2] * V['x_micro_plus', i]
                        + g_jac_x_SCALED_fun(V['x_micro_minus', i], theta_xminus).T @ lam_SAM_i)
        micro_connect_macro = cstr_op.Constraint(expr=expr_connect,
                                                 name=f'micro_connect_macro_{i}',
                                                 cstr_type='eq')
        SAM_cstrs_list.append(micro_connect_macro)
        SAM_cstrs_entry_list.append(cas.entry(f'micro_connect_macro_{i}', shape=xminus.shape))
    # MACRO INTEGRATION
    X_macro_start = model.variables_dict['x'](V['x_macro', 0])
    X_macro_end = model.variables_dict['x'](V['x_macro', -1])
    # START: connect X0_macro and the endpoint of the reelin phase, by replacing the periodicity constraint
    # reconstruct the periodicty constraint (remove state 'e') from constraint
    state_start = model.variables_dict['x'](X_macro_start)
    state_end = model.variables_dict['x'](V['x', -1])
    periodicty_expr = []
    for name in state_start.keys():
        if name != 'e':  # don't enforce periodicity on the energy
            periodicty_expr.append(state_start[name] - state_end[name])
    ocp_cstr_list.get_constraint_by_name(f'state_periodicity').expr = ca.vertcat(*periodicty_expr)
    # for macro: Baumgarte as function
    # baumgarte_cst_SX = model.constraints_dict['equality']['dynamics_constraint']
    # baumgarte_cst_fun = cas.Function('baumgarte_cst_fun', [model.variables], [baumgarte_cst_SX])
    # Macro RK scheme
    for i in range(d_SAM):
        macro_rk_cstr = cstr_op.Constraint(
            expr=V['x_macro_coll', i] - (X_macro_start.cat + cas.horzcat(*V['v_macro_coll']) @ A_macro[i, :].T),
            name=f'macro_rk_cstr_{i}',
            cstr_type='eq')
        SAM_cstrs_list.append(macro_rk_cstr)
        SAM_cstrs_entry_list.append(cas.entry(f'macro_rk_cstr_{i}', shape=xminus.shape))
    # END: connect x_plus with end of the reelout
    macro_end_cstr = cstr_op.Constraint(
        expr=X_macro_end.cat - (X_macro_start.cat + cas.horzcat(*V['v_macro_coll']) @ b_macro),
        name='macro_end_cstr',
        cstr_type='eq')
    SAM_cstrs_list.append(macro_end_cstr)
    SAM_cstrs_entry_list.append(cas.entry('macro_end_cstr', shape=xminus.shape))
    # # enforce consistency at start of reelout
    index_reelin_start = tf_regions_indices[-1][0]
    x_reelin_start = V['x', index_reelin_start]
    theta_start_reelin = struct_op.get_variables_at_time(nlp_options, V, Xdot, model.variables, index_reelin_start)[
        'theta']
    expr_inv = g_fun(x_reelin_start, theta_start_reelin)
    invariants_start_reelin_cstr = cstr_op.Constraint(expr=expr_inv,
                                                      name=f'invariants_start_reelin_cstr',
                                                      cstr_type='eq')
    SAM_cstrs_list.append(invariants_start_reelin_cstr)
    SAM_cstrs_entry_list.append(cas.entry(f'invariants_start_reelin_cstr', shape=expr_inv.shape))
    # connect endpoint of the macro-integration with start of the reelin phase with PROJECTION
    lam_SAM_reelin = V['lam_SAM', -1]
    macro_connect_reelin = cstr_op.Constraint(expr=X_macro_end.cat - x_reelin_start - g_jac_x_SCALED_fun(x_reelin_start,
                                                                                                         theta_start_reelin).T @ lam_SAM_reelin,
                                              name='macro_connect_reelin',
                                              cstr_type='eq')
    SAM_cstrs_list.append(macro_connect_reelin)
    SAM_cstrs_entry_list.append(cas.entry('macro_connect_reelin', shape=xminus.shape))
    return SAM_cstrs_entry_list, SAM_cstrs_list


def constructPiecewiseCasadiExpression(decisionVariable: ca.SX, edges: List, expressions: List[ca.SX]) -> ca.SX:
    """
    Construct a piecewise casadi expression from a list of edges and functions for a given scalar decision variable.
    For example, if the decision variable is x and the edges are [0,1,2] and the expressions are [f1(x),f2(x)] then the
    resulting expression is f1(x) if x is in [0,1) and f2(x) if x is in [1,2).
    For values outside, the function will return nan.

    DO NOT USE THIS FUNCTION FOR OPTIMIZATION, it is not differentiable at the edges.

    :param decisionVariable: the variable that is evaluated
    :param edges: a list of edge values, in ascending order
    :param expressions: a list of casadi.SX expressions.
    :return:
    """
    assert type(decisionVariable) is ca.SX, "The decision variable has to be a casadi.SX!"
    assert decisionVariable.shape == (1, 1), "The decision variable has to be a scalar!"
    assert type(edges) is list, "The edges have to be a list!"
    assert type(expressions) is list, "The functions have to be a list!"
    assert len(expressions) > 0, "There has to be at least one function!"
    assert len(edges) == len(expressions) + 1, "The number of edges has to be one more than the number of functions!"

    # check that edges are in ascending order
    assert np.all(np.diff(edges) > 0), "The edges have to be in ascending order!"

    outputExpression = ca.DM(0)

    # add nan for values outside the edges
    outputExpression += ca.if_else(decisionVariable < edges[0], ca.DM.nan(), 0)
    outputExpression += ca.if_else(decisionVariable >= edges[-1], ca.DM.nan(), 0)

    # iterate edges
    for edge_index in range(len(edges) - 1):
        # condition that we are in the interval
        _condition = (decisionVariable >= edges[edge_index]) * (decisionVariable < edges[edge_index + 1])

        # add the function to the output expression
        outputExpression += ca.if_else(_condition, expressions[edge_index], 0)

    return outputExpression


def originalTimeToSAMTime(nlpoptions,t_f_opt) -> ca.Function:
    """ construct an interpolating casadi function

    t_AWE -> t_SAM = f(t_AWE)

    that maps from the AWEbox timegrid (t in [0,sum(T_i]) to the SAM timegrid (discontinuous).
    This is useful because for interpolating, the AWEBox creates a timegrid as `np.linspace(0, sum(T_i), N)`,
     this function can then be used to translate this grid.
    """
    assert nlpoptions['SAM']['use'], "This function is only for SAM"

    d_SAM = nlpoptions['SAM']['d']
    N_SAM = nlpoptions['SAM']['N']
    n_k = nlpoptions['n_k']
    macroInt = CollocationIRK(np.array(ca.collocation_points(d_SAM, nlpoptions['SAM']['MaInt_type'])))

    # get the regions and the duration of each region
    regions_indeces = struct_op.calculate_SAM_regions(nlpoptions)
    regions_deltans = np.array([region.__len__() for region in regions_indeces])
    T_regions = (t_f_opt / n_k * regions_deltans).full().flatten()  # the duration of each discretization region
    h = N_SAM
    t_coll = [sum([h*T_regions[j]*macroInt.A[i,j] for j in range(d_SAM)]) for i in range(d_SAM)]
    t_reelout_end = sum([h*T_regions[j]*macroInt.b[j] for j in range(d_SAM)])

    t = ca.SX.sym('t')
    edges = np.cumsum(np.concatenate([[0],T_regions]))
    edges = edges - 1E-9 # shift all edged by a small amount to avoid numerical issues
    edges[-1] = edges[-1] + 2E-9 # shift the last edge by a larger amount to avoid numerical issues

    expressions = []

    # reel out
    for i in range(d_SAM):
        offset = t_coll[i]
        t_offset = edges[i] + T_regions[i]*{'BD': 1, 'CD': 0.5, 'FD': 0.0}[nlpoptions['SAM']['ADAtype']]
        expressions.append(offset + (t - t_offset))

    # reel in
    expressions.append(t_reelout_end + (t - sum(T_regions[0:d_SAM])))

    return ca.Function('t_SAM', [t], [constructPiecewiseCasadiExpression(t, edges.tolist(), expressions)])

