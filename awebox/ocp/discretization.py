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
'''
discretization code (direct collocation or multiple shooting)
creates n_l_p variables
creates n_l_p constraints
creates n_l_p outputs
python-3.5 / casadi-3.4.5
- authors: elena malz 2016
           rachel leuthold, jochem de schutter alu-fr 2017-18
'''

import casadi.tools as cas

from . import constraints

from . import collocation

from . import multiple_shooting

from . import performance

import pdb

import awebox.tools.struct_operations as struct_op

def setup_nlp_v(nlp_numerics_options, model, formulation, Collocation):

    # extract necessary inputs
    variables_dict = model.variables_dict
    nk = nlp_numerics_options['n_k']

    # check if phase fix and adjust theta accordingly
    if nlp_numerics_options['phase_fix']:
        theta = get_phase_fix_theta(variables_dict)
    else:
        theta = variables_dict['theta']

    # define interval struct entries for controls and states
    entry_tuple = (
        cas.entry('xd', repeat = [nk+1], struct = variables_dict['xd']),
        )

    # add additional variables according to provided options
    if nlp_numerics_options['discretization'] == 'direct_collocation':

        if nlp_numerics_options['collocation']['u_param'] == 'zoh':
            entry_tuple += (
                cas.entry('u',  repeat = [nk],   struct = variables_dict['u']),
            )

        # add algebraic variables at interval except for radau case
        if nlp_numerics_options['collocation']['scheme'] != 'radau':
            if nlp_numerics_options['lift_xddot']:
                entry_tuple += (
                    cas.entry('xddot', repeat = [nk], struct= variables_dict['xddot']), # depends on implementation (e.g. not present for radau collocation)
                )
            if nlp_numerics_options['lift_xa']:
                entry_tuple += (cas.entry('xa', repeat = [nk],   struct= variables_dict['xa']),) # depends on implementation (e.g. not present for radau collocation)
                if 'xl' in list(variables_dict.keys()):
                    entry_tuple += (cas.entry('xl', repeat = [nk],   struct= variables_dict['xl']),)  # depends on implementation (e.g. not present for radau collocation)

        # add collocation node variables
        d = nlp_numerics_options['collocation']['d'] # interpolating polynomial order
        coll_var = Collocation.get_collocation_variables_struct(variables_dict, nlp_numerics_options['collocation']['u_param'])
        entry_tuple += (cas.entry('coll_var', struct = coll_var, repeat= [nk,d]),)

    elif nlp_numerics_options['discretization'] == 'multiple_shooting':

        entry_tuple += (
            cas.entry('u',  repeat = [nk],   struct = variables_dict['u']),
        )

        # add slack variables for inequalities
        if nlp_numerics_options['slack_constraints'] == True and model.constraints_dict['inequality']:
            entry_tuple += (cas.entry('us',  repeat = [nk],   struct = model.constraints_dict['inequality']),)

        # multiple shooting: add algebraic variables at interval if lifted
        if nlp_numerics_options['lift_xddot']:
            entry_tuple += (
                cas.entry('xddot', repeat = [nk], struct= variables_dict['xddot']), # depends on implementation (e.g. not present for radau collocation)
            )
        if nlp_numerics_options['lift_xa']:
            entry_tuple += (cas.entry('xa', repeat = [nk],   struct= variables_dict['xa']),) # depends on implementation (e.g. not present for radau collocation)
            if 'xl' in list(variables_dict.keys()):
                entry_tuple += (cas.entry('xl', repeat = [nk],   struct= variables_dict['xl']),)  # depends on implementation (e.g. not present for radau collocation)

    # add global entries
    entry_list = [entry_tuple]
    entry_list += [
        cas.entry('theta', struct = theta),
        cas.entry('phi',   struct = model.parameters_dict['phi']),
        cas.entry('xi',    struct = formulation.xi_dict['xi'])
    ]

    # generate structure
    V = cas.struct_symMX(entry_list)

    return V

def construct_time_grids(nlp_numerics_options):

    time_grids = {}
    nk = nlp_numerics_options['n_k']
    if nlp_numerics_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        ms = False
        d = nlp_numerics_options['collocation']['d']
        scheme = nlp_numerics_options['collocation']['scheme']
        tau_root = cas.vertcat(cas.collocation_points(d, scheme))
        tcoll = []

    elif nlp_numerics_options['discretization'] == 'multiple_shooting':
        direct_collocation = False
        ms = True
        tcoll = None

    # make symbolic time constants
    if nlp_numerics_options['phase_fix']:
        tfsym = cas.SX.sym('tfsym',2)
        nk_reelout = round(nk * nlp_numerics_options['phase_fix_reelout'])

        t_switch = tfsym[0] * nk_reelout / nk
        time_grids['t_switch'] = cas.Function('tgrid_tswitch', [tfsym], [t_switch])

    else:
        tfsym = cas.SX.sym('tfsym',1)

    # initialize
    tx = []
    tu = []

    for k in range(nk+1):

        # extract correct time constant in case of phase fix
        if nlp_numerics_options['phase_fix']:
            if k < nk_reelout:
                tf = tfsym[0]
                k0 = 0.0
                kcount = k
            else:
                tf = tfsym[1]
                k0 = nk_reelout * tfsym[0]/ tf
                kcount = k - nk_reelout
        else:
            k0 = 0.0
            kcount = k
            tf = tfsym

        # add interval timings
        tx = cas.vertcat(tx, (k0 + kcount) * tf / float(nk))
        if k < nk:
            tu = cas.vertcat(tu, (k0 + kcount) * tf / float(nk))

        # add collocation timings
        if direct_collocation and (k < nk):
            for j in range(d):
                tcoll = cas.vertcat(tcoll,(k0 + kcount + tau_root[j]) * tf / float(nk))

    if direct_collocation:
        # reshape tcoll
        tcoll = tcoll.reshape((d,nk)).T
        tx_coll = cas.vertcat(cas.horzcat(tu, tcoll).T.reshape((nk*(d+1),1)),tx[-1])

        # write out collocation grids
        time_grids['coll'] = cas.Function('tgrid_coll',[tfsym],[tcoll])
        time_grids['x_coll'] = cas.Function('tgrid_x_coll',[tfsym],[tx_coll])

    # write out interval grid
    time_grids['x'] = cas.Function('tgrid_x',[tfsym],[tx])
    time_grids['u'] = cas.Function('tgrid_u',[tfsym],[tu])


    return time_grids

def setup_nlp_cost():

    cost = cas.struct_symSX([(
        cas.entry('tracking'),
        cas.entry('u_regularisation'),
        cas.entry('ddq_regularisation'),
        cas.entry('gamma'),
        cas.entry('iota'),
        cas.entry('psi'),
        cas.entry('tau'),
        cas.entry('eta'),
        cas.entry('nu'),
        cas.entry('upsilon'),
        cas.entry('fictitious'),
        cas.entry('power'),
        cas.entry('t_f'),
        cas.entry('theta_regularisation'),
        cas.entry('nominal_landing'),
        cas.entry('compromised_battery'),
        cas.entry('transition'),
        cas.entry('slack')
    )])

    return cost

def setup_nlp_p_fix(V, model):

    # fixed system parameters
    p_fix = cas.struct_symSX([(
        cas.entry('ref', struct=V),     # tracking reference for cost function
        cas.entry('weights', struct=model.variables)  # weights for cost function
    )])

    return p_fix

def setup_nlp_p(V, model):

    cost = setup_nlp_cost()
    p_fix = setup_nlp_p_fix(V, model)

    P = cas.struct_symMX([
        cas.entry('p',      struct = p_fix),
        cas.entry('cost',   struct = cost),
        cas.entry('theta0', struct = model.parameters_dict['theta0'])
    ])

    return P

def setup_integral_output_structure(nlp_numerics_options, integral_outputs):

    nk = nlp_numerics_options['n_k']

    entry_tuple = ()

    # interval outputs
    entry_tuple += (
        cas.entry('int_out', repeat = [nk+1], struct = integral_outputs),
    )

    if nlp_numerics_options['discretization'] == 'direct_collocation':
        d  = nlp_numerics_options['collocation']['d']
        entry_tuple += (
            cas.entry('coll_int_out', repeat = [nk,d], struct = integral_outputs),
        )

    Integral_outputs_struct = cas.struct_symMX([entry_tuple])

    return Integral_outputs_struct

def setup_output_structure(nlp_numerics_options, model_outputs, form_outputs):
    # create outputs
    # n_o_t_e: !!! outputs are defined at nodes where both state and algebraic variables
    # are defined. In the radau case, algebraic variables are not defined on the interval
    # nodes, note however that algebraic variables might be discontinuous so the same
    # holds for the outputs!!!
    nk = nlp_numerics_options['n_k']

    entry_tuple =  ()
    if nlp_numerics_options['discretization'] == 'direct_collocation':

        # extract collocation parameters
        d  = nlp_numerics_options['collocation']['d']
        scheme = nlp_numerics_options['collocation']['scheme']

        if scheme != 'radau':

            # define outputs on interval and collocation nodes
            entry_tuple += (
                cas.entry('outputs',      repeat = [nk],   struct = model_outputs),
                cas.entry('coll_outputs', repeat = [nk,d], struct = model_outputs),
            )

        else:

            # define outputs on collocation nodes
            entry_tuple += (
                cas.entry('coll_outputs', repeat = [nk,d], struct = model_outputs),
            )

    elif nlp_numerics_options['discretization'] == 'multiple_shooting':

        # define outputs on interval nodes
        entry_tuple += (
            cas.entry('outputs', repeat = [nk], struct = model_outputs),
        )

    Outputs = cas.struct_symSX([entry_tuple]
                           + [cas.entry('final', struct=form_outputs)])

    return Outputs

def discretize(nlp_numerics_options, model, formulation):

    # -----------------------------------------------------------------------------
    # discretization setup
    # -----------------------------------------------------------------------------
    nk = nlp_numerics_options['n_k']

    if nlp_numerics_options['discretization'] == 'direct_collocation':
        direct_collocation = True
        ms = False
        d = nlp_numerics_options['collocation']['d']
        scheme = nlp_numerics_options['collocation']['scheme']
        Collocation = collocation.Collocation(nk, d, scheme)
        Multiple_shooting = None

    elif nlp_numerics_options['discretization'] == 'multiple_shooting':
        direct_collocation = False
        ms = True
        Collocation = None
        dae = model.get_dae()
        Multiple_shooting = multiple_shooting.Multiple_shooting(nk, dae, nlp_numerics_options['integrator'])
        slacks = None

    # --------------------------------------
    # prepare model variables and dynamics
    # --------------------------------------
    variables = model.variables
    parameters = model.parameters
    variables_dict = model.variables_dict

    #---------------------------------------
    # prepare constraints structure
    #---------------------------------------
    form_constraints = formulation.constraints
    constraints_fun = formulation.constraints_fun # initial, terminal and periodicity constraints
    path_constraints = model.constraints
    path_constraints_fun = model.constraints_fun
    g_struct = constraints.setup_constraint_structure(nlp_numerics_options, model, formulation) # empty struct for collocated constraints

    #-------------------------------------------
    # DISCRETIZE VARIABLES, CREATE NLP PARAMETERS
    #-------------------------------------------
    V = setup_nlp_v(nlp_numerics_options, model, formulation, Collocation)
    P = setup_nlp_p(V, model)
    if direct_collocation:
        Xdot = Collocation.get_xdot(nlp_numerics_options, V, model)

    # construct time grids for this nlp
    time_grids = construct_time_grids(nlp_numerics_options)

    # ---------------------------------------
    # prepare outputs structure
    # ---------------------------------------
    outputs = model.outputs
    outputs_fun = model.outputs_fun

    [form_outputs, form_outputs_dict] = performance.collect_performance_outputs(nlp_numerics_options, model, V)
    form_outputs_fun = cas.Function('form_outputs_fun', [V, P], [form_outputs.cat])

    Outputs_struct = setup_output_structure(nlp_numerics_options, outputs, form_outputs)

    #-------------------------------------------
    # COLLOCATE CONSTRAINTS, OUTPUTS
    #-------------------------------------------

    # prepare listing of outputs and constraints
    Outputs_list = []
    g_list = []
    g_bounds = {'lb':[], 'ub':[]}

    # extract model.parameters from V
    param_at_time = parameters(cas.vertcat(P['theta0'], V['phi']))
    xi = V['xi']  #TODO: don't hard code!

    # parallellize constraints on collocation nodes
    if direct_collocation:
        [coll_dynamics,
        coll_constraints,
        coll_outputs,
        Integral_outputs_list,
        Integral_constraint_list] = Collocation.collocate_constraints(nlp_numerics_options, model, formulation, V, P, Xdot)

    # parallellize constraints on interval nodes
    if ms:
        [ms_xf,
        ms_z0,
        Xdot,
        ms_constraints,
        ms_outputs,
        Integral_outputs_list,
        Integral_constraint_list] = Multiple_shooting.discretize_constraints(nlp_numerics_options, model, formulation, V, P)

    # Construct list of constraints (+ bounds) and outputs
    for kdx in range(nk):

        # time constant of the following interval
        tf = struct_op.calculate_tf(nlp_numerics_options, V, kdx)

        if kdx == 0:

            # extract initial (reference) variables
            var_initial = struct_op.get_variables_at_time(nlp_numerics_options, V, Xdot, model, 0)
            var_ref_initial = struct_op.get_var_ref_at_time(nlp_numerics_options, P, V, Xdot, model, 0)

            # add initial constraints
            [g_list, g_bounds] = constraints.append_initial_constraints(
                                     g_list, g_bounds, form_constraints, constraints_fun, var_initial, var_ref_initial, xi)

        if (ms) or (direct_collocation and scheme != 'radau'):

            # at each interval node, algebraic constraints should be satisfied
            [g_list, g_bounds] = constraints.append_algebraic_constraints(
                            g_list, g_bounds, dae.z(ms_z0[:,kdx]), V, kdx)

            # at each interval node, path constraints should be satisfied
            if 'us' in list(V.keys()): # slack path constraints
                slacks = V['us',kdx]

            [g_list, g_bounds] = constraints.append_path_constraints(
                                    g_list, g_bounds, path_constraints, ms_constraints[:, kdx],slacks)

            # compute outputs for this time interval
            Outputs_list.append(ms_outputs[:,kdx])

        if direct_collocation:

            # add constraints and outputs on collocation nodes
            for ddx in range(d):

                # at each (except for first node) collocation point dynamics should meet
                [g_list, g_bounds] = constraints.append_collocation_constraints(
                                        g_list, g_bounds, coll_dynamics[:,kdx*d+ddx])

                # at each (except for first node) collocation node, path constraints should be satisfied
                [g_list, g_bounds] = constraints.append_path_constraints(
                                        g_list, g_bounds, path_constraints, coll_constraints[:,kdx*d+ddx])

                # compute outputs for this time interval
                Outputs_list.append(coll_outputs[:,kdx*d+ddx])

            # endpoint should match next start point
            [g_list, g_bounds] = Collocation.append_continuity_constraint(g_list, g_bounds, V, kdx)

        elif ms:

            # endpoint should match next start point
            [g_list, g_bounds] = Multiple_shooting.append_continuity_constraint(g_list, g_bounds, ms_xf, V, kdx)

    # extract terminal (reference) variables
    var_terminal = struct_op.get_variables_at_final_time(nlp_numerics_options, V, Xdot, model)
    var_ref_terminal = struct_op.get_var_ref_at_final_time(nlp_numerics_options, P, Xdot, model)

    # add terminal and periodicity constraints
    [g_list, g_bounds] = constraints.append_terminal_constraints(g_list, g_bounds, form_constraints, constraints_fun, var_terminal, var_ref_terminal, xi)
    [g_list, g_bounds] = constraints.append_periodic_constraints(g_list, g_bounds, form_constraints, constraints_fun, var_initial, var_terminal)

    if direct_collocation:
        [g_list, g_bounds] = constraints.append_integral_constraints(nlp_numerics_options, g_list, g_bounds, Integral_constraint_list,
                                                                            form_constraints, constraints_fun, V, Xdot, model, formulation.integral_constants)

    Outputs_list.append(form_outputs_fun(V, P))

    # Create Outputs struct and function
    Outputs = Outputs_struct(cas.vertcat(*Outputs_list))
    Outputs_fun = cas.Function('Outputs_fun', [V, P], [Outputs.cat])

    [g_list, g_bounds] = constraints.append_wake_fix_constraints(nlp_numerics_options, g_list, g_bounds, V, Outputs, model)
    [g_list, g_bounds] = constraints.append_vortex_strength_constraints(nlp_numerics_options, g_list, g_bounds, V, Outputs, model)

    # Create Integral outputs struct and function
    Integral_outputs_struct = setup_integral_output_structure(nlp_numerics_options, model.integral_outputs)
    Integral_outputs = Integral_outputs_struct(cas.vertcat(*Integral_outputs_list))
    Integral_outputs_fun = cas.Function('Integral_outputs_fun', [V, P], [Integral_outputs.cat])

    # Create g struct and functions and g_bounds vectors
    [g, g_fun, g_jacobian_fun, g_bounds] = constraints.create_constraint_outputs(g_list, g_bounds, g_struct, V, P)

    Xdot_struct = struct_op.construct_Xdot_struct(nlp_numerics_options, model)
    Xdot_fun = cas.Function('Xdot_fun',[V],[Xdot])

    return V, P, Xdot_struct, Xdot_fun, g_struct, g_fun, g_jacobian_fun, g_bounds, Outputs_struct, Outputs_fun, Integral_outputs_struct, Integral_outputs_fun, time_grids, Collocation, Multiple_shooting

def get_phase_fix_theta(variables_dict):

    entry_list = []
    for name in list(variables_dict['theta'].keys()):
        if name == 't_f':
            entry_list.append(cas.entry('t_f', shape = (2,1)))
        else:
            entry_list.append(cas.entry(name, shape = variables_dict['theta'][name].shape))

    theta = cas.struct_symSX(entry_list)

    return theta
