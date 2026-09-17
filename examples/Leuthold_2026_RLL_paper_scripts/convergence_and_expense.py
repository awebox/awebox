#!/usr/bin/python3

import matplotlib
matplotlib.use("Agg")   # ← MUST be here, before pyplot

import awebox.trial as awe_trial

from examples.Leuthold_2026_RLL_paper_scripts import helpful_operations as help_op

from awebox.logger.logger import Logger as awelogger

awelogger.logger.setLevel(10)


base_name = 'vtn'

def run(inputs={}, final_homotopy_step='final'):

    # basic options
    options = {}
    options = help_op.get_basic_options_for_convergence_expense_and_comparison(options)

    # allow a reduction of the problem for testing purposed
    if 'nlp.n_k' in inputs.keys():
        n_k = inputs['nlp.n_k']
    else:
        n_k = options['nlp.n_k']
    periods_tracked = inputs['periods_tracked']
    
    wake_nodes = help_op.from_periods_tracked_to_wake_nodes(n_k, periods_tracked)
    options['model.aero.vortex.wake_nodes'] = wake_nodes

    if ('solver.hippo_strategy' in inputs.keys()) and (inputs['solver.hippo_strategy'] == False):
        inputs['solver.mu_hippo'] = 1e-2
        inputs['solver.hippo_strategy'] = False

    for name, val in inputs.items():
        if '.' in name:
            options[name] = inputs[name]

    # visualization
    options['visualization.cosmetics.save_figs'] = True
    options['visualization.cosmetics.save.format_list'] = ['pdf']
    options['visualization.cosmetics.animation.snapshot_index'] = -1
    options['visualization.cosmetics.trajectory.body_cross_sections_per_meter'] = 10 / options['user_options.kite_standard']['geometry']['b_ref']
    options['visualization.cosmetics.trajectory.wake_nodes'] = True
    options['visualization.cosmetics.trajectory.kite_aero_dcm'] = True
    options['visualization.cosmetics.trajectory.trajectory_rotation_dcm'] = True
    options['visualization.cosmetics.variables.si_or_scaled'] = 'si'
    options['visualization.cosmetics.trajectory.kite_bodies'] = True
    options['visualization.cosmetics.trajectory.reel_in_linestyle'] = '--'  
    options['visualization.cosmetics.trajectory.temporal_epigraph_length_to_span'] = 5.
    
    options['model.aero.vortex.induction_factor_normalizing_speed'] = 'u_ref'  
    options['model.aero.actuator.normal_vector_model'] = 'dual'
    options['visualization.cosmetics.temporal_epigraph_locations'] = [0.32, 'switch', 1.0] 
        
    options['model.scaling.other.print_help_with_scaling'] = True
    
    
    if ('basic_health_variant' in inputs.keys()) and (inputs['basic_health_variant'] == True):
        options = help_op.toggle_basic_health_options(options)
    
    
    # build trial and optimize
    trial_name_vortex = help_op.build_unique_trial_name(base_name, inputs)
    trial_vortex = awe_trial.Trial(options, trial_name_vortex)
    trial_vortex.build()

    trial_vortex.optimize(final_homotopy_step=final_homotopy_step)
    
    latex_dict = {'stab_derivs':
                    {'0': r'0',
                    'alpha': r'\AngleOfAttack',
                    'beta': r'\Sideslip',
                    'deltaa': r'\AileronAngle',
                    'deltae': r'\ElevatorAngle',
                    'deltar': r'\RudderAngle',
                    'p': r'\rollRate',
                    'q': r'\pitchRate',
                    'r': r'\yawRate',
                    'CL': r'\CL',
                    'CD': r'\CD',
                    'CS': r'\CS',
                    'CX': r'\CX',
                    'CY': r'\CY',
                    'CZ': r'\CZ',
                    'Cl': r'\Cl',
                    'Cm': r'\Cm',
                    'Cn': r'\Cn',
                    },
                  'model_var_bounds':
                    {'theta.diam_t': r'\MainTetherDiameter',
                     'theta.t_f': r'\OptimizationPeriod',
                     'x.l_t': r'\MainTetherLength',
                     'x.dl_t': r'\MainTetherSpeed',
                     'u.ddl_t': r'\MainTetherAcceleration',
                     'x.q': r'\NodePosition',
                     'x.dq': r'\NodeVelocity',
                     'x.omega': r'\KiteAngularVelocity',
                     'x.delta': r'\KiteControlSurfaceDeflection',
                     'z.lambda': r'\NodeTensionPerLength',
                     'u.ddelta': r'\KiteControlSurfaceDeflectionRate',
                    },
                  'model_ineq_bounds':
                      {
                        'tether_force_max': r'\UpperBound{\TensionForce}',
                        'tether_force_min': r'\LowerBound{\TensionForce}',
                        'airspeed_max': r'\UpperBound{{\AirSpeed_\eff}}',
                        'airspeed_min': r'\LowerBound{{\AirSpeed_\eff}}',
                        'alpha_ub': r'\UpperBound{\AngleOfAttack}',
                        'alpha_lb': r'\LowerBound{\AngleOfAttack}',
                        'beta_ub': r'\UpperBound{\SideSlip}',
                        'beta_lb': r'\LowerBound{\SideSlip}',
                        'rotation_max': r'\UpperBound{\yawAngle}'
                      },
                  'environment':
                      {
                        't_ref': r'\Reference{\AirTemperature}',
                        'gamma_air': r'\AirTemperatureGradient',
                        'rho_ref': r'\Reference{\AirDensity}',
                        'g': r'\GravityAcceleration',
                        'r': r'\SpecificGasConstant',
                        'gamma': r'\AirPolytropic',
                        'mu_ref': r'\Reference{\AirDynamicViscosity}',
                        'c_sutherland': r'\SutherlandConstant',
                          'u_ref': r'\ReferenceWindSpeed',
                          'z_ref': r'\WindReferenceHeight',
                          'exp_ref': r'\PowerWindRoughnessExponent'
                     },
                  'model_dimensions':
                      {
                          'nx': r'\NumberOfStates',
                          'nu': r'\NumberOfControls',
                          'nz': r'\NumberOfAlgebraics',
                          'np_var': r'\NumberOfParameters',
                          'np_fix': r'\NumberOfPassedOptionParameters'
                      },
                  'kite':
                        {'kite_dof': r' $\DOF$ ',
                        'm_k': r'\KiteMass',
                        'j': r'\KiteMomentOfInertia',
                        's_ref': r'\PlanformArea',
                        'b_ref': r'\Wingspan',
                        'c_ref': r'\MAC'
                         }
                  }
    trial_vortex.make_report(to_echo_or_latex='latex', latex_dict=latex_dict, save=True)
    trial_vortex.print_cost_information()
    help_op.save_results_including_figures(trial_vortex, options)

    return None

    
def call_by_pt(n_k, pt, inputs={}, final_homotopy_step='final'):
    import gc
    from glob import glob
    print('n_k: ' + str(n_k) + '; pt: ' + str(pt))

    if pt > -1e-10:
        inputs['nlp.n_k'] = n_k
        inputs['periods_tracked'] = pt

        trial_name = help_op.build_unique_trial_name(base_name, inputs)
        if not glob('*' + trial_name + '*'):
            trial = run(inputs, final_homotopy_step=final_homotopy_step)
            del trial
        gc.collect()
        
    return None


def call_by_wake_nodes(n_k, wake_nodes, inputs={}, final_homotopy_step='final'):
    pt = help_op.from_wake_nodes_to_periods_tracked(n_k, wake_nodes)
    call_by_pt(n_k, pt, inputs, final_homotopy_step=final_homotopy_step)
    return None


def call_by_memory(n_k, memory_gb, inputs={}, final_homotopy_step='final'):

    if 'nlp.collocation.d' in inputs.keys():
        collocation_d = inputs['nlp.collocation_d']
    else:
        collocation_d = 4
    
    # curve fit for memory [GB]: 3.56715 + 0.00121953 V
    aa = 5.79095
    bb = 0.0010823
    def estimate_periods_tracked(aa, bb, n_k, d, kites, mem_gb):
        p1 = -aa - 3. * bb * (1. + n_k + d * n_k) * kites + mem_gb
        p2 = 3. * bb * n_k * (1. + n_k + d * n_k) * kites
        pt = p1 / p2
        return pt

    pt = estimate_periods_tracked(aa, bb, n_k, collocation_d, 2, memory_gb)
    call_by_pt(n_k, pt, inputs, final_homotopy_step=final_homotopy_step)
    return None
    

if __name__ == "__main__":
    n_k = 5
    
    #total_memory_gb = 30
    #target_memory_gb = 0.3 * total_memory_gb
    #call_by_memory(n_k, target_memory_gb)

    inputs = {'visualization.cosmetics.induction.n_points_contour':5}
    call_by_wake_nodes(n_k, 2, inputs=inputs, final_homotopy_step='initial_guess')
    #call_by_pt(n_k, 0.01)
