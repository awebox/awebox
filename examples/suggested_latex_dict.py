
def get_suggested_latex_dictionary():
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
                     'theta.diam_s': r'\SecondaryTetherDiameter',
                     'theta.l_s': r'\SecondaryTetherLength',
                     'theta.t_f': r'\OptimizationPeriod',
                     'x.l_t': r'\MainTetherLength',
                     'x.dl_t': r'\MainTetherSpeed',
                     'x.ddl_t': r'\MainTetherAcceleration',
                     'u.ddl_t': r'\MainTetherAcceleration',
                     'x.q': r'\NodePosition',
                     'x.dq': r'\NodeVelocity',
                     'x.omega': r'\KiteAngularVelocity',
                     'x.delta': r'\KiteControlSurfaceDeflection',
                     'z.lambda': r'\NodeTensionPerLength',
                     'u.ddelta': r'\KiteControlSurfaceDeflectionRate',
                     'u.dddl_t': r'\MainTetherJerk'
                    },
                  'model_ineq_bounds':
                      {
                        'tether_force_max': r'\UpperBound{\TensionForce}',
                        'tether_force_min': r'\LowerBound{\TensionForce}',
                        'tether_stress': r'{{}\SafetyFactor}_{\Stress}',
                        'acceleration': r'{\factor}_{\acceleration}$',
                        'airspeed_max': r'\UpperBound{{\AirSpeed_\eff}}',
                        'airspeed_min': r'\LowerBound{{\AirSpeed_\eff}}',
                        'alpha_ub': r'\UpperBound{\AngleOfAttack}',
                        'alpha_lb': r'\LowerBound{\AngleOfAttack}',
                        'beta_ub': r'\UpperBound{\SideSlip}',
                        'beta_lb': r'\LowerBound{\SideSlip}',
                        'rotation_roll': r'\UpperBound{\abs{\rollAngle}}',
                        'rotation_pitch': r'\UpperBound{\abs{\pitchAngle}}',
                        'rotation_yaw': r'\UpperBound{\abs{\yawAngle}}',
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
                        'z0_air': r'\LogWindRoughnessLength',
                        'exp_ref': r'\PowerWindRoughnessExponent',
                        'p_ref': r'\Reference{\AirPressure}'
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
                        'c_ref': r'\MAC',
                        'ar': r'\AR'
                         },
                  'tether':
                      {'cd_model': r' $coef. model$ ',
                       'cd': r'\CD',
                       'tether_drag_model': r' $drag model$ ',
                       'rho': r'\TetherDensity',
                       'control_var': r'\control_\tether',
                       'aero_elements': r'\NumberOfElements',
                       'max_stress': r'\TetherYieldStress'
                       }
                  }
    return latex_dict