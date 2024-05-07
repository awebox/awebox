"""
Second, dual-aircraft case study as described in the paper

De Schutter, J.; Leuthold, R.C.; Bronnenmeyer, T.; Malz, E.; Gros, S.; Diehl, M. 
"AWEbox: an Optimal Control Framework for Single- and Multi-Aircraft
Airborne Wind Energy Systems". 
Preprints 2022, 2022120018 (doi: 10.20944/preprints202212.0018.v1).

:author: Jochem De Schutter
"""

import awebox as awe
import reference_options as ref
import pickle
import copy

if __name__ == "__main__":


    HOMOTOPY = 'SIPH' # 'SIPH' / 'PIPH'
    results_folder = 'results_cs2_{}/'.format(HOMOTOPY)

    import pathlib
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True) 

    # set reference options
    options = ref.set_reference_options(user = 'A')
    options = ref.set_dual_kite_options(options)

    # choose homotopy method
    options['solver.homotopy_method.type'] = 'scheduled'
    options['solver.homotopy_method.gamma'] = 'penalty'
    options['solver.homotopy_method.psi'] = 'penalty'
    options['solver.max_iter'] = 2000
    options['visualization.cosmetics.interpolation.N'] = 2
    options['solver.max_iter_hippo'] = 100

    # build trial
    trial = awe.Trial(options)
    trial.build()

    # solve for optimal design
    timings = {}
    trial.optimize(options_seed = options, intermediate_solve = True)
    timings['u_ref10_intermediate'] = copy.deepcopy(trial.optimization.t_wall)
    intermediate_sol_design = copy.deepcopy(trial.solution_dict)
    trial.optimize(options_seed = options, warmstart_file = intermediate_sol_design, intermediate_solve=False, recalibrate_viz = False)
    timings['u_ref10_final'] = copy.deepcopy(trial.optimization.t_wall)
    trial.write_to_csv(file_name = results_folder + 'sweep_u_ref10', frequency=10., rotation_representation='dcm')

    # define wind speed sweep
    u_ref_up = list(range(10,21))
    u_ref_down = list(range(0,11))
    u_ref_down.reverse()

    u_ref_tot = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]

    # fix params
    fixed_params = {}
    for theta in trial.model.variables_dict['theta'].keys():
        if theta != 't_f':
            fixed_params[theta] = trial.optimization.V_final['theta', theta].full()[0][0]
    options['user_options.trajectory.fixed_params'] = fixed_params

    print('Fixing optimal design parameters:')
    print(fixed_params)

    intermediate_sol = None # init

    if HOMOTOPY == 'SIPH':
        
        # SIPH sweep
        hom_steps = 10
        for idx, u_ref in enumerate(u_ref_up):

            if idx > 0:

                intermediate_time = 0.0

                for jdx in range(1, hom_steps+1):

                    step = jdx/hom_steps
                    options['user_options.wind.u_ref'] = (1-step)*u_ref_up[idx-1] + step*u_ref

                    print('================================')
                    print('Solve trial for u_ref = {} m/s'.format(options['user_options.wind.u_ref']))
                    print('================================')

                    if idx == 1 and jdx == 1:
                        trial.optimize(options_seed = options, warmstart_file = intermediate_sol_design, intermediate_solve = True)
                    else:
                        trial.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve = True)

                    intermediate_time += copy.deepcopy(trial.optimization.t_wall['optimization'])
                    timings['u_ref{}_intermediate_status'.format(round(u_ref))] = copy.deepcopy(trial.optimization.return_status_numeric['optimization'])
                    intermediate_sol = copy.deepcopy(trial.solution_dict)

                timings['u_ref{}_intermediate'.format(round(u_ref))] = intermediate_time

                trial.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=False, recalibrate_viz=False)
                trial.write_to_csv(file_name = results_folder + 'sweep_u_ref{}'.format(round(u_ref)), frequency=10., rotation_representation='dcm')
                timings['u_ref{}_final'.format(round(u_ref))] = copy.deepcopy(trial.optimization.t_wall['optimization'])
                timings['u_ref{}_final_status'.format(round(u_ref))] = copy.deepcopy(trial.optimization.return_status_numeric['optimization'])

        # perform sweep
        for idx, u_ref in enumerate(u_ref_down):

            if idx > 0:

                intermediate_time = 0.0

                for jdx in range(1, hom_steps+1):

                    step = jdx/hom_steps
                    options['user_options.wind.u_ref'] = (1-step)*u_ref_down[idx-1] + step*u_ref

                    print('================================')
                    print('Solve trial for u_ref = {} m/s'.format(options['user_options.wind.u_ref']))
                    print('================================')

                    if idx == 1 and jdx == 1:
                        trial.optimize(options_seed = options, warmstart_file = intermediate_sol_design, intermediate_solve = True)
                    else:
                        trial.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve = True)

                    intermediate_time += copy.deepcopy(trial.optimization.t_wall['optimization'])
                    timings['u_ref{}_intermediate_status'.format(round(options['user_options.wind.u_ref']))] = copy.deepcopy(trial.optimization.return_status_numeric['optimization'])
                    intermediate_sol = copy.deepcopy(trial.solution_dict)

                timings['u_ref{}_intermediate'.format(round(u_ref))] = intermediate_time

                trial.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=False, recalibrate_viz = False)
                trial.write_to_csv(file_name = results_folder + 'sweep_u_ref{}'.format(round(u_ref)), frequency=10., rotation_representation='dcm')
                timings['u_ref{}_final'.format(round(u_ref))] = copy.deepcopy(trial.optimization.t_wall['optimization'])
                timings['u_ref{}_final_status'.format(round(u_ref))] = copy.deepcopy(trial.optimization.return_status_numeric['optimization'])

        pickle.dump(timings, open(results_folder+'u_ref_sweep_timings.p','wb'))

    elif HOMOTOPY == 'PIPH':

        # PIPH sweep
        for idx, u_ref in enumerate(u_ref_tot):

            options['user_options.wind.u_ref'] = u_ref
            trial.optimize(options_seed = options, recalibrate_viz=False)
            trial.write_to_csv(file_name = results_folder + 'sweep_u_ref{}'.format(round(u_ref)), frequency=10., rotation_representation='dcm')
            timings['u_ref{}_final'.format(round(u_ref))] = copy.deepcopy(trial.optimization.t_wall['optimization'])
            timings['u_ref{}_final_status'.format(round(u_ref))] = copy.deepcopy(trial.optimization.return_status_numeric['optimization'])

        pickle.dump(timings, open(results_folder+'u_ref_sweep_timings_extra.p','wb'))
