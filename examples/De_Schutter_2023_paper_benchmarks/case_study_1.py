"""
First, single-aircraft case study as described in the paper

De Schutter, J.; Leuthold, R.C.; Bronnenmeyer, T.; Malz, E.; Gros, S.; Diehl, M. 
"AWEbox: an Optimal Control Framework for Single- and Multi-Aircraft
Airborne Wind Energy Systems". 
Preprints 2022, 2022120018 (doi: 10.20944/preprints202212.0018.v1).

:author: Jochem De Schutter
"""

import awebox as awe
import reference_options as ref
import numpy as np
import pickle
import strip_saved_results as strip
import random

def get_init_scenarios(user = 'A'):
    
    init_bounds = {
        'flight_speed':    [20.0, 60.0],
        'tether_length':   [300.0, 600.0],
        'elevation_angle': [15.0, 35.0],
        'cone_angle':      [10.0, 30.0],
        'phase_angle':     [0, 2*np.pi],
        'tether_diameter': [1e-3, 5.0e-3]
    }

    if user == 'B':
        opt_params = pickle.load(open('./opt_params_final.p','rb'))
        opt_params['elevation_angle'] = np.rad2deg(opt_params['elevation_angle'])
        opt_params['cone_angle'] = np.rad2deg(opt_params['cone_angle'])
        init_bounds_B = {}
        factor = 3 # reduce range
        for i, (k,v) in enumerate(init_bounds.items()):
            offset = (v[1]-v[0])/(2*factor)
            init_bounds_B[k] = [max(opt_params[k]-offset, v[0]), min(opt_params[k]+offset, v[1])]
        init_bounds = init_bounds_B

    random.seed(121)

    init_scenarios = []
    for k in range(100):
        init_scenarios.append([
            ('inclination_deg', random.uniform(init_bounds['elevation_angle'][0], init_bounds['elevation_angle'][1])),
            ('l_t', random.uniform(init_bounds['tether_length'][0], init_bounds['tether_length'][1])),
            ('theta.diam_t', random.uniform(init_bounds['tether_diameter'][0], init_bounds['tether_diameter'][1])),
            ('cone_deg', random.uniform(init_bounds['cone_angle'][0], init_bounds['cone_angle'][1])),
            ('groundspeed', random.uniform(init_bounds['flight_speed'][0], init_bounds['flight_speed'][1])),
            ('psi0_rad', random.uniform(init_bounds['phase_angle'][0], init_bounds['phase_angle'][1]))
        ])

    return init_scenarios

def get_save_name(options, homotopy_method, init_scenarios, init):

    # construct save name
    kites = str(len(options['user_options.system_model.architecture'].keys()))
    tau = str(-int(np.log10(options['solver.mu_hippo'])))
    nk = str(options['nlp.n_k'])
    hs = str(int(options['solver.homotopy_step.gamma']*10))
    init = init_scenarios.index(init)
    save_name = '{}_k{}_nk{}_tau{}_hs{}_ni_split_init{}'.format(homotopy_method, kites, nk, tau, hs, init)

    return save_name

if __name__ == "__main__":

    # dual kite sim
    USER = 'A' # 'A' / 'B'
    # set reference options
    options = ref.set_reference_options(user = USER)

    results_folder = 'results_cs1_{}/'.format(USER)

    import pathlib
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True) 

    trial = awe.Trial(options)
    trial.build()

    # solve reference solution
    for homotopy_method in ['NH', 'CIPH', 'PIPH']:

        if homotopy_method == 'CIPH':
            trial.options['solver']['homotopy_method']['type'] = 'scheduled'
            trial.options['solver']['homotopy_method']['gamma'] = 'classic'
            trial.options['solver']['homotopy_method']['psi'] = 'classic'
        elif homotopy_method == 'PIPH':
            trial.options['solver']['homotopy_method']['type'] = 'scheduled'
            trial.options['solver']['homotopy_method']['gamma'] = 'penalty'
            trial.options['solver']['homotopy_method']['psi'] = 'penalty'
        elif homotopy_method == 'PCIPH':
            trial.options['solver']['homotopy_method']['type'] = 'scheduled'
            trial.options['solver']['homotopy_method']['gamma'] = 'penalty'
            trial.options['solver']['homotopy_method']['psi'] = 'classic'
        elif homotopy_method == 'SIPH':
            trial.options['solver']['homotopy_method']['type'] = 'single'
        elif homotopy_method == 'NH':
            options['solver.homotopy_method.type'] = 'scheduled'

        save_name = '{}_ref'.format(homotopy_method)
        if homotopy_method != 'NH':
            # optimize
            trial.optimize()
            strip.save_homotopy(trial, save_name, folder = results_folder)
        else:
            options = ref.set_reference_options(user = USER)
            options['solver.mu_hippo'] = 1e0
            trial_raw = awe.Trial(options)
            trial_raw.build()
            trial_raw.optimize(final_homotopy_step = 'initial_guess')
            trial_raw.solution_dict['final_homotopy_step'] = 'final'
            trial_raw.optimize(options_seed = options, warmstart_file = trial_raw.solution_dict)
            strip.save_homotopy(trial_raw, save_name, folder = results_folder)

    # perform initialization sweep
    init_scenarios = get_init_scenarios(USER)
    for scenario in init_scenarios:

        for homotopy_method in  ['NH', 'CIPH', 'PIPH']:

            # reset options
            options = ref.set_reference_options(user = USER)

            # switch to classical continuation if applicable
            if homotopy_method == 'CIPH':
                options['solver.homotopy_method.type'] = 'scheduled'
                options['solver.homotopy_method.gamma'] = 'classic'
                options['solver.homotopy_method.psi'] = 'classic'
            elif homotopy_method == 'PIPH':
                trial.options['solver']['homotopy_method']['type'] = 'scheduled'
                trial.options['solver']['homotopy_method']['gamma'] = 'penalty'
                trial.options['solver']['homotopy_method']['psi'] = 'penalty'
            elif homotopy_method == 'PCIPH':
                options['solver.homotopy_method.type'] = 'scheduled'
                options['solver.homotopy_method.gamma'] = 'penalty'
                options['solver.homotopy_method.psi'] = 'classic'
            elif homotopy_method == 'SIPH':
                options['solver.homotopy_method.type'] = 'single'
            elif homotopy_method == 'NH':
                options['solver.homotopy_method.type'] = 'scheduled'


            # update initialization scenario
            for k in range(len(scenario)):
                options['solver.initialization.{}'.format(scenario[k][0])] = scenario[k][1]

            # save name
            save_name = get_save_name(options, homotopy_method, init_scenarios, scenario)

            # build and optimize trial
            if homotopy_method != 'NH':
                trial.optimize(options)
                # save solution
                strip.save_homotopy(trial, save_name, folder = results_folder)
            else:
                trial_raw.optimize(options, final_homotopy_step = 'initial_guess')
                trial_raw.solution_dict['final_homotopy_step'] = 'final'
                trial_raw.optimize(options_seed = options, warmstart_file = trial_raw.solution_dict)

                # save solution
                strip.save_homotopy(trial_raw, save_name, folder = results_folder)
