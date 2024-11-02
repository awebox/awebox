import casadi as cd
import awebox.opts.kite_data.ampyx_data as ampyx_data
import awebox as awe
import awebox.trial as awe_trial

def test_parallelization():
    N = 300
    x = cd.MX.sym('x'); z = cd.MX.sym('z'); p = cd.MX.sym('p')
    dae = {'x': x, 'z': z, 'p': p, 'ode': 0, 'alg': z}
    func = cd.integrator('func', 'idas', dae)
    F = func.map(N, 'thread')
    F(x0=0, z0=0, p=0)

def get_basic_trial_options():
    trial_options = {}
    trial_options['user_options.system_model.architecture'] = {1: 0}
    trial_options['user_options.system_model.kite_dof'] = 3
    trial_options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    trial_options['user_options.trajectory.system_type'] = 'lift_mode'
    trial_options['user_options.trajectory.lift_mode.windings'] = 1
    trial_options['model.tether.aero_elements'] = 1
    trial_options['user_options.induction_model'] = 'not_in_use'
    trial_options['nlp.collocation.u_param'] = 'zoh'
    trial_options['nlp.n_k'] = 5
    trial_options['solver.max_iter'] = 2
    return trial_options

def test_parallel_model_generation():
    trial_options = get_basic_trial_options()
    for parallelization_type in ['serial', 'concurrent_futures']:
        trial_options['model.construction.parallelization.type'] = parallelization_type
        trial_name = parallelization_type
        trial = awe_trial.Trial(trial_options, trial_name)
        trial.build()
        trial.optimize(final_homotopy_step='initial')

def test_parallel_nlp_generation():
    trial_options = get_basic_trial_options()
    for parallelization_type in ['serial', 'concurrent_futures']:
        trial_options['nlp.parallelization.type'] = parallelization_type
        trial_name = parallelization_type
        trial = awe_trial.Trial(trial_options, trial_name)
        trial.build()
        trial.optimize(final_homotopy_step='initial')


def test_parallel_solver_generation():
    trial_options = get_basic_trial_options()
    for parallelization_type in ['serial', 'multiprocessing_pool']:
        trial_options['solver.construction.parallelization.type'] = parallelization_type
        trial_name = parallelization_type
        trial = awe_trial.Trial(trial_options, trial_name)
        trial.build()
        trial.optimize(final_homotopy_step='initial')

if __name__ == "__main__":
    test_parallelization()
    test_parallel_model_generation()
    test_parallel_nlp_generation()
    test_parallel_solver_generation()
