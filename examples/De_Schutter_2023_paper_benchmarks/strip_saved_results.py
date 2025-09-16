import pickle

def strip_homotopy(homotopy):
    x_stripped = {}
    for x in list(homotopy['x'].keys()):
        x_stripped[x] = [homotopy['x'][x][0], homotopy['x'][x][-1]]
    t_stripped = {}
    for t in list(homotopy['t'].keys()):
        t_stripped[t] = [homotopy['t'][t][0], homotopy['t'][t][-1]]
    u_stripped = {}
    for u in list(homotopy['u'].keys()):
        u_stripped[u] = [homotopy['u'][u][0], homotopy['u'][u][-1]]
    z_stripped = {}
    for z in list(homotopy['z'].keys()):
        z_stripped[z] = [homotopy['z'][z][0], homotopy['z'][z][-1]]

    stripped = {
        'phi': homotopy['phi'],
        'x': x_stripped,
        'u': u_stripped,
        'z': z_stripped,
        'theta': homotopy['theta'],
        'P': homotopy['P'],
        'success': homotopy['success'],
        't': t_stripped,
        't_wall': homotopy['t_wall'],
        'iterations': homotopy['iterations'],
        'cost': homotopy['cost']
    }
    return stripped

def save_homotopy(trial, save_name, folder = 'results'):
    cb = trial.optimization.awe_callback
    if not trial.options['solver']['record_states']:
        cb = fill_in_cb(cb, trial.optimization.V_init, trial.model, trial.nlp)
        cb = fill_in_cb(cb, trial.optimization.V_final, trial.model, trial.nlp)

    homotopy = {
        'phi': cb.phi_dict,
        'x': cb.x_dict,
        'z': cb.z_dict,
        'u': cb.u_dict,
        'theta': cb.theta_dict,
        't': cb.t_dict,
        'P': [0, trial.visualization.plot_dict['power_and_performance']['avg_power']],
        'lam_x': cb.lam_x_dict,
        'lam_g': cb.lam_g,
        'g_dict': cb.g_dict,
        'lbg': trial.optimization.arg['lbg'],
        'success': trial.optimization.return_status_numeric['optimization'],
        't_wall': trial.optimization.t_wall,
        'iterations': trial.optimization.iterations,
        'cost': cb.cost_dict
    }
    pickle.dump(homotopy, open('{}/{}.p'.format(folder, save_name),'wb'))
    pickle.dump(strip_homotopy(homotopy), open('{}/{}_stripped.p'.format(folder, save_name),'wb'))

def fill_in_cb(cb, V, model, nlp):
    for x in list(model.variables_dict['x'].keys()):
        for dim in range(model.variables_dict['x'][x].shape[0]):
            cb.x_dict[x+'_'+str(dim)].append(cb.extract_x_vals(V, x, dim))
    
    for u in list(model.variables_dict['u'].keys()):
        for dim in range(model.variables_dict['u'][u].shape[0]):
            cb.u_dict[u+'_'+str(dim)].append(cb.extract_u_vals(V, u, dim))

    for z in list(model.variables_dict['z'].keys()):
        for dim in range(model.variables_dict['z'][z].shape[0]):
            cb.z_dict[z+'_'+str(dim)].append(cb.extract_z_vals(V, z, dim))

    for theta in list(model.variables_dict['theta'].keys()):
        for dim in range(model.variables_dict['theta'][theta].shape[0]):
            cb.theta_dict[theta+'_'+str(dim)].append(V['theta',theta, dim])

    for t in list(cb.t_dict.keys()):
        cb.t_dict[t].append(nlp.time_grids[t](V['theta','t_f']))

    return cb
