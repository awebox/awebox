
import awebox as awe
import awebox.mdl.model as model
import awebox.mdl.architecture as archi
import casadi as ca
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.opts.options as opts

EXTERNAL_FORCES = True

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options_seed = {}
options_seed['user_options.system_model.architecture'] = {1:0}
options_seed = set_ampyx_ap2_settings(options_seed)
options_seed['user_options.trajectory.system_type'] = 'lift_mode'

# indicate desired environment
options_seed['params.wind.z_ref'] = 100.0
options_seed['params.wind.power_wind.exp_ref'] = 0.15
options_seed['user_options.wind.model'] = 'power'
options_seed['user_options.wind.u_ref'] = 10.

# flag for external forces
if EXTERNAL_FORCES:
    options_seed['model.aero.fictitious_embedding'] = 'substitute'

options = opts.Options()
options.fill_in_seed(options_seed)
architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
options.build(architecture)

system_model = model.Model()
system_model.build(options['model'], architecture)
scaling = system_model.scaling
dae = system_model.get_dae()

# fill in system design parameters
p0 = dae.p(0.0)

# fill in tether diameter (and possible other parameters)
theta = system_model.variables_dict['theta'](0.0)
theta['diam_t'] = 2e-3 / scaling['theta']['diam_t']
theta['t_f'] = 1.0
p0['theta'] = theta.cat

# get numerical parameters
params = system_model.parameters(0.0)

#f ill in numerical parameters
param_num = options['model']['params']
for param_type in list(param_num.keys()):
    if isinstance(param_num[param_type],dict):
        for param in list(param_num[param_type].keys()):
            if isinstance(param_num[param_type][param],dict):
                for subparam in list(param_num[param_type][param].keys()):
                    params['theta0',param_type,param,subparam] = param_num[param_type][param][subparam]

            else:
                params['theta0',param_type,param] = param_num[param_type][param]
    else:
        params['theta0',param_type] = param_num[param_type]

if EXTERNAL_FORCES:
    params['phi', 'gamma'] = 1

p0['param'] = params

# make casadi collocation integrator
ts = 0.001 # sampling time
int_opts = {}
int_opts['tf'] = ts
int_opts['number_of_finite_elements'] = 40
int_opts['collocation_scheme'] = 'radau'
int_opts['interpolation_order'] = 4
int_opts['rootfinder'] = 'fast_newton'
integrator = ca.integrator('integrator', 'collocation', dae.dae, int_opts)

# simulate system (TBD)
# u0 = system_model.variables_dict['u'](0.0) # initialize controls
# x0 = ... / scaling['x']['q10'] # initialize system state (scaled!)
# z0 = dae.z(0.0) # algebraic variables initial guess

# for k in range(Nsim):

#     # fill in controls
#     u0['f_fict10'] = ... / scaling['u']['f_fict10'] # external force
#     u0['m_fict10'] = ... / scaling['u']['f_fict10'] # external moment
#     u0['ddl_t'] = ... / scaling['u']['ddl_t'] # tether acceleration control (scaled)
#     u0['ddelta10'] = 0.0 / scaling['u']['ddelta10'] # not relevant in case of external forces (scaled)
    
#     # fill controls into dae parameters
#     p0['u'] = u0

#     # if desired, change model parameter (e.g. wind speed, relevant for tether drag)
#     params['theta0', 'wind', 'u_ref'] = 10.0
#     p0['param'] = params

#     # evaluate integrator
#     out = integrator(x0 = x0, p = p0, z0 = z0)
#     z0 = out['zf']
#     x0 = out['xf']
