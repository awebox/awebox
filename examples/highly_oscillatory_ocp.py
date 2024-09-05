#!/usr/bin/python3
"""
Circular pumping trajectory for the Ampyx AP2 aircraft.
"""

import awebox as awe
import awebox.opts.options as awe_opts
import awebox.mdl.model as awe_mdl
import awebox.mdl.architecture as awe_archi
import awebox.ocp.formulation as awe_formulation
import awebox.ocp.highly_oscillatory_nlp as awe_HONlp
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_ampyx_ap2_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# initialize awebox objects
opts = awe_opts.Options()
opts.fill_in_seed(options)
model = awe_mdl.Model()
formulation = awe_formulation.Formulation()
HONlp = awe_HONlp.HighlyOscillatoryNLP()

# build nlp
architecture = awe_archi.Architecture(opts['user_options']['system_model']['architecture'])
opts.build(architecture)
model.build(opts['model'], architecture)
formulation.build(opts['formulation'], model)
HONlp.build(opts['nlp'], model, formulation)
