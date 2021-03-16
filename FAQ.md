# `awebox` FAQ


### How do I warmstart from a previous run?

1. make sure that your previous run was saved in .dict format as "trialname.dict"
in your run file

2. allow pickling
```
import pickle
```

3. load the previous run:
```
warmstart_dict = open('/home/usr/...path_to_file.../trialname.dict', "rb")
warmstart_file = pickle.load(warmstart_dict)
```

4. decrease the interior-point homotopy parameter 'mu', so that the hippo strategy doesn't relax mu too far away from the loaded solution:
```
options['solver']['mu_hippo'] = 1.e-6
```

5. include warmstart_file information into the trial.optimize() call:
```
trial.optimize(warmstart_file = warmstart_file)
```

### How do I access a part of the solved optimization variables, such as the system parameters theta?

```
theta = trial.optimization.V_final['theta']
```

### How do I access the solved performance metrics, like the power harvesting factor 'zeta' or the average power ?

    plot_dict = trial.visualization.plot_dict
    zeta = plot_dict['power_and_performance']['zeta']
    avg_power = plot_dict['power_and_performance']['avg_power']



### How do I get the IPOPT verbose output to print during the optimization process?

include the following into the preamble of the run file: 

```
from awebox.logger.logger import Logger as awelogger
awelogger.logger.setLevel(10)
```

### The sub-problems of the homotopy converge nicely until we get to the power problems. what should I do?

You may need to re-tune the scaling values for the tether tensions, the energy, and the power. To do this:

1. first, cap the number of iterations at a small value:
```
options['solver']['max_iter'] = 250
```

2. second, create a sweep over some new scaling values: 
```
sweep_opts = [
    (['model', 'scaling_overwrite', 'xa', 'lambda'], [1.e2, 1.e3, 1.e4]),
    (['model', 'scaling_overwrite', 'xd', 'e'], [1.e3, 1.e4, 1.e5]),
    (['solver', 'cost_overwrite', 'power', 1], [1.e-2, 1.e-1, 1.e0])
]
```

3. run the tuning sweep (with the same "options" that you've previously defined), and plotting the convergence and solution comparisons: 
```
sweep = awe.Sweep(name = 'tuning_sweep', options = options, seed = sweep_opts)
sweep.run()
sweep.plot('comp_stats', 'comp_convergence')
plt.show()
```

4. chose the sweep trial with the "best" convergence behavior, and use these values to re-set your original trial scalings: 
```
lambda_scaling = 1e2
energy_scaling = 1e3
power_cost = 1e-1
options['model']['scaling_overwrite']['xa']['lambda'] = lambda_scaling
options['model']['scaling_overwrite']['xd']['e'] = energy_scaling
options['solver']['cost_overwrite']['power'][1] = power_cost
```

5. don't forget to un-cap the number of iterations!
```
options['solver']['max_iter'] = 2e3
```


### What does it mean when I get an error message: "Process finished with exit code 137 (interrupted by signal 9: SIGKILL)"

This likely means that the memory useage was too high. maybe try breaking your sweeps up into smaller groups.


### How do I get to a specific collocation variable in the solution? (Eg, one of the lifted aerodynamic variables, like 'f_aero21'...)

```
 V_final = trial.optimization.V_final
 V_final['coll_var', :, :, 'xl', 'f_aero21']
```

### How do I find out what parameters were used for a given trial, after the fact? (for example, the kite span?)

```
plot_dict = trial.visualization.plot_dict
b_ref = plot_dict['options']['params']['geometry']['b_ref']
```

### How do I get an animated output of the trajectory ("monitor plot")?

1. make sure that you have the 'ffmpeg' movie writer installed on your computer

2. include 'animation' into the list of plots desired:
```
plot_list = ['animation']
```

3. then, ask trial to create the plots
```
trial.plot(plot_list)
```

Please be aware that creating the animation is a slow process.

### How do I check if my problem is healthy? (And/or, how do I find out which constraints are making my problem unhealthy?)

The awebox has a built-in health-checker! 

0. Please be aware that the health-checker is slow, and requires a lot of memory. For this reason, we recommend that you only run the health-checker on a minimal version of your problem - even if the solution of that minimal version would not normally satisfy the quality control checks.
For example, we should drastically reduce the number of control intervals and/or reduce the winding number to 1:
```
options['nlp']['n_k'] = 5
options['user_options']['trajectory']['lift_mode']['windings'] = 1
```

1. Now, we need to turn the health-checker on. You can specify when the health-checker should run: after every sub-problem along the homotopy, after a sub-problem that fails, and/or at the end of the final homotopy step, by enabling any combination of the following:
```
options['solver']['health_check']['when']['autorun'] = True
options['solver']['health_check']['when']['failure'] = True
options['solver']['health_check']['when']['final'] = True
```

2. If we want to manually look over the KKT matrix, and the jacobian of the active and equality constraints, and the reduced hessian, we should use the following option. Be awere that these plots need to be closed manally, and so will interrupt a batch-run. 
```
options['solver']['health_check']['when']['spy_matrices'] = True
```

3. Finally, if you suspect that an LICQ, SOSC or other conditioning problem exists, the debugger attached to the health checker can help you find the problem. This identification will give more detailed information about potentially problematic constraints if you use the following option. Please be aware, that this will slow down the problem discretization, and should generally be turned off. 
```
options['nlp']['collocation']['name_constraints'] = True
```
(This option is currently only available for direct-collocation problems. Stay tuned for its introduction into multiple-shooting problems as well!)

Happy Debugging!

### When I try to run a script that imports the awebox (like one of the included examples), I get a "No module named 'awebox'" error. What should I do? 

This error manifests like:

```
user@computer:~/awebox/examples$ python3 single_kite_lift_mode_simple.py 
Traceback (most recent call last):
  File "single_kite_lift_mode_simple.py", line 3, in <module>
    import awebox as awe
ImportError: No module named 'awebox'
```

Check that the location of the awebox is in your PYTHONPATH from your terminal. ( note, that the "awebox" folder that we're talking about here, is the folder that contains sub-folders ["awebox", "docs", "examples", ...]. It is not the "awebox" folder that contains sub-folders ["mdl", "ocp", "viz", ...]. )

```
user@computer:~$ python3 -c "import sys; print('\n'.join(sys.path))"
['', '/home/user', '/usr/local/lib', ..., '/home/user/path_to_the_awebox/awebox', ...]
```

The printed list should include the path to the awebox. If it doesn't, then add it to the list by typing (in the terminal):

```
user@computer:~$ export PYTHONPATH="${PYTHONPATH}:/home/user/path_to_the_awebox/awebox"
```


### When I try to run a script that uses the awebox (like one of the included examples), I get a "No module named 'casadi'" error. What should I do? 

This error manifests like:
```
user@computer:~/awebox/examples$ python3 single_kite_lift_mode_simple.py 
Traceback (most recent call last):
  File "single_kite_lift_mode_simple.py", line 3, in <module>
    import awebox as awe
  File "/home/user/awebox/awebox/__init__.py", line 26, in <module>
    from .trial import Trial
  File "/home/user/awebox/awebox/trial.py", line 31, in <module>
    import awebox.trial_funcs as trial_funcs
  File "/home/user/awebox/awebox/trial_funcs.py", line 35, in <module>
    import awebox.tools.vector_operations as vect_op
  File "/home/user/awebox/awebox/tools/vector_operations.py", line 36, in <module>
    import casadi.tools as cas
ImportError: No module named 'casadi'
```

Use pip3 to install casadi! 
```
user@computer:~$ pip3 install casadi
```

### When I try to run a script that uses the awebox (like one of the included examples), I get an Invalid Option error after 1 iteration. What should I do? 

This problem would manifest like:
```
INFO:	Initial solution...
INFO:	
INFO:	
INFO:	ERROR: Solver FAILED, not moving on to next step...
INFO:	solver return status..........: Invalid_Option                
INFO:	number of iterations..........: 1                 
```

You probably don't have the HSL/COIN solvers installed correctly on your computer. you can test that this is the problem, by adding the following line to your python3 script: 

```
options['solver']['linear_solver'] = 'mumps'
```

if, when you run the script, the optimization itself succeeds, then you should focus your attention on correctly installing the solvers. you can find these instructions at: 
https://github.com/casadi/casadi/wiki/Obtaining-HSL


### On the Obtaining-HSL page install instructions, they keep referring to a path (where_you_want_to_install) and (hsl_install_directory). if I don't use the --prefix flag, where does HSL install normally?

to /usr/local/lib

### When I try to run a script that uses the awebox with the linear solver MA57, I get an 'Incorrect objective type' error. What should I do? 

This error manifests like:

```
INFO:	Initial solution...
INFO:
...
This is IPOPT version 3.12.3, running with linear solver ma57
...
Runtime parameters:
    Objective type: Unknown!
    Coarsening type: METIS_CTYPE_RM
    Initial partitioning type: METIS_IPTYPE_GROW
    Refinement type: Unknown!
...
Input Error: Incorrect objective type.
...
```

You probably don't have the HSL/COIN solvers installed correctly on your computer. In particular, you should return to the steps of "make"ing the hsl solvers, paying special attention to the flags.

### Where are the time-varying performance outputs (eg. trial.optimization.output_vals[1]['coll_outputs', :, :, 'performance', 'loyd_factor'])) actually defined?

These are defined in mdl/aero/indicators.

### Where are the global performance outputs (eg. 'power_and_performance') actually defined?

These are defined in opti/diagnostics.

### My awebox script gets "Killed" unexpectedly. What should I do? 

This error manifests like:
```
INFO:	Building optimization...
INFO:	initialize callback...
INFO:	generate solvers...
Killed
```

You may not have enough memory to construct the desired OCP. Can you make your problem smaller somehow? 


### I keep getting a warning message about SO(3), even though the IPOPT output says that my optimization solved. What should I do? 

This error manifests like: 

```
...
INFO:	solver return status..........: Solve_Succeeded    
...
"WARNING:	given rotation matrix is not a member of SO(3)."
...
```

When we generate the time-series information for plotting, we have to interpolate the discretized optimization variables.
This includes the rotation matrix. If there are too few data points found from the optimization, then this
interpolation may create rotation matrices that are not (even roughly) rotation matrices.

If this is a problem for you, it's most likely that you don't have enough collocation intervals in your problem!
Please increase this number using the options inputs:

```
options['nlp']['n_k'] = n_k
```


### Does trial.optimization.V_final give values that are 'scaled' or 'SI'?

the 'SI' version!

However, if you want the scaled version, you can instead query:
```
sol = trial.optimization.solution
V_solution_scaled = trial.nlp.V(sol['x'])
```

### What are the units of the power output? (aka. the given 'SI' unit of power...)

The power is defined according to:

    segment tension = (lambda_multiplier) * (segment length)

    power = (tension at bottom segment) * (tether reelout speed),

which means that the units of power are:

    units of tension = [N] = (units of lambda multiplier) * [m]

            ==> units of lambda multiplier = [N/m]

    units of power = (units of lambda multiplier) * [m] * [m/s] = [Nm / s] = [W]

(However, if you use sweep.plot('comp_stats'), the plotted "power_output_kw" has units of [kW].)


### Is there a version of the HSL solver "xxx" binary/dynamic-link-library/etc. available for operating system "yyy"?

We're not sure... But, if it exists, you will be able to find it at:
http://www.hsl.rl.ac.uk/ipopt/

In the worst case, we'd recommend contacting the HSL developers. Their contact info can be found at:
http://www.hsl.rl.ac.uk/contact.html

Good luck!


### How do I know if the trials in my sweep, actually converged?

Include the flag "comp_convergence" into the plot_list.

```
sweep.plot(['comp_convergence'])
```

A converged trial will have a "return_status_numeric" value of 1 or 2


### How do I know if any particular trial, actually converged?

You can check the return_status_numeric of the trial, with the knowledge that

```
return_status_numeric = trial.return_status_numeric
```

A converged trial will have a "return_status_numeric" value of 1 or 2.

### I want to run an axi-symmetric simulation (where the wind axis is parallel to the main tether). What should I do?

You will need to remove the ground constraints, remove gravity, and make the wind and atmosphere uniform. You can do this with:

```
options['user_options']['wind']['model'] = 'uniform'
options['user_options']['atmosphere'] = 'uniform'
options['model']['system_bounds']['xd']['q'] = [np.array([-cas.inf, -cas.inf, -cas.inf]), np.array([cas.inf, cas.inf, cas.inf])]
options['model']['system_bounds']['xd']['wz_ext'] = [-cas.inf, cas.inf]
options['model']['system_bounds']['xd']['wz_int'] = [-cas.inf, cas.inf]
options['params']['atmosphere']['g'] = 0.
```

It's also helpful to initialize according to the axi-symmetry that you're expecting:
```
options['solver']['initialization']['inclination_deg'] = 0.
```

Please note that at least one of the solution quality control tests will automatically fail, because there will be node locations "below-ground".


### How do I find out how much the various terms within the objective acutally add to the objective, after my trial has been optimized?

```
trial.print_cost_information()
```

### How do I test if I've installed the COIN/HSL linear solvers correctly?

We suggest running the following tests:

1. Does the casadi example NLP rocket solve in its given form ('mumps' linear solver)?
https://github.com/casadi/casadi/blob/master/docs/examples/python/rocket.py

(If this fails, the problem is with your casadi installation.)

2. Does the casadi example NLP rocket solve with the linear solver "ma57"?
modify the line
```
opts = {"ipopt.tol":1e-10, "expand":True}
```
to read:
```
opts = {"ipopt.tol":1e-10, "expand":True, "ipopt.linear_solver":"ma57"}
```

(If this fails, the problem is with the COIN/HSL solver installation.)

3. Does the awebox example file single_kite_lift_mode_simple.py run?

(If this fails, the problem is with the awebox. Please report the bug, so that we can fix it! Thanks in advance!)