#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
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
###################################
# Class Trial contains information and methods to perform one
# optimization of a tree-structured multiple-kite system
###################################

import awebox.tools.print_operations as print_op
import awebox.trial_funcs as trial_funcs
import awebox.ocp.nlp as nlp
import awebox.opti.optimization as optimization
import awebox.sim as sim
import awebox.mdl.model as model
import awebox.mdl.architecture as archi
import awebox.ocp.formulation as formulation
import awebox.viz.visualization as visualization
import awebox.quality as quality
import awebox.tools.save_operations as data_tools
import awebox.opts.options as opts
import awebox.tools.struct_operations as struct_op
from awebox.logger.logger import Logger as awelogger
from tabulate import tabulate
import copy

# These are a series of helper functions for the output method
# This is probably not the best place for it ... but just putting it here for the convenience of getting something basic working
################################################################################################################################

import numpy as np
import casadi

# convert casadi DM data structures to either numpy arrays or simple scalar
def convert_casadi_structure(casadi_structure):
    if casadi_structure.shape == (1,1):
        return float(casadi_structure[0,0])
    return np.array(casadi_structure)

# This method converts the awebox output data format into a list of tensors
# many of the high-order data structure (vectors, matrices) are stored as a list of vectors, which may not be convenient for some data processing
# list_of_vectors is the awebox output
# reshape_arg should be a dictionary of dimenions when the tensor is a matrix or higher order, otherwise the result is a vector
def convert_from_casadi_index_to_time_tensor_index(list_of_vectors, reshape_arg=None):
    # get the size of the data structure
    tensor_size=len(list_of_vectors)
    time_size=list_of_vectors[0].shape[0]
    # pre-allocate the output list
    output_list=[None]*time_size
    # loop through the data structure to re-arrange the data
    for j in range(time_size):
        tensor=None
        if tensor_size>1:
            # create our tensor
            tensor=np.zeros(tensor_size)
            # loop through the list entries to grab all the tensor values
            for i in range(tensor_size):
                tensor[i]=list_of_vectors[i][j]
            # re-shape tensor if needed
            if not reshape_arg is None:
                tensor = np.reshape(tensor, reshape_arg)
        elif tensor_size==1:
            tensor=float(list_of_vectors[0][j])
        # save the tensor
        output_list[j]=tensor
    # return our result
    return output_list

# Adds entry to output structure
def add_entry_to_output_structure(base_dict, key_str, entry_value, reshape_arg=None, add_help=False, help_str='Undefined help string'):
    # perform some conversions on the data to make it easier for the user
    my_entry_value=entry_value
    # first convert if it's a casadi data matrix
    if isinstance(entry_value, casadi.casadi.DM):
        my_entry_value=convert_casadi_structure(entry_value)
    # if it's a list, ensure that there are no casadi data matrices, furthermore, switch the indexing scheme
    elif isinstance(entry_value, list):
        my_entry_value=[]
        for vec in entry_value:
            if isinstance(vec, casadi.casadi.DM):
                my_entry_value.append(convert_casadi_structure(vec))
            else:
                my_entry_value.append(vec)
        if len(my_entry_value)>0:
            if isinstance(my_entry_value[0], np.ndarray):
                my_entry_value=convert_from_casadi_index_to_time_tensor_index(my_entry_value, reshape_arg)
    # add the value
    base_dict[key_str]=my_entry_value
    # add the help string
    if add_help:
        # if there is no help dictionary, then create one
        if not 'help' in base_dict:
            # init the help dictionary
            base_dict['help']={}
            # add a help string for the help dictionary
            base_dict['help']['help']="A data structure that describes the entries of the dictionary"
        # add the help string to the help entry
        base_dict['help'][key_str]=help_str
    # return the result
    return base_dict

################################################################################################################################

class Trial(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def __init__(self, seed, name = 'trial'):

        # check if constructed with solved trial dict
        if 'solution_dict' in seed.keys():

            self.__solution_dict = seed['solution_dict']
            self.__visualization = visualization.Visualization()
            self.__visualization.options = seed['solution_dict']['options']
            self.__visualization.plot_dict = seed['plot_dict']
            self.__visualization.create_plot_logic_dict()
            self.__options = seed['solution_dict']['options']

        else:

            self.__options_seed   = seed
            self.__options        = opts.Options()
            self.__options.fill_in_seed(self.__options_seed)
            self.__model          = model.Model()
            self.__formulation    = formulation.Formulation()
            self.__nlp            = nlp.NLP()
            self.__optimization   = optimization.Optimization()
            self.__visualization  = visualization.Visualization()
            self.__quality        = quality.Quality()
            self.__name           = name    #todo: names used as unique identifiers in sweep. smart?
            self.__type           = 'Trial'
            self.__status         = None
            self.__timings        = {}
            self.__solution_dict  = {}
            self.__save_flag      = False

            self.__return_status_numeric = -1

            self._freeze()

    def build(self, is_standalone_trial=True):

        if is_standalone_trial:
            print_op.log_license_info()

        if self.__options['user_options']['trajectory']['type'] == 'mpc':
            raise ValueError('Build method not supported for MPC trials. Use PMPC wrapper instead.')

        awelogger.logger.info(60*'=')
        awelogger.logger.info(12*' '+'Building trial "%s" ...', self.__name)
        awelogger.logger.info(60*'=')
        awelogger.logger.info('')

        architecture = archi.Architecture(self.__options['user_options']['system_model']['architecture'])
        self.__options.build(architecture)
        self.__model.build(self.__options['model'], architecture)
        self.__formulation.build(self.__options['formulation'], self.__model)
        self.__nlp.build(self.__options['nlp'], self.__model, self.__formulation)
        self.__optimization.build(self.__options['solver'], self.__nlp, self.__model, self.__formulation, self.__name)
        self.__visualization.build(self.__model, self.__nlp, self.__name, self.__options)
        self.__quality.build(self.__options['quality'], self.__name)
        self.set_timings('construction')
        awelogger.logger.info('Trial "%s" built.', self.__name)
        awelogger.logger.info('Trial construction time: %s',print_op.print_single_timing(self.__timings['construction']))
        awelogger.logger.info('')

    def optimize(self, options_seed = [], final_homotopy_step = 'final',
                 warmstart_file = None, vortex_linearization_file = None, debug_flags = [],
                 debug_locations = [], save_flag = False, intermediate_solve = False, recalibrate_viz = True):

        if not options_seed:
            options = self.__options
        else:
            # regenerate nlp bounds for parametric sweeps
            options = opts.Options()
            options.fill_in_seed(options_seed)
            architecture = archi.Architecture(self.__options['user_options']['system_model']['architecture'])
            options.build(architecture)
            import awebox.mdl.dynamics as dyn
            self.__model.generate_scaled_variable_bounds(options['model'])
            self.__nlp.generate_variable_bounds(options['nlp'], self.__model)

        # get save_flag
        self.__save_flag = save_flag

        if self.__options['user_options']['trajectory']['type'] == 'mpc':
            raise ValueError('Optimize method not supported for MPC trials. Use PMPC wrapper instead.')

        awelogger.logger.info(60*'=')
        awelogger.logger.info(12*' '+'Optimizing trial "%s" ...', self.__name)
        awelogger.logger.info(60*'=')
        awelogger.logger.info('')


        self.__optimization.solve(options['solver'], self.__nlp, self.__model,
                                  self.__formulation, self.__visualization,
                                  final_homotopy_step, warmstart_file, vortex_linearization_file,
                                  debug_flags = debug_flags, debug_locations =
                                  debug_locations, intermediate_solve = intermediate_solve)

        self.__solution_dict = self.generate_solution_dict()

        self.set_timings('optimization')

        self.__return_status_numeric = self.__optimization.return_status_numeric['optimization']

        if self.__optimization.solve_succeeded:
            awelogger.logger.info('Trial "%s" optimized.', self.__name)
            awelogger.logger.info('Trial optimization time: %s',print_op.print_single_timing(self.__timings['optimization']))

        else:

            awelogger.logger.info('WARNING: Optimization of Trial (%s) failed.', self.__name)

        if (not intermediate_solve and recalibrate_viz):
            cost_fun = self.nlp.cost_components[0]
            cost = struct_op.evaluate_cost_dict(cost_fun, self.optimization.V_opt, self.optimization.p_fix_num)
            self.visualization.recalibrate(self.optimization.V_opt,self.visualization.plot_dict, self.optimization.output_vals,
                                            self.optimization.integral_outputs_final, self.options, self.optimization.time_grids,
                                            cost, self.name, self.__optimization.V_ref, self.__optimization.global_outputs_opt)

            # perform quality check
            self.__quality.check_quality(self)

        # print solution
        self.print_solution()

        # save trial if option is set
        if self.__save_flag is True or self.__options['solver']['save_trial'] == True:
            saving_method = self.__options['solver']['save_format']
            self.save(saving_method = saving_method)

        awelogger.logger.info('')

    def plot(self, flags, V_plot=None, cost=None, parametric_options=None, output_vals=None, sweep_toggle=False, fig_num = None):

        if V_plot is None:
            V_plot = self.__solution_dict['V_opt']
            recalibrate = False
        if parametric_options is None:
            parametric_options = self.__options
        if output_vals == None:
            output_vals = self.__solution_dict['output_vals']
        if cost == None:
            cost = self.__solution_dict['cost']
        time_grids = self.__solution_dict['time_grids']
        integral_outputs_final = self.__solution_dict['integral_outputs_final']
        V_ref = self.__solution_dict['V_ref']
        trial_name = self.__solution_dict['name']

        self.__visualization.plot(V_plot, parametric_options, output_vals, integral_outputs_final, flags, time_grids, cost, trial_name, sweep_toggle, V_ref, self.__optimization.global_outputs_opt, 'plot',fig_num, recalibrate = recalibrate)

        return None

    def set_timings(self, timing):
        if timing == 'construction':
            self.__timings['construction'] = self.model.timings['overall'] + self.formulation.timings['overall'] \
                                            + self.nlp.timings['overall'] + self.optimization.timings['setup']
        elif timing == 'optimization':
            self.__timings['optimization'] = self.optimization.timings['optimization']

    def save(self, saving_method = 'dict', fn = None):

        # log saving method
        awelogger.logger.info('Saving trial ' + self.__name + ' using ' + saving_method)

        # set savefile name to trial name if unspecified
        if not fn:
            fn = self.__name

        # choose correct function for saving method
        if saving_method == 'awe':
            self.save_to_awe(fn)
        elif saving_method == 'dict':
            self.save_to_dict(fn)
        else:
            awelogger.logger.error(saving_method + ' is not a supported saving method. Trial ' + self.__name + ' could not be saved!')

        # log that save is complete
        awelogger.logger.info('Trial (%s) saved.', self.__name)
        awelogger.logger.info('')
        awelogger.logger.info(print_op.hline('&'))
        awelogger.logger.info(print_op.hline('&'))
        awelogger.logger.info('')
        awelogger.logger.info('')

    def get_output_data_structure(self, add_help=False):

        output_structure={}
        
        src=self.visualization.plot_dict

        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'name'                  , src['name']                                                      , None , add_help, "The name given for the problem")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'n_k'                   , src['n_k']                                                       , None , add_help, "The number of control points used to solve the problem [-]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'u_ref'                 , src['u_ref']                                                     , None , add_help, "The reference wind speed for the problem [m/s]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'interpolated_time'     , src['time_grids']['ip']                                          , None , add_help, "The time for interpolated output quantities [s]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'x'                     , {}                                                               , None , add_help, "The state data for the solution")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'delta10'               , src['x']['delta10']                                              , None , add_help, "Control surface deflection [ degree ??? radian ??? ]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'dl_t'                  , src['x']['dl_t']                                                 , None , add_help, "Reel-out speed [m/s]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'dq10'                  , src['x']['dq10']                                                 , None , add_help, "Aircraft velocity [m/s]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'l_t'                   , src['x']['l_t']                                                  , None , add_help, "Tether length [m]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'omega10'               , src['x']['omega10']                                              , None , add_help, "Angular velocity [ radians/s ??? degrees/s ??? ]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'q10'                   , src['x']['q10']                                                  , None , add_help, "Position [m]")
        output_structure['x']                               =add_entry_to_output_structure(output_structure['x']                               , 'r10'                   , src['x']['r10']                                                  , (3,3), add_help, "Aircraft orientation [-]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'z'                     , {}                                                               , None , add_help, "Lagrange multipliers for the constraint equations in the dynamic problem")
        output_structure['z']                               =add_entry_to_output_structure(output_structure['z']                               , 'lambda10'              , src['z']['lambda10']                                             , None , add_help, "Lagrange multiplier for the tether length constraint")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'u'                     , {}                                                               , None , add_help, "Control input to the problem")
        output_structure['u']                               =add_entry_to_output_structure(output_structure['u']                               , 'ddelta10'              , src['u']['ddelta10']                                             , None , add_help, "Aircraft control surface deflection [ degree? radian? ]")
        output_structure['u']                               =add_entry_to_output_structure(output_structure['u']                               , 'ddl_t'                 , src['u']['ddl_t']                                                , None , add_help, "Reel-out acceleration [m/s^2]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'outputs'               , {}                                                               , None , add_help, "Output data derived from the problem solution")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'masses'                , {}                                                               , None , add_help, "Various mass values for the system")
        output_structure['outputs']['masses']               =add_entry_to_output_structure(output_structure['outputs']['masses']               , 'm_tether1'             , src['outputs']['masses']['m_tether1']                            , None , add_help, "Mass of the tether [kg]")
        output_structure['outputs']['masses']               =add_entry_to_output_structure(output_structure['outputs']['masses']               , 'ground_station'        , src['outputs']['masses']['m_tether1']                            , None , add_help, "Mass of the ground station [kg]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'e_kinetic'             , {}                                                               , None , add_help, "Kinetic energy in the system")
        output_structure['outputs']['e_kinetic']            =add_entry_to_output_structure(output_structure['outputs']['e_kinetic']            , 'q10'                   , src['outputs']['e_kinetic']['q10']                               , None , add_help, "Kinetic energy associated with the aircraft [J]")
        output_structure['outputs']['e_kinetic']            =add_entry_to_output_structure(output_structure['outputs']['e_kinetic']            , 'ground_station'        , src['outputs']['e_kinetic']['ground_station']                    , None , add_help, "Kinetic energy associated with the ground station [J]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'e_potential'           , {}                                                               , None , add_help, "Kinetic energy in the system")
        output_structure['outputs']['e_potential']          =add_entry_to_output_structure(output_structure['outputs']['e_potential']          , 'q10'                   , src['outputs']['e_potential']['q10']                             , None , add_help, "Potential energy associated with the aircraft [J]")
        output_structure['outputs']['e_potential']          =add_entry_to_output_structure(output_structure['outputs']['e_potential']          , 'ground_station'        , src['outputs']['e_potential']['ground_station']                  , None , add_help, "Potential energy associated with the ground station [J]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'tether_length'         , {}                                                               , None , add_help, "Tether kinematic output")
        output_structure['outputs']['tether_length']        =add_entry_to_output_structure(output_structure['outputs']['tether_length']        , 'c10'                   , src['outputs']['tether_length']['c10']                           , None , add_help, "Tether length [m]")
        output_structure['outputs']['tether_length']        =add_entry_to_output_structure(output_structure['outputs']['tether_length']        , 'dc10'                  , src['outputs']['tether_length']['dc10']                          , None , add_help, "Reel-out speed [m/s]")
        output_structure['outputs']['tether_length']        =add_entry_to_output_structure(output_structure['outputs']['tether_length']        , 'ddc10'                 , src['outputs']['tether_length']['ddc10']                         , None , add_help, "Reel-out acceleration [m/s^2]")
        output_structure['outputs']['tether_length']        =add_entry_to_output_structure(output_structure['outputs']['tether_length']        , 'rot_kinematics10'      , src['outputs']['tether_length']['rot_kinematics10']              , (3,3), add_help, "The direction of the tether [-]")
        output_structure['outputs']['tether_length']        =add_entry_to_output_structure(output_structure['outputs']['tether_length']        , 'orthonormality10'      , src['outputs']['tether_length']['orthonormality10']              , (3,3), add_help, "The residual for R^T*R-I=0 for the above rotation matrix [-]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'aerodynamics'          , {}                                                               , None , add_help, "Various mass values for the system")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CA1'                   , src['outputs']['aerodynamics']['CA1']                            , None , add_help, "Coefficient for the X component of aerodynamic force in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CY1'                   , src['outputs']['aerodynamics']['CY1']                            , None , add_help, "Coefficient for the Y component of aerodynamic force in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CN1'                   , src['outputs']['aerodynamics']['CN1']                            , None , add_help, "Coefficient for the Z component of aerodynamic force in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CD1'                   , src['outputs']['aerodynamics']['CD1']                            , None , add_help, "Coefficient for the X component of aerodynamic force in the wind frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CS1'                   , src['outputs']['aerodynamics']['CS1']                            , None , add_help, "Coefficient for the Y component of aerodynamic force in the wind frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CL1'                   , src['outputs']['aerodynamics']['CL1']                            , None , add_help, "Coefficient for the Z component of aerodynamic force in the wind frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'Cl1'                   , src['outputs']['aerodynamics']['Cl1']                            , None , add_help, "Coefficient for the X component of aerodynamic moment in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'Cm1'                   , src['outputs']['aerodynamics']['Cm1']                            , None , add_help, "Coefficient for the Y component of aerodynamic moment in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'Cn1'                   , src['outputs']['aerodynamics']['Cn1']                            , None , add_help, "Coefficient for the Z component of aerodynamic moment in the body frame, f = dynamic_press*S*C [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'air_velocity1'         , src['outputs']['aerodynamics']['air_velocity1']                  , None , add_help, "The air velocity seen by the aircraft (i.e. wind-kite_velocity) [m/s]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'airspeed1'             , src['outputs']['aerodynamics']['airspeed1']                      , None , add_help, "The magnitude of the air velocity (i.e. ||wind-kite_velocity||) [m/s]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'u_infty1'              , src['outputs']['aerodynamics']['u_infty1']                       , None , add_help, "The wind speed at the kite altitude, including shear [m/s]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'air_density1'          , src['outputs']['aerodynamics']['air_density1']                   , None , add_help, "The air density [kg/m^3]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'dyn_pressure1'         , src['outputs']['aerodynamics']['dyn_pressure1']                  , None , add_help, "The dynamic pressure (i.e. 1/2*rho*airspeed^2) [Pa]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'LoverD1'               , src['outputs']['aerodynamics']['LoverD1']                        , None , add_help, "The lift of drag ratio [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'ehat_chord1'           , src['outputs']['aerodynamics']['ehat_chord1']                    , None , add_help, "The chord direction of the body frame [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'ehat_span1'            , src['outputs']['aerodynamics']['ehat_span1']                     , None , add_help, "The span direction of the body frame [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'ehat_up1'              , src['outputs']['aerodynamics']['ehat_up1']                       , None , add_help, "The upwards direction of the body frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_aero_body1'          , src['outputs']['aerodynamics']['f_aero_body1']                   , None , add_help, "The aerodynamic force in the body frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_aero_control1'       , src['outputs']['aerodynamics']['f_aero_control1']                , None , add_help, "The aerodynamic force in the frame defined for the control (similar to body frame) [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_aero_earth1'         , src['outputs']['aerodynamics']['f_aero_earth1']                  , None , add_help, "The aerodynamic force in the global reference frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_aero_wind1'          , src['outputs']['aerodynamics']['f_aero_wind1']                   , None , add_help, "The aerodynamic force in the wind reference frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_lift_earth1'         , src['outputs']['aerodynamics']['f_lift_earth1']                  , None , add_help, "The lift force in the global reference frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_drag_earth1'         , src['outputs']['aerodynamics']['f_drag_earth1']                  , None , add_help, "The drag force in the global reference frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'f_side_earth1'         , src['outputs']['aerodynamics']['f_side_earth1']                  , None , add_help, "The side force in the global reference frame [N]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CL_var1'               , src['outputs']['aerodynamics']['CL_var1']                        , None , add_help, " ??? still not sure ??? ")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CD_var1'               , src['outputs']['aerodynamics']['CD_var1']                        , None , add_help, " ??? still not sure ??? ")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'CS_var1'               , src['outputs']['aerodynamics']['CS_var1']                        , None , add_help, " ??? still not sure ??? ")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'ortho_resi1'           , src['outputs']['aerodynamics']['ortho_resi1']                    , None , add_help, "Orthonormality constraint residual (i.e. r^T*r-I=0) [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'wingtip_ext1'          , src['outputs']['aerodynamics']['wingtip_ext1']                   , None , add_help, "Exterior wing tip position [m]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'wingtip_int1'          , src['outputs']['aerodynamics']['wingtip_int1']                   , None , add_help, "Interior wing tip position [m]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'fstar_aero1'           , src['outputs']['aerodynamics']['fstar_aero1']                    , None , add_help, "Defined as air_velocity^T*ehat_chord/c_ref [1/s]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'r1'                    , src['outputs']['aerodynamics']['r1']                             , (3,3), add_help, "The orientation matrix for the aircraft [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'm_aero_body1'          , src['outputs']['aerodynamics']['m_aero_body1']                   , None , add_help, "The aerodynamic moment forces in the body frame [Nm]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'mach1'                 , src['outputs']['aerodynamics']['mach1']                          , None , add_help, "The Mach number [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'reynolds1'             , src['outputs']['aerodynamics']['reynolds1']                      , None , add_help, "The Reynolds number [-]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'alpha1'                , src['outputs']['aerodynamics']['alpha1']                         , None , add_help, "The angle of attack [radians]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'beta1'                 , src['outputs']['aerodynamics']['beta1']                          , None , add_help, "The side-slip angle [radians]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'alpha_deg1'            , src['outputs']['aerodynamics']['alpha_deg1']                     , None , add_help, "The angle of attack [degrees]")
        output_structure['outputs']['aerodynamics']         =add_entry_to_output_structure(output_structure['outputs']['aerodynamics']         , 'beta_deg1'             , src['outputs']['aerodynamics']['beta_deg1']                      , None , add_help, "The side-slip angle [degrees]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'environment'           , {}                                                               , None , add_help, "The environmental data for the simulation")
        output_structure['outputs']['environment']          =add_entry_to_output_structure(output_structure['outputs']['environment']          , 'windspeed1'            , src['outputs']['environment']['windspeed1']                      , None , add_help, "The wind speed used in the simulation [m/s]")
        output_structure['outputs']['environment']          =add_entry_to_output_structure(output_structure['outputs']['environment']          , 'pressure1'             , src['outputs']['environment']['pressure1']                       , None , add_help, "The atmospheric pressure used in the simulation [Pa]")
        output_structure['outputs']['environment']          =add_entry_to_output_structure(output_structure['outputs']['environment']          , 'temperature1'          , src['outputs']['environment']['temperature1']                    , None , add_help, "The temperature used in the simulation [C]")
        output_structure['outputs']['environment']          =add_entry_to_output_structure(output_structure['outputs']['environment']          , 'density1'              , src['outputs']['environment']['density1']                        , None , add_help, "The density used in the simulation [kg/s^3]")
        output_structure['outputs']                         =add_entry_to_output_structure(output_structure['outputs']                         , 'local_performance'     , {}                                                               , None , add_help, "Data structure that gives additional performance figures for the simulation")
        output_structure['outputs']['local_performance']    =add_entry_to_output_structure(output_structure['outputs']['local_performance']    , 'tether_force10'        , src['outputs']['local_performance']['tether_force10']            , None , add_help, "The tether force [N]")
        output_structure['outputs']['local_performance']    =add_entry_to_output_structure(output_structure['outputs']['local_performance']    , 'tether_stress10'       , src['outputs']['local_performance']['tether_stress10']           , None , add_help, "The tether stress [Pa]")
        output_structure                                    =add_entry_to_output_structure(output_structure                                    , 'power_and_performance' , {}                                                               , None , add_help, "Data structure that gives additional performance figures for the simulation")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'e_final'               , src['power_and_performance']['e_final']                          , None , add_help, "The total energy generated within 1 cycle [J]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'time_period'           , src['power_and_performance']['time_period']                      , None , add_help, "The period of a cycle [s]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'avg_power'             , src['power_and_performance']['avg_power']                        , None , add_help, "The average power over 1 cycle [W]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'zeta'                  , src['power_and_performance']['zeta']                             , None , add_help, " ??? not sure what this is ")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'power_per_surface_area', src['power_and_performance']['power_per_surface_area']           , None , add_help, "The power generated per unit of wing area ??? [W/m^2]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'z_av'                  , src['power_and_performance']['z_av']                             , None , add_help, " ??? average altitude ??? [m]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'elevation'             , src['power_and_performance']['elevation']                        , None , add_help, " ??? the elevation angle [rad/deg???] or elevation above sea-level???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'azimuth'               , src['power_and_performance']['azimuth']                          , None , add_help, " ??? some sort of gemetric angle quantity, but not sure what? Degree? Radian?")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'cone_angle'            , src['power_and_performance']['cone_angle']                       , None , add_help, "The cone angle of the cycle [rad??? deg???]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'dq_final'              , src['power_and_performance']['dq_final']                         , None , add_help, "The final velocity of the aircraft [m/s]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'dq10_av'               , src['power_and_performance']['dq10_av']                          , None , add_help, "The average velocity of the aircraft [m/s]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'cmax'                  , src['power_and_performance']['cmax']                             , None , add_help, " ??? Not sure what this is ??? ")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'tension_max'           , src['power_and_performance']['tension_max']                      , None , add_help, "The maximum tension force [N]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'tension_avg'           , src['power_and_performance']['tension_avg']                      , None , add_help, "The average tension force [N]")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'eff_sideforce_loss'    , src['power_and_performance']['eff_sideforce_loss']               , None , add_help, " ??? An efficiency metric for the side-force losses ???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'eff_overall'           , src['power_and_performance']['eff_overall']                      , None , add_help, " ??? An efficiency metric for the system ???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'eff_tether_drag_loss'  , src['power_and_performance']['eff_tether_drag_loss']             , None , add_help, " ??? An efficiency metric for the tether drag losses ???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'eff_drag_loss'         , src['power_and_performance']['eff_drag_loss']                    , None , add_help, " ??? An efficiency metric for the aircraft drag losses ???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'control_frequency10'   , src['power_and_performance']['control_frequency10']              , None , add_help, " ??? Frequency of control input/outptu [Hz] ???")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'winding_axis'          , src['power_and_performance']['winding_axis']                     , None , add_help, " ??? axis of rotation for the reel-in drum ??? ")
        output_structure['power_and_performance']           =add_entry_to_output_structure(output_structure['power_and_performance']           , 'winding1'              , src['power_and_performance']['winding1']                         , None , add_help, " ??? something to do with the winding ??? ")

        return output_structure;

    def print_solution(self):

        # the actual power indicators
        if 'e' in self.__model.integral_outputs.keys():
            e_final = self.__optimization.integral_outputs_final['int_out',-1,'e']
        else:
            e_final = self.__optimization.V_final['x', -1, 'e'][-1]

        time_period = self.__optimization.global_outputs_opt['time_period'].full()[0][0]
        avg_power = e_final / time_period

        headers = ['Parameter / output', 'Optimal value', 'Dimension']

        table = [['Average power output', str(avg_power/1e3), 'kW']]
        table.append(['Time period', str(time_period), 's'])
        import numpy as np
        theta_info = {
            'diam_t': ('Main tether diameter', 1e3, 'mm'),
            'diam_s': ('Secondary tether diameter', 1e3, 'mm'),
            'l_s': ('Secondary tether length', 1, 'm'),
            'l_t': ('Main tether length', 1, 'm'),
            'l_i': ('Intermediate tether length', 1, 'm'),
            'diam_i': ('Intermediate tether diameter', 1e3, 'mm'),
            'l_t_full': ('Wound tether length', 1, 'm'),
            'P_max': ('Peak power', 1e3, 'kW'),
            'ell_radius': ('Ellipse radius', 1, 'm'),
            'ell_elevation': ('Ellipse elevation', 180.0/np.pi, 'deg'),
            'ell_theta': ('Ellipse division angle', 180.0/np.pi, 'deg'), 
            'a': ('Average induction', 1, '-'),
        }

        for theta in self.model.variables_dict['theta'].keys():
            if theta != 't_f':
                info = theta_info[theta]
                table.append([
                    info[0],
                    str(round(self.__optimization.V_final['theta', theta].full()[0][0]*info[1],3)),
                    info[2]]
                    )
        print(tabulate(table, headers=headers))

    def save_to_awe(self, fn):

        # reset multiple_shooting trial
        if self.__options['nlp']['discretization'] == 'multiple_shooting':
            self.__nlp = nlp.NLP()
            self.__optimization = optimization.Optimization()
            self.__visualization = visualization.Visualization()

        # pickle data
        data_tools.save(self, fn, 'awe')

    def save_to_dict(self, fn):

        # create dict to be saved
        data_to_save = {}

        # store necessary information
        data_to_save['solution_dict'] = self.generate_solution_dict()
        data_to_save['plot_dict'] = self.__visualization.plot_dict

        # pickle data
        data_tools.save(data_to_save, fn, 'dict')

    def generate_solution_dict(self):

        solution_dict = {}

        # seeding data
        solution_dict['time_grids'] = self.__optimization.time_grids
        solution_dict['name'] = self.__name

        # parametric sweep data
        solution_dict['V_opt'] = self.__optimization.V_opt
        solution_dict['V_final'] = self.__optimization.V_final
        solution_dict['V_ref'] = self.__optimization.V_ref
        solution_dict['options'] = self.__options
        solution_dict['output_vals'] = [
            copy.deepcopy(self.__optimization.output_vals[0]),
            copy.deepcopy(self.__optimization.output_vals[1]),
            copy.deepcopy(self.__optimization.output_vals[2])
        ]
        solution_dict['integral_outputs_final'] = self.__optimization.integral_outputs_final
        solution_dict['stats'] = self.__optimization.stats
        solution_dict['iterations'] = self.__optimization.iterations
        solution_dict['timings'] = self.__optimization.timings
        cost_fun = self.__nlp.cost_components[0]
        cost = struct_op.evaluate_cost_dict(cost_fun, self.__optimization.V_opt, self.__optimization.p_fix_num)
        solution_dict['cost'] = cost

        # warmstart data
        solution_dict['final_homotopy_step'] = self.__optimization.final_homotopy_step
        solution_dict['Xdot_opt'] = self.__nlp.Xdot(self.__nlp.Xdot_fun(self.__optimization.V_opt))
        solution_dict['g_opt'] = self.__nlp.g(self.__nlp.g_fun(self.__optimization.V_opt, self.__optimization.p_fix_num))
        solution_dict['opt_arg'] = self.__optimization.arg

        return solution_dict

    def print_cost_information(self):

        sol = self.optimization.solution
        V_solution_scaled = self.nlp.V(sol['x'])

        p_fix_num = self.optimization.p_fix_num

        cost_fun = self.nlp.cost_components[0]
        cost_dict = struct_op.evaluate_cost_dict(cost_fun, V_solution_scaled, p_fix_num)

        message = '... cost components at solution are:'
        awelogger.logger.info(message)

        print_op.print_dict_as_table(cost_dict)

        awelogger.logger.info('')

        total_dict = {'total_cost': self.nlp.f_fun(V_solution_scaled, p_fix_num)}
        print_op.print_dict_as_table(total_dict)

        return None

    def write_to_csv(self, file_name=None, frequency=30., rotation_representation='euler'):

        if file_name is None:
            file_name = self.name
        trial_funcs.generate_trial_data_csv(self, file_name, frequency, rotation_representation)

        return None

    def generate_optimal_model(self, param_options = None):
        return trial_funcs.generate_optimal_model(self, param_options= param_options)

    @property
    def options_seed(self):
        return self.__options_seed

    @property
    def options(self):
        return self.__options

    @options.setter
    def options(self, value):
        print('Cannot set options object.')

    @property
    def status(self):
        status_dict = {}
        status_dict['model'] = self.__model.status
        status_dict['nlp'] = self.__nlp.status
        status_dict['optimization'] = self.__optimization.status
        return status_dict

    @status.setter
    def status(self, value):
        print('Cannot set status object.')

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        print('Cannot set model object.')

    @property
    def nlp(self):
        return self.__nlp

    @nlp.setter
    def nlp(self, value):
        print('Cannot set nlp object.')

    @property
    def optimization(self):
        return self.__optimization

    @optimization.setter
    def optimization(self, value):
        print('Cannot set optimization object.')

    @property
    def formulation(self):
        return self.__formulation

    @formulation.setter
    def formulation(self, value):
        print('Cannot set formulation object.')

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        print('Cannot set type object.')

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        print('Cannot set name object.')

    @property
    def timings(self):
        return self.__timings

    @timings.setter
    def timings(self, value):
        print('Cannot set timings object.')

    @property
    def visualization(self):
        return self.__visualization

    @visualization.setter
    def visualization(self, value):
        print('Cannot set visualization object.')

    @property
    def quality(self):
        return self.__quality

    @quality.setter
    def quality(self, value):
        print('Cannot set quality object.')

    @property
    def return_status_numeric(self):
        return self.__return_status_numeric

    @return_status_numeric.setter
    def return_status_numeric(self, value):
        print('Cannot set return_status_numeric object.')

    @property
    def solution_dict(self):
        return self.__solution_dict

    @solution_dict.setter
    def solution_dict(self, value):
        print('Cannot set solution_dict object.')

def generate_initial_state(model, V_init):
    x0 = model.struct_list['x'](0.)
    for name in list(model.struct_list['x'].keys()):
        x0[name] = V_init['x',0,0,name]
    return x0
