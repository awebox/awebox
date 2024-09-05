#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
import os
import casadi as cas


def get_mass_inertia_AP2(A_scaled,kappa):

    # import numpy as np
    
    b_ref = 5.5  # [m]
    A_ref = 3.  # [m^2]
    c_ref = 0.55  # [m]
    AR = b_ref / c_ref

    b_scaled = np.round(np.sqrt(A_scaled*AR),3)
    c_scaled = np.round(b_scaled/AR,2)
    
    m_ref = 36.8  # [kg]
    J_ref = np.array([[25., 0.0, 0.47],
                      [0.0, 32., 0.0],
                      [0.47, 0.0, 56.]])
   
    m_scaled = m_ref*(b_scaled/b_ref)**kappa
    J_scaled = J_ref*(b_scaled/b_ref)**(kappa+2)
    
    return(m_scaled,J_scaled,b_scaled,c_scaled)

def get_l_init(kappa,A_scaled,hws_mean):
    
    if hws_mean <6: 
        l_init = 400
    elif hws_mean < 10:
        l_init = 600
    elif hws_mean <14:
        l_init  = 1000
    else:
        l_init = 1200
    return(l_init)


def get_l_init_hws_log(hws_mean):
    
    if hws_mean <8 :
        l_init = 800
    elif hws_mean<10:
        l_init = 1000
    else:
        l_init = 1100
        
    return(l_init)

def get_d_tether(A_scaled):
    
    if A_scaled == 10:
        d_tether = 0.0055
        F_constr = 34*1000
        
    # elif A_scaled == 20:
    #     d_tether = 0.0078
    #     F_constr = 60*1000
        
    # elif A_scaled == 50:
    #     d_tether = 0.0123
    #     F_constr = 140*1000
        
    else:
        d_20 = 0.0078
        F_20 = 60*1000
        
        d_tether = np.round(d_20*np.sqrt(A_scaled/20),4)
        F_constr = np.round(F_20*A_scaled/20,4)

    return(d_tether,F_constr)

def get_l_init_hws(hws_mean):
    
    
    if hws_mean <10 :
        l_init = 800
    elif hws_mean<13:
        l_init = 1000
    else:
        l_init = 1200
        
    return(l_init)

def export_results_log(filename,folder_dir,location,A_scaled,kappa,hws_ref,file_out):

    import pickle
    import numpy as np
    import os
    
    data_in                  = pickle.load(open(folder_dir + filename + '.dict','rb'))
       
    if file_out + '.p' in os.listdir(folder_dir):
        data_out = pickle.load( open(folder_dir + file_out + ".p","rb" ) )
    else:
        data_out = {}
        
    if not location in data_out:
        data_out[location]={}
    if not hws_ref in data_out[location]:
        data_out[location][hws_ref]={}
           
    tether_diameter_constr              = data_in['plot_dict']['options']['model']['system_bounds']['theta']['diam_t'][0]
    tether_crosssection                 = tether_diameter_constr**2*np.pi/4
    X,Y,Z                               = [np.array(np.squeeze(data_in['plot_dict']['x']['q10'][i])) for i in range(3)]
    tether_max_stress                   = np.array(data_in['plot_dict']['options']['params']['tether']['max_stress'])
    tether_speed_max                    = np.round(np.max(np.squeeze(np.array(data_in['plot_dict']['x']['dl_t'][0]))),1)

    
    data_out[location][hws_ref]['U'] = np.array(data_in['plot_dict']['outputs']['aerodynamics']['u_infty1'])
    data_out[location][hws_ref]['windspeed'] = np.mean(np.array(data_in['plot_dict']['outputs']['environment']['windspeed1']))
    data_out[location][hws_ref]['X']          = X
    data_out[location][hws_ref]['Y']          = Y
    data_out[location][hws_ref]['Z']          = Z
    data_out[location][hws_ref]['time']       = np.array(data_in['plot_dict']['time_grids']['ip'])
    data_out[location][hws_ref]['p_current']  = np.squeeze(np.array(data_in['plot_dict']['outputs']['performance']['p_current']))
    data_out[location][hws_ref]['f_Loyd']     = np.array(data_in['plot_dict']['outputs']['performance']['loyd_factor'])
    data_out[location][hws_ref]['p_avg']      = np.array(data_in['plot_dict']['power_and_performance']['avg_power'])
    data_out[location][hws_ref]['e_final']    = np.array(data_in['plot_dict']['power_and_performance']['e_final'])
    data_out[location][hws_ref]['l_tether']   = np.squeeze(np.array(data_in['plot_dict']['x']['l_t'][0]))
    data_out[location][hws_ref]['alpha']      = np.array(data_in['plot_dict']['outputs']['aerodynamics']['alpha_deg1']) 
    data_out[location][hws_ref]['beta']       = np.array(data_in['plot_dict']['outputs']['aerodynamics']['beta_deg1']) 
    data_out[location][hws_ref]['tether_force']  = np.array(data_in['plot_dict']['outputs']['local_performance']['tether_force10'][0])
    data_out[location][hws_ref]['tether_stress'] = np.array(data_in['plot_dict']['outputs']['local_performance']['tether_stress10'][0])
    data_out[location][hws_ref]['tether_force_constr']    = data_in['plot_dict']['options']['model']['params']['model_bounds']['tether_force_limits'][1]
    data_out[location][hws_ref]['tether_f_safety_internal'] = np.array(data_in['plot_dict']['options']['params']['tether']['stress_safety_factor'])
    data_out[location][hws_ref]['tether_max_force']       = np.array(tether_crosssection*tether_max_stress)
    data_out[location][hws_ref]['tether_force_fsafety']   = np.round((tether_diameter_constr**2/4*np.pi*3.6e9)/data_out[location][hws_ref]['tether_force_constr'],1)
    data_out[location][hws_ref]['p_avg']              = np.squeeze(np.array(data_in['plot_dict']['power_and_performance']['avg_power']))
    data_out[location][hws_ref]['tether_speed']       = np.squeeze(np.round(np.array(data_in['plot_dict']['x']['dl_t'][0]),2))
    data_out[location][hws_ref]['tether_speed_min']   = np.round(np.min(np.squeeze(np.array(data_in['plot_dict']['x']['dl_t'][0]))),1)
    data_out[location][hws_ref]['tether_speed_max']   = np.round(np.max(np.squeeze(np.array(data_in['plot_dict']['x']['dl_t'][0]))),1)
    data_out[location][hws_ref]['p_max_constr']       = np.round(tether_speed_max.max()*data_out[location][hws_ref]['tether_force_constr'])
    data_out[location][hws_ref]['diam_tether']        =      np.sqrt(np.divide(data_out[location][hws_ref]['tether_force'] ,data_out[location][hws_ref]['tether_stress'])*4/np.pi)
    data_out[location][hws_ref]['mass']               = np.round(data_in['plot_dict']['options']['params']['geometry']['m_k'],1)
    data_out[location][hws_ref]['airspeed']           = np.round(data_in['plot_dict']['outputs']['aerodynamics']['airspeed1'],1)
    data_out[location][hws_ref]['alpha']              = np.round(data_in['plot_dict']['outputs']['aerodynamics']['alpha_deg1'],1)
    data_out[location][hws_ref]['beta']               = np.round( data_in['plot_dict']['outputs']['aerodynamics']['beta_deg1'],1)
    

    data_out[location][hws_ref]['Z_in']               = np.arange(10,1010,10)
    data_out[location][hws_ref]['u_in']               = hws_ref * np.log10(np.arange(10,1010,10) / z0_air) / np.log10(10 / z0_air)
    data_out[location][hws_ref]['v_in']               = np.zeros(len(data_out[location][hws_ref]['u_in']))
    data_out[location][hws_ref]['hws_mean_in']        = np.mean(data_out[location][hws_ref]['u_in'])

    pickle.dump( data_out, open(folder_dir + file_out + ".p", "wb" ) )
    
    return(data_out)


#########################
# DEFINE MASS & INERTIA #
#########################
for A_scaled  in [80]:
    kappa = 3.3
    m_scaled,J_scaled,b_scaled,c_scaled = get_mass_inertia_AP2(A_scaled,kappa)
    
    ############################
    # DEFINE TETHER CONSTRAINT #
    ############################
    diam_t , F_constr = get_d_tether(A_scaled) # from paper table 3 on page 16
    
    for location in ['offshore']: #
    
        max_iter = 3000
            
        folder_dir = './' + str(int(kappa*10)) + '/'
        
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
            
        if location =='onshore':
            z0_air = 0.1
        else:
            z0_air = 0.001
            
        dl_dt_min   = -15 
        dl_dt_max   = 10
        ddl_dt_max  = 10
        dddl_dt_max = 100
       
        n_DOF = 6
        file_out = 'log_wind_A_' + str(A_scaled) + '_kappa_' + str(int(kappa*10)) + '_diamt_' + str(int(diam_t*10000)) 
    
        for hws_ref in np.arange(6,22,3):
    
            ########################
            # SET-UP TRIAL OPTIONS #
            ########################
            options = {}
    
    
            logging.basicConfig(filemode='w',format='%(levelname)s:    %(message)s', level=logging.DEBUG)                    
            
            options['nlp.collocation.u_param'] = 'poly' 
            options['solver.max_iter'] = max_iter 
    
            options['user_options.system_model.architecture']     = {1:0}
            options['user_options.system_model.kite_dof']         = n_DOF
            options['user_options.kite_standard']                 = awe.ampyx_data.data_dict()
            
            options['user_options.trajectory.type']               = 'power_cycle'
            options['user_options.trajectory.system_type']        = 'lift_mode'
            options['user_options.trajectory.lift_mode.windings'] = 5
            
            options['user_options.induction_model']             = 'not_in_use'
            options['user_options.tether_drag_model']           = 'split'
            
            options['model.geometry.overwrite.m_k']             = m_scaled
            options['model.geometry.overwrite.j']               = J_scaled
            options['model.geometry.overwrite.b_ref']           = b_scaled
            options['model.geometry.overwrite.c_ref']           = c_scaled   
                
            options['model.aero.overwrite.alpha_max_deg']       = 30   
            options['model.aero.overwrite.alpha_min_deg']       = -10   
    
    
            options['params.tether.stress_safety_factor']       = 1# remove : set to 1
            options['model.system_bounds.theta.diam_t']         = np.array([diam_t,diam_t])
            options['model.system_bounds.x.dl_t']               = [dl_dt_min, dl_dt_max]  
            options['params.tether.max_stress']                 = 3.6e12 #3.6e9

    
            options['model.model_bounds.tether_force.include']  = True
            options['params.model_bounds.tether_force_limits']  = np.array([1e0, F_constr])
            options['model.system_bounds.x.q']                  = [np.array([-cas.inf, -cas.inf, 10.0]), np.array([cas.inf, cas.inf, cas.inf])]
            options['model.system_bounds.x.q'][0][2]            = 50 + A_scaled/2 #[np.array([-1000, -1000, 50 + A_scaled/2]), np.array([5000, 5000, 5000])]
            
            options['model.model_bounds.airspeed.include']      = False
            options['params.model_bounds.airspeed_limits']      = np.array([10,  85])
            options['model.system_bounds.x.l_t']                = [1.0e-2, 2.0e3]
        
            options['solver.initialization.l_t']                = get_l_init_hws(hws_ref)# initial guess
            options['solver.initialization.inclination_deg']    = 40 # initial guess
            options['solver.initialization.max_cone_angle_single'] = 40 # initial guess
            options['solver.initialization.groundspeed']        = 85 # default 60
                        
            options['user_options.wind.model']                  = 'log_wind'
            options['user_options.wind.u_ref']                  = float(hws_ref)
            options['params.wind.log_wind.z0_air']              = z0_air
            
            ##################
            # OPTIMIZE TRIAL #
            ##################
            
            filename = 'log_A_' + str(A_scaled) + '_' + location + '_ref_w_tether_constr_' +str(hws_ref)
            
            trial = awe.Trial(options, folder_dir + filename)
            trial.build()
            trial.optimize()
            # trial.plot('level_2')
            trial.save()
            # plt.show()
            
            export_results_log(filename,folder_dir,location,A_scaled,kappa,hws_ref,file_out)
                
        
    plt.close('all')

   
    #%% 
    file_AWEbox = pickle.load(open(folder_dir +file_out+'.p','rb'))
    
    fig_trajectory = plt.figure(figsize=(13,7))
    ax_hws      = fig_trajectory.add_subplot(1, 3, 1) 
    ax_traj_1   = fig_trajectory.add_subplot(2, 3, 2) 
    ax_traj_2   = fig_trajectory.add_subplot(2, 3, 5) 
    ax_force    = fig_trajectory.add_subplot(4, 3, 3)
    ax_v_t      = fig_trajectory.add_subplot(4, 3, 6)
    ax_alpha    = fig_trajectory.add_subplot(4, 3, 9)
    ax_power    = fig_trajectory.add_subplot(4, 3, 12)
    ax_hodo     = fig_trajectory.add_subplot(5,5,1)
                   
         
    ax_hws.set_position([0.08, 0.54, 0.2, 0.36])
    ax_traj_1.set_position([0.37, 0.54, 0.24, 0.36])
    ax_traj_2.set_position([0.37, 0.1, 0.24, 0.36])
    ax_power.set_position([0.72, 0.1, 0.26, 0.17])
    ax_alpha.set_position([0.72, 0.31, 0.26, 0.17])
    ax_v_t.set_position([0.72, 0.52, 0.26, 0.17])
    ax_force.set_position([0.72, 0.73, 0.26, 0.17])
    ax_hodo.set_position([0.08, 0.1, 0.2, 0.36])
    
    
    file_list = list(file_AWEbox[location].keys())
    for ident in file_list:
    
        X       = file_AWEbox[location][ident]['X']
        Y       = file_AWEbox[location][ident]['Y']
        Z       = file_AWEbox[location][ident]['Z']
        l_tether       = file_AWEbox[location][ident]['l_tether']
        
        u_in    = file_AWEbox[location][ident]['u_in']
        v_in    = file_AWEbox[location][ident]['v_in']
        z_in    = file_AWEbox[location][ident]['Z_in']
        hws_in  = np.sqrt(u_in**2+v_in**2)
        F_t     = file_AWEbox[location][ident]['tether_force']
        time    = file_AWEbox[location][ident]['time']
        alpha   = np.squeeze(file_AWEbox[location][ident]['alpha'])
        v_t     = file_AWEbox[location][ident]['tether_speed']
        p_t     = file_AWEbox[location][ident]['p_current']
        stress_t= file_AWEbox[location][ident]['tether_stress']
        hws_oper  = np.sqrt(file_AWEbox[location][ident]['U'][0]**2+file_AWEbox[location][ident]['U'][1]**2)
             
    
        ax_traj_1.plot(X,Z)
        ax_traj_2.plot(X,Y)
        ax_hodo.plot(time,l_tether)
        
        
        # ax_hws.plot(hws,z_in)
        ax_hws.plot(hws_oper,Z)
        ax_force.plot(time,F_t/1000)
        ax_v_t.plot(time,v_t)
        ax_power.plot(time,p_t/1000)
        ax_alpha.plot(time,alpha)
        
        ax_hws.grid(True)
        ax_hws.set_xlabel('U [m/s]')
        ax_hws.set_ylabel('z [m]',labelpad=-5)
        ax_hws.set_ylim(0,1000)
        ax_hws.set_xlim(0,30)
    
     
        # ax_hodo.plot(u_in,v_in)
        ax_hodo.grid(True)
        ax_hodo.set_xlabel('u [m/s]')
        ax_hodo.set_ylabel('v [m/s]',labelpad=-5)
        # ax_hodo.set_ylim(-15,15)
        # ax_hodo.set_xlim(0,30)
    
        
        ax_traj_1.grid(True)
        ax_traj_1.set_xlabel('x [m]')
        ax_traj_1.set_ylabel('z [m]',labelpad=5)
        ax_traj_1.set_ylim(0,1000)
        ax_traj_1.set_yticks(np.arange(0,1100,250))
        ax_traj_1.set_xlim(0,1000)
    
        ax_traj_2.grid(True)
        ax_traj_2.set_xlabel('x [m]')
        ax_traj_2.set_ylabel('y [m]',labelpad=-10)
        ax_traj_2.set_ylim(-750,500)
        ax_traj_2.set_yticks(np.arange(-750,510,250))
        ax_traj_2.set_xlim(0,1000)
        
        ax_force.grid(True)
        ax_force.set_ylabel('$F_{tether} [kN]$',labelpad=19)
        ax_force.set_xlim(0,52)
        ax_force.set_xticks(np.arange(0,52,10))
        ax_force.set_xticklabels('')
        # ax_force.set_ylim(0, F_constr/1000+0.1*F_constr/1000)
        ax_force.plot([0,50],[F_constr/1000,F_constr/1000],ls='-',c='k')
     
        ax_v_t.grid(True)
        ax_v_t.set_ylabel('$v_{tether}$ [$ms^{-1}$]',labelpad=18)
    #        ax_v_t.set_xlabel('t [s]')
        ax_v_t.set_xlim(0,52)
        ax_v_t.set_xticks(np.arange(0,52,10))
        ax_v_t.plot([0,50],[-15,-15],'k')
        ax_v_t.set_xticklabels('')
        
        ax_alpha.grid(True)
        ax_alpha.set_ylabel(r'$\alpha$ [$^{\circ}$]',labelpad=28)
    #        ax_alpha.set_xlabel('t [s]')
        ax_alpha.set_xlim(0,52)
        ax_alpha.set_xticks(np.arange(0,52,10))
    
        ax_alpha.set_xticklabels('')
        ax_alpha.set_yticks(np.arange(-15,31,10))
        ax_alpha.set_ylim(-15,30)
    
        ax_power.grid(True)
        ax_power.set_ylabel('$P_{current}$ [kW]',labelpad=0)
        ax_power.set_xlabel('t [s]')
        ax_power.set_xlim(0,52)
        ax_power.set_xticks(np.arange(0,52,10))
         
        
        figure_str = 'timeseries_log_A_' +str(A_scaled) + '_kappa_' + str(kappa) 
    
        fig_trajectory.savefig(folder_dir + figure_str + '.png')    
                    
        
    
    fig       = plt.figure(figsize=(13,7))
    ax_diam_t = fig.add_subplot(4, 1, 1) #,projection='3d'
    ax_F_t  = fig.add_subplot(4, 1, 2) #,projection='3d'
    ax_air  = fig.add_subplot(4, 1, 3) #,projection='3d'
    ax_p  = fig.add_subplot(4, 1, 4) #,projection='3d'
    
    
    ax_diam_t.plot([file_AWEbox[location][ident]['windspeed'] for ident in file_list],[file_AWEbox[location][ident]['diam_tether'][0]*1000 for ident in file_list],'-o')
    ax_diam_t.grid(True)
    ax_diam_t.set_ylabel('$d_{tether}$ [mm]')
    ax_diam_t.set_ylim(diam_t*1000-10,diam_t*1000+10)
    
    ax_F_t.plot([file_AWEbox[location][ident]['windspeed'] for ident in file_list],[np.max(file_AWEbox[location][ident]['tether_force'])/1000 for ident in file_list],'-o')
    ax_F_t.grid(True)
    ax_F_t.set_ylabel('$F_{tether}$ [kN]')
    
    ax_air.plot([file_AWEbox[location][ident]['windspeed'] for ident in file_list],[np.max(file_AWEbox[location][ident]['airspeed']) for ident in file_list],'-o')
    ax_air.grid(True)
    ax_air.set_ylabel('$v_{air}$ [m/s]')
    
    ax_p.plot([file_AWEbox[location][ident]['windspeed'] for ident in file_list],[file_AWEbox[location][ident]['p_avg']/1000 for ident in file_list],'-o')
    ax_p.grid(True)
    ax_p.set_ylabel('$p_{avg}$ [kW]')
    
    figure_str = 'Ft_vair_power_log_A_' +str(A_scaled) + '_kappa_' + str(kappa) 
    
    fig.savefig(folder_dir + figure_str + '.png')    

