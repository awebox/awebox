#!/usr/bin/python3

import awebox as awe
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
import os
import casadi as cas
def get_main_wind_components(u_data,v_data,z_data):
    # cluster u,v profiles as main directon and deviation
    # hws=np.sqrt(u_NoOBS[:,idx_alt_min:idx_alt_max]**2+v_NoOBS[:,idx_alt_min:idx_alt_max]**2)

#    dir_xy          = np.arctan2(v_data[:,:idx_out_max],u_data[:,:idx_out_max])
    hws = np.sqrt(u_data**2+v_data**2)
    hws_mean = np.mean(np.interp(np.arange(100,401,25),z_data,hws))
    u_mean = np.mean(np.interp(np.arange(100,401,25),z_data,u_data))
    v_mean = np.mean(np.interp(np.arange(100,401,25),z_data,v_data))
    
    dir_mean        = np.arctan2(u_mean,v_mean)
    u_deviation     = np.cos(dir_mean)*u_data - np.sin(dir_mean)*v_data
    u_main          = np.sin(dir_mean)*u_data + np.cos(dir_mean)*v_data
    
    return(u_main,u_deviation)

def sort_clusters(centroids,Z_mean,cluster_index,location):
    
    if location == 'onshore':
        
        cluster_mean  = [np.mean(np.interp(np.arange(100,401,25),Z_mean,idx[::2])) for idx in centroids]
        k_cluster_sorted      = np.argsort(cluster_mean)
    elif location == 'offshore':
        cluster_mean  = [np.mean(np.interp(np.arange(100,401,25),Z_mean,idx)) for idx in centroids]
        k_cluster_sorted      = np.argsort(cluster_mean)


    return(k_cluster_sorted)

def get_cluster_wind(location,str_cluster):
    import pickle

    if location == 'onshore':
        file = '/media/msommerf/dataexchange/Phd_awebox/wind_data/from website/Pritzwalk_cluster_wind_interp.p'
        
        
    elif location == 'offshore':
        file = '/media/msommerf/dataexchange/Phd_awebox/wind_data/from website/FINO3_cluster_wind_interp.p'
    
    wind_data           = pickle.load(open(file,'rb'))
    cluster_name_idx    = np.where([str_cluster in k_str for k_str in list(wind_data['cluster_object'].keys())])[0][0]
    cluster_name        = list(wind_data['cluster_object'].keys())[cluster_name_idx]
    
    cluster_index       = wind_data['cluster_object'][cluster_name].labels_ 
    u_main              = wind_data['wind_data']['u_main']
    u_deviation         = wind_data['wind_data']['u_deviation']
    Z_mean              = wind_data['wind_data']['Z_mean']
    
    centroids           = wind_data['cluster_object'][cluster_name].cluster_centers_   
    k_cluster_sorted    = sort_clusters(centroids,Z_mean,cluster_index,location)
    
    return(u_main,u_deviation,Z_mean,cluster_index,k_cluster_sorted)


def get_percentile_wind(u_main,u_deviation,Z_mean,cluster_index,k_cluster,p_percentile):
    
    u_main_cluster      = u_main[cluster_index==k_cluster]
    u_deviation_cluster = u_deviation[cluster_index==k_cluster]
    
    hws                 = np.sqrt(u_main_cluster**2+u_deviation_cluster**2)
    hws_mean            = [np.mean(np.interp(np.arange(100,401,25),Z_mean,idx)) for idx in hws]
    hws_percentile      = np.percentile(hws_mean,p_percentile,interpolation='nearest')
    idx_percentile      = np.where(hws_percentile==hws_mean)[0][0]
    
    u_percentile        = u_main_cluster[idx_percentile+1]
    v_percentile        = u_deviation_cluster[idx_percentile+1]
    
    return(u_percentile,v_percentile,Z_mean,np.round(hws_percentile,2))

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

# def get_l_init(kappa,A_scaled):
    
#     if kappa < 2.6:
#         l_init = ((200/30*A_scaled +1166+2/3)-500)/7*hws_mean+500-5*((200/30*A_scaled +1166+2/3)-500)/7
#     else:
#         l_init = ((300/30*A_scaled +1300)-500)/7*hws_mean+500-5*((300/30*A_scaled +1300)-500)/7
        
#     return(l_init)
def get_l_init_hws(hws_mean):
    
    
    if hws_mean <10 :
        l_init = 600
    elif hws_mean<13:
        l_init = 800
    else:
        l_init = 1000
        
    return(l_init)

def get_d_tether(A_scaled):
    
    # if A_scaled == 20:
    #     d_tether = 0.0078
    #     F_constr = 60*1000
        
    #  else:
    d_20 = 0.0078
    F_20 = 60*1000
    
    d_tether = np.round(d_20*np.sqrt(A_scaled/20),4)
    F_constr = np.round(F_20*A_scaled/20,4)


    return(d_tether,F_constr)



def export_results(filename,folder_dir,location,A_scaled,kappa,str_cluster,k_cluster,p_percentile,u_percentile,v_percentile,Z_mean,hws_mean,file_out):
# =============================================================================
#     define this function!!
# =============================================================================
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
    if not str_cluster in data_out[location]:
        data_out[location][str_cluster]={}
    if not str(k_cluster) in data_out[location][str_cluster]:
        data_out[location][str_cluster][str(k_cluster)]={}
    if not 'p_' + str(p_percentile) in data_out[location][str_cluster][str(k_cluster)]:
        data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)] = {}
        
        
    tether_diameter_constr              = data_in['plot_dict']['options']['model']['system_bounds']['theta']['diam_t'][0]
    tether_crosssection                 = tether_diameter_constr**2*np.pi/4
    X,Y,Z                               = [np.array(np.squeeze(data_in['plot_dict']['xd']['q10'][i])) for i in range(3)]
    tether_max_stress                   = np.array(data_in['plot_dict']['options']['params']['tether']['max_stress'])
    tether_speed_max                    = np.round(np.max(np.squeeze(np.array(data_in['plot_dict']['xd']['dl_t'][0]))),1)

    
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['U'] = np.array(data_in['plot_dict']['outputs']['aerodynamics']['u_infty1'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['windspeed'] = np.mean(np.array(data_in['plot_dict']['outputs']['environment']['windspeed1']))
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['X']          = X
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['Y']          = Y
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['Z']          = Z
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['time']       = np.array(data_in['plot_dict']['time_grids']['ip'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['p_current']  = np.squeeze(np.array(data_in['plot_dict']['outputs']['performance']['p_current']))
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['f_Loyd']     = np.array(data_in['plot_dict']['outputs']['performance']['loyd_factor'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['p_avg']      = np.array(data_in['plot_dict']['power_and_performance']['avg_power'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['e_final']    = np.array(data_in['plot_dict']['power_and_performance']['e_final'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['l_tether']   = np.squeeze(np.array(data_in['plot_dict']['xd']['l_t'][0]))
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['alpha']      = np.array(data_in['plot_dict']['outputs']['aerodynamics']['alpha_deg1']) 
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['beta']       = np.array(data_in['plot_dict']['outputs']['aerodynamics']['beta_deg1']) 
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force']  = np.array(data_in['plot_dict']['outputs']['local_performance']['tether_force10'][0])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_stress'] = np.array(data_in['plot_dict']['outputs']['local_performance']['tether_stress10'][0])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force_constr']    = data_in['plot_dict']['options']['model']['params']['model_bounds']['tether_force_limits'][1]
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_f_safety_internal'] = np.array(data_in['plot_dict']['options']['params']['tether']['stress_safety_factor'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_max_force']       = np.array(tether_crosssection*tether_max_stress)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force_fsafety']   = np.round((tether_diameter_constr**2/4*np.pi*3.6e9)/data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force_constr'],1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['p_avg']              = np.squeeze(np.array(data_in['plot_dict']['power_and_performance']['avg_power']))
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_speed']       = np.squeeze(np.round(np.array(data_in['plot_dict']['xd']['dl_t'][0]),2))
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_speed_min']   = np.round(np.min(np.squeeze(np.array(data_in['plot_dict']['xd']['dl_t'][0]))),1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_speed_max']   = np.round(np.max(np.squeeze(np.array(data_in['plot_dict']['xd']['dl_t'][0]))),1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['p_max_constr']       = np.round(tether_speed_max.max()*data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force_constr'])
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['diam_tether']        = np.sqrt(np.divide(data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_force'] ,data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['tether_stress'])*4/np.pi)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['mass']               = np.round(data_in['plot_dict']['options']['params']['geometry']['m_k'],1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['airspeed']           = np.round(data_in['plot_dict']['outputs']['aerodynamics']['airspeed1'],1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['alpha']              = np.round(data_in['plot_dict']['outputs']['aerodynamics']['alpha_deg1'],1)
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['beta']               = np.round( data_in['plot_dict']['outputs']['aerodynamics']['beta_deg1'],1)
    

    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['Z_in']               = Z_mean
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['u_in']               = u_percentile
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['v_in']               = v_percentile
    data_out[location][str_cluster][str(k_cluster)]['p_' + str(p_percentile)]['hws_mean_in']        = hws_mean

    pickle.dump( data_out, open(folder_dir + file_out + ".p", "wb" ) )
    
    return(data_out)
#########################
# DEFINE MASS & INERTIA #
#########################
for A_scaled  in [80]:
    for kappa in [3.3]:
        m_scaled,J_scaled,b_scaled,c_scaled = get_mass_inertia_AP2(A_scaled,kappa)
        
        ############################
        # DEFINE TETHER CONSTRAINT #
        ############################
        diam_t , F_constr = get_d_tether(A_scaled) # from paper table 3 on page 16
        
        for location in ['onshore']: #
            if location =='onshore':
                z0_air = 0.1
            else:
                z0_air = 0.001
            for str_cluster in ['k_20']: #'k_50','k_100'
                u_main,u_deviation,Z_mean,cluster_index,k_cluster_sorted = get_cluster_wind(location,str_cluster)
                
                file_out = 'A_' + str(A_scaled) + '_kappa_' + str(int(kappa*10))
    
                max_iter = 3000
                    
                # folder_dir = '/media/msommerf/dataexchange/Phd_awebox/wes_123_sizing/awebox_rerun/results/A' + str(A_scaled) +'/kappa'  + str(int(kappa*10)) + '/'
                folder_dir = '/media/msommerf/dataexchange/Phd_awebox/wes_123_sizing/awebox_rerun/results/A_' + str(A_scaled) + '/kappa_' + str(int(kappa*10)) + '/' + str_cluster +'/'
        
                if not os.path.exists(folder_dir):
                    os.makedirs(folder_dir)
                    
                for k_cluster in k_cluster_sorted:#[np.where(k_cluster_sorted==4)[0][0]:]: # [np.where(k_cluster_sorted==9)[0][0]:]: #enumerate(k_cluster_sorted)
                    p_List = [5,50,95]#[5,50,95]
                   
                    for p_percentile in p_List:
                
        
                        u_percentile,v_percentile,Z_mean,hws_mean = get_percentile_wind(u_main,u_deviation,Z_mean,cluster_index,k_cluster,p_percentile)
                        print(str(k_cluster) + '; percentile(' +str(p_percentile) +'): ' + str(np.round(np.mean(np.sqrt(u_percentile**2+v_percentile**2)),1)))
                                    
                        dl_dt_min = -15 
                        dl_dt_max = 10
                        ddl_dt_max = 10
                        dddl_dt_max = 100
                       
                        n_DOF = 6
                    
                        if hws_mean>4.5:
                
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
                        
                            options['solver.initialization.l_t']                = get_l_init_hws(hws_mean)# initial guess
                            options['solver.initialization.inclination_deg']    = 40 # initial guess
                            options['solver.initialization.max_cone_angle_single'] = 20# initial guess
                            options['solver.initialization.groundspeed']        = 85 # default 60
                           
                            k_heightlvl = [1,2,4,7,9,11,16,18] 
                            options['user_options.wind.model']                  = 'WRF'
                            options['user_options.wind.WRF_heightsdata']        = Z_mean[k_heightlvl]#[::k_heightlvl]
                            options['user_options.wind.WRF_winddata']           = np.stack((u_percentile,v_percentile)).T[k_heightlvl]#[::k_heightlvl]
                            options['user_options.wind.u_ref']                  = hws_mean # hws_100 #hws_10
                            options['params.wind.z_ref']                        = 100 # default is 10
                    
                            ##################
                            # OPTIMIZE TRIAL #
                            ##################
                            
                            filename = 'A_'+str(A_scaled)+'_m_'+str(int(m_scaled))+'_Uref_' +str(hws_mean) + '_' + location + '_'+ str_cluster+ '_' + str(k_cluster) + '_p_' + str(p_percentile)
                            
                            trial = awe.Trial(options, folder_dir + filename)
                            trial.build()
                            trial.optimize()
                            # trial.plot('level_2')
                            trial.save()
                            # plt.show()
                            
    
