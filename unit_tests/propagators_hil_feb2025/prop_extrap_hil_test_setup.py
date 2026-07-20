#%% IMPORTS

## Prep test data for J2 proapgator and host pos/attitude extrapolators
# Feb 2025

# Req columns: 
# t_GPS;s_DATA IN : Timestamped data, as inputs from sat bus [dt = 1s/10s/whatever]
# t_GPS_now : GPS time in the PRESENT; 5 ms steps
# t_GPS; s_true: true data, timestamped data as truth; 5 ms steps

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
import scipy as sp
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import plotting_tools.modular_plotting as modplot
# path jazz
path_cwd = os.getcwd()

csv_output_path = r'orbital_simulations/leo_for_prop_hil/leo_leo'

fname_simparam = 'simulation_parameters.json'
fname_states = 'states_fine.dat' # 1s steps
# fname_states = 'state_history.dat' # 60s steps

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop
import prediction_methods.interpolators as interp
import prediction_methods.error_generation as errgen
importlib.reload(out)
import attitude_tools.attitude_simulation as att_sim
import basic_tools.in_out as io
import pointing_calculations.ae_calculation as ae_calc
import plotting_tools.combined_plots as cmb_plt   
import attitude_tools.conversions as conv
import pointing_calculations.ae_calculation as ae_calc
import tudat_tools.astro_simulations.astro_moon_rooftop_azel as where_sun


## Loading satellite orbital data
importlib.reload(j2prop)
import tudat_tools.data_processing.data_processing_utilities as dputil
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path,
                                                                 state_name=fname_states)

host_chosen = simulation_parameters['sat_names'][2]
target_chosen = simulation_parameters['sat_names'][-3]
output_folder = r'outputs/tables/prop_hil_2025' 
label = 'leo_leo'
full_output_folder = fr'{output_folder}/{label}' 

check_setup_aer = 1
check_inputs = 1
# ae_desired = [-89.7, 0.46]
ae_desired = [45, 0.46]

# ae_desired = [45, 15]
make_single_output = 0
make_full_output = 1
currrent_year = 2025
nr_days = 1
day_used = 28
month_used = 2

t_minimum = 21 # minutes
t_max = 25
t_from_0 = data_raw[:,[0]] - data_raw[0,[0]]
ii_0 = np.where(t_from_0 > t_minimum*60)[0][0]
ii_f = np.where(t_from_0 < t_max*60)[0][-1]
t_sliced = t_from_0[ii_0:ii_f,[0]]
t_from_0 = t_sliced - t_sliced[0]
r_host = data_raw[ii_0:ii_f,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[ii_0:ii_f,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
r_target = data_raw[ii_0:ii_f,simulation_parameters['r_index'][target_chosen]]
v_target = data_raw[ii_0:ii_f,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
states_host = np.hstack((r_host, v_host))
states_target= np.hstack((r_target, v_target))
aer_all = ae_calc.calc_ae_full(
    states_host = states_host,
    states_target= states_target,
    rotation_function=1,
    default_offset='down_along',
)
# UTC to Local time conversion
if month_used > 3 and month_used < 11:
    dt_local2utc = -2
else:
    dt_local2utc = -1
## Input desired Az/El to Collimator


save_all_pointing_inputs = 1
body = f'Colimator_{ae_desired}'
output_folder = r'outputs/tables/prop_testing'

if make_full_output: 
    mounting_rpy = None # [R, P, Y]
    make_plots = 0
    save_comb = 0 # save all inputs/outputs in a single 
    save_separately = 1 # save all inputs/outputs in a separate csv's
    calc_qdot = 1
    force_unity_quat = 1
    output_only_azel = 0

    
    h_start = 0
    h_end = 24
    t_start_local = dt.datetime(currrent_year, month_used, day_used, h_start, 0, 0) 
    full_output_folder = f'{output_folder}//{body}_{t_start_local.date().isoformat()}'
    zip_name = f'{body}_{t_start_local.date().isoformat()}.zip'
    try:            
        os.mkdir(full_output_folder)
        print(f'Made folder {full_output_folder}')
    except:
        pass
    dt_mins = 1 / 60 # 1 second timesteps

    n_digits_used = len(str(dt_mins*60))-2 # 3 digits for 5 ms

    dt_loop = dt.timedelta(minutes = dt_mins)
    constant_dt_upd = dt.timedelta(seconds = 1)
    loop_length = r_host.shape[0]
    # loop_length = int(60 / dt_mins* (h_end-h_start + (nr_days-1)*24))
    
    # placeholders
    cest_hoursmins = []
    t_gps_storage = np.zeros((loop_length, 1))
    s_host_storage = np.zeros((loop_length, 6)) # r, v [m, m/s] of HQ in ECI
    s_target_storage = np.zeros((loop_length, 6)) 
    q_eci2bf_storage = np.zeros((loop_length, 8)) # scalar-first quat from ECI to NED
    q_bf2gf_storage = np.zeros((loop_length, 4)) # scalar-first quat from NED to Terminal Global Frame (North West Up)
    ae_data_storage = np.zeros((loop_length, 2)) # Az, El [rad]
    
    # get J2000 time
    dt_gps2j2000 = t_conv.dt_gps2j2000tt()
    t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000
    datetime_update_rate = int(1/dt_mins/60)

    # Compute for provided host, target states
    for ii, s_ii in enumerate(states_host):
        if ii == 0:
            t_current = t_start_local
            t_j2000_current = t_j2000_start        
        s_host = s_ii
        s_target = states_target[ii,:]
        inputs, ae_2_sun = where_sun.ae_oh2col(ae_desired, 
                                               t_j2000 = t_j2000_current,
                                               s_host=s_host,
                                               s_target=s_target)

        cest_hoursmins.append(t_current.time().isoformat())
        #r_gs_eci, r_sun_eci, 
        # quat_eci2bf, quat_bf2gf 1 - ae_2_sun (array): Azimuth, Elevation to sun [deg]
        t_gps_storage[ii,:] = np.round(inputs[0], n_digits_used)
        s_host_storage[ii,:] = inputs[1]
        s_target_storage[ii,:] = inputs[2]
        q_eci2bf_storage[ii,:4] = inputs[3]
        q_bf2gf_storage[ii,:] = inputs[4]
        ae_data_storage[ii,:] = np.deg2rad(ae_2_sun) # rad
        t_j2000_current  = t_j2000_current + dt_mins*60
        
        if ii % datetime_update_rate == 0 and ii >= datetime_update_rate:
            # Update CEST date-time tracker every 1 second
            t_current = t_current + constant_dt_upd
    # index tracker to loop states forward and back for entire day

    if calc_qdot:
        # get EA
        ii_calculated = ii
        ea_all = np.zeros((ii,3))
        omega_all = np.zeros((ii,3))
        for jj, q_ii in enumerate(q_eci2bf_storage[:ii,:]):
            ea_ii = conv.convert_dcm2ea(conv.convert_quat2dcm(q_ii)) # deg
            ea_all[jj,:] = ea_ii

        # get EA dot
        ea_dot = np.zeros(ea_all.shape)
        for ii in range(3):
            ea_dot[:,ii] = np.gradient(ea_all[:,ii], t_gps_storage[:ii_calculated].flatten())
            
        # get Omega
        # get q; qdot
        q_full = np.zeros((ea_all.shape[0],8))
        
        for ii, ea_ii in enumerate(ea_all):
            omega_ii = conv.calc_omega(ea_ii, ea_dot[ii,:], deg = 1) # deg/s            
            q_recal, q_dot = conv.calc_qdot(ea_ii, omega_ii, q = None)
            # q_recal, q_dot = conv.calc_qdot(ea_ii, omega_ii, q = q_eci2bf_storage[ii,:4])
            q_full[ii,:4] = q_recal.flatten()
            q_full[ii,4:] = q_dot.flatten()
            omega_all[ii,:] = omega_ii
        q_eci2bf = q_full
        q_eci2bf_storage[:ii_calculated,:] = q_eci2bf
    q_eci2bf_storage[:,4:] = np.gradient(q_eci2bf_storage[:,:4], t_gps_storage.flatten(), axis = 0)
    jj = 0
    jj_limit = np.shape(states_host)[0]
    
    dt_mins = 1 / 60 # 1 second timesteps
    for ii in range(loop_length):
        if ii >= jj:
            # start storing
            t_gps_storage[ii,:] = np.round(t_gps_storage[0,:] + ii*dt_mins*60, n_digits_used)
            s_host_storage[ii,:] = s_host_storage[jj,:]
            s_target_storage[ii,:] = s_target_storage[jj,:]
            q_eci2bf_storage[ii,:] = q_eci2bf_storage[jj,:]
            q_bf2gf_storage[ii,:] = q_bf2gf_storage[jj,:]
            ae_data_storage[ii,:] = ae_data_storage[jj,:]
            t_j2000_current  = t_j2000_current + dt_mins*60
        
        jj += 1
        if jj == jj_limit:
            jj = 0        
        
    if check_inputs:
        f_host = cmb_plt.plot_states(s_host_storage, t_in = t_gps_storage-t_gps_storage[0], title = 'Host')        
        f_target = cmb_plt.plot_states(s_target_storage, t_in = t_gps_storage-t_gps_storage[0], title = 'Target')
        f_attitude = cmb_plt.plot_quats(q_eci2bf_storage, t_in = t_gps_storage-t_gps_storage[0], title = 'Attitude')        
        plt.show()
        
    if save_all_pointing_inputs:
        output_success, df = io.save_azel(t_gps_storage,
                                          s_host_storage,
                                          s_target_storage,
                                          q_eci2bf_storage,
                                          q_bf2gf_storage,
                                          ae_data_storage,
                                          fname = 'gs2moon_data',
                                          full_folder = full_output_folder,
                                          zip_name = zip_name,
                                          make_zip = 1)    

    
    if 1:
        title_reftime = 'ref_time'
        output_times_dict = { 'day_used' : [day_used],
            'month_used' : [month_used],
            'h_start' : [h_start],
            'h_end' : [h_end],
            't_res' : [dt_mins*60]}
        df_date = pd.DataFrame.from_dict(output_times_dict)
        df_date.to_csv(f"{full_output_folder}//{title_reftime}.csv", index = False)
    print(f'Loop done. Start : {t_start_local} \nFinal time : {t_current}')
    
        


if check_setup_aer:    
    aer_for_gradient = np.copy(aer_all)
    aer_for_gradient[:,0] = np.unwrap(aer_for_gradient[:,0])
    aer_gradient_all = np.gradient(aer_for_gradient, t_from_0.flatten(), axis = 0)
    aer_gradient_all[:,:2] = np.rad2deg(aer_gradient_all[:,:2])
    aer_all_deg = np.copy(aer_all)
    aer_all_deg[:,:2] = np.rad2deg(aer_all_deg[:,:2])
    f, axs = cmb_plt.plot_aer(t_from_0,
                              aer_all_deg,
                              unit = 'deg',
                              title = f'{host_chosen}-{target_chosen} NO attitude added',
                              setting='normal',
                              )
    
    f, axs = cmb_plt.plot_aer(t_from_0,
                              aer_gradient_all,
                              unit = 'deg',
                              title = f'{host_chosen}-{target_chosen} NO attitude added',
                              )    
    plt.show()