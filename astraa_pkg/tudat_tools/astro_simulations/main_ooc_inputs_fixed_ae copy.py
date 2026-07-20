### Feb 2025
# Input: required Az/el for moon-scanning
# Output: csv file of az/el/etc
import os
import numpy as np
import pandas as pd
import datetime as dt
import sys
import importlib
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import json
import attitude_tools.conversions as conv
import paa_tools.paa_calculation as paa_calc
import pointing_calculations.ae_calculation as ae_calc
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as cmbplt
import basic_tools.time_conversion as t_conv
import basic_tools.parsing as parse
import basic_tools.vector_operations as vec_op
import basic_tools.data_loading as load
import basic_tools.in_out as io
import astronomy_tools.astro_targets as where_astro_stuff
import importlib
import matplotlib.dates as mdates
import tudat_tools.tudat_converter as tud
import tudat_tools.astro_simulations.astro_moon_rooftop_azel as where_sun
importlib.reload(ae_calc)
importlib.reload(tud)

currrent_year = 2025
nr_days = 1
day_used = 12
month_used = 2

# UTC to Local time conversion
if month_used > 3 and month_used < 11:
    dt_local2utc = -2
else:
    dt_local2utc = -1
## Input desired Az/El to Collimator
ae_desired = [-89.77, 0.46]


make_single_output = 0
make_full_output = 1

save_all_pointing_inputs = 1
body = f'Colimator_{ae_desired}'
output_folder = r'outputs/tables/col_testing'
# fixed inputs for pos/att
# s_host = np.array([1e7, 1e7, 1e7, 1e3, 1e3, 1e3])
# s_target = np.array([1e7, -1e8, 1e8, 1e3, 1e3, 1e3])
## uncomment  s_host and s_target below to get inputs with only Target Position non-zero
s_host = np.array([0, 0, 0, 0, 0, 0])
# s_target = np.array([
#     10000 * np.sin(np.deg2rad((90 - ae_desired[1]))) * np.cos(np.deg2rad(ae_desired[0])),
#     10000 * np.sin(np.deg2rad((90 - ae_desired[1]))) * np.sin(np.deg2rad(ae_desired[0])),
#     10000 * np.cos(np.deg2rad((90 - ae_desired[1]))),
#     0,0,0]) * 10

s_target = np.array([1e6, -1e6, 1e5, 0, 0, 0])
if make_single_output:
    t_start_local = dt.datetime.now() +dt.timedelta(minutes = 1)
    dt_gps2j2000 = t_conv.dt_gps2j2000tt()
    t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000
    t_j2000_current = t_j2000_start
    t_offset_min = 0
    timedelta = dt.timedelta(minutes = t_offset_min)
    # dt_local2utc
    t_start_local  = t_start_local  + timedelta
    t_utc = t_start_local+ dt.timedelta(hours = -dt_local2utc)
    # set other date/time
    inputs, ae_2_sun = where_sun.ae_oh2col(ae_desired, 
                                               t_j2000 = t_j2000_current,
                                               s_host=s_host,
                                               s_target=s_target)
    # can compare to https://www.timeanddate.com/moon/germany/munich
    print(f'''{body} vector at {t_start_local.isoformat()}        
    Outputs : Az = {ae_2_sun[0]:.2f} deg, El = {ae_2_sun[1]:.2f} deg
    EL  {np.deg2rad(ae_2_sun[1])*1e6 :.1f} urad 
    ''') 

elif make_full_output: # MOON-ANGLE CSV GENERATION FOR SCANS AND TRACKING AND WHATNOT
    mounting_rpy = None # [R, P, Y]
    make_plots = 0
    save_comb = 0 # save all inputs/outputs in a single 
    save_separately = 1 # save all inputs/outputs in a separate csv's
    calc_qdot = 1
    # ADDED TO GENERATE FULL-INPUTS WITH MANUAL AZ CORRECTION INCLUDED IN BF QUATERNION
    force_unity_quat = 1
    ## Only giving az;el;etc to Moon (used in initial scans)
    # setting to 0 will also output pos host (Mynaric hq), pos target, attitude, etc
    output_only_azel = 0
    ## SETTING TO get final rotation with mounting offset quaternion equal to unity

    # for easier parsing 
    # Set up date and resolution

    
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
    loop_length = int(60 / dt_mins* (h_end-h_start + (nr_days-1)*24))
    
    # placeholders
    cest_hoursmins = []
    t_gps = np.zeros((loop_length, 1))
    states_gs = np.zeros((loop_length, 6)) # r, v [m, m/s] of HQ in ECI
    states_sun = np.zeros((loop_length, 6)) # r, v [m, m/s]. v = 0 for sun, not needed
    q_eci2bf = np.zeros((loop_length, 4)) # scalar-first quat from ECI to NED
    q_bf2gf = np.zeros((loop_length, 4)) # scalar-first quat from NED to Terminal Global Frame (North West Up)
    ae_data = np.zeros((loop_length, 2)) # Az, El [rad]
    
    # get J2000 time
    dt_gps2j2000 = t_conv.dt_gps2j2000tt()
    t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000
    datetime_update_rate = int(1/dt_mins/60)


    for ii in range(loop_length):
        if ii == 0:
            t_current = t_start_local
            t_j2000_current = t_j2000_start
        inputs, ae_2_sun = where_sun.ae_oh2col(ae_desired, 
                                               t_j2000 = t_j2000_current,
                                               s_host=s_host,
                                               s_target=s_target)

        cest_hoursmins.append(t_current.time().isoformat())
        #r_gs_eci, r_sun_eci, 
        # quat_eci2bf, quat_bf2gf 1 - ae_2_sun (array): Azimuth, Elevation to sun [deg]
        ae_data[ii,:] = np.deg2rad(ae_2_sun) # rad
        t_gps[ii,:] = np.round(inputs[0], n_digits_used)
        states_gs[ii,:] = inputs[1]
        states_sun[ii,:3] = inputs[2][:3]
        q_eci2bf[ii,:] = inputs[3]
        q_bf2gf[ii,:] = inputs[4]
        t_j2000_current  = t_j2000_current + dt_mins*60
        
        if ii % datetime_update_rate == 0 and ii >= datetime_update_rate:
            # Update CEST date-time tracker every 1 second
            t_current = t_current + constant_dt_upd

    cest_hoursmins = np.array(cest_hoursmins).reshape((loop_length, 1))
    
    if 1: # May 30 Added ref tiem tracker
        title_reftime = 'ref_time'
        output_times_dict = { 'day_used' : [day_used],
            'month_used' : [month_used],
            'h_start' : [h_start],
            'h_end' : [h_end],
            't_res' : [dt_mins*60]}
        df_date = pd.DataFrame.from_dict(output_times_dict)
        df_date.to_csv(f"{full_output_folder}//{title_reftime}.csv", index = False)
    print(f'Loop done. Start : {t_start_local} \nFinal time : {t_current}')

    if calc_qdot:
        # get EA
        ea_all = np.zeros((q_eci2bf.shape[0],3))
        omega_all = np.zeros((q_eci2bf.shape[0],3))
        for ii, q_ii in enumerate(q_eci2bf):
            ea_ii = conv.convert_dcm2ea(conv.convert_quat2dcm(q_ii)) # deg
            ea_all[ii,:] = ea_ii

        # get EA dot
        ea_dot = np.zeros(ea_all.shape)    
        for ii in range(3):
            ea_dot[:,ii] = np.gradient(ea_all[:,ii], t_gps.flatten())
            
        # get Omega
        # get q; qdot
        q_full = np.zeros((ea_all.shape[0],8))
        
        for ii, ea_ii in enumerate(ea_all):
            omega_ii = conv.calc_omega(ea_ii, ea_dot[ii,:], deg = 1) # deg/s            
            q_recal, q_dot = conv.calc_qdot(ea_ii, omega_ii)
            q_full[ii,:4] = q_recal.flatten()
            q_full[ii,4:] = q_dot.flatten()
        q_eci2bf = q_full

    
    if save_all_pointing_inputs:
        output_success, df = io.save_azel(t_gps,
                                          states_gs,
                                          states_sun,
                                          q_eci2bf,
                                          q_bf2gf,
                                          ae_data,
                                          fname = 'gs2moon_data',
                                          full_folder = full_output_folder,
                                          zip_name = zip_name,
                                          make_zip = 1)
        