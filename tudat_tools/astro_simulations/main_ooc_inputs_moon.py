### Oct 2023
# code to output inputs - host pos/vel, target pos/vel, host attitude in quat/quat rate, mounting offset
# and expected Az/el for moon-scanning on Mynaric's rooftop
import matplotlib.pyplot as plt
# import splines.quaternion
import pathlib
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

### OPTIONS
# Compute and print Az/El for a single input as a dt.datetime object
# by default - for current time
make_single_output = 0

save_all_pointing_inputs = 1 # output host pos/vel/att, target pos, etc AND Az El

### make table of full inputs/outputs (states of ground station, target [sun], 
# attitude - quaternion from ECI to global frame (NED) and 
# quaternion mounting_offset (body-frame NED to terminal frame NWU))
# SINGLE DAY
output_day_tabular_data = 1

# illumination + big time steps + months(s) of data for moon scan windows
# MULTI-DAY
output_long_window_data =  0

currrent_year = 2025
nr_days = 1
day_used = 2
month_used = 10

# UTC to Local time conversion
if month_used > 3 and month_used < 11:
    dt_local2utc = -2
else:
    dt_local2utc = -1
body = 'sun'
# body = 'moon'
pt_direction = 'NWU' # North, West, Up terminal pointing
# pt_direction = 'NED' # North, West, Up terminal pointing for CALSPAN debugging


output_folder = r'outputs/tables/sun_vector'
tud_rotator = tud.tudat_predictor()
if make_single_output:    

    t_offset_min = 0
    t_start_local = dt.datetime.now() +dt.timedelta(minutes = 1)
    timedelta = dt.timedelta(minutes = t_offset_min)
    # dt_local2utc
    t_start_local  = t_start_local  + timedelta
    t_utc = t_start_local+ dt.timedelta(hours = -dt_local2utc)
    # set other date/time
       # test time_utc to time now
    t_utc = dt.datetime.now() 
        # end test 

    inputs, ae_now = where_sun.ae_roof2sun(t_utc, tud_rotator, mode = body, pointing = 'NWU')
    # can compare to https://www.timeanddate.com/moon/germany/munich
    print(f'''{body} vector at {t_start_local.isoformat()}        
    Outputs : Az = {ae_now[0]:.2f} deg, El = {ae_now[1]:.2f} deg
    EL  {np.deg2rad(ae_now[1])*1e6 :.1f} urad 
    ''') 

if output_day_tabular_data: # MOON-ANGLE CSV GENERATION FOR SCANS AND TRACKING AND WHATNOT
    # body = 'moon'
    # pt_direction = 'NWU' # North, West, Up terminal pointing
    mounting_rpy = None # [R, P, Y]
    make_plots = 0
    save_comb = 0 # save all inputs/outputs in a single 
    save_separately = 1 # save all inputs/outputs in a separate csv's
    calc_qdot = 1
    # ADDED TO GENERATE FULL-INPUTS WITH MANUAL AZ CORRECTION INCLUDED IN BF QUATERNION
    force_unity_quat = 1
    if force_unity_quat:
        print(f'!!!!!!!!!!Unity MO quaternion option is ON!!!!!!!!!!')
    # MAKE SURE ITS THE SAME AS THE DERIVED VALUE
    # 180 FITTING FOR DEC 6 2023 trials
    manual_az_correction = 0
    # manual_az_correction = 180 # deg
    if manual_az_correction:
        print(f'\n!!!!!!!!!!\nManual AZ CORRECTION = {manual_az_correction:.0f} deg. SET TO 0 IF YOU DONT KNOW WHAT IT DOES\n!!!!!!!!!!\n')
    ## Only giving az;el;etc to Moon (used in initial scans)
    # setting to 0 will also output pos host (Mynaric hq), pos target, attitude, etc
    output_only_azel = 0
    ## SETTING TO get final rotation with mounting offset quaternion equal to unity

    # for easier parsing 
    # Set up date and resolution

    
    h_start = 0# TODO remove
    min_start = 0
    h_end = 24
    t_start_local = dt.datetime(currrent_year, month_used, day_used, h_start, min_start, 0) 
    full_output_folder = f'{output_folder}//{body}_{t_start_local.date().isoformat()}'
    zip_name = f'{body}_{t_start_local.date().isoformat()}.zip'
    try:            
        os.mkdir(full_output_folder)
        print(f'Made folder {full_output_folder}')
    except:
        pass
    # dt_mins = 1 # 10 mins -> 6x24 runs for day-long output
    dt_mins = 1 / 60 # 1 second timesteps
    # dt_mins = 1 / 60 / 200 # 5 milisecond timesteps
    # dt_mins = 1 / 60 / 2 # .5 second timesteps
    # dt_mins = 1 / 60 / 20 # .05 second steps -> ~4 urad 

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
    dt_gps2j2000 = t_conv.dt_gps2j2000tt() # t_j2000 = t_gps + dt_gps2j2000
    t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000
    datetime_update_rate = int(1/dt_mins/60)


    for ii in range(loop_length):
        if ii == 0:
            t_current = t_start_local
            t_j2000_current = t_j2000_start
        inputs, ae_2_sun = where_sun.ae_roof2sun(t_current, tud_rotator, mode = body,
                                        pointing=pt_direction,
                                        manual_az_correction = manual_az_correction, 
                                        mounting_offset = mounting_rpy, 
                                        t_j2000 = t_j2000_current, dt_gps2j2000 = dt_gps2j2000, 
                                        force_unity_quaternion = force_unity_quat)

        cest_hoursmins.append(t_current.time().isoformat())
        #r_gs_eci, r_sun_eci, 
        # quat_eci2bf, quat_bf2gf 1 - ae_2_sun (array): Azimuth, Elevation to sun [deg]
        ae_data[ii,:] = np.deg2rad(ae_2_sun) # rad
        t_gps[ii,:] = np.round(inputs[0], n_digits_used)
        states_gs[ii,:] = inputs[1]
        states_sun[ii,:3] = inputs[2]
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
        


if output_long_window_data:
    ### TO PROCESS OUTPUT
    # TO FILTER TABLE: SCRIPT IN ANALYSES->OOC->MOON_VIS_CONDITIOSN->main_sleepover.py
    ### TO PROCESS OUTPUT
    # pt_direction = 'NWU' # North, West, Up terminal pointing
    mounting_rpy = None # [R, P, Y]
    save_separately = 1 # save all inputs/outputs in a separate csv's
    
    ## Only giving az;el;etc to Moon (used in initial scans)
    output_only_azel = 1 
    current_year = 2024
    nr_days = 180
    day_used = 8
    month_used = 4
    h_start = 0
    h_end = 24
    if month_used > 3 and month_used < 11:
        dt_local2utc = -2
    else:
        dt_local2utc = -1

    t_start_local = dt.datetime(current_year, month_used, day_used, h_start, 0, 0) 
    full_output_folder = f'{output_folder}//{body}_{t_start_local.date().isoformat()}_big'
    try:            
        os.mkdir(full_output_folder)
        print(f'Made folder {full_output_folder}')
    except:
        pass
    dt_mins = 3600 / 60 # 60 minute timesteps

    n_digits_used = len(str(dt_mins*60))-2 # 3 digits for 5 ms

    dt_loop = dt.timedelta(minutes = dt_mins)
    constant_dt_upd = dt.timedelta(seconds = dt_mins*60)
    loop_length = int(60 / dt_mins* (h_end-h_start + (nr_days-1)*24))
    
    # placeholders
    cest_hoursmins = []
    cest_days = []
    t_gps = np.zeros((loop_length, 1))
    ae_data = np.zeros((loop_length, 2)) # Az, El [rad]
    illumination = np.zeros((loop_length,1))
    day_seconds = np.zeros((loop_length,1))
    # get J2000 time
    dt_gps2j2000 = t_conv.dt_gps2j2000tt() # t_j2000 = t_gps + dt_gps2j2000
    t_j2000_start = t_conv.utc2gws(t_start_local+ dt.timedelta(hours = dt_local2utc)) + dt_gps2j2000


    for ii in range(loop_length):
        if ii == 0:
            t_current = t_start_local
            t_j2000_current = t_j2000_start
        inputs, ae_2_sun = where_sun.ae_roof2sun(t_current, tud_rotator, mode = body,
                                    pointing=pt_direction, mounting_offset = mounting_rpy, 
                                    t_j2000 = t_j2000_current, dt_gps2j2000 = dt_gps2j2000)

        cest_hoursmins.append(t_current.time().isoformat())
        cest_days.append(t_current.date().isoformat())
        day_seconds[ii,:] = t_current.hour*3600+t_current.minute*60 + t_current.second
        #r_gs_eci, r_sun_eci, 
        ae_data[ii,:] = ae_2_sun # deg
        t_gps[ii,:] = np.round(inputs[0], n_digits_used)
        # Get moon illumination
        r_moon_ii, illumination_ii = where_astro_stuff.compute_moon_vector_eci(inputs[0], what_brightness=1)
        illumination[ii] = illumination_ii
        t_j2000_current  = t_j2000_current + dt_mins*60
                    
        t_current = t_current + constant_dt_upd

    cest_hoursmins = np.array(cest_hoursmins).reshape((loop_length, 1))
    cest_days = np.array(cest_days).reshape((loop_length, 1))
    if save_separately:
        # AZ EL ANGLES IN deg
        if output_only_azel:
            data_output = [ae_data]
            titles = [ f'Long_aeILUM_gs2{body}']
            columns = [
                    ['t_gps_s', 'CEST_yr_mo_d','CEST_hms', 's_of_day', 'az_deg', 'el_deg', 'illum'],
                    ]
        for ii, title in enumerate(titles):
            
            output_df = pd.DataFrame(data = np.hstack((t_gps, cest_days, cest_hoursmins, day_seconds, data_output[ii], illumination)), columns = columns[ii])

            print(f'Saving {title}')
            try:                                
                output_df.to_csv(f"{full_output_folder}//{title}.csv", index = False)
            except:
                os.mkdir(full_output_folder)
                output_df.to_csv(f"{full_output_folder}//{title}.csv", index = False)
            print(f'Saved data MOON vector data')