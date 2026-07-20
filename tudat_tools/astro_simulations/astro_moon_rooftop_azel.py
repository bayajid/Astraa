### THE CODE to get sun's azimuth and elevation from the roof.
# Outputs all kinds of stuff:
# such as true sun vector in ECI
# attitude from NED or some other frame to ECI
# ECEF and ECI positions
# Long/Lat altitude
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
import astronomy_tools.element_conversion as elem_conv
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
import astronomy_tools.astro_targets as where_sun
import tudat_tools.tudat_converter as tud
import attitude_tools.attitude_resolution as att_res
import importlib
import matplotlib.dates as mdates
importlib.reload(ae_calc)
importlib.reload(tud)

def ae_roof2sun(t_in_cest, TUD_rotator = None, mode = 'sun', coord_hq = [48.137017, 11.419067, 0], #567.5
                pointing = 'NWU', manual_az_correction = 0, mounting_offset = None, sun_vec = None,
                t_j2000 = None, dt_gps2j2000 = None, force_unity_quaternion = 0, print_outputs = 0):
    """Integrated function to compute Azimuth and Elevation angles of the sun
    from the roof of Mynaric headquarters

    Args:
        t_in_cest (dt.datetime object): Chosen date+time eg dt.datetime(2023, 5, 3, 6, 0, 0) for May 3rd
        TUD_rotator (class, optional): custom TUD rotator class. Defaults to None.
        coord_hq (list, optional): Latitude/longitude of Headquaters. 
            Defaults to [48.137017, 11.419067] for Triebwerk
            [48.090557, 11.309388, 0] for old Mynaric office
            
        pointing (str, optional): Mounting offset setting, to rotate from NED to glboal frame.
            Defaults to 'NWU' North West Up
        mounting_offset [RPY] - Roll, Pitch, Yaw angles for a 3-2-1 mounting offset. Set pointing = 'custom'
        force_unity_quaternion (bool, optional)- whether the Body-frame quaternion also includes the required mounting offset rotation
         to provide default pointing (ie [0 1 0 0] for NWU pointing) with a unity monuting offset [1 0 0 0] in SW
        print_outputs (int, optional): _description_. Defaults to 0.

    Returns:
        0 - inputs (tuple), t_gps, r_gs_eci, r_sun_eci, quat_eci2bf, quat_bf2gf
        1 - ae_2_sun (array): Azimuth, Elevation to sun [deg]
    """    

    if mode == 'sun':
        fct_astrobody = where_sun.compute_sun_vector_eci_better
    elif mode == 'moon':
        fct_astrobody = where_sun.compute_moon_vector_eci

    # TODO add expected sun vector error from skyfield
    if type(TUD_rotator) == type(None):
        TUD_rotator = tud.tudat_predictor()
    
    # If no J2000 time is input, use CEST
    if type(t_j2000) == type(None):
        time_utc = t_in_cest + dt.timedelta(hours = -2)
        t_gps = t_conv.utc2gws(time_utc)
        TUD_rotator.set_time(time_utc)
    else:
        ### FASTER TIME PROCESSING. Runs if J2000 != None
        t_gps = t_j2000 - dt_gps2j2000 
        TUD_rotator.set_time(t_j2000)
    
    # get ECI position of ground headquarters
    # hq_ecef = TUD_rotator.convert_lla2ecef(coord_hq)
    hq_ecef = elem_conv.lla_2_ecef(coord_hq)
    hq_ecef = np.concatenate((hq_ecef, [0,0,0]))
    r_gs_eci = TUD_rotator.rotate_ecef2eci(hq_ecef)
    
    if mode == 'sun' or mode == 'moon':
        # get ECI sun-vector
        if type(sun_vec) == type(None):
            r_target_eci = fct_astrobody(t_gps)
        else:
            r_target_eci = sun_vec
    elif mode == 'innsbruck':
        # get Innsbruck ECI position
        coord_target = [47.2675, 11.391, 574] # lat long h
        target_ecef = elem_conv.lla_2_ecef(coord_target)
        target_ecef = np.concatenate((target_ecef, [0,0,0]))
        r_target_eci = TUD_rotator.rotate_ecef2eci(target_ecef)
    
    # get attitude
    rot_ecef2ned = TUD_rotator.rot_ecef2ned(coord_hq)
    # rot_ecef2eci = tud_conv.rot_ecef2eci()
    rot_eci2ecef = TUD_rotator.rot_eci2ecef()
    if pointing == 'NED': # TODO add more terminal orientation options
        rot_bf2gf = np.eye(3) # TODO this will represent the mounting offset. Should be added as a separate input.
        # and set to a constant
        quat_bf2gf = conv.convert_dcm2quat(rot_bf2gf)
    elif pointing == 'NWU': # align global frame with North - West - Up
        # elevation - positive up. Azimuth positive west-wards
        if 0: # 180 deg singularity of converting DCM to quat
            rot_bf2gf = conv.convert_ea2dcm([180,0,0])
            quat_bf2gf = conv.convert_dcm2quat(rot_bf2gf)
        else:
            rot_bf2gf = conv.convert_ea2dcm([180,0,0])
            quat_bf2gf = np.array([0,1,0,0])
    else:
        rot_bf2gf = conv.convert_ea2dcm(mounting_offset)
        quat_bf2gf = conv.convert_dcm2quat(rot_bf2gf)
    
    # ALWAYS NWU
    # rot_bf2gf = conv.convert_ea2dcm([180,0,0])
    # quat_bf2gf = conv.convert_dcm2quat(rot_bf2gf)
    
    # Convert to qutaernions
    # Dec 2023 addition
    # Aritificial quaternion to keep normal Az/El but change the body-frame attitude
    # such that the terminal points in the right direction w/ full inputs
    # without having to use the true mounting offset quaternion
        # _returned - attitude such that a unity MO quaternion leads to right az/el
        # _true -  actual BF for terminal on earth surface and MO to align it with NWU
    quat_bf2gf_returned = quat_bf2gf
    rot_eci2bf_true = rot_ecef2ned @ rot_eci2ecef
    rot_eci2bf_returned = rot_eci2bf_true 
    if force_unity_quaternion:
        quat_bf2gf_returned = np.array([1,0,0,0])
        # Also include manual az correction as a rotation about Z
        if manual_az_correction == 0:
            rot_eci2bf_returned = rot_bf2gf @ rot_eci2bf_true
        else:
            rot_manual_az = conv.convert_ea2dcm([0, 0, manual_az_correction])
            rot_eci2bf_returned = rot_manual_az @ rot_bf2gf @ rot_eci2bf_true
    quat_eci2bf_true = conv.convert_dcm2quat(rot_eci2bf_true)    
    quat_eci2bf_returned = conv.convert_dcm2quat(rot_eci2bf_returned) 
    
    # Get Az/El
    inputs = (t_gps, r_gs_eci, r_target_eci, quat_eci2bf_returned, quat_bf2gf_returned)
    # inputs = (t_gps, r_gs_eci, r_target_eci, quat_eci2bf, quat_bf2gf)
    ae_2_sun = np.rad2deg(ae_calc.calc_ae_full(r_gs_eci, r_target_eci, attitude_eci2bf = quat_eci2bf_true,
                                                attitude_mountingoffset=quat_bf2gf)[0][:2])
    
    
    if print_outputs:
        print(f'\nInputs: CEST input : {function}')
        print(f'\nGS Long: {coord_hq[1]:.3f} deg, Lat {coord_hq[0]:.3f} deg, Alt {coord_hq[2]:.0f} km.')
        print(f'Outputs: Sun Az : {ae_2_sun[0]:.0f} deg, El : {ae_2_sun[1]:.0f} deg')
    return inputs, ae_2_sun

def ae_oh2col(ae_desired, 
              t_j2000 = None,
              q_mo = np.array([1,0,0,0]),
              s_host = [1e7, 1e7, 1e7, 1e3, 1e3, 1e3],
              s_target = [1e7, 1e8, 1e8, 1e3, 1e3, 1e3],
              reset_attitude_to_unity=1
              ):
    # Provides inputs with attitude for desired Az/El [deg]
    
    # get attitude
    quat_eci2bf_command_all = att_res.calc_attitude_for_ae_single(s_host, s_target, q_mo, ae_desired)
    if reset_attitude_to_unity:
        quat_eci2bf_command_all[0] = 1
        quat_eci2bf_command_all[1] = 0
        quat_eci2bf_command_all[2] = 0
        quat_eci2bf_command_all[3] = 0
    quat_bf2gf_returned = q_mo
    if t_j2000 is not None:
        t_gps = t_j2000 - t_conv.dt_gps2j2000tt() 
    else:
        t_gps = 1.3e9
    # Get Az/El
    inputs = (t_gps, s_host, s_target, quat_eci2bf_command_all, quat_bf2gf_returned)
    ae_2_sun = np.rad2deg(ae_calc.calc_ae_full(s_host, s_target, attitude_eci2bf = quat_eci2bf_command_all,
                                                attitude_mountingoffset=quat_bf2gf_returned)[0][:2])
    return inputs, ae_2_sun
if __name__ == '__main__':
    ## Tests, debugging, prepping different outputs
    # make_day_analysis = 0 # get sun vectors and AE angles for entire day
    
    # make plots for entire day
    make_day_analysis = 1

    # Compute and print Az/El for a single input as a dt.datetime object
    make_single_output = 0
    
    ### make table of full inputs/outputs (states of ground station, target [sun], 
    # attitude - quaternion from ECI to global frame (NED) and 
    # quaternion mounting_offset (body-frame NED to terminal frame NWU))
    output_tabular_data = 1
    output_long_window_data =  0 # illumination + big time steps + months(s) of data for moon scan windows
    # Unit test option, see Az/El with LOS pointing to north pole
    make_NP_pointing_test = 0
    
    # body = 'moon'
    body = 'sun'
    

    output_folder = r'outputs/tables/sun_vector'
    tud_rotator = tud.tudat_predictor()
    # TODO add tolerance for microseconds
    if make_single_output:    

        t_offset_min = 0
        t_start_cest = dt.datetime.now() +dt.timedelta(minutes = 1)
        timedelta = dt.timedelta(minutes = t_offset_min)
        t_start_cest  = t_start_cest  + timedelta
        # set other date/time
        # t_start_cest = dt.datetime(2023, 6, 1, 20, 54, 0) 
        inputs, ae_now = ae_roof2sun(t_start_cest, tud_rotator, mode = body, pointing = 'NWU')
        # can compare to https://www.timeanddate.com/sun/germany/munich
        print(f'''{body} vector at {t_start_cest.isoformat()}        
        Outputs : Az = {ae_now[0]:.2f} deg, El = {ae_now[1]:.2f} deg
        EL  {np.deg2rad(ae_now[1])*1e6 :.1f} urad 
        ''') 

    if output_tabular_data: # MOON-ANGLE CSV GENERATION FOR SCANS AND TRACKING AND WHATNOT
        pt_direction = 'NWU' # North, West, Up terminal pointing
        mounting_rpy = None # [R, P, Y]
        make_plots = 0
        save_comb = 0 # save all inputs/outputs in a single 
        save_separately = 0 # save all inputs/outputs in a separate csv's
        
        ## Only giving az;el;etc to Moon (used in initial scans)
        # setting to 0 will also output pos host (Mynaric hq), pos target, attitude, etc
        output_only_azel = 0 

        # for easier parsing 
        # Set up date and resolution 11,4,0,24,1.0
        # TODO set date for desired moon angles
        nr_days = 1
        day_used = 11
        month_used = 4
        h_start = 0
        h_end = 24
        t_start_cest =  dt.datetime.now() + dt.timedelta(minutes = 1) #dt.datetime(2025, month_used, day_used, h_start, 0, 0) 
        full_output_folder = f'{output_folder}//{body}_{t_start_cest.date().isoformat()}'
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
        t_j2000_start = t_conv.utc2gws(t_start_cest+ dt.timedelta(hours = -2)) + dt_gps2j2000
        datetime_update_rate = int(1/dt_mins/60)


        for ii in range(loop_length):
            if ii == 0:
                t_current = t_start_cest
                t_j2000_current = t_j2000_start
            inputs, ae_2_sun = ae_roof2sun(t_current, tud_rotator, mode = body,
                                           pointing=pt_direction, mounting_offset = mounting_rpy, 
                                           t_j2000 = t_j2000_current, dt_gps2j2000 = dt_gps2j2000)

            cest_hoursmins.append(t_current.time().isoformat())
            #r_gs_eci, r_sun_eci, 
            # quat_eci2bf, quat_bf2gf 1 - ae_2_sun (array): Azimuth, Elevation to sun [deg]
            ae_data[ii,:] = ae_2_sun # deg
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
        
        if save_comb and 0: # does not work yet TODO fix
            combined_outputs_dict = {
                'CEST_date' : cest_hoursmins,
                'GPS time' : t_gps,
                'AE_sun' : ae_data,
                'rv_GS_eci' : states_gs,
                'rv_sun_eci' : states_sun,
                'q_eci2bf' : q_eci2bf,
                'q_bf2gf' : q_bf2gf,
            }
            df_combined = pd.DataFrame.from_dict(combined_outputs_dict)
            title = 'combined_inout'
            print(f'Saving {title}')
            df_combined.to_csv(f"{full_output_folder}//{title}.csv", index = False)       
        if save_separately:
            # AZ EL ANGLES IN URAD
            if output_only_azel:
                data_output = [np.deg2rad(ae_data)*1e6]
                titles = [ f'ae_gs2{body}']
                columns = [
                        ['t_gps_s', 'CEST_hms', 'az_urad', 'el_urad'],
                        ]
            else:
                data_output = [states_gs, states_sun, np.deg2rad(ae_data)*1e6, q_eci2bf, q_bf2gf]
                titles = [ 'states_gs_eci', f'states_{body}_eci', f'ae_gs2{body}', 'attitude_eci2bf', 'attitude_bf2gf']
                columns = [
                        ['t_gps_s', 'CEST_hms', 'x_m', 'y_m', 'z_m', 'vx_ms', 'vy_ms', 'vz_ms'],
                        ['t_gps_s', 'CEST_hms', 'x_m', 'y_m', 'z_m', 'vx_ms', 'vy_ms', 'vz_ms'],
                        ['t_gps_s', 'CEST_hms', 'az_urad', 'el_urad'],
                        ['t_gps_s', 'CEST_hms', 'q_s', 'q_1', 'q_2', 'q_3'],
                        ['t_gps_s', 'CEST_hms', 'q_s', 'q_1', 'q_2', 'q_3']
                        ]
            for ii, title in enumerate(titles):
                
                output_df = pd.DataFrame(data = np.hstack((t_gps, cest_hoursmins, data_output[ii])), columns = columns[ii])

                print(f'Saving {title}')
                try:                                
                    output_df.to_csv(f"{full_output_folder}//{title}.csv", index = False)
                except:
                    os.mkdir(full_output_folder)
                    output_df.to_csv(f"{full_output_folder}//{title}.csv", index = False)
                print(f'Saved data sun vector data')
        if 1: # May 30 Added ref tiem tracker
            title_reftime = 'ref_time'
            output_times_dict = { 'day_used' : [day_used],
                'month_used' : [month_used],
                'h_start' : [h_start],
                'h_end' : [h_end],
                't_res' : [dt_mins*60]}
            df_date = pd.DataFrame.from_dict(output_times_dict)
            df_date.to_csv(f"{full_output_folder}//{title_reftime}.csv", index = False)
        print(f'Loop done. Start : {t_start_cest} \nFinal time : {t_current}')

        if make_plots:
            add_crit_timelines = 1
            if add_crit_timelines:
                t_sunrise_cest = dt.datetime(2025, month_used, day_used, 5, 55, 0)
                t_sunset_cest = dt.datetime(2025, month_used, day_used, 20, 29, 0)
                t_noon_cest = dt.datetime(2025, month_used, day_used, 13, 10, 0)
                crit_times = [t_sunrise_cest, t_noon_cest, t_sunset_cest]
                labels = ['sunrine', 'noon', 'sunset']
                colors = ['y', 'black', 'r']
            az_limits = [-180, 180]
            el_limits = [-60, 70]
            f, axs = plt.subplots(2)
            dates = mdates.date2num(cest_hoursmins)
            for ii, ax in enumerate(axs):
                ax.plot_date(dates, ae_data[:,ii], '-')
                ax.set_ylabel(['Az','El'][ii] + ' [deg]')
                ax.set_xlabel('Time CEST')
                # ax.set_ylim([az_limits, el_limits][ii])
                if ii == 0:
                    ax = cmbplt.autoscale_yaxis(ax, 1, 1, n_bins = 7)
                else:
                    ax = cmbplt.autoscale_yaxis(ax, 1, 1, n_bins = 9)
                if add_crit_timelines:
                    for jj, c in enumerate(colors):
                        ax.plot([crit_times[jj], crit_times[jj]], [ax.get_ylim()[0], ax.get_ylim()[1]], c = c, label = labels[jj] + ' May 3rd')
                    ax.legend()
                ax.grid()
            plt.gcf().autofmt_xdate()
            myFmt = mdates.DateFormatter('%H:%M')
            plt.gca().xaxis.set_major_formatter(myFmt)
            f.suptitle(f'Sun pointing Az/El for {pt_direction} Headquarters ROOF position\non date {t_start_cest.date()}')
            if 0:
                fig_title = 'sun_roof_NEDLLAfix'
                bplt.savefig(f, fig_title, subfolder = 'sun_vector')
    
    if output_long_window_data and not output_tabular_data:
        pt_direction = 'NWU' # North, West, Up terminal pointing
        mounting_rpy = None # [R, P, Y]
        save_separately = 1 # save all inputs/outputs in a separate csv's
        
        ## Only giving az;el;etc to Moon (used in initial scans)
        output_only_azel = 1 
        nr_days = 60
        day_used = 1
        month_used = 10
        h_start = 0
        h_end = 24
        t_start_cest = dt.datetime(2025, month_used, day_used, h_start, 0, 0) 
        full_output_folder = f'{output_folder}//{body}_{t_start_cest.date().isoformat()}_big'
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
        t_j2000_start = t_conv.utc2gws(t_start_cest+ dt.timedelta(hours = -2)) + dt_gps2j2000


        for ii in range(loop_length):
            if ii == 0:
                t_current = t_start_cest
                t_j2000_current = t_j2000_start
            inputs, ae_2_sun = ae_roof2sun(t_current, tud_rotator, mode = body,
                                        pointing=pt_direction, mounting_offset = mounting_rpy, 
                                        t_j2000 = t_j2000_current, dt_gps2j2000 = dt_gps2j2000)

            cest_hoursmins.append(t_current.time().isoformat())
            cest_days.append(t_current.date().isoformat())
            day_seconds[ii,:] = t_current.hour*3600+t_current.minute*60 + t_current.second
            #r_gs_eci, r_sun_eci, 
            ae_data[ii,:] = ae_2_sun # deg
            t_gps[ii,:] = np.round(inputs[0], n_digits_used)
            # Get moon illumination
            r_moon_ii, illumination_ii = where_sun.compute_moon_vector_eci(inputs[0], what_brightness=1)
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
           
    if make_day_analysis:
        pt_direction = 'NWU' # North, West, Up terminal pointing
        make_plots = 0
        
        # Set up date
        day_used = 1
        month_used = 10
        t_start_cest = dt.datetime(2025, month_used, day_used, 0, 0, 0)
        dt_mins = 5 # 10 mins -> 6x24 runs for day-long output
        dt_loop = dt.timedelta(minutes = dt_mins)
        loop_length = int(60/dt_mins*24)

        
        # placeholders
        cest_hoursmins = []
        ae_data = np.zeros((loop_length, 2))
        for ii in range(loop_length):
            if ii == 0:
                t_current = t_start_cest
            t_current = t_current + dt_loop
            inputs, ae_2_sun = ae_roof2sun(t_current, tud_rotator, pointing=pt_direction, mode = body)

            cest_hoursmins.append(t_current)
            ae_data[ii,:] = ae_2_sun
        print(f'Loop done. Final time : {t_current}')
        if make_plots:
            add_crit_timelines = 0
            if add_crit_timelines:
                t_sunrise_cest = dt.datetime(2025, month_used, day_used, 5, 55, 0)
                t_sunset_cest = dt.datetime(2025, month_used, day_used, 20, 29, 0)
                t_noon_cest = dt.datetime(2025, month_used, day_used, 13, 10, 0)
                crit_times = [t_sunrise_cest, t_noon_cest, t_sunset_cest]
                labels = ['sunrine', 'noon', 'sunset']
                colors = ['y', 'black', 'r']
            az_limits = [-180, 180]
            el_limits = [-60, 70]
            f, axs = plt.subplots(2)
            dates = mdates.date2num(cest_hoursmins)
            for ii, ax in enumerate(axs):
                ax.plot(dates, ae_data[:,ii], '-')
                ax.set_ylabel(['Az','El'][ii] + ' [deg]')
                ax.set_xlabel('Time CEST')
                # ax.set_ylim([az_limits, el_limits][ii])
                if ii == 0:
                    ax = cmbplt.autoscale_yaxis(ax, 1, 1, n_bins = 7)
                else:
                    ax = cmbplt.autoscale_yaxis(ax, 1, 1, n_bins = 9)
                if add_crit_timelines:
                    for jj, c in enumerate(colors):
                        ax.plot([crit_times[jj], crit_times[jj]], [ax.get_ylim()[0], ax.get_ylim()[1]], c = c, label = labels[jj] + ' May 3rd')
                    ax.legend()
                ax.grid()
            plt.gcf().autofmt_xdate()
            myFmt = mdates.DateFormatter('%H:%M')
            plt.gca().xaxis.set_major_formatter(myFmt)
            f.suptitle(f'{body} pointing Az/El for {pt_direction} Headquarters ROOF position\non date {t_start_cest.date()}')
            if 0:
                fig_title = 'sun_roof_NEDLLAfix'
                bplt.savefig(f, fig_title, subfolder = 'sun_vector')
        
    if make_NP_pointing_test: # TODO move to separate function for test purposes
        sun_vec_np_eci = np.array([0, 0, 6378e3]) # NORTH POLE
        t_start_cest = dt.datetime(2025, 1, 10, 0, 0, 0)
        ae_np = ae_roof2sun(t_start_cest, tud_rotator, sun_vec = sun_vec_np_eci)
        print(f'''{body} vector PARTIAL TEST. False sun vector input - pointing at north pole
        Expected AE : Az = 0, El slightly negative
        Outputs : Az = {ae_np[0]:.2f} deg, El = {ae_np[1]:.2f} deg
        ''')            