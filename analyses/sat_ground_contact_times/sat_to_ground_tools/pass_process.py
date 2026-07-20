#%% Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# import plotting_functions as pl
from analyses.sat_ground_contact_times.sat_to_ground_tools.gt_calc import make_lat_long_plot, make_ground_track_plot, calc_gt, dict_2_array, mod360_deg
from analyses.sat_ground_contact_times.sat_to_ground_tools.old_simulate_ground_track import simulate_ground_track, calc_fov_points, find_gs_in_fov, calc_required_fov, calc_sc_nadir_coord, calc_area_access_el, calculate_gs_visibility, calc_vis_area_point
import time
from analyses.sat_ground_contact_times.sat_to_ground_tools.GS_coordinates import output_dict_overview, sin_rho_gs, rho_gs, rho, sin_rho, eps_min, eta_lct, R_E, phi_lct
def process_gt_to_passes(gt_df,
                        gs_index_used = 1,
                        save_folder = r'outputs/tables/contact_times',
                        month_done = 'Oct',
                        save_clean = 1,
                        save_raw = 1,
                        dt_used = 10):
    # gt_df - dataframe with [0-time, 1 - dates, 2- long deg, 3 - lat deg, -1 heading [rad]]
    run_time_start = time.time()   
    dates = gt_df.iloc[:,1].values
    gs_dict = output_dict_overview['GS']

    t_vec = gt_df.iloc[:,0].values
    t_vec = np.round((t_vec - t_vec[0])*86400,0)
    longlat = gt_df.iloc[:,[1,2]].values # rad
    heading = gt_df.iloc[:,[3]].values # rad
    heading_deg =heading
    simulated_ground_track = [t_vec, longlat, heading_deg] # s, deg, deg, deg
    gs_dict, output_dict_full,  output_dict_raw = calculate_gs_visibility(gs_dict,
                                                                        simulated_ground_track,
                                                                        rho,
                                                                        sin_rho,
                                                                        eps_min,
                                                                        eta_lct,
                                                                        phi_lct,
                                                                        R_E = R_E,
                                                                        gs_used = gs_index_used,
                                                                        check_gs_visibility = True,
                                                                        n_digits = 2,                                                                     
                                                                        dt_used = dt_used
                                                                        )
                                                                        
    print(f'GS Visibility calculated. Executed in {time.time() -run_time_start:.1f} s')
    #%% Process outputs OBERPFAFFENHOFEN ONLY
    output_single_raw_df = pd.DataFrame(output_dict_raw)
    output_single_raw_df = output_single_raw_df.transpose()
    output_single_raw_df.insert(0, 't', output_single_raw_df.pop('t'))
    output_single_raw_df['slant_range'] = output_single_raw_df['slant_range']/1e3 # Convert slant range to [km]
    output_single_raw_visible = output_single_raw_df[output_single_raw_df['is_visible'] == True]
    ii_vis = output_single_raw_visible.iloc[:,0].values/dt_used
    output_single_raw_visible['date'] = dates[ii_vis.astype(int)]
    gs_coord = gs_dict[gs_index_used]['long/lat']
    if save_raw:
        try:
            output_name_raw = f'raw_outputs_all_{gs_coord}.csv' 
            output_single_raw_df.to_csv(f'{save_folder}/{output_name_raw}')
            print(f'Saved {save_folder}/{output_name_raw}')
        except:
            print('Raw file too large to save')
    else:
        print('Raw passes not saved')
    output_name_vis = f'raw_outputs_visible_{gs_coord}_.csv'
    if save_clean:
        output_single_raw_visible.to_csv(f'{save_folder}/{output_name_vis}')
        print(f'Saved {save_folder}/{output_name_vis}')    #%% Generate FOV bounds for each grid point- Verification purposes 

    ## Get passes dataframe
    pass_dict =gs_dict[gs_index_used]
    passes_dict = pass_dict['passes']
    
    peak_gs_el = []
    pass_nr = []
    start_date = []
    start_t_s = []
    length_observable = []
    # length_in_horizon = []

    for ii, key in enumerate(passes_dict.keys()):
        pass_dat = passes_dict[key]
        if passes_dict[key]['length']>0:
            pass_ind = ii+1
            peak_el_val = pass_dat['gs_elevation_max']
            date_start_pass = dates[pass_dat['ii_start']]
            t_start_pass = pass_dat['t_start']
            t_length_pass = pass_dat['length']
            # t_length_vis = pass_dat['length_vis']


            pass_nr.append(pass_ind)
            peak_gs_el.append(peak_el_val)
            start_date.append(date_start_pass)
            start_t_s.append(t_start_pass)
            length_observable.append(t_length_pass)
            # length_in_horizon.append(t_length_vis)
        
    dict_output = {}
    dict_output['peak_gs_el'] = peak_gs_el
    dict_output['pass_nr'] = pass_nr
    dict_output['start_date'] = start_date
    dict_output['start_t_s'] = start_t_s
    dict_output['length_observable'] = length_observable
    # dict_output['length_in_horizon'] = length_in_horizon

    output_df = pd.DataFrame.from_dict(dict_output)
    output_df.to_csv(f'{save_folder}/pass_overview_{gs_coord}.csv', index = False)
    print(f'Saved {save_folder}/pass_overview_{gs_coord}.csv')
    
    return output_df