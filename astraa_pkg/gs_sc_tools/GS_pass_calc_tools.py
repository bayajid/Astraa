#%% Imports
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import matplotlib.pyplot as plt
# import plotting_functions as pl
from gs_sc_tools.gt_calc import make_lat_long_plot, make_ground_track_plot, calc_gt, dict_2_array, mod360_deg
from gs_sc_tools.gt_simulation_tools import simulate_ground_track, calc_fov_points, find_gs_in_fov, calc_required_fov, calc_sc_nadir_coord, calc_area_access_el, calculate_gs_visibility, calc_vis_area_point
from gs_sc_tools.GS_coordinates import output_dict_overview, sin_rho_gs, rho_gs, rho, sin_rho, eps_min, eta_lct, R_E, phi_lct
import time
def process_gt_to_passes(gt_df,
                        tit_app = None,
                        gs_index_used = 1,
                        save_folder = r'sat_GS_contacts',
                         month_done = None,
                         check_fov_lims = 0,
                         save_clean = 1,
                         save_raw = 1,
                         dt_used = 10):
    """Function to compute viewing angles and contact times
    from a provided ground track and chosen ground station index

    Args:
        gt_df (pandas DF): Dataframe with columns of time, long, lat and heading
        h_sat (float): Sat altitude [km]
        gs_index_used (int, optional): Index of chosen ground station. 1 for Wessling. Defaults to 1.
        save_folder (string, optional): path to save results. Defaults to r'monthly_outputs/contact_times'.
        month_done (string, optional): Deprecated, only used to label outputs. Defaults to None.
        save_clean (int, optional): Whether clean outputs are saved. Defaults to 1.
        save_raw (bool, optional): whether raw outputs are saved. Defaults to 1.
        dt_used (int, optional): time-step of data. Defaults to 10.

    Returns:
        _type_: _description_
    """    
    # gt_df - dataframe with [0-time, 1 - dates, 2- long deg, 3 - lat deg, -1 heading [rad]]
    if type(month_done) == type(None): 
        month_done = 'full_period'
    # get title based on sat altitude
    if type(tit_app) == type(None):
        title_append = ''
    else:
        title_append = tit_app

    run_time_start = time.time()   
    dates = gt_df.iloc[:,1].values
    gs_dict = output_dict_overview['GS']

    t_vec = gt_df.iloc[:,0].values
    t_vec = np.round((t_vec - t_vec[0])*86400,0)
    longlat = gt_df.iloc[:,[2,3]].values
    heading = gt_df.iloc[:,[-1]].values # rad
    heading_deg = np.rad2deg(heading)
    simulated_ground_track = [t_vec, longlat, heading_deg]
    gs_dict, output_dict_full,  output_dict_raw = calculate_gs_visibility(gs_dict,
                                                                        simulated_ground_track,
                                                                        rho,
                                                                        sin_rho,
                                                                        eps_min,
                                                                        eta_lct,
                                                                        phi_lct,
                                                                        check_fov_lims=check_fov_lims,
                                                                        R_E = R_E,
                                                                        gs_used = gs_index_used,
                                                                        check_gs_visibility = True,
                                                                        n_digits = 2,                                                                     
                                                                        dt_used = dt_used
                                                                        )
    
    print(f'GS Visibility calculated for {month_done} Executed in {time.time() -run_time_start:.1f} s')

    output_single_raw_df = pd.DataFrame(output_dict_raw)
    output_single_raw_df = output_single_raw_df.transpose()
    output_single_raw_df.insert(0, 't', output_single_raw_df.pop('t'))
    output_single_raw_df['gs_azimuth_rate'] = np.gradient(np.unwrap(output_single_raw_df['gs_azimuth'], 300), output_single_raw_df['t'])
    output_single_raw_df['sc_azimuth_rate'] = np.gradient(np.unwrap(output_single_raw_df['sc_azimuth'], 300), output_single_raw_df['t'])

    output_single_raw_df['slant_range'] = output_single_raw_df['slant_range']/1e3 # Convert slant range to [km]
    output_single_vis = output_single_raw_df[output_single_raw_df['is_visible'] == True]
    ii_vis = output_single_vis.iloc[:,0].values/dt_used
    output_single_vis['date'] = dates[ii_vis.astype(int)]
    gs_coord = gs_dict[gs_index_used]['long/lat']


    if save_raw:
        try:
            output_name_raw = f'raw_outputs_all_{title_append}{gs_coord}_{month_done}.csv' 
            output_single_raw_df.to_csv(f'{save_folder}\\{output_name_raw}')
            print(f'Saved {save_folder}\\{output_name_raw}')
        except:
            os.mkdir(save_folder)
            output_single_raw_df.to_csv(f'{save_folder}\\{output_name_raw}')
            print(f'Saved {save_folder}\\{output_name_raw}')            
    else:
        print('Raw passes not saved')
    output_name_vis = f'raw_outputs_visible_{title_append}{gs_coord}_{month_done}.csv'
    if save_clean:
        try:
            output_single_vis.to_csv(f'{save_folder}\\{output_name_vis}')
            print(f'Saved {save_folder}\\{output_name_vis}')
        except:
            os.mkdir(save_folder)
            output_single_vis.to_csv(f'{save_folder}\\{output_name_raw}')
            print(f'Saved {save_folder}\\{output_name_vis}')

    ## Get passes dataframe
    pass_dict =gs_dict[gs_index_used]
    passes_dict = pass_dict['passes']
    
    peak_gs_el = []
    pass_nr = []
    start_date = []
    start_t_s = []
    length_observable = []
    az_gs_max = []
    az_sc_max = []
    # length_in_horizon = []
    for ii, key in enumerate(passes_dict.keys()):
        pass_dat = passes_dict[key]
        if passes_dict[key]['length']>0:
            pass_ind = ii+1
            peak_el_val = pass_dat['gs_elevation_max']
            date_start_pass = dates[pass_dat['ii_start']]
            t_start_pass = pass_dat['t_start']
            t_length_pass = pass_dat['length']
            pass_dat['gs_elevation_max']
            az_rates_gs = output_single_raw_df['gs_azimuth_rate'].iloc[pass_dat['ii_start']:pass_dat['ii_end']]
            az_rates_sc= output_single_raw_df['sc_azimuth_rate'].iloc[pass_dat['ii_start']:pass_dat['ii_end']]
            pass_dat['gs_az_rate_max'] = np.max(np.abs(az_rates_gs))
            pass_dat['sc_az_rate_max'] = np.max(np.abs(az_rates_sc))
            # t_length_vis = pass_dat['length_vis']
            az_gs_max.append(np.max(np.abs(az_rates_gs)))
            az_sc_max.append(np.max(np.abs(az_rates_sc)))
            pass_nr.append(pass_ind)
            peak_gs_el.append(peak_el_val)
            start_date.append(date_start_pass)
            start_t_s.append(t_start_pass)
            length_observable.append(t_length_pass)
            # length_in_horizon.append(t_length_vis)
        
    dict_output = {}
    dict_output['peak_gs_el'] = peak_gs_el
    dict_output['peak_azrate_gs'] = az_gs_max
    dict_output['peak_azrate_sc'] = az_sc_max
    dict_output['pass_nr'] = pass_nr
    dict_output['start_date'] = start_date
    dict_output['start_t_s'] = start_t_s

    dict_output['length_observable'] = length_observable
    # dict_output['length_in_horizon'] = length_in_horizon

    output_df = pd.DataFrame.from_dict(dict_output)
    output_df.to_csv(f'{save_folder}\\pass_overview_{title_append}{month_done}.csv', index = False)
    print(f'Saved {save_folder}\\pass_overview_{title_append}{month_done}.csv')
    
    return output_df, output_single_vis