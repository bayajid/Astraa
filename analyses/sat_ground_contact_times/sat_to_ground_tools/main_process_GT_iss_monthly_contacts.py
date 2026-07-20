#%% Script to process monthly ISS ground-tracks and viewing angles into contact times
# output Ground track/pass information
import numpy as np
import pandas as pd
from sgp4 import exporter, omm
from sgp4.api import Satrec
from astropy.time import Time
import matplotlib.pyplot as plt
from gt_calc import make_lat_long_plot, make_ground_track_plot, calc_gt, dict_2_array, mod360_deg
from old_simulate_ground_track import simulate_ground_track, calc_fov_points, find_gs_in_fov, calc_required_fov, calc_sc_nadir_coord, calc_area_access_el, calculate_gs_visibility, calc_vis_area_point
import gt_calc as gt_tools
from GS_coordinates import gs_dict, gs_l3haris, gs_coordinates
import sys, os
# Add parent directory to paths
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import plotting_functions as pl
import pass_process as gs_pass_tools
#%%
month_starts_iso = [f'2022-{mo}-01 00:00:00' for mo in range(1,13)]

dt = 10

path_gt = 'monthly_outputs'
gt_data_used = 'ground_tracks_Oct'

months = [
'jan',
'feb',
'mar',
'apr',
'may',
'jun',
'jul',
'aug',
'sep',
'oct',
'nov',
'dec',


]
if 1: # run for all months
    for ii, ind_mo in enumerate(month_starts_iso):
        gt_data_used = f'ground_tracks_{ii+1}'
        data_gt = pd.read_csv(f'{path_gt}/{gt_data_used}.csv')
        long = data_gt['long'].values
        lat = data_gt['lat'].values
        t_vec = data_gt['jd'].values
        
        head_recalc = gt_tools.calc_heading_fromgt(t_vec, long, lat)[:,1]
        data_gt = data_gt.iloc[:-1,:]
        data_gt['heading'] = head_recalc
        data_gt.to_csv(f'{path_gt}/{gt_data_used}.csv', index = 0)
        output_dict = gs_pass_tools.process_gt_to_passes(data_gt, month_done=months[ii])
else: # run for a single month
    for ii in [7]:
        gt_data_used = f'ground_tracks_{ii+1}'

        data_gt = pd.read_csv(f'{path_gt}/{gt_data_used}.csv')



        # heading_angles = gt_tools.calc_heading_fromgt()
        output_dict = gs_pass_tools.process_gt_to_passes(data_gt, month_done=months[ii])    