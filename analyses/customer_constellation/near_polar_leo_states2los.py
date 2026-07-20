## Script to compute AER and rates for a chosen host-satellite
# in link cases, fitting the Terran constellation
# Date August 7, 2023

#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations/terran_near_polar_split/NearPolar12x244.00h'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
save_folder = r'outputs/tables/terran_const'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import plotting_tools.combined_plots as combplt
import plotting_tools.modular_plotting as modplot
import attitude_tools.terminal_rotations as lct_rot
import pointing_calculations.ae_calculation as ae_calc
import basic_tools.time_conversion as t_conv
import tudat_tools.tudat_converter as tudatconv
from tudat_tools.data_processing.data_saving_utilities import dict2txt
import tudat_tools.data_processing.data_processing_utilities as dputil
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

# pointing_ref = 'along_track_up'
# pointing_ref = 'cross_track_up'
pointing_ref = 'behind_track_up'
# Looks like this is Terran's, except elevation sign is swapped
# pointing_ref = 'behind_track_down'
## Loading satellite orbital data
# use targets fitting for verification
verification_mode = 0
# visualize orbits
vis_3d_mode = 0
limit_rows = 0 # cut down datapoitns used for faster runs
# output aer;aer_dot for host
save_csv = 1
limit_targets = 0
host_ind_chosen = '0_0'
opposite_plane_ind = 11

if limit_rows:
    rows_used = 1e3
else:
    rows_used = None
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = rows_used)

sat_names = simulation_parameters['sat_names']
target_names_plane4 = [ii for ii in sat_names if '_4_' in ii]
target_names_plane5 = [ii for ii in sat_names if '_5_' in ii]
target_names_plane6 = [ii for ii in sat_names if '_6_' in ii]
target_names_plane7 = [ii for ii in sat_names if '_7_' in ii]

host_pack = [[ii, name] for ii, name in enumerate(sat_names) if host_ind_chosen in name][0]
host_name = host_pack[1]
host_state_ind = [1+ ii + host_pack[0]*6 for ii in range(6)]

t_j2000 = data_raw[:,0]
s_host = data_raw[:,host_state_ind]
r_host = s_host[:,[0,1,2]]
v_host = s_host[:,[3,4,5]]
nrows = s_host.shape[0]
#%%
## get targets
if not verification_mode:
    targets_whole_plane = [sat for sat in sat_names if f'_{opposite_plane_ind}_' in sat]
    # if opposite_plane_ind == 5:
    #     targets_whole_plane = target_names_plane5
    # elif opposite_plane_ind == 6:
    #     targets_whole_plane = target_names_plane6
    # elif opposite_plane_ind == 4:
    #     targets_whole_plane = target_names_plane4
    # elif opposite_plane_ind == 7:
    #     targets_whole_plane = target_names_plane7
else:
    targets_whole_plane = ['sat_leo_t_0_1', # coplanar - to leader
                           'sat_leo_t_1_0', # crossplane adjacent
                           'sat_leo_t_1_1',
                           ] 
    # targets_whole_plane = ['sat_leo_t_0_1'] 
# targets_chosen = targets_whole_plane[11:12]
targets_chosen = targets_whole_plane
los_dict = {}
for ii, sat_target in enumerate(targets_chosen):
    target_ind = simulation_parameters['state_ind'][sat_names.index(sat_target)]
    s_target = data_raw[:,[1+ ii + target_ind for ii in range(6)]]
    r_target = s_target[:,[0,1,2]]
    # get ECI LOS
    los_eci = r_target - r_host
    los_dict[sat_target] = {}
    los_dict[sat_target]['r'] = r_target
    los_dict[sat_target]['v'] = s_target[:,[3,4,5]]
    los_dict[sat_target]['los_eci'] = los_eci

importlib.reload(lct_rot)

rotations_eci2lct = lct_rot.calc_quat_eci2lct(r_host, v_host, default_pointing = pointing_ref)
quat_eci2lct = rotations_eci2lct[0]
dcm_eci2lct = rotations_eci2lct[1]

aer_storage = np.zeros((nrows, int(1+len(targets_chosen)*6)))
t_ref = t_j2000 - t_j2000[0]
aer_storage[:,0] = t_ref
aer_label_dict = {}
for ii, sat_target in enumerate(targets_chosen):
    # Az, El [rad], slant-range [m]
    aer_host2target = ae_calc.calc_ae_full(s_host, los_dict[sat_target]['r'], attitude_eci2bf = quat_eci2lct)
    ae_deg = np.rad2deg(aer_host2target[:,:2])
    a_rate = np.gradient(ae_deg[:,0], t_j2000).reshape((nrows, 1))
    e_rate = np.gradient(ae_deg[:,1], t_j2000).reshape((nrows, 1))
    r_rate = np.gradient(aer_host2target[:,2],t_j2000).reshape((nrows, 1))
    ii_target = [1+ii*6 + jj for jj in range(6)]
    aer_storage[:,ii_target] = np.hstack((ae_deg, aer_host2target[:,[2]], a_rate, e_rate, r_rate))
    aer_label_dict[sat_target] = ii_target
    
#%% save
if not verification_mode:
    if save_csv:
        fname_aer = f'seam_aer_{host_name}_to_{opposite_plane_ind}'
        fname_indices = f'seam_ind_{host_name}_to_{opposite_plane_ind}'
        np.savetxt(X = aer_storage, fname = f'{save_folder}/{fname_aer}.csv', delimiter = ',')
        dict2txt(aer_label_dict, fname_indices, save_folder)
    
# Verify - check specific link variables
importlib.reload(combplt)
importlib.reload(modplot)
if verification_mode and not vis_3d_mode:
    # make 2d aer plots
    for sat in aer_label_dict.keys():
        f, ax = combplt.plot_aer(
            aer= aer_storage[:,aer_label_dict[sat][:3]],
            t = t_ref,
            title = f'AER HOST {host_name} to TARGET {sat}',
              setting = '')
if vis_3d_mode:
    # make 3d plots
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    # Host sat
    # f, ax = modplot.add_orbit_basic(f, ax, c = 'b', state_i = r_host, label_f='', s = 3, alpha = 0.2)
    f, ax = modplot.add_orbit_basic(f, ax, states = r_host, c = 'b', label = '', linewidth = 3)
    f, ax = modplot.add_scatters_simple(f, ax, c = 'b', data_used = r_host, label_f = f'HOST {host_name}', s = 50)
    f, ax = modplot.add_ref_frame(f, ax, chosen_setting=1, rot_gf = dcm_eci2lct[0], origin = r_host[0,:])
    if verification_mode:
        # Coplanar target
        ind_0 = 0
        # Cross-plane target
        ind_1 = 1
    else:        
        ind_0 = 10
        ind_1 = 11

    f, ax = modplot.add_scatters_simple(f, ax, c = 'r', data_used = los_dict[targets_chosen[ind_0]]['r'], label_f = targets_chosen[ind_0], s = 50)
    f, ax = modplot.add_scatters_simple(f, ax, c = 'm', data_used = los_dict[targets_chosen[ind_1]]['r'], label_f = targets_chosen[ind_1], s = 50)
    f, ax = modplot.add_orbit_basic(f, ax, states = los_dict[targets_chosen[1]]['r'], c = 'm', label = '', linewidth = 3)
        
    f, ax = modplot.add_glossary_basic(f, ax, title = 'Verification - Terran Constellation at t=0')
    
    # ax.view_init(30,70)
    # ax.view_init(10,-50)
    ax.view_init(45,-50)
else:
    for ii, sat in enumerate(aer_label_dict.keys()):
        if ii in [8,9,10,11,12,13]:
            f, ax = combplt.plot_aer(
                aer= aer_storage[:,aer_label_dict[sat][:3]],
                t = t_ref,
                title = f'AER HOST {host_name} to TARGET {sat}',
                r_lim = 1,
                setting = '')