#%% July 26, 2023 - code to compute attitude profiles and evaluate
# the resulting Mounting offset resolution by moon calibrations
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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import plotting_tools.basic_plotting as bplt

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
# Moon calibration condition file
calibration_time_path = r'outputs\tables\moon_conops_conditions\final_overview'
position_path = r'outputs\tables\moon_conops_conditions\filtered'


# pass_overviews = pd.read_csv(calibration_time_path)

t_used = '62d'
chosen_index = 2
files_all = os.listdir(calibration_time_path)
files_all = [f for f in files_all if t_used in f]
# sys.exit()
files_filtered = [ii for ii in files_all if t_used in ii]
for ii, file in enumerate(files_filtered):
    print(f'{ii} -> {file}')
fname_passdata = files_filtered[chosen_index]
sat_host = fname_passdata[:fname_passdata.index('_62')]

fname_positions = [ii for ii in os.listdir(position_path) if sat_host in ii][0]

path_data = fr'{calibration_time_path}/{fname_passdata}'
path_positions = fr'{position_path}/{fname_positions}'
print(f'\nInd = {chosen_index} -> {fname_passdata}')
if 'sat_leo_eq' in sat_host:
    sat_name = 'LEO 1000 km altitude, Equatorial.'
elif 'sat_leo_incl' in sat_host:
    sat_name = 'LEO 1000 km altitude, Inclined 53 deg.'
elif 'sat_leo_polar' in sat_host:
    sat_name = 'LEO 1000 km altitude, Near-Polar 89 deg.'
elif 'sat_meo' in sat_host:
    sat_name = 'MEO 13880 km altitude, Equatorial.'

dat_pass = pd.read_csv(path_data)
dat_pos = pd.read_csv(path_positions)
print(f'---Loaded \"{fname_passdata}\" AND \"{fname_positions}\"---\n')
#%% Unpack all required inputs based on chosen pass index
pass_chosen = 1
add_noise = 1
pass_stats = dat_pass[dat_pass['pass_nr'] == pass_chosen]
pass_position_data = dat_pos[dat_pos['pass_nr'] == pass_chosen]
print(f'''
Chosen pass nr : {pass_chosen}
stats : {pass_stats}

Number of rows available : {pass_position_data.shape[0]}
----Noise USAGE---- : {bool(add_noise)}
      ''')

r_host = pass_position_data.iloc[:,[1,2,3]].values
v_host = pass_position_data.iloc[:,[-4, -3, -2]].values
t_gps = pass_position_data.iloc[:,[0]].values
r_moon = pass_position_data.iloc[:,[4,5,6]].values
illumination = pass_position_data.iloc[:,[7]].values
dt_data = t_gps[1] - t_gps[0]
# Get attitude
constrain_azel = 0
constrain_attitude = 1

A_limits = [-195, 195]
E_limits = [15, 85]

AE_chosen_1 = [1, 75]
AE_chosen_2 = [25, 60] # 1/12 deg/s attitude?

# Start with 2 scans. TODO - prepare AE chosen for entire pointing model calculation
nrows = 2
AE_chosen_all = np.zeros((nrows,2))
AE_chosen_all[0,:] = AE_chosen_1
AE_chosen_all[1,:] = AE_chosen_2
# Get host_att_1; host_att_2

mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

# Placeholders for commanded attitude
ea_eci2bf_command_all = np.zeros((nrows, 6)) # EA; EA_rate [deg, deg/s]
quat_eci2bf_command_all = np.zeros((nrows, 4)) # scalar-first
make_3d_plot = 1 
scan_gap = 1
ii_ae = 0 
ii_scans = [1]
ii_scans.append(ii_scans[0] + scan_gap)
rotational_rate = 0.1
# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
## COMPUTE ATTITUDE
if constrain_azel:
    quat_eci2bf_command_all, ea_eci2bf_command_all = att_res.find_attitude_for_ae(pass_position_data, AE_chosen_all)
    print(f'Attitude computed for {nrows} points')
    print(f'Euler rates : {np.round(ea_eci2bf_command_all[0,3:],4)} deg/s for AE : {*AE_chosen_all,} deg')
elif constrain_attitude:
    # attitude_profiles = ['earth_point', 'earth_roll', 'sun_roll', 'sun_point']
    attitude_profiles = ['earth_point']
    # attitude_profiles = ['earth_roll']
    # attitude_profiles = ['sun_roll']
    # attitude_profiles = ['sun_point']
    print(f'Using defined attitude profiles : {attitude_profiles}')
    attitude_eci2bf = []
    for ii, att_prof in enumerate(attitude_profiles):
        quat_eci2bf, rot_eci2bf = att_sim.calc_quat_eci2bf(r_host, v_host, att_prof, t_gps = t_gps.flatten(), roll_velocity=rotational_rate)
        attitude_eci2bf.append(quat_eci2bf)
    # TODO save attitude
make_3d_plot = 1
if make_3d_plot:
    # making 3d plots
    import plotting_tools.modular_plotting as modplot
    importlib.reload(modplot)
    length = 1e3
    ii_used = [1, 5, 10, 15, 20] # for orbit
    ii_frame = [1, 5, 10, 15, 20]
    frame_ind_0 = [10, 10]
    # frame_ind_1 = [10,10]
    # frame_ind_2 = [20,20]
    len_los = 1e7

    frame_plotted = 'BF'
    if frame_plotted == 'BF':
        rot_shown = rot_eci2bf
    
    ftitle = f'''Attitude type : {att_prof}; roll = {rotational_rate} deg p.s.
    '''

    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    f, ax = modplot.add_orbit_basic(f, ax, r_host[ii_used,:], label = f'{sat_name} Orbit', c = 'b', linewidth = 3)    
    f, ax = modplot.add_ref_frame(f, ax, chosen_setting= 1, 
                                    rot_gf = rot_shown[frame_ind_0[0]], 
                                    origin = r_host[frame_ind_0[1],:]
                                    )
    
    f, ax = modplot.add_single_los(f, ax, 
                                    state_h = r_host[frame_ind_0[1],:],
                                    state_t = r_moon[frame_ind_0[0],:],
                                    normalize = 1,
                                    len_normalized = len_los,
                                    label_used= 'Moon direction',
                                    color = 'c'
                                    )
    f, ax = modplot.add_single_los(f, ax, 
                                    state_h = np.array([0,0,0]),
                                    state_t = where_sun.compute_sun_vector_eci_better(float(t_gps[0])),
                                    normalize = 1,
                                    color='yellow',
                                    len_normalized = len_los,
                                    label_used= 'Sun direction'
                                                                        )
    for ii_f in ii_frame:
        f, ax = modplot.add_ref_frame(f, ax, chosen_setting= 1, 
                                rot_gf = rot_shown[ii_f], 
                                origin = r_host[ii_f,:]
                                )

    modplot.set_axes_equal(ax, axlim = 10e6)
    # FOR EQUATORIAL
    # ax.view_init(6,-25)
    # # ax.view_init(90,50)
    
    # FORP OLAR
    ax.view_init(5,30)
    # ax.view_init(90,30)
    f, ax = modplot.add_glossary_basic(f, ax, title = ftitle)
    bplt.autosave(f, subfolder = '3d_moonConops_debug')
                                