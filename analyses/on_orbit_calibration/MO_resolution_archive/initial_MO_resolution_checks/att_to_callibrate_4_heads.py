#%%
# This code is used to analyze whether all 4 optical heads
# can be callibrated at once. 
# the challenge considered will be finding a body-frame attitude
# such that each optical head has a clear view of the moon and is
# not blocked by a neighboring terminal.
# to do so, a minimum of 15 deg elevation angle is imposed
# 10 comes from a neighboring terminal 1 m away
# with 17cm of height blocking the direct view of the terminal behind
# +5 more for mounting offset/scan width considerations
## IMPORTS
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
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.tudat_converter as tudatconv
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec
import plotting_tools.basic_plotting as bplt
import astronomy_tools.astro_targets as where_sun
import basic_tools.in_out as savedat
import pointing_calculations.ae_calculation as ae_calc

# try to keep Az/El stable
el_min = 20
az_chosen = 1

mounting_offset_rpy = [5, 1, -3] # MOUNTING OFFSET random 3-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

tracked_body = 'moon'
# which moon vector to is the truth. 
# ephemerides - most precise option. 
# approx - approximate maths expression (at least 170 uradprecision)
moon_truth_used = 'ephemerides'
## Loading satellite orbital data
# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\moontrackers\leomeo_mixincl7d'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = 10e3)

# host_chosen = 'leo_host_polar'
# host_chosen = 'leo_host_incl'
host_chosen = 'meo_eq'

if host_chosen == 'leo_host_polar':
    ind_r = [1,2,3]
elif host_chosen == 'leo_host_incl':
    ind_r = [7,8,9]
elif host_chosen == 'meo_eq':
    ind_r = [16,17,18]

# time slicing
t_j2000 = data_raw[:,0]
t_fromstart = t_j2000 - t_j2000[0]
r_host = data_raw[:,ind_r]
v_host = data_raw[:,[ii+3 for ii in ind_r]]
moon_from_sf = where_sun.body_fromsp(t_j2000[0])
## Getting body-frame attitude
t_chosen = t_j2000
nrows = t_chosen.shape[0]

dt = t_j2000[1] - t_j2000[0]
save_csv = 0

make_plots = 1
plot_dazdel = 0
plot_dlos = 0
plot_3d = 1

AE_req = [az_chosen, el_min]

# placeholders
ea_eci2bf = np.zeros((nrows, 3))
rot_eci2bf = np.zeros((nrows, 3, 3))
r_moon_eci_true = np.zeros((nrows, 3)) # m, m, m
ae_moon = np.zeros((nrows, 3)) # t_gps; Az; El [deg]
los_bf = np.zeros((nrows, 3)) # x, y, z [m]
dt_gps2j2000 = t_conv.dt_gps2j2000tt() # t_j2000 = t_gps + dt_gps2j2000
t_gps = np.zeros((nrows, 1)) # t_gps [s]
aer_lct = np.zeros((nrows, 3)) # az [rad], el [rad], r [m]
r_target = np.zeros((nrows, 3))
quat_eci2bf = np.zeros((nrows, 4))
quat_mo =  np.zeros((nrows, 4))
for ii, t_j2000_ii in enumerate(t_chosen):
    t_gps_ii = t_j2000_ii - dt_gps2j2000 
    r_host_ii = r_host[ii,:]
    v_host_ii = v_host[ii,:]
    ## get approx moon vector

    r_moon_true = moon_from_sf.get_sun(t_j2000_ii - t_j2000[0], body = 'moon')

    r_host2moon = r_moon_true - r_host_ii # los ECI
    
    ## get attitude
    los_req_lct = ae_calc.xyz_from_aer([AE_req[0], AE_req[1], np.linalg.norm(r_host2moon)])
    if 1: # mounting offset
        rot_bf2lct = np.eye(3, dtype = float)
    else:
        pass
    # convert to body-frame attitude
    los_req_bf = rot_bf2lct.transpose() @ los_req_lct
    los_req_bf_norm = los_req_bf / np.linalg.norm(los_req_bf)
    r_host2moon_norm = r_host2moon / np.linalg.norm(r_host2moon)
    rot_eci2bf_req = los_req_bf_norm.reshape((3,1)) @ r_host2moon_norm.reshape((1,3))
    rot_eci2bf[ii] = rot_eci2bf_req
    ea_eci2bf[ii,:] = conv.convert_dcm2ea(rot_eci2bf_req)
    t_gps[ii] = t_gps_ii
    r_target[ii,:] = r_moon_true

    q_eci2bf = conv.convert_dcm2quat(rot_eci2bf_req)
    q_mounting_offset = conv.convert_dcm2quat(rot_bf2lct)

    ae_lct = ae_calc.calc_ae_full(r_host_ii, r_moon_true, q_eci2bf, q_mounting_offset)[0]
    aer_lct[ii,:] = ae_lct
    quat_eci2bf[ii,:] = q_eci2bf
    quat_mo[ii,:] = q_mounting_offset
print(f'DONE calculating body-frame attitude for Az/El moon angles of Az= {AE_req[0]}; El = {AE_req[1]} deg.')
#%%
if make_plots:
    # make Euler angle plots
    if 0: 
        unit = 'min'
        if unit == 'min':
            t_plotted = t_fromstart / 60
            xlims = [0, 240]
            if 'meo' in host_chosen:
                xlims = [0, 480]
        elif unit == 's':
            t_plotted = t_fromstart
            xlims = [0, 7200]
        f, axs = plt.subplots(nrows = 2)
        ax = axs[0]
        for ii in range(3):
            ax.plot(t_plotted, ea_eci2bf[:,ii], label = 'RPY'[ii])
        # ax.legend()
        ax.set_ylabel('Euler Angle [deg]')
        ax.set_xlim(xlims[0], xlims[1])
        ax.grid('on')
        ax = axs[1]
        for ii in range(3):
            ax.plot(t_plotted, np.gradient(ea_eci2bf[:,ii], t_fromstart), label = 'RPY'[ii])
            ax.set_ylabel('Euler rate [deg/s]')
        ax.legend()
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xlabel(f't [{unit}]')
        ax.grid('on')
        f.suptitle(f'Body-frame Euler Angles and Rates for el_min = {el_min:.0f} deg; Az = {az_chosen} deg')
        bplt.autosave(f, subfolder = 'MoonConops')
    
    # make LCT Az El and FOV limit plots
    if 1:
        unit = 'min'
        if unit == 'min':
            t_plotted = t_fromstart / 60
            xlims = [0, 240]
            if 'meo' in host_chosen:
                xlims = [0, 480]
        elif unit == 's':
            t_plotted = t_fromstart
            xlims = [0, 7200]
        
        f, axs = plt.subplots(nrows = 2)

        ax = axs[0]
        
        ax.plot(t_plotted, np.rad2deg(aer_lct[:,0]))
        # ax.legend()
        ax.set_ylabel('Azimuth [deg]')
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim([0.9, 1.1])
        ax.grid('on')
        ax = axs[1]
        ax.plot(t_plotted, np.rad2deg(aer_lct[:,1]), label = 'LCT Elevation')
        # ax.legend()
        ax.set_ylabel('Elevation [deg]')
        ax.legend()
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xlabel(f't [{unit}]')
        ax.set_ylim([0, 30])
        ax.plot([t_plotted[0], t_plotted[-1]], [11.3, 11.3], c = 'r', label = 'el_min')
        ax.grid('on')
        ax.legend()
        f.suptitle(f'LCT Global Frame Azimuth and Elevation')
        bplt.autosave(f, subfolder = 'MoonConops')
        pass
if save_csv:
    importlib.reload(savedat)
    savedat.save_azel(
                      t_gps,
                      r_host,
                      r_target,
                    quat_eci2bf,
                    quat_mo,
                      aer_lct,
                      fname = f'{host_chosen}_pt_io_mooncall_fixed_a{az_chosen}el{el_min}'
                      )
    