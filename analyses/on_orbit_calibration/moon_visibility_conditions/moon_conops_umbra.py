#%%
# This code is used to analyze the time that the host satellite
# is within complete darkness from the sun (umbra) and has a visible moon
# the goal is to see what time window we get for these ideal conditions
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
import plotting_tools.modular_plotting as modplot
import astronomy_tools.astro_targets as where_sun
import basic_tools.in_out as savedat
import pointing_calculations.ae_calculation as ae_calc

# try to keep Az/El stable

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
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = 5*106*6)

# host_chosen = 'leo_host_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'leo_host_eq'
host_chosen = 'meo_eq'

if host_chosen == 'leo_host_polar':
    ind_r = [1,2,3]
elif host_chosen == 'leo_host_incl':
    ind_r = [7,8,9]
elif host_chosen == 'leo_host_eq':
    ind_r = [13, 14, 15]    
elif host_chosen == 'meo_eq':
    ind_r = [19,20,21]

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
r_inum = np.zeros((nrows, 3))
#%%
ii_inum = 0
r_penum = np.zeros((nrows, 3))
ii_penum = 0
r_sun = np.zeros((nrows, 3))
ii_sun = 0
importlib.reload(where_sun)
for ii, t_j2000_ii in enumerate(t_chosen):
    t_gps_ii = t_j2000_ii - dt_gps2j2000 
    r_host_ii = r_host[ii,:]
    v_host_ii = v_host[ii,:]
    ## get approx moon vector

    r_sun_true = moon_from_sf.get_sun(t_j2000_ii - t_j2000[0], body = 'sun')

    # um, penum = where_sun.check_shadow(r_host_ii, r_sun_true)
    um, penum = where_sun.check_shadow_canonical(r_host_ii, r_sun_true)
    if um:
        r_inum[ii_inum,:] = r_host_ii
        ii_inum+=1
    elif penum:
        r_penum[ii_penum,:] = r_host_ii
        ii_penum+=1
    else:
        r_sun[ii_sun,:] = r_host_ii
        ii_sun+=1
# print('In Umbra : ', ii_inum)
print('In Penumbra : ', ii_penum*10/60)
# print('In sun : ', ii_sun)
print(f'Length umbra : {ii_inum*10/60:.2f} min')
print(f'Length Sun : {ii_sun*10/60:.2f} min')
r_inum = r_inum[:ii_inum,:]
r_penum = r_penum[:ii_penum,:]
r_sun = r_sun[:ii_sun,:]
#%%
importlib.reload(modplot)
if plot_3d:
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    f, ax = modplot.add_arc(f, ax, 'y', r_sun, label_f = 'Sun Illuminated')
    if ii_inum > 0:
        f, ax = modplot.add_arc(f, ax, 'black', r_inum, label_f = 'Umbra')
    if ii_penum > 0:
        f, ax = modplot.add_arc(f, ax, 'b', r_penum, label_f = 'Penumbra')
    f, ax = modplot.add_single_los(f, ax, state_h = np.array([0,0,0]), state_t = r_sun_true/1e4/2, label_used = 'Sun Vector')
    f, ax = modplot.add_glossary_basic(f,ax, title = f'{host_chosen} Sun Shadow Analysis')

    ax.view_init(25,190)