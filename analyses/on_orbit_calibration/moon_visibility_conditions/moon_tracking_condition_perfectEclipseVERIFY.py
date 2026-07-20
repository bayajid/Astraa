#%%
# Date - July 20, 2023
# Updated from moon_tracking_condition_slice 
# to include solar eclipse checks.
# Goal is to check when the satellite sees the Moon
# and it is fully illuminated
# and the satellite is in the Sun's umbra
# Thank you again Gokhan, I could not do this alone
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
path_cwd = os.getcwd()
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.astro_targets as where_sun
import basic_tools.time_conversion as t_conv
import link_processing_tools.visibility_checks as vis_check
import basic_tools.data_loading as dat_load
import plotting_tools.modular_plotting as modplot
# path jazz
downsample = 0
t_req = 600
save_csv = 0
# length_chosen = 1 # days
# length_chosen = 7  # days
length_chosen = 62 # days
# csv_output_path = fr'orbital_simulations\moontrackers\leomeo_mixincl{length_chosen:.0f}d'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
fname_dep_var = 'dependent_variables.dat'
output_folder = r'orbital_simulations\srp'
## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
# PLOT OPTIONS
plot_2d = 1 # 2d difference - when eclispe and when nah
plot_3d = 0 # eclipse plots in 3d - limit nrows to see single orbit for this
plot_single = 0 # whether a single point is plotted in 3d (debug purposes)


nrows = 80e3
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = output_folder, 
                                                                 nrows = nrows)
dep_var = dat_load.open_dat(filename=fname_dep_var,
                            folder_path=output_folder, nrows = nrows)
hosts_available = simulation_parameters['sat_names']
# 'leo_polar'
# 'leo_incl'
# 'leo_eq'
# 'meo_eq'
host_index = 3
host_used = hosts_available[host_index]

t_j2000 = data_raw[:,0]
t_gps = t_j2000 + t_conv.dt_j2000tt2gps()
nrows = len(t_j2000)

r_index = simulation_parameters['r_index'][host_used]
v_index = [3 + r for r in r_index]

# Get host orbit
r_host = data_raw[:,r_index]
v_host = data_raw[:,v_index] 
srp_norm = dep_var[:,host_index+1]

importlib.reload(where_sun)
# t_gps = t_gps[[1500]]
# r_host = r_host[[1500], :]
# v_host = v_host[1500, :]
# srp_norm = srp_norm[[1500]]
# nrows = 1
# get moon orbit
r_moon = np.zeros((nrows,3))
moon_illumination = np.zeros((nrows,1))
in_umbra = np.zeros((nrows))
moon_from_sf = where_sun.body_fromsp(t_j2000[0])
r_sun = np.zeros((nrows, 3))

r_inum = np.zeros((nrows, 3))
ii_inum = 0
for ii, t_gps_ii in enumerate(t_gps):
    r_host_ii = r_host[ii,:]
    r_moon_ii, illumination = where_sun.compute_moon_vector_eci(t_gps_ii, what_brightness=1)
    r_sun_true = moon_from_sf.get_sun(t_gps_ii - t_gps[0], body = 'sun')
    um, penum = where_sun.check_shadow_canonical(r_host_ii, r_sun_true)
    r_moon[ii,:] = r_moon_ii
    moon_illumination[ii] = illumination
    in_umbra[ii] = um
    r_sun[ii,:] = r_sun_true
    if um:
        r_inum[ii_inum,:] = r_host_ii
        ii_inum +=1

r_inum = r_inum[:ii_inum,:]
#%%
importlib.reload(modplot)
tudat_umbra = [0 if acc != 0 else 1 for ii, acc in enumerate(srp_norm)]
ii_tudat_umbra = [ii for ii, um in enumerate(tudat_umbra) if um == 1]
print(f'{host_used}')
print(f'TUDAT umbra pts:{sum(tudat_umbra)}; own -> {sum(in_umbra)}. Diff : {(sum(tudat_umbra)-sum(in_umbra))} or {(sum(tudat_umbra)-sum(in_umbra))/nrows*100:.3f}%')

if plot_2d:
    f, axs = plt.subplots(nrows = 2)
    t_plotted = t_gps - t_gps[0]
    axs[0].plot(t_plotted/3600, in_umbra)
    axs[0].set_title('Canonical model')
    axs[1].plot(t_plotted/3600, srp_norm)
    axs[1].set_title('TUDAT Solar Accel Norm')
    for ax in axs:
        ax.set_xlim([0,5])
        ax.set_ylabel('In Eclipse')
        ax.set_xlabel('t [hr]')
    f.set_tight_layout('tight')
if plot_3d and not plot_single:
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    f, ax = modplot.add_arc(f, ax, 'y', r_host, label_f = 'All orbit', size = 5)    
    f, ax = modplot.add_arc(f, ax, 'black', r_inum, label_f = 'Umbra', size = 35)
    f, ax = modplot.add_arc(f, ax, 'red', r_host[ii_tudat_umbra,:], label_f = 'SRP = 0', size = 20)
    f, ax = modplot.add_single_los(f, ax, state_h = np.array([0,0,0]), state_t = r_sun[-1]/1e4/2, label_used = 'Sun Vector')
    f, ax = modplot.add_single_los(f, ax, state_h = np.array([0,0,0]), state_t = r_sun[0,:]/1e4/2, label_used = 'Sun Vector')
    f, ax = modplot.add_glossary_basic(f,ax, title = f'{host_used} Sun Shadow Analysis')

    ax.view_init(90,190)
if plot_3d and plot_single:
    # limited 3d plot
    ind_tried = 1500
    ii_plotted = [ind_tried, ind_tried+1] 
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    f, ax = modplot.add_arc(f, ax, 'y', r_host[ii_plotted,:], label_f = 'All orbit', size = 50)    
    f, ax = modplot.add_single_los(f, ax, state_h = np.array([0,0,0]), state_t = r_sun[-1]/1e4/2, label_used = 'Sun Vector')
    f, ax = modplot.add_single_los(f, ax, state_h = np.array([0,0,0]), state_t = r_sun[0,:]/1e4/2, label_used = 'Sun Vector')
    f, ax = modplot.add_glossary_basic(f,ax, title = f'{host_used} Sun Shadow Analysis')

    ax.view_init(90,190)
if 0:
    # calculate passes and check occultation
    ii_vis = vis_check.check_occultation(r_host, r_moon, R_atm = 100e3, limit_nr_links=0)
    bool_vis = [True if ii in ii_vis else False for ii in range(nrows)]
    # store
    data = np.hstack((t_gps.reshape((nrows, 1)), r_host, r_moon, moon_illumination, np.array(bool_vis).reshape((nrows, 1)), in_umbra))
    df_stored = pd.DataFrame(data = data, columns = ['t_gps', 'x_h', 'y_h', 'z_h', 'x_m', 'y_m', 'z_m', 'illumination','is_visible', 'in_umbra'])
    save_title = f'{host_chosen}_conopstime_perfcond_{length_chosen:.0f}d.csv'
    if save_csv:
        df_stored.to_csv(f'{output_folder}/{save_title}', index = 0)
        print(f'Saved {save_title}')