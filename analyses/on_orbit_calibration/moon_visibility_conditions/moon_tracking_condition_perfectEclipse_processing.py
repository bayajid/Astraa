#%% Generate histograms and time-serie plots
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
import tudat_tools.data_processing.data_processing_utilities as dputil
import plotting_tools.basic_plotting as bplt
import plotting_tools.modular_plotting as modplot
# path jazz
# length_chosen = 1 # days
# length_chosen = 7  # days
length_chosen = 62 # days
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
input_folder = r'outputs\tables\moon_conops_conditions\raw'
output_folder_filtered = r'outputs\tables\moon_conops_conditions\filtered'
output_folder_overview = r'outputs\tables\moon_conops_conditions\final_overview'
## Loading satellite orbital data
t_used = f'{length_chosen:.0f}d'
chosen_index = 3
files_all = os.listdir(input_folder)
files_all = [f for f in files_all if 'perfcond' in f]
# sys.exit()
files_filtered = [ii for ii in files_all if t_used in ii]
for ii, file in enumerate(files_filtered):
    print(f'{ii} -> {file}')
f_chosen = files_filtered[chosen_index]
sat_host = f_chosen[:f_chosen.index('_con')]
if 'sat_leo_eq' in sat_host:
    sat_name = 'LEO 1000 km altitude, Equatorial.'
elif 'sat_leo_incl' in sat_host:
    sat_name = 'LEO 1000 km altitude, Inclined 53 deg.'
elif 'sat_leo_polar' in sat_host:
    sat_name = 'LEO 1000 km altitude, Near-Polar 89 deg.'
elif 'sat_meo' in sat_host:
    sat_name = 'MEO 13880 km altitude, Equatorial.'

print(f'Chosen : {chosen_index} -> {f_chosen} (Sat host : {sat_host})')

### Settings for processing
# minimum moon illumination threshold for good conditions
illumination_threshold = 70


# saving conditionals
save_csv = 1
make_plots = 1
save_plots = 0
dat = pd.read_csv(f'{input_folder}/{f_chosen}')
t_gps_full = dat['t_gps'].values
dt_standard = t_gps_full[1] - t_gps_full[0]
dat_nosun = dat[dat['in_umbra'] == 1]
dat_moonvis = dat[dat['is_visible'] == 1]
dat_moonbright = dat[dat['illumination'] > illumination_threshold]

dat_allcond = dat_nosun[dat_nosun['is_visible'] == 1]
dat_allcond = dat_allcond[dat_allcond['illumination'] > illumination_threshold]

print(f'''Filtering results. Simulation length : {length_chosen} days
      
Total data points : {dat.size}

Moon illumination above {illumination_threshold}% : {dat_moonbright.size/dat.size*100:.1f}%
Moon not occulted by Earth : {dat_moonvis.size/dat.size*100:.1f}%
Satellite in umbra : {dat_nosun.size/dat.size*100:.1f}%

Points with all conditions satisfied : {dat_allcond.size/dat.size*100:.1f}%
''')

nr_passes = 0

t_gps = dat_allcond['t_gps'].values
t_gps_from0 = t_gps - t_gps[0]  

dt_column = np.zeros((dat_allcond.shape[0],2)) # dt ii, pass index, pass length
pass_nr = 1
# dt_column[0,1] = pass_nr
for ii, t_ii in enumerate(t_gps):
    if ii > 0:
        dt_ii = t_ii - t_gps[ii-1]
        dt_column[ii,0] = dt_ii
        if dt_ii > dt_standard:
            pass_nr +=1        
    dt_column[ii,1] = pass_nr
dat_allcond = dat_allcond.assign(pass_nr = dt_column[:,1].astype(int))

# process pass times and invis times

pass_overview = np.zeros((pass_nr, 6)) # pass nr, t_total, illum_min, t_til_next_pass, ii_start, ii_end
for ii, pass_nr in enumerate(dat_allcond['pass_nr'].unique()):
    pass_data = dat_allcond[dat_allcond['pass_nr'] == pass_nr]
    pass_tvec = pass_data['t_gps'].values
    pass_start = pass_tvec[0]
    pass_indices = pass_data.index.values
    if ii == 0:
        nopass_length = pass_tvec[0] - dat['t_gps'].values[0]
    elif ii == int(max(dat_allcond['pass_nr'])-1):
        nopass_length = max([pass_start - pass_end, dat['t_gps'].values[-1] - pass_data['t_gps'].values[-1] ])
    else:
        nopass_length = pass_start - pass_end
    pass_end = pass_tvec[-1]
    pass_length = pass_end - pass_start
    ii_start = pass_indices[0]
    ii_end = pass_indices[-1]
    illum_min = np.min(pass_data['illumination'])
    pass_overview[ii,:] = [pass_nr, pass_length, illum_min, nopass_length, ii_start, ii_end]

pass_overview_df = pd.DataFrame(pass_overview, columns = ['pass_nr', 't_pass_s', 'illum_min', 't_nopass_s', 'ii_start', 'ii_end'])
if save_csv:
    save_title = f'{sat_host}_{length_chosen}d_ilum{illumination_threshold}_perf_moon_conditions.csv'
    dat_allcond.to_csv(f'{output_folder_filtered}/{save_title}', index = 0)
    print(f'Saved {save_title}')
    save_title_ov = f'{sat_host}_{length_chosen}d_ilum{illumination_threshold}_passdata.csv'
    pass_overview_df.to_csv(f'{output_folder_overview}/{save_title_ov}', index = 0)   
    print(f'Saved {save_title_ov}')


# #%%
# f, ax = plt.subplots(figsize = (8,5))
# if 'meo' in sat_host:
#     tsetting = 'hr'
# else:
#     tsetting = 'min'
# t_vec = t_gps - t_gps[0]
# if tsetting == 'min':
#     t_plot = t_vec/60
#     unit = 'min'
#     xlim = 300
#     t_min_steps = 20
# elif tsetting == 'hr':
#     t_plot = t_vec/60/60
#     unit = 'hr'
#     xlim = 14
#     t_min_steps = 1

# vis_labels = ['Visible' if ii == 1 else 'Invisible' for ii in vis_cond]
# ax.scatter(t_plot, vis_labels)
# ax.set_xlim([0,xlim])
# ax.set_xticks(np.arange(0, xlim, t_min_steps))
# for ii_start in i_start_lst:
#     ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r')
# for ii_start in i_end_lst:
#     ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r')
#     if ii_start == i_end_lst[-1]:
#         ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r', label = 'Visibility start/end')
# ax.legend()
# # ax.grid('on')
# # ax.invert_yaxis()
# ax.set_ylim(['Invisible', 'Visible'])
# ax.set_xlabel(f't [{unit}]', fontweight = 'bold')
# f.suptitle(f'Moon visibility for {sat_name}', fontweight = 'bold' )
# bplt.autosave(f, subfolder = 'MoonConops')
# #%%
# make_vistime_plot = 1
# if make_vistime_plot:
#     # time [days]
#     t_plotted = (t_start_lst-t_gps[0]/60)/1440
#     t_plotted_full = (t_gps - t_gps[0])/86400
#     f, ax = plt.subplots()
#     f.suptitle(f'Moon Visibility Times For {sat_name}')
#     ax.scatter(t_plotted[:-1], t_vis_lst,c = 'g', s = 2, label = 'Visible Moon time')
#     ax.plot(ax.get_xlim(), [T_eclp,T_eclp], c = 'm', linestyle = '--', label = 'Analytical worst-case eclipse approximation')
#     ax.plot(ax.get_xlim(), [T_vis,T_vis], c = 'orange', linestyle = '--', label = 'Analytical minimum visibility time')
#     ax.scatter(t_plotted[:-1], t_ecl_lst,c = 'r', s = 2, label = 'Moon Occulted by Earth')
#     ax.scatter(t_plotted_full, illum, c = 'y', label = 'Moon Illumination [%]', s = 15, marker = 'x')
#     ax.legend()
#     ax.set_xlim(t_plotted[0],t_plotted[-1] )
#     ax.set_xlabel('t since start [days]')
#     ax.set_ylabel('Visible time per pass [min]')
#     ax.set_ylim([0,600])
#     ax.grid('on')
#     bplt.autosave(f, subfolder = 'MoonConops')