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
length_chosen = 7  # days
# length_chosen = 62 # days
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
input_folder = r'outputs\tables\moon_vis'
## Loading satellite orbital data

t_used = f'{length_chosen:.0f}d'
chosen_index = 0
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
dat = pd.read_csv(f'{input_folder}/{f_chosen}')
dat = dat.iloc[3600:,:]
t_gps = dat['t_gps'].values
t_step = t_gps[1] - t_gps[0]
vis_cond = dat['is_visible'].values
illum = dat['illumination'].values
umbra = dat['in_umbra'].values
pos_h = dat[['x_h','y_h', 'z_h']].values
pos_t = dat[['x_m','y_m', 'z_m']].values
ii_vis = [ii for ii, vis in enumerate(vis_cond) if vis]
#%%
make_3d_plot = 0
if make_3d_plot:
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_orbit_basic(f, ax, pos_h[-6000:,:], label = 'host', c = 'y')
    # f, ax = modplot.add_orbit_basic(f, ax, pos_t, label = 'moon', c = 'b')
    f, ax = modplot.add_earth(f, ax)
    modplot.set_axes_equal(ax)
    ax.view_init(0,0)
#%%

nr_passes = 0
pass_dat = np.zeros((10000, 6)) # length, illumination, t_start, t_end, ii_start, ii_end
if len(ii_vis) != 0: # link occurs 
    nr_passes+=1
    ## Split entire link range to separate links
    ii_prev = ii_vis[0] # previous index w.r.t. all array
    ii_start = ii_vis[0] # link start index
    t_start = t_gps[0] # link start time 
    jj_start = 0 # index w.r.t. LINK array
    n_link = 1 # total number of link tracker w/ current target
    ## jj - indices in link_data. ii - indices in raw_data
    # Overview outputs (1 value per link)
    t_window_lst, type_lst, t_start_lst, t_end_lst, i_start_lst, i_end_lst, illum_lst, slant_range_max_lst= [],[],[],[],[],[],[],[]
    link_lst, az_max, el_max, daz_max, del_max, ic_host, ic_target = [],[],[],[],[],[], []
    ## Time-series outputs (all points per link)
    az_h = []
    el_h = []
    r_h = []
    # Gradients
    dr_h = []
    daz_h = []
    del_h = []
    t_h = []
    for jj, ii_next in enumerate(ii_vis):
        ii = ii_next
        d_ii = ii_next - ii_prev # difference in link array index                
        if d_ii > 1 or (ii_start == ii_vis[0] and ii_next+1 == len(ii_vis)): # new link begins - doesnt trigger for cosntant links
            if d_ii > 1:
                # get link end parameters if link end was found
                if n_link > 1:
                    t_nolink = t_start - t_end
                t_end = t_gps[ii_vis[jj-1]]
                link_type = 'window'
                t_window = t_end - t_start
                if ii_start == ii_prev: # in case detected link is only a single data point
                    ii_prev +=1 # include the next state point

                row_start = ii_start if ii_start == 0 else ii_start # EXclude data point just before link occurs
                row_end = ii_prev                                
                # extract chosen LCT azimuth, elevation, LOS vector
                illum_lst.append(illum[row_start])
                type_lst.append(link_type)
                t_window_lst.append(t_window/60)
                t_start_lst.append(t_start/60)
                i_start_lst.append(row_start)
                i_end_lst.append(row_end)
                ic_host.append(pos_h[row_start,:])
                ic_target.append(pos_t[row_start,:])
                link_lst.append(n_link)
            jj_start = jj
            ## Update indices
            n_link+=1
            ii_start = ii_next
            t_start = t_gps[ii] # start of next link time
        elif d_ii == 1: # link continues 
            pass
        ii_prev = ii_next
#%% Analytical calc
t_ecl_lst = []
t_vis_lst = []
for ii, i_end in enumerate(i_start_lst[:-1]):
    t_ecl = i_start_lst[ii+1] - i_end_lst[ii]
    t_vis = i_end_lst[ii] - i_start_lst[ii]
    t_ecl_lst.append(t_ecl*t_step/60) # min
    t_vis_lst.append(t_vis*t_step/60) # min
R_E = 6378e3
h = np.linalg.norm(pos_h[0,:])-6378e3


T_sat = 2*np.pi*np.sqrt((R_E+h)**3/3.986e14)/60 #min
T_eclp = T_sat * 2*np.arcsin(R_E / (R_E+h))/6.28
T_vis = T_sat - T_eclp
print(f'''
      At h = {h/1e3:.0f} km -> T = {T_sat:.0f} min
      Moon Eclipse (less, but worst-case [eq. orbit]) : {T_eclp:.0f} min
      -> Vis : {T_vis:.0f} min
      ''')
#%%
f, ax = plt.subplots(figsize = (8,5))
if 'meo' in sat_host:
    tsetting = 'hr'
else:
    tsetting = 'min'
t_vec = t_gps - t_gps[0]
if tsetting == 'min':
    t_plot = t_vec/60
    unit = 'min'
    xlim = 300
    t_min_steps = 20
elif tsetting == 'hr':
    t_plot = t_vec/60/60
    unit = 'hr'
    xlim = 14
    t_min_steps = 1

vis_labels = ['Visible' if ii == 1 else 'Invisible' for ii in vis_cond]
ax.scatter(t_plot, vis_labels)
ax.set_xlim([0,xlim])
ax.set_xticks(np.arange(0, xlim, t_min_steps))
for ii_start in i_start_lst:
    ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r')
for ii_start in i_end_lst:
    ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r')
    if ii_start == i_end_lst[-1]:
        ax.plot([t_plot[ii_start], t_plot[ii_start]], ['Invisible','Visible'], c = 'r', label = 'Visibility start/end')
ax.legend()
# ax.grid('on')
# ax.invert_yaxis()
ax.set_ylim(['Invisible', 'Visible'])
ax.set_xlabel(f't [{unit}]', fontweight = 'bold')
f.suptitle(f'Moon visibility for {sat_name}', fontweight = 'bold' )
bplt.autosave(f, subfolder = 'MoonConops')
#%%
make_vistime_plot = 1
if make_vistime_plot:
    # time [days]
    t_plotted = (t_start_lst-t_gps[0]/60)/1440
    t_plotted_full = (t_gps - t_gps[0])/86400
    f, ax = plt.subplots()
    f.suptitle(f'Moon Visibility Times For {sat_name}')
    ax.scatter(t_plotted[:-1], t_vis_lst,c = 'g', s = 2, label = 'Visible Moon time')
    ax.plot(ax.get_xlim(), [T_eclp,T_eclp], c = 'm', linestyle = '--', label = 'Analytical worst-case eclipse approximation')
    ax.plot(ax.get_xlim(), [T_vis,T_vis], c = 'orange', linestyle = '--', label = 'Analytical minimum visibility time')
    ax.scatter(t_plotted[:-1], t_ecl_lst,c = 'r', s = 2, label = 'Moon Occulted by Earth')
    ax.scatter(t_plotted_full, illum, c = 'y', label = 'Moon Illumination [%]', s = 15, marker = 'x')
    ax.legend()
    ax.set_xlim(t_plotted[0],t_plotted[-1] )
    ax.set_xlabel('t since start [days]')
    ax.set_ylabel('Visible time per pass [min]')
    ax.set_ylim([0,600])
    ax.grid('on')
    bplt.autosave(f, subfolder = 'MoonConops')