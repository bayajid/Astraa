#%% eRROR PLOTS of moon vector for approx and precise functions VS NASA ephemeris 
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
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt

path_main = r'analyses\moon_tracking\moon_vectors'
files_av = os.listdir(path_main)
#%%
conv_sp_to_pe = 1
if conv_sp_to_pe:
    moon_sp = pd.read_csv(f'{path_main}/{files_av[0]}', header = 0).values
    t_sp = moon_sp[:,0]
    r_moon = moon_sp[:,[1,2,3]]
    pe_sp = np.zeros((moon_sp.shape[0], moon_sp.shape[1]+1)) # tPE, dx, dy, dz
    t_gps_0 = t_sp[0]

    sf_moon = where_sun.body_fromsp(t_gps_0 + t_conv.dt_gps2j2000tt())
    for ii, t_ii in enumerate(t_sp):
        r_moon_true = sf_moon.get_sun(t_ii - t_gps_0, body = 'moon')
        pe = vec_calc.calc_dot_angle(r_moon[ii,:], r_moon_true)
        dr = r_moon_true - r_moon[ii,:]

        pe_sp[ii,0] = t_ii
        pe_sp[ii,1] = pe
        pe_sp[ii,2:] = dr
    df_pe_sp = pd.DataFrame(pe_sp, columns = ['t [t_gps]', 'pe [mrad]', 'dx', 'dy', 'dz'])
    df_pe_sp.to_csv(f'{path_main}/pe_sp_approxmoon.csv', index = 0)

#%%
files_pe = [file for file in  os.listdir(path_main) if 'pe' in file or 'Err' in file]
labels_pe = [
    'Precise, double-prec.',
    'Approx, double-prec.',
    'Approx, single-prec.',
]
zorders = [1, 0.9, 0.5]
colors = ['b', 'r', 'y']
plot_x = 'hour'
plot_x = 'day'
n_days_plotted = 31
n_hours_plotted = 4



# t_fromstart_d = err_placeholder_1hr[:,0]
# t_fromstart_d = (err_placeholder_1hr[:,0] - err_placeholder_1hr[0,0])/86400
# t_fromstart_hr = (t_fromstart_d - t_fromstart_d[0])*24

if plot_x == 'hour':
    xlim = n_hours_plotted
    # t_plotted = t_fromstart_hr
    unit = 'hr'
elif plot_x == 'day':
    xlim = n_days_plotted
    # t_plotted = t_fromstart_d
    unit = 'day'



f, ax = plt.subplots(1)
for ii, file in enumerate(files_pe):
# ax.plot(t_plotted, err_placeholder_1hr[:,1])
    data  = pd.read_csv(f'{path_main}/{file}').values
    if ii == 2:
        data[:,1] = data[:,1] * 1e3
        # print(pd.read_csv(f'{path_main}/{file}'))
    t_gps = data[:,0] - data[0,0]
    pe = data[:,1]
    if plot_x == 'hour':
        t_plotted = t_gps / 3600 
    else:
        t_plotted = t_gps / 3600 / 24
    ax.plot(t_plotted, pe, label = labels_pe[ii], zorder = zorders[ii], c = colors[ii])

ax.set_ylabel(f'PE [mrad]')
ax.grid('on')
ax.set_xlim([0, xlim])
# ax.set_ylim([0,1])
ax.set_xlabel(f't since start [{unit}]')
ax.legend()
ax.grid('on')
f.set_tight_layout('tight')
f.suptitle(f'Calculated Moon Vector Difference from DE440 Planetary and Lunar Ephemeris')
bplt.autosave(f, subfolder = 'MoonFixing')