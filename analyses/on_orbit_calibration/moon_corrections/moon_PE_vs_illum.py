#%% Script to analyze moon-vector errors over time 
# and whether they depend on the illumination percentage
import matplotlib.pyplot as plt
import numpy as np
import os 
import datetime as dt
import pandas as pd
import importlib
import os, sys
from skyfield.api import load as sfload
import skyfield.framelib as framelib
path_cwd = os.getcwd()
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import astronomy_tools.astro_targets as where_sun
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec_op
import plotting_tools.basic_plotting as bplt
importlib.reload(where_sun)
year_start = 2023
year_end = 2025

# dt_hours = 1 # hour -> 24 x 365 x 17 ~160000 computations
# dt_seconds = dt_hours*3600
dt_seconds = 3600
t_start_cest = dt.datetime(year_start, 1, 1, 0, 0, 0)
dt_hours = dt_seconds / 3600
loop_length = int((year_end-year_start)*365*24/dt_hours) 

err_placeholder = np.zeros((loop_length, 3)) # t [t_gps], pe [mrad], illum [%]

use_more_precise_moonvec = 0
save_errors = 0
t_gps_0 = t_conv.utc2gws(t_start_cest+ dt.timedelta(hours = -2))
sf_obj = where_sun.body_fromsp(t_gps_0+t_conv.dt_gps2j2000tt())
for ii in range(loop_length):
    dt_ii = ii*dt_seconds
    t_ii = t_gps_0 + dt_ii

    moon_vec_own, illum_ii = where_sun.compute_moon_vector_eci(t_ii, what_brightness=1) # approx
    
    moon_vec_sf = sf_obj.get_sun(dt_ii, body = 'moon') # truth

    pe = vec_op.calc_dot_angle(moon_vec_own, moon_vec_sf)*1e3 # [rad -> mrad]
    err_placeholder[ii,:2] = [t_ii, pe]
    err_placeholder[ii,2:] = illum_ii
error_df = pd.DataFrame(data = err_placeholder, columns = ['t [t_gps]', 'pe [mrad]', 'illum'])

if save_errors:
    title = f'MoonErr_{used_type}.csv'
    print(f'Saved {title}')
    error_df.to_csv(title, index = 0)
#%% Filter PE with illumination
illum_threshold = 95
ii_filt = [ii for ii, err in enumerate(err_placeholder[:,1]) if err_placeholder[ii,2]> illum_threshold]

#%%

f, axs = plt.subplots(3)
t_fromstart_d = err_placeholder[:,0]
t_fromstart_d = (t_fromstart_d - t_fromstart_d[0])/86400
ax = axs[0]
ax.plot(t_fromstart_d, err_placeholder[:,1])
ax.scatter(t_fromstart_d[ii_filt], err_placeholder[ii_filt,1], c = 'r', label = f'Illum > {illum_threshold}', s = 2)
ax.legend()
ax.set_ylabel(f'PE [mrad]')
ax.grid('on')
ax.set_xlim([0, max(t_fromstart_d)])

ax = axs[1]

ax.plot(t_fromstart_d, err_placeholder[:,2])
ax.set_ylabel(f'Moon Illumination [%]')
ax.set_xlim([0, max(t_fromstart_d)])
ax.set_xlabel(f't since {year_start} [days]')
ax.grid('on')

ax = axs[2]
ax.scatter(err_placeholder[:,2], err_placeholder[:,1], s = 2, alpha = 0.1)
ax.set_ylabel(f'PE [mrad]')
ax.set_xlabel(f'Illumination [%]')
ax.grid('on')

f.set_tight_layout('tight')
f.suptitle(f'Moon Illumination and Approx. Dbl-Precision Moon Vector Error')
bplt.autosave(f, subfolder = 'MoonPrecision')
#%%
make_fft_check = 0
if make_fft_check:
    freq_dom_x = np.fft.fft(err_placeholder[:,2])
    freq_dom_y = np.fft.fft(err_placeholder[:,3])
    freq_dom_z = np.fft.fft(err_placeholder[:,4])
    freq_all = [freq_dom_x, freq_dom_y, freq_dom_z]
    f, axs = plt.subplots(3)
    freq = np.fft.fftfreq(len(t_fromstart_d))
    for ii, ax in enumerate(axs):
        fft_plotted = freq_all[ii]
        ax.plot(freq, fft_plotted.real)
        ax.set_xlim([0, 0.01])