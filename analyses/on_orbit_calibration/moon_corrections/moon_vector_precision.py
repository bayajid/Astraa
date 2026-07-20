#%% Script to analyze moon-vector errors over time
# save errors in ECI x/y/z AND in pointing error []
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
year_end = 2023.1

# dt_hours = 1 # hour -> 24 x 365 x 17 ~160000 computations
# dt_seconds = dt_hours*3600
dt_seconds = 60
t_start_cest = dt.datetime(year_start, 1, 1, 0, 0, 0)
dt_hours = dt_seconds / 3600
loop_length = int((year_end-year_start)*365*24/dt_hours) 


err_placeholder = np.zeros((loop_length, 5)) # t [t_gps], pe [mrad], dx, dy, dz [m]

use_single_precision = 0
use_more_precise_moonvec = 1
truncation = 0
save_errors = 1
t_gps_0 = t_conv.utc2gws(t_start_cest+ dt.timedelta(hours = -2))
sf_obj = where_sun.body_fromsp(t_gps_0+t_conv.dt_gps2j2000tt())
for ii in range(loop_length):
    dt_ii = ii*dt_seconds
    t_ii = t_gps_0 + dt_ii

    if use_more_precise_moonvec:
        if use_single_precision:
            moon_vec_own = where_sun.where_moon_single_prec(t_ii, rotate = 1, truncation = truncation, single_precision = use_single_precision)
            used_type = f'New-Meeus Single Precision'
        else:
            used_type = 'New-Meeus'    
            moon_vec_own = where_sun.where_moon_for_real(t_ii, rotate = 1, truncation = truncation, single_precision = use_single_precision)
    else:
        used_type = 'Original'
        moon_vec_own = where_sun.compute_moon_vector_eci(t_ii) # approx
    


    moon_vec_sf = sf_obj.get_sun(dt_ii, body = 'moon') # truth

    dr = moon_vec_sf - moon_vec_own # m
    pe = vec_op.calc_dot_angle(moon_vec_own, moon_vec_sf)*1e3 # [rad -> mrad]
    err_placeholder[ii,:2] = [t_ii, pe]
    err_placeholder[ii,2:] = dr
error_df = pd.DataFrame(data = err_placeholder, columns = ['t [t_gps]', 'pe [mrad]', 'dx', 'dy', 'dz'])
if save_errors:
    title = f'MoonErr_{used_type}.csv'
    print(f'Saved {title}')
    error_df.to_csv(title, index = 0)
#%%
d_years = (year_end-year_start)*365
d_years = 5
f, axs = plt.subplots(2)
t_fromstart_d = err_placeholder[:,0]
t_fromstart_d = (t_fromstart_d - t_fromstart_d[0])/86400
ax = axs[0]
ax.plot(t_fromstart_d, err_placeholder[:,1])
ax.set_ylabel(f'PE [mrad]')
ax.grid('on')
ax.set_xlim([0, d_years])

ax = axs[1]
for ii in range(3):
    ax.plot(t_fromstart_d, err_placeholder[:,ii+2], label = 'xyz'[ii])
    ax.set_ylabel(f'Pos. Error [m]')
    ax.set_xlim([0, d_years])
ax.set_xlabel(f't since {year_start} [days]')
ax.grid('on')
f.set_tight_layout('tight')
f.suptitle(f'Approx. Moon Vector Errors - {used_type} calculation Trunc {truncation} vs de440 Planetary Ephemeris')
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