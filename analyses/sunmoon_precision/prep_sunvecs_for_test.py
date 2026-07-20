#%%
import matplotlib.pyplot as plt
# import splines.quaternion
import pathlib
import os
import numpy as np
import pandas as pd
import sys
import importlib
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])

import json
import attitude_tools.conversions as conv
import paa_tools.paa_calculation as paa_calc
import plotting_tools.modular_plotting as modplot
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec_op
import basic_tools.data_loading as load
import basic_tools.in_out as out
import astronomy_tools.astro_targets as where_sun


import astropy.coordinates as ap_coord
import astropy.time as ap_time
from skyfield.api import load as sfload
import skyfield.framelib as framelib
import astronomy_tools.astro_rotations as ast_rot
import datetime as dt
importlib.reload(out)
importlib.reload(modplot)
importlib.reload(vec_op)
importlib.reload(where_sun)
importlib.reload(t_conv)

#%% get sun _vec
# Full script to compute sun-vector Azimuth/Elevation 
# for a given LEO satellite with some provided attitude

# placeholders for test cases
r_host_in = []
t_gps_in = []
sun_vec_out = []
sun_vec_source = []

get_unit_vec = 0
## Own function
# t_gps = 1277693948.816 # 646930800.0 J2000 - [2020-Jul]
t_0 = dt.datetime(2020, 3, 3, 12, 0, 0) # utc, initial time
t_gps = t_conv.utc2gws(t_0)
time_step = 86400 * 7 + 1757 # every 7 days + 30ish mins

nr_vectors = int((10 * 365.25*86400) / time_step)


t_vector_gps_all = np.arange(t_gps, t_gps + nr_vectors*time_step, time_step)


# using skyfield get apparent sun vectors from earth
sun_app_verif = where_sun.body_fromsp(t_gps, t_type = 'gps') 


outputs = np.zeros((nr_vectors, 5)) # t, r_sun, pe_expected


for ii, t_ii in enumerate(t_vector_gps_all):
    # good
    r_sun_app_sf = sun_app_verif.get_sun(t_ii - t_gps)
    # approx
    r_sun_app_own = where_sun.compute_sun_vector_eci_better(t_ii)
    pe_exp = vec_op.calc_dot_angle(r_sun_app_sf, r_sun_app_own)
    # store
    outputs[ii,0] = t_ii
    outputs[ii,[1,2,3]] = r_sun_app_sf
    outputs[ii,[4]] = pe_exp # rad
print('outputs made')
sunvec_df = out.make_n_save('sun_vec_tests', outputs, data_cols=['t_gps','x_sun','y_sun','z_sun','pe_rad'])