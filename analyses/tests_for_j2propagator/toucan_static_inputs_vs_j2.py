# Analyzing what position/pointing errors are added
# by propagators and extrapolators being activated
# and receiving static inputs, 0 velocity inputs, updates every 5 Hz 
#%% IMPORTS
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
# csv_output_path = r'orbital_simulations\single_sat\leo_j2_prop'
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import prediction_methods.j2propagator as j2prop
import prediction_methods.interpolators as interp
## LIST COMPREHENSION ft. if else
# [x+1 if x >= 45 else x+5 for x in l]

## Loading satellite orbital data
importlib.reload(j2prop)
import tudat_tools.data_processing.data_processing_utilities as dputil
data_update_rate = 5 # Hz
cts_update_rate = 200 # Hz
# INITIAL CONDITIONS
r_0 = [11e6, 0, 0]
r_0 = np.array(r_0)
v_0 = [0, 0, 0]
v_0 = np.array(v_0)
X = np.hstack((r_0,v_0))
# J2 propagator
t_0 = 10
nr_upd_waited = 100
t_f = 1 / data_update_rate * nr_upd_waited
dt = 1 / cts_update_rate
r_host_predicted = j2prop.propagate_orbit(X, t_0, t_f, 10)
r_host_predicted = np.vstack((np.hstack(([0], X)), r_host_predicted))
t_prop = r_host_predicted[:,0]
dr_prop = r_0 - r_host_predicted[:,[1,2,3]]
update_index_coming = np.logspace(0,1, num = 2)
t_updates = [ii * (1/data_update_rate) for ii in update_index_coming]
# Quadratic Interpolator
interp_class = interp.we_interpolating()
t1 = 1.3e9
t2 = 1.3e9 + r_host_predicted[1,0]
s1 = X
s2 = r_host_predicted[0,1:]
interp_class.get_quad_interpolant([t1, t2], 
                                    r_both = np.vstack((s1[:3], s2[:3])), 
                                    v_both = np.vstack((s1[3:], s2[3:])))
t_interp = np.arange(t_prop[0], t_prop[-1], 1)
r_interp = interp_class.interpolate(t_interp + t1)
dr_interp_wj2 = r_interp - X[:3]
if 1:
    ylims = [np.min(dr_prop), np.max(dr_prop)]
    f, ax = plt.subplots()
    ax.plot(t_prop, np.linalg.norm(dr_prop, axis = 1), label = 'J2')
    # ax.plot(t_interp, np.linalg.norm(dr_interp_wj2, axis = 1), label = 'J2 + extrap')
    for ii, t_upd in enumerate(t_updates):
        ax.plot([t_upd, t_upd], ax.get_ylim(), label = f'Update {t_upd*data_update_rate:.0f}')
    ax.set_ylabel('dr [m]')
    ax.set_xlabel('t since update [s]')
    f.suptitle('Propagated vs Static Position error using J2 propagator and Quadratic Extrap.')
    ax.set_ylim(ylims)
    ax.grid('on')
    ax.legend()
    # columns = ['t_gps', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'dr_exp']
    # out.make_n_save('j2_prop_testdata', data = data_to_store, data_cols = columns, subfolder = 'j2prop_testdata')
    # #%% plot 
    # f, ax = plt.subplots()
    # ax.plot(t_j2000 - t_j2000[0], np.linalg.norm(dr, axis = 1))
    # ax.set_ylabel('dr [m]')
    # ax.set_xlabel('t [s]')