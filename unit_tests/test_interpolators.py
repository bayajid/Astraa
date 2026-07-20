## Minor script to interpolate position and velocity data using quadratic
# polynomials.
# example attitude scenario
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
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
csv_output_path = r'orbital_simulations/tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
att_fine = r'outputs/attitude_tests/QQdot_4hz'
## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv

import tudat_tools.data_processing.data_processing_utilities as dputil
import prediction_methods.interpolators as interp
import prediction_methods.attitude_prediction_methods as att_pred
save_attitude = 1
save_interpolated_attitude = 1
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = 1000)

host_chosen = 'leo_host_polar'

t_j2000 = data_raw[:,0]
t_gps = t_j2000+t_conv.dt_j2000tt2gps()
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
dt = t_gps[1]-t_gps[0]
ii0, ii1 = 0,5
t0 = t_gps[ii0]
t1 = t_gps[ii1]
r0, r1 = r_host[ii0,:], r_host[ii1,:]
v0 = v_host[ii0,:]
v1 = v_host[ii1,:]
# importlib.reload(interp)
interpolache = interp.we_interpolating_pos()
interpolache.get_quad_interpolant(t0, t1, r0, r1, v1)

t_interp = np.arange(t0, t1+dt, dt)
r_interpolated = interpolache.interpolate(t_interp)
r_true = r_host[ii0:ii1+1,:]

f, ax = plt.subplots()
t_plot = t_interp - t_interp[0]
for ii in range(3):
    ax.plot(r_true[:,ii] - r_interpolated[:,ii], label = 'xyz'[ii])
ax.legend()
ax.set_ylabel('dr [m]')
ax.set_xlabel('t from t0 [s]')
