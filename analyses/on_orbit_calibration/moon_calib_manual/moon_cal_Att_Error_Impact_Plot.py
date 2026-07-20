#%% August 21- quantify impact of error sources
# eg host/moon pos, host att, 
import scipy as sp
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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import plotting_tools.basic_plotting as bplt

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
# Moon calibration condition file
path_err = r'outputs\tables\moon_mo_res_errors'
err_all = os.listdir(path_err)

err_used = 'Att'
sat_orbit = 'LEO Polar Orbit'
err_filt = [e for e in err_all if err_used in e]
print(f'Err found :{err_filt}')
labels = ['Earth-point Stable', 'Earth-point Rolling', 'Sun-point Stable', 'Sun-point Rolling']
f, ax = plt.subplots()
for ii, err in enumerate(err_filt):
    err_data = pd.read_csv(fr'{path_err}/{err}')
    ax.plot(err_data.iloc[:,0].values, err_data.iloc[:,1].values, label = labels[ii])

ax.set_yscale('log')
ax.set_xlabel('Attitude Error Per Axis [mrad]', fontweight = 'bold')
ax.grid('on')
ax.set_xlim([0, 0.3])
ax.set_ylim([1e-2, 5e1])
ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
ax.legend()
f.suptitle(f'Mounting Offset Resolution Error for 4 Attitude Profiles \n{sat_orbit},considering only S/C Attitude Error Impact')
