## Templates for loading satellite data
# generating attitude
# and whatnot. 

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
csv_output_path = r'orbital_simulations/leo_meo_srpcheck_high_precision/leo_meo_srpcheck'#'orbital_simulations\capella_leo_polar_2000km_medium_precision\capella_leo_polar_2000km'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'

## MVP imports
import basic_tools.vector_operations as vec_calc
import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun
import attitude_tools.attitude_simulation as att_sim
import plotting_tools.basic_plotting as bplt
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as out
import pointing_calculations.ae_calculation as ae_calc

save_title = 'capella_leo_coplanar_2000km'
## Loading satellite orbital data
import tudat_tools.data_processing.data_processing_utilities as dputil
make_plot = 1
make_reverse_calc = 1
save_outputs = 1
nrows = 3600*4

data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows=nrows)

host_chosen = 'leo_polar'
target_chosen = 'meo_eq'#'leo_polar_2'
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'
t_j2000 = data_raw[:,0]
r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
r_target = data_raw[:,simulation_parameters['r_index'][target_chosen]]
v_target = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][target_chosen]]]
import tudat_tools.tudat_converter as tudatconv

## Getting ECI to RSW rotation matrix
importlib.reload(att_sim)
tudconv = tudatconv.tudat_predictor()
ii = 0
q_all, rot_eci2lct, q_dot_all = att_sim.calc_quat_eci2lct(r_host, v_host, calc_qdot=1, t_vec = t_j2000)
q_mo = np.copy(q_all)
q_mo[:,0] = 1
q_mo[:,1:] = q_mo[:,1:] * 0
#%% calc AER
states_host = np.hstack((r_host, v_host))
states_target= np.hstack((r_target, v_target))
attitude_eci2bf= np.hstack((q_all, q_dot_all))
aer_all = ae_calc.calc_ae_full(
    states_host = states_host,
    states_target= states_target,
    attitude_eci2bf= attitude_eci2bf,
    rotation_function=1,
)
#%% Save
if save_outputs:
    out.save_azel(
        t_gps = t_j2000 + t_conv.dt_j2000tt2gps(),
        s_h = states_host,
        s_t = states_target,
        q_eci2bf = attitude_eci2bf,
        q_mo = q_mo,
        ae = aer_all,
        fname = f'{save_title}_r1',
        pos_unit = 'km',
                
                )
import plotting_tools.combined_plots as cmb_plt   
if make_plot:
    cmb_plt.plot_aer(t_j2000, aer_all, unit = 'rad', setting = '', title = 'Rotation New')

if make_reverse_calc:
    aer_all_2 = ae_calc.calc_ae_full(
        states_host = states_host,
        states_target= states_target,
        attitude_eci2bf= attitude_eci2bf,
        rotation_function=2
    )
    cmb_plt.plot_aer(t_j2000, aer_all_2, unit = 'rad', setting = '', title = 'Rotation Old')
    out.save_azel(
        t_gps = t_j2000 + t_conv.dt_j2000tt2gps(),
        s_h = states_host,
        s_t = states_target,
        q_eci2bf = attitude_eci2bf,
        q_mo = q_mo,
        ae = aer_all_2,
        fname = f'{save_title}_r2',
        pos_unit = 'km',
    )