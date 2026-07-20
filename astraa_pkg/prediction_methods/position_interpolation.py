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
csv_output_path = r'orbital_simulations\tudat_raw_states'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
att_fine = r'outputs\attitude_tests\QQdot_4hz'
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

ii0, ii1 = 0,999
t0 = t_gps[ii0]
t1 = t_gps[ii1]
# r0, r1 = r_host[ii0,:], r_host[ii1,:]
# v0 = v_host[ii0,:]
# v1 = v_host[ii1,:]
# importlib.reload(interp)
# interpolache = interp.we_interpolating_pos()
# interpolache.get_quad_interpolant(t0, t1, r0, r1, v1)
make_pos_interp4hz = 0
make_both_interp_200hz = 1
if make_pos_interp4hz:
# interpolation time-vector
    dt = 0.25
   
    t_interp = np.round(np.arange(t0, t1+dt, dt), 4)
    t_interp = t_interp.reshape((t_interp.shape[0],1))

    
    

    # get velocity via splines
    v0_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], v_host[:ii1,0])
    v1_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], v_host[:ii1,1])
    v2_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], v_host[:ii1,2])
    v0_interpolated = v0_interpolator(t_interp)
    v1_interpolated = v1_interpolator(t_interp)
    v2_interpolated = v2_interpolator(t_interp)

    v_interp = np.hstack((
    v0_interpolated,
    v1_interpolated,
    v2_interpolated,
    ))
    r0_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], r_host[:ii1,0])
    r1_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], r_host[:ii1,1])
    r2_interpolator = sp.interpolate.CubicSpline(t_gps[:ii1], r_host[:ii1,2])
    r0_interpolated = r0_interpolator(t_interp)
    r1_interpolated = r1_interpolator(t_interp)
    r2_interpolated = r2_interpolator(t_interp)

    r_interp = np.hstack((
    r0_interpolated,
    r1_interpolated,
    r2_interpolated,
    ))

    path_output = r'outputs\attitude_tests'
    path_fullstates = f'{path_output}\states_fine.csv'
    states_fine = pd.DataFrame(np.hstack((t_interp, r_interp, v_interp)), columns = ['t_gps','r_x','r_y','r_z','v_x','v_y','v_z'])
    states_fine.to_csv(path_fullstates, index = 0)
    print(f'saved {path_fullstates}')
# elif make_both_interp_200hz:
pos_fine_df = pd.DataFrame(data = np.hstack((np.expand_dims(t_gps, -1), r_host, v_host)))
make_pos = 1
make_att = 0
dt = 0.005
ii_used = [0,1,2,3,4]
t_in_all = pos_fine_df.iloc[ii_used, [0]].values
r_in_all = pos_fine_df.iloc[ii_used, [1,2,3]].values
v_in_all = pos_fine_df.iloc[ii_used, [4,5,6]].values
if make_pos:
    t_interp_full = np.round(np.arange(t_in_all[0], t_in_all[-1], dt),4)
    data_out = np.zeros((t_interp_full.shape[0],4))
    pos_interpolator = interp.we_interpolating()

    n_datapoint_per_t = int(data_out.shape[0]/(len(ii_used)-1))
    ii_n = 0
    for ii, ii_0 in enumerate(ii_used): 
        if ii_0 != ii_used[-1]:
            
            # get ii_0, ii_1
            ii_used_ii = [ii_0, ii_used[ii+1]]
            t_stamps_ii = pos_fine_df.iloc[ii_used_ii, [0]].values
            # get time for interpolation
            t_interpolation_ii = np.round(np.arange(t_stamps_ii[0], t_stamps_ii[1], dt),4)
            r_in, v_in = r_in_all[[ii, ii+1],:], v_in_all[[ii, ii+1],:]
            pos_interpolator.get_quad_interpolant(t_stamps_ii.flatten(), r_in, v_in)
            # interpolate
            r_interp = pos_interpolator.interpolate(t_interpolation_ii)
            # store
            data_out[ii_n:ii_n+n_datapoint_per_t,0] =t_interpolation_ii
            data_out[ii_n:ii_n+n_datapoint_per_t,1:] = r_interp
            ii_n += n_datapoint_per_t
    
    r_input_df = pd.DataFrame(data = np.hstack((t_in_all, r_in_all, v_in_all)), columns = ['t_gps','r_x','r_y','r_z', 'v_x', 'v_y', 'v_z'])
    r_output_df = pd.DataFrame(data = data_out, columns = ['t_gps','r_x','r_y','r_z'])
    path_output = r'outputs\attitude_tests'

    path_input_df = fr'{path_output}\posinterp_input_df.csv'
    path_output_df = fr'{path_output}\posinterp_output_df.csv'

    r_input_df.to_csv(path_input_df, index = 0 )
    print(f'Saved {path_input_df}')
    r_output_df.to_csv(path_output_df, index = 0 )
    print(f'Saved {path_output_df}')
if make_att:
    # ATTITUDE
    att_fine_df = pd.read_csv(f'{att_fine}.csv')
    
    ii_used = list(range(0, 1+np.where(att_fine_df.iloc[:,0].values == t_in_all[-1])[0][0]))
    q_in_all = att_fine_df.iloc[ii_used, [2,3,4,5]].values
    qdot_in_all = att_fine_df.iloc[ii_used, [6,7,8,9]].values
    t_in_all = att_fine_df.iloc[ii_used, [0]].values
    t_interp_full = np.round(np.arange(t_in_all[0], t_in_all[-1], dt),4)
    data_out = np.zeros((t_interp_full.shape[0],5))
    att_interpolator = interp.we_interpolating()

    n_datapoint_per_t = int(data_out.shape[0]/(len(ii_used)-1))
    ii_n = 0
    for ii, ii_0 in enumerate(ii_used): 
        if ii_0 != ii_used[-1]:
            
            # get ii_0, ii_1
            ii_used_ii = [ii_0, ii_used[ii+1]]
            t_stamps_ii = att_fine_df.iloc[ii_used_ii, [0]].values
            # get time for interpolation
            t_interpolation_ii = np.round(np.arange(t_stamps_ii[0], t_stamps_ii[1], dt),4)
            q_in, qdot_in = att_fine_df.iloc[ii_used_ii, [2,3,4,5]].values, att_fine_df.iloc[ii_used_ii, [6,7,8,9]].values
            att_interpolator.get_quad_interpolant(t_stamps_ii.flatten(), q_in, qdot_in)
            # interpolate
            q_interp = att_interpolator.interpolate(t_interpolation_ii)
            # store
            data_out[ii_n:ii_n+n_datapoint_per_t,0] =t_interpolation_ii
            data_out[ii_n:ii_n+n_datapoint_per_t,1:] = q_interp
            ii_n += n_datapoint_per_t


    if 1:
        # saving    
        q_input_df = pd.DataFrame(data = np.hstack((t_in_all, q_in_all,
qdot_in_all)), columns = ['t_gps','q1','q2','q3', 'q4', 
                            'qdot1','qdot2','qdot3', 'qdot4'
                            ])
        # q_output_df = pd.DataFrame(data = np.hstack((t_interp.reshape(t_interp.shape[0],1), q_interp)), 
        #                         columns = ['t_gps','q1','q2','q3', 'q4', ])
        q_output_df = pd.DataFrame(data = data_out, 
                                columns = ['t_gps','q1','q2','q3', 'q4', ])
        path_output = r'outputs\attitude_tests'

        path_input_df = fr'{path_output}\qinterp_input_df.csv'
        path_output_df = fr'{path_output}\qinterp_output_df.csv'

        q_input_df.to_csv(path_input_df, index = 0 )
        print(f'Saved {path_input_df}')
        q_output_df.to_csv(path_output_df, index = 0 )
        print(f'Saved {path_output_df}')