#%% July 27, 2023 - functions to simulate atttiude
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

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import plotting_tools.basic_plotting as bplt

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot

import basic_tools.in_out as savedat
import pointing_calculations.ae_calculation as ae_calc
import attitude_tools.terminal_rotations as get_rot



def calc_quat_eci2lct(r_h, v_h, default_pointing = 'along_track', calc_qdot = 0, t_vec = None):
    """Function to get a quaternion from ECI to RSW -> LCT frame with default pointing in the along-track
        calculates ROT_ECI2RSW using the position/velocity vector of host
        and ROT_RSW2LCT assuming a default LCT pointing orientation
        convert the rotation matrices to a scalar-first quaternion
        07-03-2024 Added calc_qdot option

    Args:
        r_h (N x 3 array): host ECI post [m,m,m]
        v_h (N x 3 array): hot ECI vel [m/s, m/s, m/s]
        default_pointing (str, optional): LCT orientation, only supports default input. Defaults to 'along_track'.
        calc_qdot (bool, optional): whether quaternion rates should be calculated
        t_vec (N x 1 array): GPS or any other time-scale time vector
    Returns:
        q_all, rot_eci2lct: quaternion and DCM from ECI to LCT for LCT pointing in along-track direction
    """    
    nrows = r_h.shape[0]
    
    # placeholders
    rot_eci2rsw = np.zeros((nrows, 3, 3))
    rot_eci2lct = np.zeros((nrows, 3, 3)) # debug/test purposes
    q_all = np.zeros((nrows, 4))
    
    # get LCT orientation
    if default_pointing == 'along_track':
        axes_rotlct = [1, 2]
        angles_rotlct = [90, 90]
    else:
        print(f'Pointing of {default_pointing} not yet implemented')
    rot_rsw2lct = get_rot.rotmat_rsw2lct(angles_rotlct, axes_rotlct)
    # get combined rotation matrix and convert to qutaernions
    for ii, (r_h, v_h) in enumerate(zip(r_h, v_h)):
        rot_eci2rsw[ii] = get_rot.calc_rotrsweci(r_h, v_h)    
        rot_comb_ii = rot_rsw2lct @ rot_eci2rsw[ii]
        rot_eci2lct[ii] = rot_comb_ii
        q_ii = conv.convert_dcm2quat(rot_comb_ii)
        q_all[ii,:] = q_ii
    
    # compute quaternion rates
    if calc_qdot:
        q_dot_all = np.zeros(q_all.shape)
        # get Ea_all
        dcm_all = np.zeros((q_all.shape[0], 3,3))
        ea_all =  np.zeros((q_all.shape[0], 3))
        omega_all = np.zeros((q_all.shape[0], 3))
        for ii, q in enumerate(q_all):
            dcm_all[ii,:] = conv.convert_quat2dcm(q)
            ea_all[ii,:] = conv.convert_dcm2ea(dcm_all[ii,:])        
        # get EA_dot_all
        ea1_dot_all = np.gradient(ea_all[:,0], t_vec)
        ea2_dot_all = np.gradient(ea_all[:,1], t_vec)
        ea3_dot_all = np.gradient(ea_all[:,2], t_vec)
        ea_dot_all = np.hstack((ea1_dot_all.reshape(nrows,1),
                                ea2_dot_all.reshape(nrows,1),
                                ea3_dot_all.reshape(nrows,1)))
        for ii, ea in enumerate(ea_all):
            omega_ii = conv.calc_omega(ea, ea_dot_all[ii,:])
            omega_all[ii,:] = omega_ii
            q_ii, q_dot_ii = conv.calc_qdot(None, omega_ii, 1, dcm_all[ii,:])
            q_dot_all[ii,:] = q_dot_ii.flatten()
        
        # get q_dot_all
    else:
        q_dot_all = None
    
    return q_all, rot_eci2lct, q_dot_all

def calc_quat_eci2bf(r_host, 
                     v_host, 
                     att_profile = 'earth_point', 
                     t_gps = None, 
                     euler_rates = None, 
                     add_ideal_jerk = 0, 
                     add_realistic_jerk = None, 
                     roll_velocity = 0.1, 
                     calc_qdot = 0, 
                     rotation_axis = 0):
    """Function to get a quaternion from ECI to BF frame with some pointing mode
        calculates ROT_ECI2RSW using the position/velocity vector of host
        and additional rotations to match the required profile
        convert the rotation matrices to a scalar-first quaternion

    Args:
        r_h (N x 3 array): host ECI post [m,m,m]
        v_h (N x 3 array): hot ECI vel [m/s, m/s, m/s]
        att_profile (str, optional): BF orientation. Defaults to 'earth-point'.
        add_jerk (bool): whether rotational velocity sign is swapped at mid-point of data.
        roll_velocity (float): deg/s of angualr velocity given for attitude profiles
        calc_qdot  (bool): whether to compute quaternion rates from ECI to BF
    Returns:
        q_all, rot_eci2bf: quaternion and DCM from ECI to BF for BF pointing in along-track direction
    """    
    nrows = r_host.shape[0]
    
    # placeholders
    # ECI to RSW based on pos/vel (earth-pointing)
    rot_eci2rsw = np.zeros((nrows, 3, 3))
    # RSW to BF rotation component
    rot_rsw2bf = np.zeros((nrows, 3, 3)) 
    # Complete rotation
    rot_eci2bf = np.zeros((nrows, 3, 3)) 

    q_all = np.zeros((nrows, 4))
    
    # Earth-pointing RSW rotation
    for ii, (r_h, v_h) in enumerate(zip(r_host, v_host)):
        rot_eci2rsw[ii] = get_rot.calc_rotrsweci(r_h, v_h, option = 'swr')    
        
    if type(euler_rates) ==type(None):
        euler_angles_integrated = np.zeros((nrows, 3))
        if att_profile == 'earth_roll':        
            euler_rates = np.zeros((1,3))
            euler_rates[0, rotation_axis] = roll_velocity
        elif att_profile == 'sun_roll':        
            euler_rates = np.array([0,roll_velocity, 0]) # deg/s - around sun vec
        elif att_profile == 'sun_roll_perp':   # ROLL PERPENDICULAR TO SUN DIRECTION
            # euler_rates = np.array([0,0, roll_velocity]) # deg/s - around sun vec            
            euler_rates = np.array([roll_velocity,0,0]) # deg/s - around sun vec            
        elif att_profile == 'moon_roll_perp':
            # euler_rates = np.array([roll_velocity,0,0]) # deg/s - around sun vec            
            euler_rates = np.array([0,0,roll_velocity]) # deg/s - around sun vec            
        else:
                # euler_rates = np.array([roll_velocity,0,0]) # deg/s - around sun vec            
            euler_rates = np.array([roll_velocity,0,0])
        euler_angles_integrated[:,:] = euler_rates
    

    
    dt_data = t_gps[1] - t_gps[0]
    # get complete BF orientation
    if att_profile == 'earth_point':        
        rot_rsw2bf[:] = np.eye(3, dtype = float)
        for ii, (r_h, v_h) in enumerate(zip(r_h, v_h)):
            rot_eci2bf[ii] = rot_eci2rsw[ii]
    elif att_profile == 'earth_roll':
        euler_angles_integrated = np.zeros((nrows, 3))
        euler_angles_prev = np.array([0,0,0])
        
        for ii, t_ii in enumerate(t_gps):            
            euler_angles_integrated[ii,:] = euler_angles_prev + euler_rates * dt_data
            euler_angles_prev = euler_angles_integrated[ii,:]
            rot_rsw2bf[ii] = conv.convert_ea2dcm(euler_angles_prev)
    elif att_profile == 'sun_point':
        rot_rsw2bf[:] = np.eye(3, dtype = float)
        for ii, (r_h, v_h) in enumerate(zip(r_host, v_host)):
            r_sun = where_sun.compute_sun_vector_eci_better(t_gps[ii])
            y_axis = r_sun / np.linalg.norm(r_sun)
            x_axis = np.cross(y_axis, np.array([0,1,0])) # perpendicular to sun direction and Y in ECI
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            rot_sunpoint = np.vstack((x_axis, y_axis, z_axis)) #
            rot_eci2rsw[ii] = rot_sunpoint
    elif att_profile == 'moon_roll_perp':
        rot_rsw2bf[:] = np.eye(3, dtype = float)
        for ii, (r_h, v_h) in enumerate(zip(r_host, v_host)):
            r_moon = where_sun.compute_moon_vector_eci(t_gps[ii]) - r_h
            y_axis = r_moon / np.linalg.norm(r_moon)
            x_axis = np.cross(y_axis, r_h) # perpendicular to sun direction and Y in ECI
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            rot_sunpoint = np.vstack((x_axis, y_axis, z_axis)) #
            rot_eci2rsw[ii] = rot_sunpoint
    elif att_profile == 'moon_roll_perp_tilt':
            r_moon = where_sun.compute_moon_vector_eci(t_gps[ii]) - r_h
            y_axis = r_moon / np.linalg.norm(r_moon)
            x_axis = np.cross(y_axis, r_h) # perpendicular to sun direction and Y in ECI
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            
            rot_sunpoint = np.vstack((x_axis, y_axis, z_axis)) #
            rot_eci2rsw_tilt = conv.convert_eigenaxis2dcm(x_axis, np.deg2rad(0))
            rot_eci2rsw[ii] = rot_eci2rsw_tilt @ rot_sunpoint
    elif att_profile == 'sun_roll' or att_profile == 'sun_roll_perp':
        for ii, (r_h, v_h) in enumerate(zip(r_host, v_host)):
            r_sun = where_sun.compute_sun_vector_eci_better(t_gps[ii])
            y_axis = r_sun / np.linalg.norm(r_sun)
            x_axis = np.cross(y_axis, np.array([0,1,0])) # perpendicular to sun direction and Y in ECI
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            rot_sunpoint = np.vstack((x_axis, y_axis, z_axis)) #
            rot_eci2rsw[ii] = rot_sunpoint
    else:
        print(f'Pointing of {att_profile} not yet implemented')
        pass
        return 
    euler_angles_integrated = np.zeros((nrows, 3))
    euler_angles_prev = np.array([0,0,0])
    dt_data = t_gps[1] - t_gps[0]
    if type(euler_rates) != type(None):    
        for ii, t_ii in enumerate(t_gps):           
            if add_ideal_jerk:
                if ii == (len(t_gps)/2):
                    print(f'Attitude Simualtion - SWAPPING Euler rate sign')
                    euler_rates = -euler_rates
            else:
                if type(add_realistic_jerk) != type(None):
                    if ii == (len(t_gps)/2):
                        print(f'Attitude Simualtion - SWAPPING attitude jerk rate to neg : {add_realistic_jerk:.3f} deg/s')
                        add_realistic_jerk = - add_realistic_jerk #
                    euler_rates = euler_rates + np.array([add_realistic_jerk, 0, 0])*dt_data
                        

            euler_angles_integrated[ii,:] = euler_angles_prev + euler_rates * dt_data
            euler_angles_prev = euler_angles_integrated[ii,:]
            rot_rsw2bf[ii] = conv.convert_ea2dcm(euler_angles_prev)
        for ii, rot_eci2rsw_ii in enumerate(rot_eci2rsw):
            rot_comb_ii = rot_rsw2bf[ii] @ rot_eci2rsw_ii
            rot_eci2bf[ii] = rot_comb_ii
            q_all[ii,:] = conv.convert_dcm2quat(rot_comb_ii)
    else: # if no additional ea rates are given, use just default pointing
        for ii, rot_eci2rsw_ii in enumerate(rot_eci2rsw):
            q_all[ii,:] = conv.convert_dcm2quat(rot_eci2rsw_ii)
    ## Computing quaternion rates
    if calc_qdot:
        q_dot_all = np.zeros(q_all.shape)
        # get Ea_all
        dcm_all = np.zeros((q_all.shape[0], 3,3))
        ea_all =  np.zeros((q_all.shape[0], 3))
        omega_all = np.zeros((q_all.shape[0], 3))
        for ii, q in enumerate(q_all):
            dcm_all[ii,:] = conv.convert_quat2dcm(q)
            ea_all[ii,:] = conv.convert_dcm2ea(dcm_all[ii,:])        
        # get EA_dot_all
        ea1_dot_all = np.gradient(ea_all[:,0], t_gps)
        ea2_dot_all = np.gradient(ea_all[:,1], t_gps)
        ea3_dot_all = np.gradient(ea_all[:,2], t_gps)
        ea_dot_all = np.hstack((ea1_dot_all.reshape(nrows,1),
                                ea2_dot_all.reshape(nrows,1),
                                ea3_dot_all.reshape(nrows,1)))
        for ii, ea in enumerate(ea_all):
            omega_ii = conv.calc_omega(ea, ea_dot_all[ii,:])
            omega_all[ii,:] = omega_ii
            
            q_ii, q_dot_ii = conv.calc_qdot(q = None, rpy = None, w = omega_ii, deg =  1,dcm = dcm_all[ii,:])
            q_dot_all[ii,:] = q_dot_ii.flatten()
        
        # get q_dot_all
    else:
        q_dot_all = None
    # get combined rotation matrix and convert to qutaernions
    return q_all, q_dot_all, rot_eci2bf
if __name__ == '__main__':
    # example attitude scenario
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
    

    fname_states_fine = r'outputs\attitude_tests\states_fine.csv'
    ## MVP imports
    import basic_tools.vector_operations as vec_calc
    import astronomy_tools.constants as const
    import astronomy_tools.astro_targets as where_sun
    import plotting_tools.basic_plotting as bplt
    import basic_tools.time_conversion as t_conv

    import tudat_tools.data_processing.data_processing_utilities as dputil
    importlib.reload(conv)
    use_full_data = 0
    save_attitude = 1
    save_interpolated_attitude = 1
    if use_full_data:
        savename = 'QQdot_full'
        data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path, nrows = 1000)

        host_chosen = 'leo_host_polar'

        t_j2000 = data_raw[:,0]
        t_gps = t_j2000+t_conv.dt_j2000tt2gps()
        r_host = data_raw[:,simulation_parameters['r_index'][host_chosen]]
        v_host = data_raw[:,[ii+3 for ii in simulation_parameters['r_index'][host_chosen]]]
    else:
        savename = 'QQdot_4hz'
        states_host = pd.read_csv(fname_states_fine)
        t_gps = states_host.iloc[:,0].values
        t_j2000 = t_gps + t_conv.dt_gps2j2000tt()
        r_host = states_host.iloc[:,[1,2,3]].values
        v_host = states_host.iloc[:,[4,5,6]].values
        # r_host =
    nrows = r_host.shape[0]
    # earth-point + 0.84 deg/s rotation
    attitude_rate_used = 0.84 # deg/s
    euler_rates = np.array([0,0,0])
    t_vec = t_j2000 - t_j2000[0]
    q_all, q_dot_all, rot_eci2bf = calc_quat_eci2bf(r_host, v_host, t_gps = t_j2000, att_profile = 'earth_point', euler_rates = euler_rates, roll_velocity = attitude_rate_used,
                                                    calc_qdot = 1)
    if save_attitude:
        data_attitude = np.hstack((t_gps.reshape(nrows,1), t_vec.reshape(nrows,1), 
                                   q_all, q_dot_all))
        data_attitude_df = pd.DataFrame(data = data_attitude, columns = [
            't_gps', 't_s', 'q1', 'q2', 'q3','q4',
            'qdot1', 'qdot2', 'qdot3','qdot4'
        ])
        # save
        path_output = r'outputs\attitude_tests'
        path_fullatt = f'{path_output}\{savename}.csv'
        
        try:
            data_attitude_df.to_csv(path_fullatt, index = 0)
            print(f'saved {path_fullatt}')
        except:
            print(f'failed to save {path_fullatt}')
        

        
