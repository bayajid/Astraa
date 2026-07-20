import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
import sys
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import json
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import pointing_calculations.ae_calculation as ae_calc
import basic_tools.vector_operations as vec_calc
import attitude_tools.attitude_resolution as att_res
def simulate_moon_calib(ii_scans,
                      r_host, 
                      r_moon, 
                      ea_eci2bf_command_all,
                      quat_mounting_offset_t,
                      use_quest = 1,
                      quat_mounting_offset_c = None,
                      manual_error_dict = None,
                      add_noise = 1,
                      add_centroid_error = 1, 
                      add_r_host_error = 1, 
                      add_r_moon_error = 1, 
                      add_att_host_error = 1,
                      att_noise_factor = 1,
                      centroid_err_factor = 1,
                      print_cond = 1,
                      check_non_colin = 0,
                      randomize_dr_dtheta = 1,
                      randomize_magnitude = 0,
                        seed_used = 1,
                      centroid_dirction_randomizer = 0,
                      use_clean_path = 0,
                      print_full = 1):
    """Function to simulate a moon calibration procedure for a given host orbit, moon position,
    host attitude and mounting offset. 
    Provides conditionals to turn errors for these components on/off
    Originally uses TRIAD to resolve the mounting offset 
    Sep 29 updates - UTILIZIGN FOR PHASE B Simulations:
     - randomizing position error directions
     -  
    Nov 2 update - cleaner method for MO quaternion update
        - use_clean_path set to 1 - get DCM -> convert to quat -> conjugate quaternions
    Args:
        ii_scan (list): indices of scans to use
        r_host (_type_): _description_
        r_moon (_type_): _description_
        ea_eci2bf_command_all (array): array of euler angles [deg]
        q_mounting_offset_c (array): known mounting offset quaternion (defaults None)
        quat_mounting_offset (array): true mounting offset quaternion
        manual_error_dict (dict): dictionary with constant error terms. 
        add_noise (bool, optional): whether errors are enabled. Defaults to 1.
        add_centroid_error (int, optional): Bool to enable centroid detection error. Defaults to 1.
        add_r_host_error (int, optional): Bool to enable host pos error. Defaults to 1.
        add_r_moon_error (int, optional): Bool to enable moon pos error. Defaults to 1.
        add_att_host_error (int, optional): Bool to enable attitude error . Defaults to 1.
        att_noise_factor (int): factor to reduce the original attitude noise of 0.1mrad. Defaults to 1.
        check_non_colin : whether to compute angle between seen Moon Vectors
        randomize_dr_dtheta (bool): randomize direction of position and attitude errors
        seed_used (int): seed used to randomize pos/att direction errors        
        print_cond (bool, optional): whether results are printed. Defaults to 1.
        print_full (bool, optional): whether all input data is printed. Defaults to 0.

    Returns:
        ae_moon_commanded_all, ae_moon_true_all, [RAD] quat_resolved, pe_mo [URAD], angle_between_los # [rad]
    """    
    nrows = r_host.shape[0]

    ## Generate noise/errors
    err_r_host = np.zeros((nrows, 3))
    err_r_target = np.zeros((nrows, 3))
    err_att_host = np.zeros((nrows, 3))
    err_eigenaxis_centroid = np.zeros((nrows, 4)) # Axis-angle representation for centroid detection error
    if add_noise:
        if type(manual_error_dict) == type(None):
            # generation own errors with hardcoded 3-sigma values, relatively outdated
            np.random.seed(seed_used)
            for ii in range(nrows):
                err_r_host[ii,:] = np.array([6, 7, 8]) * np.random.randn()
                err_r_target[ii,:] = np.array([-6, 7, 2*8]) * np.random.randn()
                err_att_host[ii,:] = np.rad2deg([0.08e-3, 0.07e-3, 0.1e-3]) * np.random.randn() / att_noise_factor
            # Add Moon centroid error, maybe assume 2% non-illumination 
            moon_illum_frac = 0.98
            # 0.09 mrad
            err_random_rss = np.deg2rad((1-moon_illum_frac) * 0.5 * 1/2) / centroid_err_factor
            err_eigenaxis_centroid[:,-1] = err_random_rss # [rad]
        elif randomize_dr_dtheta:
            # use defined error magnitudes from the error dictionary, but 
            # np.random.seed(seed_used)
            # axis components used to generate random rotation axes
            
            nr_data_rows = err_r_host.shape[0]
            # axes
            nr_axis_comps = nr_data_rows*3
            nr_angles_to_generate = nr_data_rows * 3

            # gaussian for axis directions
            x_comps = np.random.normal(size = nr_axis_comps)
            y_comps = np.random.normal(size = nr_axis_comps)
            z_comps = np.random.normal(size = nr_axis_comps)
            # uniform for rotation angles
            rotation_angles = 2*np.pi * np.random.uniform(size = nr_angles_to_generate)
            euler_eig_all = np.zeros((3, nr_data_rows, 4)) # comps with errors - nr rows per comp - axis - angle
            
            euler_eig_all[:,:,0] = x_comps.reshape((3,nr_data_rows))
            euler_eig_all[:,:,1] = y_comps.reshape((3,nr_data_rows))
            euler_eig_all[:,:,2] = z_comps.reshape((3,nr_data_rows))
            euler_eig_all[:,:,3] = rotation_angles.reshape((3,nr_data_rows))
            
            # randomize directions
            for ii in range(nr_data_rows):
                if not randomize_magnitude:
                    mag_scales = [1,1,1,1]
                else:
                    mag_scales = np.random.randn(4)/3
                ii_err = 0
                rot_rhost = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], euler_eig_all[ii_err, ii, 3]) # rad
                
                err_r_host[ii] = mag_scales[ii_err] * rot_rhost @ manual_error_dict['err_r_host'][ii]
                ii_err = 1
                rot_rmoon = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], euler_eig_all[ii_err, ii, 3])
                err_r_target[ii] = mag_scales[ii_err] * rot_rmoon @ manual_error_dict['err_r_target'][ii]
                ii_err = 2

                att_err_ii = np.linalg.norm(manual_error_dict['err_att_host'][ii]) # [deg]
                att_err_dcm = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], np.deg2rad(att_err_ii)) # [rad]
                
                err_att_host[ii] = mag_scales[ii_err] *conv.convert_dcm2ea(att_err_dcm) # deg
            
            try:
                err_random_rss = mag_scales[3] * manual_error_dict['rss_random_errors'] + manual_error_dict['sum_mean_errors']
            except:
                print(f'Manual mean-errors missing')
                err_random_rss = mag_scales[3] * manual_error_dict['rss_random_errors']
            err_eigenaxis_centroid[:,-1] = err_random_rss
        else:
            # extract errors from dictionary
            err_r_host = manual_error_dict['err_r_host']
            err_r_target = manual_error_dict['err_r_target']
            err_att_host = manual_error_dict['err_att_host']
            err_random_rss = manual_error_dict['sum_random_errors']
            err_eigenaxis_centroid[:,-1] = err_random_rss
    if not add_noise:
        add_centroid_error = 0
        add_r_host_error = 0
        add_r_moon_error = 0
        add_att_host_error = 0
    # placeholders
    ae_moon_commanded_all = np.zeros((nrows, 2))
    ae_moon_true_all = np.zeros((nrows, 2))
    for ii, ii_scan in enumerate(ii_scans):
    
        r_host_ii = r_host[ii_scan,:]
        v_host_ii = [0,0,0] # placeholder, not needed here

        r_moon_ii = r_moon[ii_scan,:]
        # _c - COMMANDED values, contains knowledge errors, mounting offset assumed 0
        # _t - TRUE values, does not contain knowledge errors, true mounting ofset used
        r_host_ii_c = r_host_ii    

        if not add_r_host_error:
            err_r_host[ii_scan,:] = [0,0,0]
        if not add_r_moon_error:
            err_r_target[ii_scan,:] = [0,0,0]
        if not add_att_host_error:
            err_att_host[ii_scan,:] = [0,0,0]
        if not add_centroid_error:
            centroid_err_used = None
        else:
            centroid_err_used = err_random_rss

        r_host_ii_c = r_host_ii + err_r_host[ii_scan,:]
        r_host_ii_t = r_host_ii
        
        r_moon_ii_c = r_moon_ii + err_r_target[ii_scan,:]
        r_moon_ii_t = r_moon_ii
        
        ea_eci2bf_c = ea_eci2bf_command_all[ii_scan,:3] + err_att_host[ii_scan,:]
        ea_eci2bf_t = ea_eci2bf_command_all[ii_scan,:3]

        q_eci2bf_c = conv.convert_ea2quat(ea_eci2bf_c)
        q_eci2bf_t = conv.convert_ea2quat(ea_eci2bf_t)
        if type(quat_mounting_offset_c) == type(None):
            quat_mounting_offset_c = np.array([1,0,0,0]) # Unity quaternion initially used

        ae_moon_expected = ae_calc.calc_ae_full(r_host_ii_c, r_moon_ii_c, q_eci2bf_c, quat_mounting_offset_c)[0]
        ae_moon_observed = ae_calc.calc_ae_full(r_host_ii_t, r_moon_ii_t, q_eci2bf_t, quat_mounting_offset_t, 
                                                centroid_error=centroid_err_used, centroid_dirction_randomizer = centroid_dirction_randomizer)[0]
        
        ae_moon_commanded_all[ii,:] = ae_moon_expected[:2]
        ae_moon_true_all[ii,:] = ae_moon_observed[:2]
    if use_quest:
        att_res_fct = att_res.compute_q_mountingoffset_quest
    else:
        att_res_fct = att_res.get_mo_quat_fromscan
    
    quat_resolved, angle_between_los = att_res_fct(ae_moon_commanded_all, ae_moon_true_all, check_non_colin = check_non_colin)
    dcm_resolved = conv.convert_quat2dcm(quat_resolved)
    ea_resolved = conv.convert_dcm2ea(dcm_resolved)
    
    # if ooc_phase.upper() == 'B': # NOV 2: no longer used. Even in Phase A it should have an initially
    # known MO quaternion
    if use_clean_path:
        # In phase B, delta_MO is solved, so add it to the initially known MO
        quat_resolved = rot.multiply_quat_hamiltonian(quat_resolved, quat_mounting_offset_c).flatten()
        ea_resolved = conv.convert_dcm2ea(conv.convert_quat2dcm(quat_resolved))
    else:
        ea_resolved = ea_resolved + conv.convert_dcm2ea(conv.convert_quat2dcm(quat_mounting_offset_c))
        quat_resolved = conv.convert_ea2quat(ea_resolved)

    mounting_offset_rpy = conv.convert_dcm2ea(conv.convert_quat2dcm(quat_mounting_offset_t))
    pe_mo = vec_calc.get_pe_for_rot(quat_resolved, quat_mounting_offset_t) 
    if print_cond:
        if print_full and add_noise:
            print(f'''-------------------------\nErrors used
                sum_random_errors : {bool(add_centroid_error)}. Error {err_random_rss*1e3:.4f} mrad
                host pos error : {bool(add_r_host_error)}. Error = {err_r_host[ii_scan,0]:.3f}, {err_r_host[ii_scan,1]:.3f}, {err_r_host[ii_scan,2]:.3f} m
                moon pos error : {bool(add_r_moon_error)}. Error = {err_r_target[ii_scan,0]:.3f}, {err_r_target[ii_scan,1]:.3f}, {err_r_target[ii_scan,2]:.3f} m 
                host attitude error: {bool(add_att_host_error)}. Error = {np.deg2rad(err_att_host[ii_scan,0])*1e3:.3f}, {np.deg2rad(err_att_host[ii_scan,1])*1e3:.3f}, {np.deg2rad(err_att_host[ii_scan,2])*1e3:.3f} mrad -> RSS {np.deg2rad(err_att_host[ii_scan,0])*1e3:.3f}, {np.linalg.norm(np.deg2rad(err_att_host[ii_scan,:])*1e3)}
''')
        print(f'PE MO : \n{pe_mo[0][0]:.1f}, {pe_mo[1][0]:.1f}, {pe_mo[2][0]:.1f} urad')
        if print_full:
            print(f'''
            Mounting offset TRUE : {mounting_offset_rpy[0]:.3f}, {mounting_offset_rpy[1]:.3f}, {mounting_offset_rpy[2]:.3f} deg
            Mounting offset RESOLVED : {ea_resolved[0]:.3f}, {ea_resolved[1]:.3f}, {ea_resolved[2]:.3f} deg
                ''')
    return ae_moon_commanded_all, ae_moon_true_all, quat_resolved, pe_mo, angle_between_los
def simulate_ooc(ii_scans,
                    r_host, 
                    r_target, 
                    ea_eci2bf_command_all,
                    quat_mounting_offset_t,
                    quat_mounting_offset_c = None,
                    function_option = 1,
                    manual_error_dict = None,
                    check_non_colin = 0,
                    randomize_error_direction = 1,
                    randomize_magnitude = 0,
                    centroid_dirction_randomizer = 0,
                    use_clean_path = 0,
                    add_trolley_tilt = 1,
                    trolley_angle = 0,
                    seed_used = 1,
                    just_calc_ae = 0,
                    ii_loop = None):
    """March 2024 Remaking simulate_moon_calib function 
    to simplify usability and remove redundant functionality
    
    Nov 2 update - cleaner method for MO quaternion update
        - use_clean_path set to 1 - get DCM -> convert to quat -> conjugate quaternions
    Args:
        ii_scan (list): indices of scans to use
        r_host (_type_): _description_
        r_moon (_type_): _description_
        ea_eci2bf_command_all (array): array of euler angles [deg]
        q_mounting_offset_c (array): known mounting offset quaternion (defaults None)
        quat_mounting_offset (array): true mounting offset quaternion
        manual_error_dict (dict): dictionary with constant error terms. 
        add_noise (bool, optional): whether errors are enabled. Defaults to 1.
        add_centroid_error (int, optional): Bool to enable centroid detection error. Defaults to 1.
        add_r_host_error (int, optional): Bool to enable host pos error. Defaults to 1.
        add_r_moon_error (int, optional): Bool to enable moon pos error. Defaults to 1.
        add_att_host_error (int, optional): Bool to enable attitude error . Defaults to 1.
        att_noise_factor (int): factor to reduce the original attitude noise of 0.1mrad. Defaults to 1.
        check_non_colin : whether to compute angle between seen Moon Vectors
        randomize_dr_dtheta (bool): randomize direction of error inputs
        seed_used (int): seed used to randomize pos/att direction errors        
        print_cond (bool, optional): whether results are printed. Defaults to 1.
        print_full (bool, optional): whether all input data is printed. Defaults to 0.

    Returns:
        ae_moon_commanded_all, ae_moon_true_all, [RAD] quat_resolved, pe_mo [URAD], angle_between_los # [rad]
    """    
    nrows = r_host.shape[0]

    ## Generate errors for each data-point according to manual_error_dict
    err_r_host = np.zeros((nrows, 3))
    err_r_target = np.zeros((nrows, 3))
    err_att_host = np.zeros((nrows, 4))
    err_pe_direct = np.zeros((nrows, 4)) # Axis-angle representation for centroid detection error
    np.random.seed(seed_used)

    if randomize_error_direction:
        # axes
        nr_axis_comps = nrows*3
        nr_angles_to_generate = nrows * 3

        # gaussian for axis directions
        x_comps = np.random.normal(size = nr_axis_comps)
        y_comps = np.random.normal(size = nr_axis_comps)
        z_comps = np.random.normal(size = nr_axis_comps)
        
        # uniform for rotation angles
        rotation_angles = 2*np.pi * np.random.uniform(size = nr_angles_to_generate)
        euler_eig_all = np.zeros((3, nrows, 4)) # comps with errors - nr rows per comp - axis - angle
        
        euler_eig_all[:,:,0] = x_comps.reshape((3,nrows))
        euler_eig_all[:,:,1] = y_comps.reshape((3,nrows))
        euler_eig_all[:,:,2] = z_comps.reshape((3,nrows))
        euler_eig_all[:,:,3] = rotation_angles.reshape((3,nrows))
    
        los_eci = r_target[:,:3] - r_host[:,:3]
        # get axis perpendicular to LOS
        # get rotation matrix for the rotation axis 
        direct_pe_axis = np.cross([0,1,0], los_eci)
        
        # randomize directions
        for ii in range(nrows):
            
            # for pos/att
            if not randomize_magnitude:
                mag_scales = [1,1,1,1]
            else:
                mag_scales = np.random.randn(4)/3
            
            try:
                err_random_rss = mag_scales[3] * manual_error_dict['rss_random_errors'] + manual_error_dict['sum_mean_errors']
            except:
                print(f'Manual mean-errors missing')
                err_random_rss = mag_scales[3] * manual_error_dict['rss_random_errors']
            err_pe_direct[:,-1] = err_random_rss
            
            # For direct PE components
            if centroid_dirction_randomizer:
                angle_rotation = np.random.random()*2*np.pi
                rot_axis_rotation = conv.convert_eigenaxis2dcm(los_eci[ii,:], angle_rotation)            
            else:
                rot_axis_rotation = np.eye(3, dtype = float)
            dcm_direct_pe = conv.convert_eigenaxis2dcm(rot_axis_rotation @ direct_pe_axis[ii,:].reshape([3,1]), err_random_rss) # input [rad]

            ii_err = 0
            rot_rhost = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], euler_eig_all[ii_err, ii, 3]) # rad
            
            err_r_host[ii] = mag_scales[ii_err] * rot_rhost @ manual_error_dict['err_r_host']
            ii_err = 1
            
            rot_rmoon = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], euler_eig_all[ii_err, ii, 3])
            err_r_target[ii] = mag_scales[ii_err] * rot_rmoon @ manual_error_dict['err_r_target']
            ii_err = 2

            att_err_ii = np.linalg.norm(manual_error_dict['err_att_host']) # [rad]
            att_err_dcm = conv.convert_eigenaxis2dcm(euler_eig_all[ii_err, ii, :3], att_err_ii) # [rad]
            
            err_att_host[ii] = mag_scales[ii_err] *conv.convert_dcm2quat(att_err_dcm) # deg
    
            err_pe_direct[ii,:] = conv.convert_dcm2quat(dcm_direct_pe)

    # placeholders
    ae_moon_commanded_all = np.zeros((nrows, 2))
    ae_moon_tracked_all = np.zeros((nrows, 2))
    for ii, ii_scan in enumerate(ii_scans):
    
        r_host_ii = r_host[ii_scan,:]
        v_host_ii = [0,0,0] # placeholder, not needed here

        r_moon_ii = r_target[ii_scan,:]
        # _c - COMMANDED values, contains knowledge errors, mounting offset assumed 0
        # _t - TRUE values, does not contain knowledge errors, true mounting ofset used
        r_host_ii_c = r_host_ii    

        centroid_err_used = err_random_rss

        r_host_ii_c = r_host_ii + err_r_host[ii_scan,:]
        r_host_ii_t = r_host_ii
        
        r_moon_ii_c = r_moon_ii + err_r_target[ii_scan,:]
        r_moon_ii_t = r_moon_ii
    
        ea_eci2bf_t = ea_eci2bf_command_all[ii_scan,:3]

        q_eci2bf_t = conv.convert_ea2quat(ea_eci2bf_t)
        q_eci2bf_c = rot.multiply_quat_ham_matrix(err_att_host[ii_scan], q_eci2bf_t)
        if add_trolley_tilt:
            trolley_angle_rad = np.deg2rad(trolley_angle) # [rad]
            q_horizontal2tilted = np.array([np.cos(trolley_angle_rad/2), -np.sin(trolley_angle_rad/2), 0, 0])
            q_horizontal2tilted = q_horizontal2tilted / np.linalg.norm(q_horizontal2tilted)
            q_eci2bf_t = rot.multiply_quat_ham_matrix(q_horizontal2tilted, q_eci2bf_t)
                            
        # Apply PE rotation directly to body-frame attitude
        q_eci2bf_c = rot.multiply_quat_ham_matrix(err_pe_direct[ii,:], q_eci2bf_c)
        # TODO
        if type(quat_mounting_offset_c) == type(None):
            quat_mounting_offset_c = np.array([1,0,0,0]) # Unity quaternion initially used

        ae_moon_expected = ae_calc.calc_ae_full(r_host_ii_c, r_moon_ii_c, q_eci2bf_c, quat_mounting_offset_c)[0]
        ae_moon_observed = ae_calc.calc_ae_full(r_host_ii_t, r_moon_ii_t, q_eci2bf_t, quat_mounting_offset_t, 
                                                centroid_error=centroid_err_used, centroid_dirction_randomizer = centroid_dirction_randomizer)[0]
        
        ae_moon_commanded_all[ii,:] = ae_moon_expected[:2]
        ae_moon_tracked_all[ii,:] = ae_moon_observed[:2]
    if just_calc_ae: # dont resolve MO
        return ae_moon_commanded_all, ae_moon_tracked_all
    
    if function_option == 0: # TRIAD
        att_res_fct = att_res.get_mo_quat_fromscan
        fct_used = 'triad'
    elif function_option == 1: # QUEST DEBUG
        att_res_fct = att_res.compute_q_mountingoffset_quest_debug
        fct_used = 'quest'
    elif function_option == 2: # SVD
        att_res_fct = att_res.compute_q_mountingoffset_quest_debug
        fct_used = 'svd'
    
    quat_resolved, angle_between_los = att_res_fct(ae_moon_commanded_all, ae_moon_tracked_all, check_non_colin = check_non_colin, fct_option = function_option)
    # quat_resolved, angle_between_los = att_res.get_mo_quat_fromscan(ae_moon_commanded_all, ae_moon_true_all, check_non_colin = check_non_colin)
    dcm_resolved = conv.convert_quat2dcm(quat_resolved)
    ea_resolved = conv.convert_dcm2ea(dcm_resolved)
    
    # if ooc_phase.upper() == 'B': # NOV 2: no longer used. Even in Phase A it should have an initially
    # known MO quaternion
    
    ## TODO check if really deprecated, April 10, 2024
    # if use_clean_path:
    #     # In phase B, delta_MO is solved, so add it to the initially known MO
    #     ea_resolved = conv.convert_dcm2ea(conv.convert_quat2dcm(quat_resolved))
    # else:
    #     ea_resolved = ea_resolved + conv.convert_dcm2ea(conv.convert_quat2dcm(quat_mounting_offset_c))
    #     quat_resolved = conv.convert_ea2quat(ea_resolved)
    quat_resolved = rot.multiply_quat_hamiltonian(quat_resolved, quat_mounting_offset_c).flatten()
    # quat_total = 
    mounting_offset_rpy = conv.convert_dcm2ea(conv.convert_quat2dcm(quat_mounting_offset_t))
    if add_trolley_tilt:
        # include tilt in true MOQ
        quat_compared = [quat_resolved, rot.multiply_quat_ham_matrix(q_horizontal2tilted, quat_mounting_offset_t)]
    else:
        quat_compared = [quat_resolved, quat_mounting_offset_t]
    pe_mo = vec_calc.get_pe_for_rot(quat_compared[0], quat_compared[1]) 
    
    return ae_moon_commanded_all, ae_moon_tracked_all, quat_resolved, pe_mo, angle_between_los, fct_used
def check_moon_scan_possibilities(t_gps,
                                  r_host, 
                      r_moon, 
                      ea_eci2bf_command_all,
                      quat_mounting_offset,
                      el_constraint = 20,
                      print_cond = 1,
                      print_full = 1):
    """Function to convert a time-series of pos/attitude inputs 
    to Az/El to the Moon and output time-windows where scans are possible
    # TODO Finish.

    Args:
        r_host (array): host positions
        r_moon (array): moon pos
        ea_eci2bf_command_all (_type_): Euler angle body-frame attitude [deg]
        quat_mounting_offset (_type_): mounting offste qutaernion
        ae_contraint (list, optional): Min Az/El values that constrain the Moon-view . Defaults to [15,0].
        print_cond (int, optional): _description_. Defaults to 1.
        print_full (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """    
    nrows = t_gps.shape[0]
    scan_dat = np.zeros((nrows, 4)) # t, vis, az, el
    
    for ii, t_gps_ii in enumerate(t_gps.flatten()):
        q_eci2bf_ii = conv.convert_ea2quat(ea_eci2bf_command_all[ii,:3])
        ae_moon = ae_calc.calc_ae_full(r_host[ii,:], 
                                       r_moon[ii,:], 
                                       q_eci2bf_ii, 
                                       quat_mounting_offset)[0][:2]
        ae_moon_deg = np.rad2deg(ae_moon)
        if ae_moon_deg[1] >= el_constraint:
            visible_cond = 1
        else:
            visible_cond = 0
        
        scan_dat[ii, :] = [t_gps_ii, visible_cond, ae_moon_deg[0], ae_moon_deg[1]]

    # 
    scan_df = scan_dat
    return scan_dat