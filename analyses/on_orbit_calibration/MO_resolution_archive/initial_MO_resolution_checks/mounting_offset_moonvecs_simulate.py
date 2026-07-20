#%% July 2023
## The purpose of this script is to emulate a spacecraft position/attitude
# scenario, where the terminal has an unknown mounting offset and isu sing
# moon for callibration purposes. The question - can the mounting offset (3-axis rotation)
# be determined by only using a single astronomical body (moon) by varying
# the time, SC position and SC orientation.
# in more scientific terms- are multiple moon-vector non-collinear 
# in the moving and rotating (non-inertial) satellite body-frame
## IMPORTS
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
import tudat_tools.data_processing.data_processing_utilities as dputil
import tudat_tools.tudat_converter as tudatconv
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import basic_tools.time_conversion as t_conv
import basic_tools.vector_operations as vec
import plotting_tools.basic_plotting as bplt
import astronomy_tools.astro_targets as where_sun

import pointing_calculations.ae_calculation as ae_calc
# Uknown mounting offset definition
mounting_offset_rpy = [5, 10, 14] # MOUNTING OFFSET random 3-axis rotation
# mounting_offset_rpy = [0, 5, 0] # MOUNTING OFFSET random 1-axis rotation
# mounting_offset_rpy = [0, 0, 5] # MOUNTING OFFSET random 1-axis rotation
rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)

# attitude_type = 'stable' # 3-axis stabilized, X along radial, Y along cross-track, Z cross-track
# attitude_type = 'spin' # 
# attitude_type = 'moonpoint_stable' # 
# attitude_type = 'moonpointY_spin' # 
# attitude_type = 'moonpointX_spin' # 
attitude_type = 'tumble' # 

# tracked_body = 'moon'
# which moon vector to is the truth. 
# ephemerides - most precise option. 
# approx - approximate maths expression (at least 170 uradprecision)
moon_truth_used = 'ephemerides'
# moon_truth_used = 'approx_max'
add_solararray_frame = 1
not_use_sp = 1
output_interm_los = 1
add_moon_vec_error = 0

# Rotational velocity vector
omega_used = 0.1 # deg/s
if 'X_spin' in attitude_type:
    omega_vec = np.array([omega_used, 0, 0])
elif 'Y_spin' in attitude_type:
    omega_vec = np.array([0, omega_used, 0])
elif attitude_type == 'spin':
    omega_vec = np.array([omega_used, 0, 0])
elif attitude_type == 'tumble':
    omega_vec = np.array([omega_used, 2*omega_used, -3*omega_used])
else: # stable
    omega_vec = np.array([0,0,0])
    
## Loading satellite orbital data
# path jazz
path_cwd = os.getcwd()
csv_output_path = r'orbital_simulations\moontrackers\leomeo_mixincl7d'
fname_simparam = 'simulation_parameters.json'
fname_states = 'state_history.dat'
data_raw, simulation_parameters = dputil.load_constellation_data(full_path = csv_output_path)

host_chosen = 'leo_host_polar'
# host_chosen = 'leo_target_polar'
# host_chosen = 'leo_host_incl'
# host_chosen = 'meo_target_incl'

sp_moom_source = 'corrected_sp'

if sp_moom_source == 'corrected_sp':
    moon_vector_computed = pd.read_csv(r'analyses\moon_tracking\MO_resolution_archive\initial_MO_resolution_checks\SPmoon_LINEAR_3600.000.csv').values
    r_moon = moon_vector_computed[:,2:]
    t_gps =  moon_vector_computed[:,0]
elif sp_moom_source == 'sp_raw':
    pass # TODO
    

# time slicing
t_j2000 = data_raw[:,0]
t_fromstart = t_gps - t_gps[0]
r_host = data_raw[:,[1,2,3]]
v_host = data_raw[:,[ii+3 for ii in [1,2,3]]]
moon_from_sf = where_sun.body_fromsp(t_j2000[0])
## Getting body-frame attitude
tudconv = tudatconv.tudat_predictor()
tt_chosen = np.arange(0, 3600, 10).astype(int) # 100 sec
t_fromstart_chosen = t_fromstart[tt_chosen] 
nrows = tt_chosen.shape[0]

print(f'\nChosen attitude : {attitude_type}')
save_csv = 1
make_plots = 1
plot_dazdel = 0
plot_dlos = 0
plot_3d = 1

# placeholders
rot_eci2bf = np.zeros((nrows, 3, 3))
pe_moonvec = np.zeros((nrows, 1))
r_moon_eci_true = np.zeros((nrows, 3)) # t_gps; Az; El [deg]
ae_moon_searched = np.zeros((nrows, 3)) # t_gps; Az; El [deg]
ae_moon_seen = np.zeros((nrows, 3)) # t_gps; Az; El [deg]
los_bf_searched = np.zeros((nrows, 3)) # x, y, z [m]
los_bf_seen = np.zeros((nrows, 3)) # x, y, z [m]
los_lct_searched = np.zeros((nrows, 3)) # x, y, z [m]
los_lct_seen = np.zeros((nrows, 3)) # x, y, z [m]

dt_gps2j2000 = t_conv.dt_gps2j2000tt() # t_j2000 = t_gps + dt_gps2j2000

for ii, tt in enumerate(tt_chosen):
    t_j2000_ii = t_j2000[tt]
    t_gps_ii = t_j2000_ii - dt_gps2j2000 
    r_host_ii = r_host[tt,:]
    v_host_ii = v_host[tt,:]
    ## get approx moon vector
    if not_use_sp:
        r_moon_obcalc = where_sun.compute_moon_vector_eci(t_gps_ii)
    else:
        r_moon_obcalc = r_moon[tt,:]

    if moon_truth_used == 'ephemerides':
        r_moon_true = moon_from_sf.get_sun(t_j2000_ii - t_j2000[0], body = 'moon')
    else:
        r_moon_true = r_moon_obcalc    
    r_host2moon = r_moon_obcalc - r_host_ii
    
    ## get attitude
    rpy_rotationpart = omega_vec * tt  # Roll Pitch Yaw around stable attitude mode
    rot_rotationpart = conv.convert_ea2dcm(rpy_rotationpart)

    if 'moon' not in attitude_type:  
        # stable attitude - aligned with velocity/radial vector (RSW frame)
        rot_eci2stable = tudconv.calc_rotrsweci(r_h = r_host_ii, v_h = v_host_ii)
    elif 'moonpoint' in attitude_type: # stable attitude - aligned with moon vector (moon-pointing)
        if 'X_spin' in attitude_type:
            # X aligned with moon
            x_axis = r_host2moon / np.linalg.norm(r_host2moon)
            y_axis = np.cross(x_axis, np.array([0,1,0])) # perpendicular to moon direction and Y in ECI
            y_axis = y_axis / np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            rot_eci2stable = np.vstack((x_axis, y_axis, z_axis)) #
            
        else:            
            # Y aligned with moon
            y_axis = r_host2moon / np.linalg.norm(r_host2moon)
            x_axis = np.cross(y_axis, np.array([0,1,0])) # perpendicular to moon direction and Y in ECI
            x_axis = x_axis / np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis) # complete RH frame
            rot_eci2stable = np.vstack((x_axis, y_axis, z_axis)) #
            if 1: # add constant offset from moon-pointing
                rot_solararray = conv.convert_ea2dcm([0, 0, 0])
                rot_eci2stable = rot_solararray @ rot_eci2stable

    rot_bf_from_eci_ii = rot_rotationpart @  rot_eci2stable
    rot_eci2bf[ii] = rot_bf_from_eci_ii
    
    # convert eci2body-frame attitude to quaternion
    quat_bf_from_eci_ii = conv.convert_dcm2quat(rot_bf_from_eci_ii)
    
    # get Az El
    # Searched - assuming no mounting offset and point at approx moon-calculated position
    ae_moon_searched_ii = ae_calc.calc_ae_full(r_host_ii, r_moon_obcalc, attitude_eci2bf= quat_bf_from_eci_ii,
                                          attitude_mountingoffset= None, output_interm_los=output_interm_los)
    # Seen - az/el of TUE moon with mounting offset
    ae_moon_seen_ii = ae_calc.calc_ae_full(r_host_ii, r_moon_true, attitude_eci2bf= quat_bf_from_eci_ii,
                                          attitude_mountingoffset= quat_mounting_offset, output_interm_los = output_interm_los)
    # debug purposes
    if 0:
        r_t = r_moon_true - r_host_ii
        r_t_bf = rot_eci2stable @ r_t
        r_t_bf_rot = rot_bf_from_eci_ii @ r_t
        quat_from_dcm = conv.convert_dcm2quat(rot_bf_from_eci_ii)
        dcm_back_from_quat = conv.convert_quat2dcm(quat_from_dcm/np.linalg.norm(quat_from_dcm))
        diff = dcm_back_from_quat - rot_bf_from_eci_ii
    
    ## Store outputs
    pe_moonvec[ii,:] = vec.calc_dot_angle(r_moon_true, r_moon_obcalc)*1e3 # mrad
    r_moon_eci_true[ii,:] = r_moon_true
    ae_moon_searched[ii,:] = np.rad2deg(ae_moon_searched_ii[:3])
    ae_moon_seen[ii,:] = np.rad2deg(ae_moon_seen_ii[:3])
    los_bf_searched[ii,:] = ae_moon_searched_ii[6:9]
    los_bf_seen[ii,:] = ae_moon_seen_ii[6:9]
    los_lct_searched[ii,:]  = ae_moon_searched_ii[9:12]
    los_lct_seen[ii,:]  = ae_moon_seen_ii[9:12]

print(f'DONE calculating expcted and true moon vector azimuth and elevation. \nMAX PE from moon-vec : {np.max(pe_moonvec):.1f} urad.')
if save_csv:
    data_stored = np.hstack((tt_chosen.reshape((tt_chosen.shape[0],1)), ae_moon_searched, ae_moon_seen[:,:2], pe_moonvec))
    data_df = pd.DataFrame(data = data_stored, columns=
                           ['t_s', 'a_true', 'e_true', 'r', 'a_meas', 'e_meas', 'pe_mrad'])
    df_title = f'AE_moonSim_Att{attitude_type}MO{mounting_offset_rpy}_nosp{not_use_sp}.csv'
    save_folder = r'outputs\tables\simulated_moon_tracks'
    data_df.to_csv(f'{save_folder}\{df_title}', index=False)
    print(f'CSV saved as {df_title}')
if make_plots:
    if plot_dazdel: # dAz, dEl
        f_title = f'moon expected vs real. \nAttitude : {attitude_type}. \nMounting offset (unknown) : {mounting_offset_rpy} deg'

        f, axs = plt.subplots(nrows = 4)
        for ii, ax in enumerate(axs[:2]):
            ax.plot(t_fromstart_chosen, ae_moon_searched[:,ii], label = 'true')
            ax.plot(t_fromstart_chosen, ae_moon_seen[:,ii], label = 'seen')
            ax.set_ylabel(['Az', 'El'][ii] + ' [deg]')
            ax.grid('on')
            ax.legend()
            axs[ii].set_ylim([np.min([np.min(np.min(ae_moon_searched[:,ii])), np.min(ae_moon_seen[:,ii])]) - 30 , 
                            np.max([np.max(np.max(ae_moon_searched[:,ii])), np.max(ae_moon_seen[:,ii])]) + 30 ])

        for ii, ax in enumerate(axs[2:]):
            ax.plot(t_fromstart_chosen, ae_moon_searched[:,ii] - ae_moon_seen[:,ii])        
            ax.set_ylabel(['D Az', 'D El'][ii] + ' [deg]')
            ax.grid('on')
        ax.plot(t_fromstart_chosen, ae_moon_searched[:,ii] - ae_moon_seen[:,ii])        
        ax.set_xlabel('t [s]')
        f.set_tight_layout('tight')
        f.suptitle(f_title)
        bplt.savefig(f, f'Delta_AE_moon_ExpvTrue_{attitude_type}')
    if plot_dlos:
        f_title = f'moon intermediate LOS comparison. \nAttitude : {attitude_type}. \nMounting offset (unknown) : {mounting_offset_rpy} deg'
        f, axs = plt.subplots(nrows = 4, figsize = (8,10)) # 1 - dLOS BF. 2 - dLOS LCT. 3 - PE BF. 4 - PE LCT
        for ii, ax in enumerate(axs):
            if ii == 0:
                y_plotted = los_bf_searched - los_bf_seen
                ylabel = 'dLOS BF [m]'
                leg = 'XYZ'
            elif ii == 1:
                y_plotted = los_lct_searched - los_lct_seen
                ylabel = 'dLOS LCT [m]'
            elif ii == 2:
                y_plotted = [np.rad2deg(vec.calc_dot_angle(los_true, los_seen))
                              for los_true, los_seen in zip(los_bf_searched, los_bf_seen)]
                leg = ['PE']
                ylabel = 'PE BF [deg]'
            elif ii == 3:
                y_plotted = [np.rad2deg(vec.calc_dot_angle(los_true, los_seen))
                              for los_true, los_seen in zip(los_lct_searched, los_lct_seen)]
                leg = ['PE']
                ylabel = 'PE LCT [deg]'

            if ii < 2:
                for jj in range(3):
                    ax.plot(t_fromstart_chosen, y_plotted[:,jj], label = leg[jj])
                ax.legend()
            else:
                ax.plot(t_fromstart_chosen, y_plotted)
            
            ax.set_ylabel(ylabel)
            ax.grid('on')
       
        ax.set_xlabel('t [s]')
        f.set_tight_layout('tight')
        f.suptitle(f_title)
        bplt.savefig(f, f'Delta_LOS_moon_ExpvTrue_ATT{attitude_type}_MO{mounting_offset_rpy}')
#%%    
if plot_3d:
    # making 3d plots
    import plotting_tools.modular_plotting as modplot
    importlib.reload(modplot)
    length = 1e3
    ii_used = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180] # for orbit
    ii_frame = [0, 18]
    frame_ind_0 = [0, 0]
    frame_ind_1 = [9, 80]
    frame_ind_2 = [17, 180]
    len_los = 1e7

    frame_plotted = 'BF'
    if frame_plotted == 'BF':
        rot_shown = rot_eci2bf
    elif frame_plotted == 'GF':
        rot_shown = rot_eci2bf

    ftitle = f'''Mounting offset: {mounting_offset_rpy}
    Frame visualized: {frame_plotted}. Attitude :{attitude_type}
    Point 0 AE exp. :{np.round(ae_moon_searched[frame_ind_0[0],:2],2)} deg. AE seen. :{np.round(ae_moon_seen[frame_ind_0[0],:2],2)} deg.
    Point 1 AE exp. :{np.round(ae_moon_searched[frame_ind_1[0],:2],2)} deg. AE seen. :{np.round(ae_moon_seen[frame_ind_1[0],:2],2)} deg.
    '''
    f, ax = modplot.make_3dplot()
    f, ax = modplot.add_earth(f, ax)
    f, ax = modplot.add_orbit_basic(f, ax, r_host[ii_used,:], label = 'LEO Polar host orbit', c = 'b', linewidth = 3)    
    f, ax = modplot.add_ref_frame(f, ax, chosen_setting= 1, 
                                    rot_gf = rot_shown[frame_ind_0[0]], 
                                    origin = r_host[frame_ind_0[1],:]
                                    )
    f, ax = modplot.add_ref_frame(f, ax, chosen_setting= 1, 
                            rot_gf = rot_shown[frame_ind_1[0]], 
                            origin = r_host[frame_ind_1[1],:]
                            )
    
    f, ax = modplot.add_single_los(f, ax, 
                                    state_h = r_host[frame_ind_0[1],:],
                                    state_t = r_moon_eci_true[frame_ind_0[0],:],
                                    normalize = 1,
                                    len_normalized = len_los,
                                    label_used= ''
                                    )
    f, ax = modplot.add_single_los(f, ax, 
                                    state_h = r_host[frame_ind_1[1],:],
                                    state_t = r_moon_eci_true[frame_ind_1[0],:],
                                    normalize = 1,
                                    len_normalized = len_los,
                                    label_used= 'LOS 2 moon'
                                                                        )
    f, ax = modplot.add_ref_frame(f, ax, chosen_setting= 1, 
                            rot_gf = rot_shown[frame_ind_2[0]], 
                            origin = r_host[frame_ind_2[1],:]
                            )
    
    f, ax = modplot.add_single_los(f, ax, 
                                    state_h = r_host[frame_ind_2[1],:],
                                    state_t = r_moon_eci_true[frame_ind_2[0],:],
                                    normalize = 1,
                                    len_normalized = len_los,
                                    label_used= '')
    modplot.set_axes_equal(ax, axlim = 10e6)
    f, ax = modplot.add_glossary_basic(f, ax, title = ftitle)
    bplt.savefig(f, name = f'3D_MtOff{mounting_offset_rpy}_Att_{attitude_type}')
