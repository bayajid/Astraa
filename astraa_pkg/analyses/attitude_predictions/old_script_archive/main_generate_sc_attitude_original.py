#%% Functi ont ogenerate some crazy spacecraft attitude
# with angular velocities and rates. Several possible settings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# own funtions
import importlib

import attitude_tools.conversions as tools_attitude
importlib.reload(rot)
# path jazz
path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/outputs'
save_outputs = 1
#%% Generating the TRUE attitude data
# inputs
# 3-2-1 Euler Angle convetion
plot_omegas = 0
att_ea_0 = np.array([5, 10, 45]) # Roll; pitch; yaw angles [deg]

# Settings for attitude 
# setting =  'rotate_swap'
# setting = 'rotate_all_axes'
# setting = 'rotate_all_pred084'
# setting = 'rotate_all_pred084_swap' # add the jerk at t=40
# setting = 'rotate_roll_yaw'
# setting = 'rotate_yaw'
# setting = 'rotate_azel_octhw467'
setting = 'customer_may5' # attitude simulated from given rates. Est and True
# euler angles and rates available (but rates are identical)
if setting == 'rotate_all_axes':
    # rotate at max velocities and rates in X, Y, Z
    # Angular Velocities about X/Y/Z [deg/s]
    om_1 = 0.84
    om_2 = 0.84
    om_3 = 0.84
    
    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0.012
    om_dot_2 = 0.012
    om_dot_3 = 0.012
elif setting == 'rotate_swap':
    # rotate at max velocities and rates in X, Y, Z
    # Angular Velocities about X/Y/Z [deg/s]
    om_1 = 0.84
    om_2 = 0.84
    om_3 = 0.84
    
    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0.012
    om_dot_2 = 0.012
    om_dot_3 = 0.012

    omega_dot_swapped_vec = np.array([-om_dot_1, -om_dot_2, -om_dot_3])

elif setting == 'rotate_roll_yaw':
    # rotate at max velocities and rates in X and Z
    om_1 = 0.84
    om_2 = 0
    om_3 = 0.84
    
    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0.012
    om_dot_2 = 0
    om_dot_3 = 0.012
elif setting == 'rotate_yaw':
    # rotate at max velocities and rates in Z
    om_1 = 0
    om_2 = 0
    om_3 = 0.84
    
    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0
    om_dot_2 = 0
    om_dot_3 = 0.012

    om_ddot_swap_1 = 0
    om_ddot_swap_2 = 0
    om_ddot_swap_3 = 0
elif setting == 'rotate_all_pred084':
    # rotate at intial velocities and max rates
    om_1 = 0.34
    om_2 = 0.34
    om_3 = 0.34
    # velocity reaches 0.84 deg/s at t=40
    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0.012
    om_dot_2 = 0.012
    om_dot_3 = 0.012
    # placeholder
    om_ddot_1 = 0.0
    om_ddot_2 = 0.0
    om_ddot_3 = 0.0
elif setting == 'rotate_azel_octhw467':
    ## Initial conditions
    om_1 = 0.84
    om_2 = 0.84
    om_3 = 0.84
    # t=0 until t = 37
    om_dot_1 = 0
    om_dot_2 = 0
    om_dot_3 = 0
    om_ddot_1 = 0
    om_ddot_2 = 0
    om_ddot_3 = 0
    # t=37 until t = 40
    om_ddot_start_1 = 0
    om_ddot_start_2 = 0
    om_ddot_start_3 = 0
    # t=40+
    om_ddot_swap_1 = 0
    om_ddot_swap_2 = -0.5
    om_ddot_swap_3 = -0.5
    om_lim = 0.84
omega_vec = np.array([om_1, om_2, om_3])
omega_dot_vec = np.array([om_dot_1, om_dot_2, om_dot_3])
omega_ddot_vec = np.array([om_ddot_1, om_ddot_2, om_ddot_3])
omega_dot_swapped_vec = np.array([-om_dot_1, -om_dot_2, -om_dot_3])
t_0 = 0
t_step = 0.01
t_end = 100
t_vec_prop = np.arange(t_0, t_end+t_step, t_step)

nrows = len(t_vec_prop)
#%% Simulate attitude and measurements
if 1:
    np.set_printoptions(precision = 3)
    print(f'-----SIMULATING true SC attitude-----')
    print(f'ATTITUDE SETTING : {setting.upper()}')
    print(f'''                  INPUTS
    initial roll, pitch, yaw : {att_ea_0} [deg] 
    initial angular velocity : {omega_vec} [deg/s]
    constant angular rate : {omega_dot_vec} [deg/s^2]
    --------------------------------------------------
    Calculating attitude kinematics for {t_end} s, with {t_step} steps
    --------------------------------------------------
    ''')
    # create placeholders   
    ea_all = np.zeros((nrows, 3))
    ea_dot_all = np.zeros((nrows, 3))
    omega_vec_all = np.zeros((nrows, 3))
    # plotting purposes only
    omega_dot_vec_all = np.zeros((nrows, 3))
    omega_ddot_vec_all = np.zeros((nrows, 3))
    quat_all = np.zeros((nrows, 4))    

    use_deg = 1 # conditional if deg are used
    # generate attitude

    omega_dot_vec_used = omega_dot_vec
    if 'octhw' not in setting:
        for ii, t_ii in enumerate(t_vec_prop):
            if ii == 0:
                ## initialize
                om_0 = omega_vec                
                ea_0 = att_ea_0

            if t_ii >= 40 and 'swap' in setting:
                omega_dot_vec_used = omega_dot_swapped_vec
                print(f't = {t_ii} We swapping accelerations to {omega_dot_vec_used} deg/s^2')

            # calc angular velocities
            om_ii = om_0 + t_step * omega_dot_vec_used # [deg/s]
            # calc euler angle rates
            ea_rate = rot.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            omega_vec_all[ii,:] = om_0

            # update
            ea_0 = ea_ii
            om_0 = om_ii         
    elif 'octhw' in setting:
        # propagate omega AND omega_dot, different Swap-time
        t0 = 37.5
        for ii, t_ii in enumerate(t_vec_prop):
            if ii == 0:
                ## initialize
                ea_0 = att_ea_0
                om_0 = omega_vec
                om_dot_0 = omega_dot_vec
                om_ddot_0 = omega_ddot_vec
            elif t_ii >= t0+.5 and t_ii < t0+1:
                om_ddot_0 = np.array([om_ddot_swap_1, om_ddot_swap_2, om_ddot_swap_3])
            elif t_ii >= t0+1 and t_ii < t0+2:
                om_ddot_0 = np.array([om_ddot_start_1, om_ddot_start_2, om_ddot_start_3])
                print(f't = {t_ii:.2f} We jerking with {om_ddot_0} deg/s^3. Rate : {om_dot_ii} deg/s^2')
            elif t_ii >= t0+2 and t_ii < t0+3:
                om_ddot_0 = -np.array([om_ddot_swap_1, om_ddot_swap_2, om_ddot_swap_3])
                print(f't = {t_ii:.2f} We jerking with {om_ddot_0} deg/s^3. Rate : {om_dot_ii} deg/s^2')
            elif t_ii > t0 + 3.7:
                om_ddot_0 = np.array([0,0,0])
                om_dot_0 = np.array([0,0,0])
                om_0 = np.array([om_lim, om_lim, om_lim])
            # calc angular velocities
            om_ii = om_0 + t_step * om_dot_0 # [deg/s]
            om_dot_ii = om_dot_0 + t_step * om_ddot_0 #[deg/s^2]
            # calc euler angle rates
            ea_rate = rot.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            omega_vec_all[ii,:] = om_0
            omega_dot_vec_all[ii,:] = om_dot_ii
            omega_ddot_vec_all[ii,:] = om_ddot_0
            # update
            ea_0 = ea_ii
            om_dot_0 = om_dot_ii
            om_0 = om_ii      
    print(f'''                  DONE
    Final roll, pitch, yaw : {ea_ii} [deg] 
    Final angular velocity : {om_ii} [deg/s]
    constant angular rate : {omega_dot_vec_used} [deg/s^2]
    --------------------------------------------------
    Calculated attitude kinematics for {t_end} s, with {t_step} steps
    --------------------------------------------------
    ''')        
    if save_outputs:
        print('Saving')
        ## Save
        data = np.hstack((t_vec_prop.reshape((nrows, 1)), ea_all, ea_dot_all, omega_vec_all))
        output_df = pd.DataFrame(columns = ['t_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'rollrate_degs', 'pitchdate_degs', 'yawrate_degs','om_x_degs', 'om_y_degs', 'om_z_degs'],
        data = data)

        output_path = fr'{path_outputs}/true_attitude_{setting}.csv'
        output_df.to_csv(output_path, index=False)
        print(f'Saved true attitude to \n{output_path} \nas true_attitude_{setting}.csv')
    else:
        print('Outputs not saved')
#%%
if plot_omegas:
    omegas_nested = [omega_vec_all, omega_dot_vec_all, omega_ddot_vec_all]
    f, axs = plt.subplots(nrows = 3)

    for ii, ax in enumerate(axs):
        w_plot = ea_dot_all[ii]
        for jj in range(3):
            ax.plot(t_vec_prop, w_plot[:,jj], label = 'xyz'[jj], marker = '>o '[jj], markevery = 100)
        
        # ax.plot([39,39], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'r', label = 'jerk start')
        ax.plot([39.5, 39.5], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'y', label = 'dat 1')
        ax.plot([40, 40], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'r', label = 'dat fin/pred.')
        ax.set_ylabel('w' + [' [deg/s]', '_dot [deg/s^2]', '_ddot [deg/s^3]'][ii])
        ax.set_ylim([-1, 1])
        ax.set_xlim([35, 45])
        ax.set_yticks([-.84, -.5, -.27, 0, .27, .5, .84])
        ax.grid()
        ax.legend()
        ax.set_xlabel('t [s]')
    # f.set_tight_layout('tight')
    print('Plotting omega, omega_dot and omega_ddot')
    plt.show()
plot_rates = 1
if plot_rates:
    f, axs = plt.subplots(3, 2)
    for ncol in range(2):
        dat_plotted = [ea_all, ea_dot_all][ncol]        
        for nrow in range(3):
            ax = axs[nrow, ncol]
            ax.plot(t_vec_prop, dat_plotted[:,nrow])
            ax.set_ylabel('RPY'[nrow] + [' [deg]', ' dot [deg/s]'][ncol])
            ax.grid('on')
    f.set_tight_layout('tight')

