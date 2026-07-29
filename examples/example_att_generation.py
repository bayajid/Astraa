#%% Script used to generate rocketlab attitude scenario data
# March 2015
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# own funtions
import sys
import importlib
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import plotting_tools.basic_plotting as bplt
import attitude_tools.conversions as conv
# path jazz
path_cwd = os.getcwd()
path_outputs = fr'{path_cwd}/outputs/tables/rocketlab_quatpred'

save_outputs = 1
#%% Generating the TRUE attitude data
# inputs
# 3-2-1 Euler Angle convetion
plot_omegas = 1
att_ea_0 = np.array([5, 10, 45]) # Roll; pitch; yaw angles [deg]
t_0 = 0
t_step = 0.01
t_end = 100
# Settings for attitude 
# setting =  'rotate_swap'
# setting = 'rotate_all_axes'
# setting = 'rotate_all_pred084'
# setting = 'rotate_all_pred084_swap' # add the jerk at t=40
# setting = 'rotate_roll_yaw'
# setting = 'rotate_yaw'
# setting = 'rotate_azel_octhw467'
# setting = 'customer_may5' # attitude simulated from given rates. Est and True
setting = 'rocketlab_march'
# euler angles and rates available (but rates are identical)
if setting == 'rocketlab_march':
    # rotate at max velocities and rates in X, Y, Z
    # Angular Velocities about X/Y/Z [deg/s]
    om_1 = 0.069 / np.sqrt(3) 
    om_2 = -0.069 / np.sqrt(3) 
    om_3 = 0.069 / np.sqrt(3) 
    w_all = np.array([om_1, om_2, om_3])
    w_all /= np.sqrt(3)    # Angular rates about X/Y/Z [deg/s^2]
    om_dot_1 = 0.0837 / np.sqrt(3)
    om_dot_2 = -0.0837 / np.sqrt(3)
    om_dot_3 = 0.0837 / np.sqrt(3)
    om_dot_limit = 0.0837
    w_dot_all = np.array([om_dot_1, om_dot_2, om_dot_3])
    w_dot_all /= np.sqrt(3)
    om_ddot_1 = 0.5756 / np.sqrt(3)
    om_ddot_2 = -0.5756 / np.sqrt(3)
    om_ddot_3 = 0.5756 / np.sqrt(3)
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
    
    om_ddot_1 = 0
    om_ddot_2 = 0
    om_ddot_3 = 0

    omega_dot_swapped_vec = np.array([-om_dot_1, -om_dot_2, -om_dot_3])

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

t_vec_prop = np.arange(t_0, t_end+t_step, t_step)
quat_quatrate_storage = np.zeros((len(t_vec_prop), 9)) # t; q; qdot
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
    if 'octhw' not in setting and 0:
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
            ea_rate = conv.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            omega_vec_all[ii,:] = om_0

            # update
            ea_0 = ea_ii
            om_0 = om_ii         
    elif 'rocketlab' in setting:
        # propagate omega AND omega_dot, different Swap-time
        t0 = 0
        acc_go_mode = 1
        acc_hold_mode = 0
        t_hold_scenario_end = 50
        dt_hold = 1.5 # nr f seconds to hold acceleraiton
        for ii, t_ii in enumerate(t_vec_prop):
            if ii == 0:
                ## initialize
                ea_0 = att_ea_0
                om_0 = omega_vec
                om_dot_0 = omega_dot_vec*1
                om_ddot_0 = om_ddot_1
            if np.linalg.norm(om_dot_0) >= om_dot_limit:
                if t_ii > t_hold_scenario_end:
                    dt_hold = 0.5
                if acc_go_mode:
                    acc_hold_mode = 1
                
                if acc_go_mode and acc_hold_mode: # modes switched - set times; turn go off
                    t_0 = t_ii
                    t_f = t_ii + dt_hold
                    acc_go_mode = 0
                
                if t_ii <= t_f:
                    # hold rate
                    om_ddot_0 = 0 * omega_ddot_vec
                else:
                    om_ddot_0 = -np.sign(om_dot_0[0])*omega_ddot_vec
                    acc_go_mode = 1
                    acc_hold_mode = 0
            
            # calc angular velocities
            om_ii = om_0 + t_step * om_dot_0 # [deg/s]
            om_dot_ii = om_dot_0 + t_step * om_ddot_0 #[deg/s^2]
            # calc euler angle rates
            ea_rate = conv.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            quat_ii, quat_rate_ii = conv.calc_qdot(ea_ii, om_ii, deg=1)
            
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            omega_vec_all[ii,:] = om_0
            omega_dot_vec_all[ii,:] = om_dot_ii
            omega_ddot_vec_all[ii,:] = om_ddot_0
            quat_quatrate_storage[ii,:] = np.vstack((t_ii, quat_ii, quat_rate_ii)).transpose()

            # update
            ea_0 = ea_ii
            om_dot_0 = om_dot_ii
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
            ea_rate = conv.calc_ea_dot(ea_0, om_ii, use_deg)
            # calc true attitude
            ea_ii = ea_0 + t_step * ea_rate
            
            # store
            ea_all[ii,:] = ea_0
            ea_dot_all[ii,:] = ea_rate
            omega_vec_all[ii,:] = om_0
            omega_dot_vec_all[ii,:] = om_dot_ii
            omega_ddot_vec_all[ii,:] = om_ddot_0
            quat_ii, quat_rate_ii = conv.calc_qdot(ea_ii, om_ii, deg=1)
            quat_quatrate_storage[ii,:] = np.hstack(([t_ii], quat_ii, quat_rate_ii))
            # update
            ea_0 = ea_ii
            om_dot_0 = om_dot_ii
            om_0 = om_ii      
    print(f'''                  DONE
        Final roll, pitch, yaw : {ea_ii} [deg] 
        Final angular velocity : {om_ii} [deg/s]
        MAX angular velocity : {np.max(np.linalg.norm(omega_vec_all, axis = 1)):.3f} [deg/s]
        constant angular rate : {omega_dot_vec_used} [deg/s^2]
        --------------------------------------------------
        Calculated attitude kinematics for {t_end} s, with {t_step} steps
        --------------------------------------------------
        ''')
    print(f"Total rotation magnitude:{np.rad2deg(2 * np.arccos(quat_quatrate_storage[-1,1])):.3f} deg")
    if save_outputs:
        print('Saving')
        ## Save
        data = np.hstack((t_vec_prop.reshape((nrows, 1)), ea_all, ea_dot_all, omega_vec_all))
        output_df = pd.DataFrame(columns = ['t_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'rollrate_degs', 'pitchdate_degs', 'yawrate_degs','om_x_degs', 'om_y_degs', 'om_z_degs'],
        data = data)

        output_path = fr'{path_outputs}/true_attitude_{setting}.csv'
        output_df.to_csv(output_path, index=False)
        # save quaternions
        data = quat_quatrate_storage
        output_df = pd.DataFrame(columns = ['t_s', 'qc', 'q1', 'q2', 'q3', 'qdotc', 'qdot1', 'qdot2', 'qdot3'],
        data = data)

        output_path = fr'{path_outputs}/true_quat{setting}.csv'
        output_df.to_csv(output_path, index=False)
        print(f'Saved true attitude to \n{output_path} \nas true_attitude_{setting}.csv')
    else:
        print('Outputs not saved')
#%%
plot_omegas = 1
plot_rates = 1
plot_quaternions = 1
if plot_omegas:
    omegas_nested = [omega_vec_all, omega_dot_vec_all, omega_ddot_vec_all]
    f, axs = plt.subplots(nrows = 3, sharex=True)

    for ii, ax in enumerate(axs):
        w_plot = omegas_nested[ii]
        for jj in range(3):
            ax.plot(t_vec_prop, w_plot[:,jj], label = 'xyz'[jj], marker = '>o '[jj], markevery = 100)
        ax.plot(t_vec_prop, np.linalg.norm(w_plot,axis=1), label = 'norm', marker = '>o '[jj], markevery = 100)
        # ax.plot([39,39], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'r', label = 'jerk start')
        # ax.plot([39.5, 39.5], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'y', label = 'dat 1')
        # ax.plot([40, 40], [-5+ax.get_ylim()[0], 5+5*ax.get_ylim()[1]], c = 'r', label = 'dat fin/pred.')
        ax.set_ylabel('w' + [' [deg/s]', '_dot [deg/s^2]', '_ddot [deg/s^3]'][ii])
        # ax.set_ylim([-0.6, 0.6])
        # ax.set_xlim([35, 45])
        # ax.set_yticks([-.84, -.5, -.27, 0, .27, .5, .84])
        ax.grid()
        ax.legend()
        ax.set_xlabel('t [s]')
    # f.set_tight_layout('tight')
    print('Plotting omega, omega_dot and omega_ddot')
    bplt.savefig(f, f'angular_rates_{setting}', tag_option=1)
    # plt.show()
if plot_quaternions:
    f, axs = plt.subplots(4, 2)
    for ncol in range(2):
        dat_plotted = quat_quatrate_storage[:,1+ncol*4:(1+ncol)*4+1]
        for nrow in range(4):
            ax = axs[nrow, ncol]
            ax.plot(t_vec_prop, dat_plotted[:,nrow])
            ax.set_ylabel(['q'+'c123'[nrow], 'qdot'+'c123'[nrow]][ncol])
            ax.grid('on')
    f.set_tight_layout('tight')
    bplt.savefig(f, f'quaternions_{setting}', tag_option=1)

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

plt.show()