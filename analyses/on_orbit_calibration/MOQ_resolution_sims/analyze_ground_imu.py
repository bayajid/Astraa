#%% May 5, simulating OOC-A on-ground test with Mynaric's IMU (0.5 deg RMS for static pitch/roll)
# https://www.vectornav.com/docs/default-source/product-brief/vn-300-product-brief.pdf?sfvrsn=f96cc41_2
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
fname_moon = 'analyses/on_orbit_calibration/MOQ_resolution_sims/gs2moon_data.csv' # r'analyses\on_orbit_calibration\MOQ_resolution_sims\gs2moon_data.csv'

## MVP imports
import basic_tools.vector_operations as vec_calc
import basic_tools.time_conversion as t_conv
import basic_tools.in_out as io
import plotting_tools.basic_plotting as bplt
import plotting_tools.plotting_utilities as plt_util

import astronomy_tools.constants as const
import astronomy_tools.astro_targets as where_sun

import attitude_tools.attitude_resolution as att_res
import attitude_tools.conversions as conv
import attitude_tools.rotations as rot
import attitude_tools.attitude_simulation as att_sim

import basic_tools.in_out as savedat

import pointing_calculations.ae_calculation as ae_calc
import pointing_calculations.simulate_moon_scan as moon_scan
import tudat_tools.data_processing.data_processing_utilities as dputil
importlib.reload(bplt)
importlib.reload(moon_scan)
importlib.reload(att_res)
debug_mode = 0

# imu_chosen = 'vectornav300'
# imu_chosen = 'WT61C TTL'
# imu_chosen = 'iNAT-RQT-400x'
imu_chosen = 'PE_Budget_v2'
use_triad = 1

title_append = 'imu_performance'

data_df = pd.read_csv(fname_moon)  

data_sliced = data_df.values[0:60000,:]#.values[106000:141000,:]
if debug_mode:
    # data_sliced = data_sliced[:10,:]
    trolley_angles_used = [0.5]
    downsampling_step = 100
    ii_sliced = np.arange(0, data_sliced.shape[0]+downsampling_step, downsampling_step)
    ii_sliced[-1] = ii_sliced[-1]-1
    data_sliced = data_sliced[ii_sliced,:]
q_bf_full = data_sliced[:,16:24]
q_bf_full[:,:4] = q_bf_full[:,:4]
q_bf_full[:,4:] = q_bf_full[:,4:] 

t_gps = data_sliced[:,0]
s_host = data_sliced[:,4:10]
s_target = data_sliced[:,10:16]
r_host_full = s_host[:,:3]
v_host_full = s_host[:,3:]
r_target_full = s_target[:,:3]

# DEBUG/VERIFICATION PURPOSES
# n_samples = 30000
n_samples = 1000
if not use_triad:
    nr_inputs = 6
    function_option = 2 # SVD
    # nr_inputs = 2
else:
    nr_inputs = 2
    function_option = 0 # TRIAD

# select update rates
att_update_rate = 2 # [Hz]
pos_update_rate = 0.1 # [Hz]

nrows = r_host_full.shape[0]
mounting_offset_rpy = [5, 4, 2.5] # MOUNTING OFFSET random 3-axis wrotation
error_mounting_offset = 65e-3 # 65e-3 # mrad, Mounting offset error

rot_bf2lct = conv.convert_ea2dcm(mounting_offset_rpy)
quat_mounting_offset = conv.convert_ea2quat(mounting_offset_rpy)
quat_mounting_offset_known = conv.convert_dcm2quat(conv.convert_eigenaxis2dcm([-1,2,3],error_mounting_offset) @ conv.convert_quat2dcm(quat_mounting_offset))
quat_mounting_offset_known = quat_mounting_offset_known / np.linalg.norm(quat_mounting_offset_known)

# TODO select attitude profile
importlib.reload(att_sim)
importlib.reload(conv)
importlib.reload(att_res)
storage_folder = 'ooc_ground_test_using_IMU'
## COMPUTE ATTITUDE
# 
err_r_host_default = np.array([6, 7, 8]) # RANDOMIZE DIRECTION? ACTUALLY TODO what magnitude??
err_r_target_default = np.array([-60, -70, 80]) # RANDOMIZE DIRECTION? 100 m target position error expected
# Attitude error ~ 0.5 mrad
if imu_chosen == 'WT61C TTL':
    att_err = 0.05 # deg
elif imu_chosen == 'vectornav300':
    att_err = 0.5 # deg
elif imu_chosen == 'iNAT-RQT-400x':
    att_err = 0.01 # deg
elif imu_chosen == 'PE_Budget_v2':
    att_err = np.rad2deg(0.5e-3) # deg
    
err_att_host_default = np.array([att_err*17e-3]) # rad # RANDOMIZE DIRECTION?

# mrad, RSS components
# RSS_components = {
#     'gimbal_control' : 0.4,
#     'thermal' : 0.8,
#     'other' : 0.1,
#     'encoder_jump':0.767*57, 
# }
import numpy as np
import matplotlib.pyplot as plt

# Your error components dictionary 
# Most values represent standard deviations for Gaussian components
# encoder_jump is special: std=0.767, can occur 0-57 times
RSS_components = {
    'gimbal_control': 0.4,
    'thermal': 0.8    
}

# Special component: encoder jump
encoder_jump_std = 0 # 0.767
max_encoder_jumps =0 # 57

def calculate_rss_with_variation(components_dict, encoder_std, max_jumps, num_samples=1000):
    """
    Calculate RSS with random variation of components
    
    Parameters:
    components_dict: dict with component names and their std deviations (Gaussian)
    encoder_std: standard deviation for each encoder jump
    max_jumps: maximum number of encoder jumps possible
    num_samples: number of Monte Carlo samples
    
    Returns:
    rss_samples: array of RSS values
    rss_mean: mean RSS value
    rss_std: standard deviation of RSS
    jump_counts: array of encoder jump counts for each sample
    """
    
    # Generate random samples for Gaussian components
    component_samples = {}
    for component, std_dev in components_dict.items():
        component_samples[component] = np.random.normal(0, std_dev, num_samples)
    
    # Generate encoder jump data
    jump_counts = np.random.randint(0, max_jumps + 1, num_samples)  # 0 to max_jumps inclusive
    
    # Calculate RSS for each sample
    rss_samples = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Sum of squares from Gaussian components
        gaussian_sum_squares = sum(component_samples[comp][i]**2 for comp in components_dict.keys())
        
        # Add encoder jump contribution
        # Each jump contributes encoder_std^2 to the sum of squares
        encoder_contribution = jump_counts[i] * (encoder_std**2)
        
        # Total RSS
        total_sum_squares = gaussian_sum_squares + encoder_contribution
        rss_samples[i] = np.sqrt(total_sum_squares)
    
    return rss_samples, np.mean(rss_samples), np.std(rss_samples), jump_counts

# Method 1: Monte Carlo simulation
print("=== Monte Carlo Simulation ===")
rss_samples, rss_mean, rss_std, jump_counts = calculate_rss_with_variation(
    RSS_components, encoder_jump_std, max_encoder_jumps, 10000
)

print(f"RSS Mean: {rss_mean:.3f}")
print(f"RSS Std Dev: {rss_std:.3f}")
print(f"RSS Range (±2σ): {rss_mean - 2*rss_std:.3f} to {rss_mean + 2*rss_std:.3f}")
print(f"Average encoder jumps per sample: {np.mean(jump_counts):.1f}")
print(f"Max encoder jumps observed: {np.max(jump_counts)}")

# Method 2: Analytical approach (for comparison)
print("\n=== Analytical Approach ===")
# For the Gaussian components
gaussian_rss = np.sqrt(sum(std**2 for std in RSS_components.values()))
# Expected encoder contribution (average jumps * encoder_std^2)
expected_jumps = max_encoder_jumps / 2  # uniform distribution mean
expected_encoder_contribution = expected_jumps * (encoder_jump_std**2)
analytical_rss = np.sqrt(gaussian_rss**2 + expected_encoder_contribution)
print(f"Analytical RSS (Gaussian only): {gaussian_rss:.3f}")
print(f"Expected encoder jumps: {expected_jumps:.1f}")
print(f"Analytical RSS (with expected encoder): {analytical_rss:.3f}")

# Single random sample example
print("\n=== Single Random Sample ===")
np.random.seed(42)  # For reproducibility
single_sample = {}
for component, std_dev in RSS_components.items():
    single_sample[component] = np.random.normal(0, std_dev)
    print(f"{component}: {single_sample[component]:.3f}")

# Random number of encoder jumps
single_jumps = np.random.randint(0, max_encoder_jumps + 1)
encoder_contribution = single_jumps * (encoder_jump_std**2)
print(f"encoder_jumps: {single_jumps} jumps (contribution: {np.sqrt(encoder_contribution):.3f})")

single_rss = np.sqrt(sum(val**2 for val in single_sample.values()) + encoder_contribution)
print(f"Single sample RSS: {single_rss:.3f}")

# Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Histogram of RSS values
plt.subplot(1, 3, 1)
plt.hist(rss_samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(rss_mean, color='red', linestyle='--', label=f'Mean: {rss_mean:.2f}')
plt.axvline(analytical_rss, color='green', linestyle='--', label=f'Analytical: {analytical_rss:.2f}')
plt.xlabel('RSS Value')
plt.ylabel('Density')
plt.title('Distribution of RSS Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Encoder jump distribution
plt.subplot(1, 3, 2)
plt.hist(jump_counts, bins=range(0, max_encoder_jumps + 2), density=True, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(np.mean(jump_counts), color='red', linestyle='--', label=f'Mean: {np.mean(jump_counts):.1f}')
plt.xlabel('Number of Encoder Jumps')
plt.ylabel('Density')
plt.title('Distribution of Encoder Jump Counts')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: RSS vs Encoder Jumps scatter plot
plt.subplot(1, 3, 3)
# Take a sample for clearer visualization
sample_idx = np.random.choice(len(rss_samples), 1000, replace=False)
plt.scatter(jump_counts[sample_idx], rss_samples[sample_idx], alpha=0.6, s=1)
plt.xlabel('Number of Encoder Jumps')
plt.ylabel('RSS Value')
plt.title('RSS vs Encoder Jump Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Function for easy reuse
def get_random_rss(components_dict, encoder_std, max_jumps, seed=None):
    """Get a single random RSS value"""
    if seed is not None:
        np.random.seed(seed)
    
    # Gaussian components
    gaussian_sum_squares = 0
    for component, std_dev in components_dict.items():
        random_value = np.random.normal(0, std_dev)
        gaussian_sum_squares += random_value**2
    
    # Encoder jumps
    num_jumps = np.random.randint(0, max_jumps + 1)
    encoder_sum_squares = num_jumps * (encoder_std**2)
    
    total_rss = np.sqrt(gaussian_sum_squares + encoder_sum_squares)
    return total_rss, num_jumps

# Example usage
print(f"\nRandom RSS examples:")
for i in range(5):
    rss_val, jumps = get_random_rss(RSS_components, encoder_jump_std, max_encoder_jumps)
    print(f"Sample {i+1}: RSS={rss_val:.3f}, Encoder jumps={jumps}")

mean_components = {
    'pointing_model_residuals' : 0.5
}
RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad
sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
# moon_ilum_fraction = 0.9

moon_ilum_fraction = 0.99
centroid_err = 0.25*17*(1-moon_ilum_fraction)
title_append = f'{title_append}_{imu_chosen}'

errors_chosen = {
        'err_r_host': err_r_host_default,
        'err_r_target': err_r_target_default,
        'err_att_host': err_att_host_default,
        'sum_mean_errors': sum_mean_errors,
        'rss_random_errors': RSS_random_errors,
    }

randomize_magnitude = 0

ii_scans = list(np.arange(0,nr_inputs,1))

error_text_box = f'''PEB Components Used:
r_h : {np.linalg.norm(errors_chosen['err_r_host']):.1f} m
r_t : {np.linalg.norm(errors_chosen['err_r_target']):.1f} m
att_h : {np.linalg.norm(errors_chosen['err_att_host'])*1e3:.1f} mrad
rss : {np.linalg.norm(errors_chosen['rss_random_errors'])*1e3:.1f} mrad
mean : {np.linalg.norm(errors_chosen['sum_mean_errors'])*1e3:.1f} mrad'''

pe_mo = []
err_att = []
non_colin_angle = []
t_gap = []
# loop over angles:
mean_components = {
    'pointing_model_residuals' : 0.5,
    'centroid_detection' : centroid_err
}
nr_recalcs = 3
for jj in range(nr_recalcs):
    for ii in range(nr_inputs, nrows-1):
        if nr_inputs == 2:
            t_gap_ii = t_gps[ii-1] - t_gps[0]
            
            r_host = r_host_full[[0,ii+1],:]
            r_target = r_target_full[[0,ii+1],:]
            quat_eci2bf = q_bf_full[[0, ii+1], :4]
        elif nr_inputs == 6:
            t_gap_ii = t_gps[ii-1] - t_gps[0]
            ii_gap = int(ii/(nr_inputs-1))
            ii_used = [0, ii_gap*1, ii_gap*2, ii_gap*3, ii_gap*4, ii]
            # ii_used = np.arange(0,nr_inputs,1)*ii_gap
            r_host = r_host_full[ii_used, :]
            r_target = r_target_full[ii_used, :]
            quat_eci2bf = q_bf_full[ii_used, :4]
        ea_eci2bf = np.array([conv.convert_quat2ea(q_ii) for q_ii in quat_eci2bf])
        # title_append = f'{title_append}_moon{moon_ilum_fraction}'

        RSS_random_errors = np.linalg.norm(list(RSS_components.values()))*1e-3 # rad
        sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
        total_random_error_input = RSS_random_errors + sum_mean_errors
        total_random_error_input = total_random_error_input # [rad]
        # run-scan # TODO integrate QUEST into simulating moon scan code
        ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle, fct_used = moon_scan.simulate_ooc(
            ii_scans = ii_scans,
            function_option = function_option,
            r_host = r_host, 
            r_target = r_target, 
            ea_eci2bf_command_all = ea_eci2bf,
            quat_mounting_offset_t = quat_mounting_offset,
            quat_mounting_offset_c = quat_mounting_offset_known,
            manual_error_dict=errors_chosen,
            check_non_colin=1,
            randomize_error_direction = 1,
            centroid_dirction_randomizer = 1,
            randomize_magnitude = randomize_magnitude,
            seed_used = ii*(jj+1),
            ii_loop = ii,
            add_trolley_tilt= False,
            trolley_angle = 0
        )
            # store
        pe_mo.append(np.max(po_ii)/1e3) # mrad
        non_colin_angle.append(np.rad2deg(obs_angle)) # deg
        t_gap.append(t_gap_ii)
data_df = pd.DataFrame.from_dict({
    'pe_mo' : pe_mo,
    'ang_sep' : non_colin_angle,
    't_gap' : t_gap
})
data_df = data_df.dropna()
angle_bins = []
pe_3sig = []
d_angle = 1
pe_2sig = []
t_bins = []
for angle_0 in range(0, 180, d_angle): # get 3-sigma PE
    angle_range = [angle_0, angle_0 + d_angle]
    df_sliced_low = data_df[data_df['ang_sep'] > angle_range[0]]
    df_sliced_high = df_sliced_low[df_sliced_low['ang_sep'] < angle_range[1]]
    angle_bins.append(np.average(angle_range))
    t_bins.append(df_sliced_high['t_gap'].quantile(1))
    pe_3sig.append(df_sliced_high['pe_mo'].quantile(0.997))
    pe_2sig.append(df_sliced_high['pe_mo'].quantile(0.95))

importlib.reload(io)
df_dict = {
    'ang_sep' : angle_bins,
    'pe_3sig': pe_3sig,
    'pe_2sig': pe_2sig,
    't_gap_s' : t_bins
}
# output_df = pd.DataFrame.from_dict(df_dict)
df_title = f'ooc_{fct_used}_{title_append}_sample{nr_inputs}_datapt{n_samples}'
output_df = io.save_dict_2_csv(df_dict, df_title, subfolder = storage_folder)
# n_samples, datapt_used, datapt_spacing, weights?
dr_h = np.linalg.norm(err_r_host_default) # m
dr_t = np.linalg.norm(err_r_target_default) # m
d_theta = np.linalg.norm(np.deg2rad(err_att_host_default)*1e3) # mrad
d_rand = RSS_random_errors*1e3 # mrad
#%%
if 1:

    importlib.reload(plt_util)
    plot_all = 1
    plot_sep = 0
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    if plot_all:
        append_title = f'\n 3sig PEB used: Pos H = {dr_h:.1f} m; T = {dr_t:.1f} m; Att H : {d_theta:.2f} mrad; Other: {d_rand:.2f} mrad; \n'
    elif plot_sep:
        # plot and save 1 by 1 
        append_title = ''
    f, ax = plt.subplots()
    ax.scatter(non_colin_angle, pe_mo, color = 'g', alpha = 0.1)
    ax.plot(angle_bins, pe_3sig, label = '3-sigma', c = 'r')
    ax.plot(angle_bins, pe_2sig, label = '2-sigma', c = 'm')
    ax.plot(ax.get_xlim(), [3,3], label = '3 mrad', c = 'y')
    ax.legend(bbox_to_anchor=(1, 0.901))
    ax.set_xlabel('Angle between scanned LOS [deg]', fontweight = 'bold')
    ax.set_ylabel('Mounting Offset Resolution Error [mrad]', fontweight = 'bold')
    ax.grid('on')
    #ax.set_xlim([5, 170])
        # ax.set_xscale('log')
        # ax.set_yscale('log')
    #ax.set_ylim([-0.1,20])
    title = f'''Phase A GROUND. Initial MOQ Error : {error_mounting_offset*1e3:.1f} mrad. 
    Centroid Det : {centroid_err:.2f} mrad; Moon Illum: {moon_ilum_fraction*100:.0f}%
    IMU: {imu_chosen} Algorithm used: '''
    f.suptitle(f'{title}{fct_used.upper()}')
    # Reformat to get angle from 0:90 for non-colinearity
    ax.text(140.85, 0.2, 
            error_text_box,
            fontsize = 12,            
            bbox = {'boxstyle': 'square',
                    'facecolor':'peachpuff',                    
                    }
            )
    figname = f'ooc_{fct_used}_{title_append}_sample{nr_inputs}_datapt{n_samples}'
    bplt.savefig(f, figname, subfolder =storage_folder, y_coord_tag = -4)
    


plt.show()
# %%
