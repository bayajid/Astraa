# Improved version addressing the identified issues

import numpy as np
import pandas as pd
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
import time
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
imu_chosen = 'iNAT-RQT-400x'
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
error_mounting_offset = 1.5e-3 # 65e-3 # mrad, Mounting offset error

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
    
err_att_host_default = np.array([att_err*17e-3]) # rad # RANDOMIZE DIRECTION?


def get_random_rss_sample(components_dict, encoder_std, max_jumps, seed=None):
    """Get a single random RSS value using Monte Carlo approach"""
    if seed is not None:
        np.random.seed(seed)
    
    # Gaussian components
    gaussian_sum_squares = 0
    for component, std_dev in components_dict.items():
        random_value = np.random.normal(0, std_dev)
        gaussian_sum_squares += random_value**2
    
    # Random encoder jumps
    num_jumps = np.random.randint(0, max_jumps + 1)
    encoder_sum_squares = num_jumps * (encoder_std**2)
    
    total_rss = np.sqrt(gaussian_sum_squares + encoder_sum_squares)
    return total_rss, num_jumps

# Key improvements to the original code:

# 1. Use actual Monte Carlo RSS values instead of fixed norm
RSS_components = {
    'gimbal_control': 0.4,
    'thermal': 0.8,
    'other': 0.1,
}
encoder_jump_std = 0.767
max_encoder_jumps = 57
moon_ilum_fraction = 0.99
centroid_err = 0.25*17*(1-moon_ilum_fraction)
title_append = f'{title_append}_{imu_chosen}'

mean_components = {
    'pointing_model_residuals': 0.5,
    'centroid_detection': centroid_err  # varies with moon illumination
}
sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3

# 2. Improved loop structure with proper random sampling
pe_mo = []
err_att = []
non_colin_angle = []
t_gap = []
rss_values_used = []  # Track actual RSS values used
encoder_jumps_used = []  # Track encoder jumps

nr_recalcs = 5  # Increased for better statistics
np.random.seed(42)  # Set overall seed for reproducibility
randomize_magnitude = 0
ii_scans = list(np.arange(0,nr_inputs,1))

# for jj in range(nr_recalcs):
#     print(f"Starting recalculation {jj+1}/{nr_recalcs}")
    
#     for ii in range(nr_inputs, nrows-1):
#         # Generate unique seed for this iteration
#         iteration_seed = 1000 + jj * 10000 + ii
        
#         # 3. Use actual Monte Carlo RSS sampling
#         RSS_random_errors_sample, encoder_jumps = get_random_rss_sample(
#             RSS_components, encoder_jump_std, max_encoder_jumps, seed=iteration_seed
#         )
#         RSS_random_errors_sample *= 1e-3  # Convert to radians
        
#         # Track values for analysis
#         rss_values_used.append(RSS_random_errors_sample * 1e3)  # Store in mrad
#         encoder_jumps_used.append(encoder_jumps)
        
#         # 4. Improved error dictionary with varying RSS
#         errors_chosen = {
#             'err_r_host': err_r_host_default,
#             'err_r_target': err_r_target_default,
#             'err_att_host': err_att_host_default,
#             'sum_mean_errors': sum_mean_errors,
#             'rss_random_errors': RSS_random_errors_sample,  # Now varies per iteration
#         }

#         if nr_inputs == 2:
#             t_gap_ii = t_gps[ii-1] - t_gps[0]
            
#             r_host = r_host_full[[0,ii+1],:]
#             r_target = r_target_full[[0,ii+1],:]
#             quat_eci2bf = q_bf_full[[0, ii+1], :4]
#         elif nr_inputs == 6:
#             t_gap_ii = t_gps[ii-1] - t_gps[0]
#             ii_gap = int(ii/(nr_inputs-1))
#             ii_used = [0, ii_gap*1, ii_gap*2, ii_gap*3, ii_gap*4, ii]
#             # ii_used = np.arange(0,nr_inputs,1)*ii_gap
#             r_host = r_host_full[ii_used, :]
#             r_target = r_target_full[ii_used, :]
#             quat_eci2bf = q_bf_full[ii_used, :4]
#         ea_eci2bf = np.array([conv.convert_quat2ea(q_ii) for q_ii in quat_eci2bf])

#         # sum_mean_errors = np.sum(list(mean_components.values())) * 1e-3
#         total_random_error_input = RSS_random_errors_sample + sum_mean_errors
#         total_random_error_input = total_random_error_input # [rad]
        
        
#         # Rest of the simulation code remains the same...
#         # [Original simulation code here]
        
#         # Example of how to continue:
#         ae_moon_commanded_all, ae_moon_true_all, quat_resolved, po_ii, obs_angle, fct_used = moon_scan.simulate_ooc(
#             ii_scans = ii_scans,
#             function_option = function_option,
#             r_host = r_host, 
#             r_target = r_target, 
#             ea_eci2bf_command_all = ea_eci2bf,
#             quat_mounting_offset_t = quat_mounting_offset,
#             quat_mounting_offset_c = quat_mounting_offset_known,
#             manual_error_dict=errors_chosen,
#             check_non_colin=1,
#             randomize_error_direction = 1,
#             centroid_dirction_randomizer = 1,
#             randomize_magnitude = randomize_magnitude,
#             seed_used = ii*(jj+1),
#             ii_loop = ii,
#             add_trolley_tilt= False,
#             trolley_angle = 0
#         )
#             # store
#         pe_mo.append(np.max(po_ii)/1e3) # mrad
#         non_colin_angle.append(np.rad2deg(obs_angle)) # deg
#         t_gap.append(t_gap_ii)

# # 5. Enhanced data analysis with RSS tracking
# data_df = pd.DataFrame.from_dict({
#     'pe_mo': pe_mo,
#     'ang_sep': non_colin_angle,
#     't_gap': t_gap,
#     'rss_used': rss_values_used,
#     'encoder_jumps': encoder_jumps_used,
#     'recalc_id': np.repeat(range(nr_recalcs), len(pe_mo)//nr_recalcs)  # Track which recalc
# })


# 6. Improved statistical analysis
def analyze_results_by_angle(df, d_angle=5, quantiles=[0.95, 0.997], min_samples=3):
    """
    Enhanced analysis function to bin results by angle and compute statistics
    
    Parameters:
    df: DataFrame with columns ['pe_mo', 'ang_sep', 't_gap', 'rss_used', 'encoder_jumps']
    d_angle: angle bin width in degrees
    quantiles: list of quantiles to compute
    min_samples: minimum samples per bin to include in results
    
    Returns:
    DataFrame with binned analysis results
    """
    print(f"Analyzing results by angle with {d_angle}° bins...")
    
    results = {
        'angle_bins': [],
        'sample_counts': [],
        'mean_rss': [],
        'std_rss': [],
        'mean_encoder_jumps': [],
        'mean_pe': [],
        'std_pe': []
    }
    
    # Add quantile columns with consistent naming
    for q in quantiles:
        if q == 0.95:
            results['pe_95th'] = []
        elif q == 0.997:
            results['pe_997th'] = []
        else:
            results[f'pe_{int(q*1000)}th'] = []
    
    angle_ranges = range(0, 180, d_angle)
    
    for angle_0 in angle_ranges:
        angle_range = [angle_0, angle_0 + d_angle]
        
        # Filter data for this angle bin
        mask = (df['ang_sep'] >= angle_range[0]) & (df['ang_sep'] < angle_range[1])
        df_slice = df[mask]
        
        # Store angle bin center
        results['angle_bins'].append(np.mean(angle_range))
        results['sample_counts'].append(len(df_slice))
        
        if len(df_slice) >= min_samples:
            # Basic statistics
            results['mean_rss'].append(df_slice['rss_used'].mean())
            results['std_rss'].append(df_slice['rss_used'].std())
            results['mean_encoder_jumps'].append(df_slice['encoder_jumps'].mean())
            results['mean_pe'].append(df_slice['pe_mo'].mean())
            results['std_pe'].append(df_slice['pe_mo'].std())
            
            # Quantiles with consistent naming
            for q in quantiles:
                if q == 0.95:
                    results['pe_95th'].append(df_slice['pe_mo'].quantile(q))
                elif q == 0.997:
                    results['pe_997th'].append(df_slice['pe_mo'].quantile(q))
                else:
                    results[f'pe_{int(q*1000)}th'].append(df_slice['pe_mo'].quantile(q))
        else:
            # Fill with NaN for insufficient data
            for key in ['mean_rss', 'std_rss', 'mean_encoder_jumps', 'mean_pe', 'std_pe']:
                results[key].append(np.nan)
            for q in quantiles:
                if q == 0.95:
                    results['pe_95th'].append(np.nan)
                elif q == 0.997:
                    results['pe_997th'].append(np.nan)
                else:
                    results[f'pe_{int(q*1000)}th'].append(np.nan)
    
    return pd.DataFrame(results)

# 7. Parallel processing option for large simulations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def run_single_recalc(recalc_params):
    """
    Function to run a single recalculation - designed for parallel processing
    
    Parameters:
    recalc_params: tuple containing (recalc_id, nr_inputs, nrows, base_errors)
    
    Returns:
    dict with results from this recalculation
    """
    recalc_id, nr_inputs, nrows, base_errors = recalc_params
    print(f"  Running recalculation {recalc_id + 1}")
    
    # Local results for this recalculation
    local_results = {
        'recalc_id': [],
        'pe_mo': [],
        'non_colin_angle': [],
        't_gap': [],
        'rss_values': [],
        'encoder_jumps': [],
        'iteration_id': []
    }
    
    for ii in range(nr_inputs, min(nrows-1, nr_inputs + 200)):  # Limit for demo
        # Generate unique seed for this iteration
        iteration_seed = 1000 + recalc_id * 10000 + ii
        
        # Generate RSS sample with varying encoder jumps
        RSS_sample, encoder_jumps = get_random_rss_sample(
            RSS_components, encoder_jump_std, max_encoder_jumps, seed=iteration_seed
        )
        RSS_sample *= 1e-3  # Convert to radians
        
        # Create error dictionary for this iteration
        errors_chosen = {
            'err_r_host': base_errors['err_r_host'],
            'err_r_target': base_errors['err_r_target'], 
            'err_att_host': base_errors['err_att_host'],
            'sum_mean_errors': base_errors['sum_mean_errors'],
            'rss_random_errors': RSS_sample,
        }
        
        # Mock time gap (would be real data in actual implementation)
        t_gap_ii = ii * 0.1  # 0.1 second intervals
        r_host = r_host_full[[0,ii+1],:]
        r_target = r_target_full[[0,ii+1],:]
        quat_eci2bf = q_bf_full[[0, ii+1], :4]
        ea_eci2bf = np.array([conv.convert_quat2ea(q_ii) for q_ii in quat_eci2bf])
    
        
        # Run the simulation (this would be the actual moon scan simulation)
        try:
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
            seed_used = iteration_seed ,
            ii_loop = ii,
            add_trolley_tilt= False,
            trolley_angle = 0)
            
            # Store results
            local_results['recalc_id'].append(recalc_id)
            local_results['pe_mo'].append(np.max(po_ii) / 1e-3)  # Convert to mrad
            local_results['non_colin_angle'].append(np.rad2deg(obs_angle))
            local_results['t_gap'].append(t_gap_ii)
            local_results['rss_values'].append(RSS_sample * 1e3)  # Store in mrad
            local_results['encoder_jumps'].append(encoder_jumps)
            local_results['iteration_id'].append(ii)
            
        except Exception as e:
            print(f"    Error in iteration {ii}: {e}")
            continue
    
    return local_results

def run_improved_simulation(nr_recalcs=3, use_parallel=True, max_workers=None):
    """
    Main simulation function that integrates everything
    
    Parameters:
    nr_recalcs: number of recalculations (Monte Carlo runs)
    use_parallel: whether to use parallel processing
    max_workers: number of parallel workers (None = auto-detect)
    
    Returns:
    tuple: (raw_results_df, binned_analysis_df)
    """
    print(f"Starting improved simulation with {nr_recalcs} recalculations...")
    
    # Prepare base error parameters
    base_errors = {
        'err_r_host': err_r_host_default,
        'err_r_target': err_r_target_default,
        'err_att_host': err_att_host_default,
        'sum_mean_errors': sum_mean_errors,
    }
    
    # Prepare parameters for each recalculation
    recalc_params_list = [
        (jj, nr_inputs, nrows, base_errors) 
        for jj in range(nr_recalcs)
    ]
    
    start_time = time.time()
    
    if use_parallel and nr_recalcs > 1:
        # Parallel execution
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count() // 2, nr_recalcs)
        
        print(f"Using parallel processing with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_list = list(executor.map(run_single_recalc, recalc_params_list))
    else:
        # Sequential execution
        print("Using sequential processing...")
        results_list = [run_single_recalc(params) for params in recalc_params_list]
    
    execution_time = time.time() - start_time
    print(f"Simulation completed in {execution_time:.2f} seconds")
    
    # Combine all results into a single DataFrame
    combined_data = {
        'pe_mo': [],
        'ang_sep': [],  # non_colin_angle
        't_gap': [],
        'rss_used': [],  # rss_values
        'encoder_jumps': [],
        'recalc_id': [],
        'iteration_id': []
    }
    
    for result_dict in results_list:
        for key in combined_data.keys():
            if key == 'ang_sep':
                combined_data[key].extend(result_dict['non_colin_angle'])
            elif key == 'rss_used':
                combined_data[key].extend(result_dict['rss_values'])
            else:
                combined_data[key].extend(result_dict[key])
    
    # Create DataFrame
    data_df = pd.DataFrame(combined_data)
    data_df = data_df.dropna()  # Remove any invalid entries
    
    print(f"Generated {len(data_df)} valid data points")
    print(f"Angle separation range: {data_df['ang_sep'].min():.1f}° - {data_df['ang_sep'].max():.1f}°")
    print(f"PE range: {data_df['pe_mo'].min():.3f} - {data_df['pe_mo'].max():.3f} mrad")
    
    # Analyze results by angle bins
    binned_results = analyze_results_by_angle(data_df, d_angle=5)
    
    return data_df, binned_results

def plot_improved_results(data_df, binned_results, save_plots=False):
    """
    Create comprehensive plots of the simulation results
    """
    if len(data_df) == 0:
        print("No data to plot!")
        return None
    
    # Create figure with explicit size
    plt.style.use('default')  # Reset any custom styles
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Scatter plot of all data points
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(data_df['ang_sep'], data_df['pe_mo'], 
                         c=data_df['rss_used'], alpha=0.7, s=10, cmap='viridis')
    ax1.set_xlabel('Angle Separation [deg]')
    ax1.set_ylabel('Mounting Offset Error [mrad]')
    ax1.set_title('All Simulation Points\n(colored by RSS error)')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('RSS Error [mrad]')
    
    # Plot 2: Binned quantiles (only if we have binned data)
    ax2 = plt.subplot(2, 3, 2)
    if not binned_results.empty and 'pe_95th' in binned_results.columns:
        valid_mask = (~np.isnan(binned_results['pe_95th'])) & (~np.isnan(binned_results['pe_997th']))
        if valid_mask.sum() > 0:
            angles = binned_results['angle_bins'][valid_mask]
            pe_95 = binned_results['pe_95th'][valid_mask]
            pe_997 = binned_results['pe_997th'][valid_mask]
            
            ax2.plot(angles, pe_95, 'b-', linewidth=2, marker='o', label='95th percentile')
            ax2.plot(angles, pe_997, 'r-', linewidth=2, marker='s', label='99.7th percentile (3σ)')
            ax2.axhline(y=3.0, color='orange', linestyle='--', linewidth=2, label='3 mrad requirement')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor binned analysis', 
                    transform=ax2.transAxes, ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'No binned data\navailable', 
                transform=ax2.transAxes, ha='center', va='center')
    
    ax2.set_xlabel('Angle Separation [deg]')
    ax2.set_ylabel('Mounting Offset Error [mrad]')
    ax2.set_title('Performance vs Angle Separation')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample count per bin
    ax3 = plt.subplot(2, 3, 3)
    if not binned_results.empty:
        ax3.bar(binned_results['angle_bins'], binned_results['sample_counts'], 
                width=8, alpha=0.7, color='lightblue', edgecolor='navy')
        ax3.set_title('Sample Distribution by Angle Bin')
    else:
        ax3.text(0.5, 0.5, 'No binned data', transform=ax3.transAxes, ha='center', va='center')
    ax3.set_xlabel('Angle Separation [deg]')
    ax3.set_ylabel('Number of Samples')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RSS error distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(data_df['rss_used'], bins=30, alpha=0.7, color='green', edgecolor='black')
    mean_rss = data_df['rss_used'].mean()
    ax4.axvline(mean_rss, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_rss:.2f}')
    ax4.set_xlabel('RSS Error [mrad]')
    ax4.set_ylabel('Frequency')
    ax4.set_title('RSS Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Encoder jumps distribution
    ax5 = plt.subplot(2, 3, 5)
    max_jumps = int(data_df['encoder_jumps'].max())
    bins = range(0, max_jumps + 2)
    ax5.hist(data_df['encoder_jumps'], bins=bins, 
             alpha=0.7, color='orange', edgecolor='black')
    mean_jumps = data_df['encoder_jumps'].mean()
    ax5.axvline(mean_jumps, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_jumps:.1f}')
    ax5.set_xlabel('Number of Encoder Jumps')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Encoder Jumps Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance by recalculation
    ax6 = plt.subplot(2, 3, 6)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, recalc_id in enumerate(sorted(data_df['recalc_id'].unique())):
        subset = data_df[data_df['recalc_id'] == recalc_id]
        color = colors[i % len(colors)]
        ax6.scatter(subset['ang_sep'], subset['pe_mo'], 
                   alpha=0.6, s=8, label=f'Recalc {recalc_id + 1}', color=color)
    ax6.set_xlabel('Angle Separation [deg]')
    ax6.set_ylabel('Mounting Offset Error [mrad]')
    ax6.set_title('Results by Recalculation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('improved_simulation_results.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'improved_simulation_results.png'")
    
    # Force display
    plt.show()#(block=False)
    plt.pause(0.1)
    
    return fig

def main():
    print("=== Running Complete Improved Simulation ===")
    
    # Run the simulation with both functions properly integrated
    raw_data, binned_analysis = run_improved_simulation(
        nr_recalcs=3,           # Number of Monte Carlo runs
        use_parallel=False,     # Sequential for debugging
        max_workers=2           # Number of parallel workers
    )
    
    if len(raw_data) == 0:
        print("No data generated - check simulation parameters")
        return
    
    # Display summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total data points: {len(raw_data)}")
    print(f"RSS error mean: {raw_data['rss_used'].mean():.3f} ± {raw_data['rss_used'].std():.3f} mrad")
    print(f"PE mean: {raw_data['pe_mo'].mean():.3f} ± {raw_data['pe_mo'].std():.3f} mrad")
    print(f"Encoder jumps mean: {raw_data['encoder_jumps'].mean():.1f}")
    print(f"Angle range: {raw_data['ang_sep'].min():.1f}° to {raw_data['ang_sep'].max():.1f}°")
    
    # Show binned analysis sample (with error checking)
    if not binned_analysis.empty and len(binned_analysis) > 0:
        print(f"\n=== Binned Analysis (first 5 bins) ===")
        # Only show columns that exist
        available_cols = ['angle_bins', 'sample_counts', 'mean_pe', 'mean_rss']
        if 'pe_95th' in binned_analysis.columns:
            available_cols.append('pe_95th')
        if 'pe_997th' in binned_analysis.columns:
            available_cols.append('pe_997th')
        
        display_cols = [col for col in available_cols if col in binned_analysis.columns]
        print(binned_analysis.head(5)[display_cols].round(3))
    else:
        print("No binned analysis data available")
    
    # Create comprehensive plots
    try:
        fig = plot_improved_results(raw_data, binned_analysis, save_plots=False)
        if fig:
            print("Plots displayed successfully")
        else:
            print("Failed to create plots")
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    print("=== Simulation Complete ===")
    return raw_data, binned_analysis
if __name__ == "__main__":
    raw_data, binned_analysis = main()