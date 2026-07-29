
# %%
"""
#     Comparison between SLERP and HERMITE Interpolation for Quaternions.
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from scipy.interpolate import CubicSpline
# import quaternion_slerp_squad as quat_slerp

# # ====================== QUATERNION MATH ======================
# # ========
# datadir = os.path.join(os.path.dirname(__file__), 'output_data/tables/custom_quatpred')
# true_file = os.path.join(datadir, 'true_quat_custom.csv')

# df = pd.read_csv(true_file)
# # sign fix
# q_raw = df.iloc[:, 1:5].values.copy()


# # ====================== USER SETTINGS ======================
# fix_sign_swap       = False         # ← Set False to keep raw sign jumps
# desired_freq_hz     = 1        # ← Your desired output update rate (e.g. 100, 50, 20 Hz)
# latency_seconds     = 5             # ← Added delay: 0.0 = no delay, 0.045 = 45 ms latency
# # ===========================================================
# if fix_sign_swap:
#     for i in range(1, len(q_raw)):
#         if np.dot(q_raw[i], q_raw[i-1]) < 0:
#             q_raw[i:] = -q_raw[i:]   # cumulative flip from first sign jump onward
#             break                    # usually only one big jump per dataset
# df.iloc[:, 1:5] = q_raw

# t_high = df.iloc[:, 0].values
# q_high = df.iloc[:, 1:5].values  # [w, x, y, z]
# qdot_high = df.iloc[:, 5:9].values

# dot = np.sum(q_high[1:] * q_high[:-1], axis=1)
# print("Biggest sign jump at t =", t_high[1:][np.argmin(dot)])

# # Diagnostics
# freq_true = 1.0 / np.mean(np.diff(t_high))
# print(f"True data frequency: {freq_true:.3f} Hz → dt = {1/freq_true*1000:.3f} ms")

# omegas = np.array([quat_slerp.angular_velocity_from_qdot(q, qd) for q, qd in zip(q_high, qdot_high)])
# print(f"Max angular rate: {np.max(np.linalg.norm(omegas, axis=1))*180/np.pi:.2f} deg/s")
# # Normalize high-rate truth
# q_high = quat_slerp.normalize(q_high)

# ## ====================== DOWNSAMPLE + LATENCY ======================
# dt_output = 1.0 / desired_freq_hz

# # Generate output timestamps: start after latency, then regular intervals
# t_start = t_high[0] + latency_seconds
# t_end   = t_high[-1]

# # Build regular timestamp grid
# t_key = np.arange(t_start, t_end + dt_output/2, dt_output)   # +dt/2 to include last if close

# # Map each desired time to nearest true sample (left-side for causality)
# indices = np.searchsorted(t_high, t_key, side='right') - 1
# indices = np.clip(indices, 0, len(t_high)-1)

# # Final delayed + downsampled data
# t_key       = t_high[indices]
# q_key       = q_high[indices]
# qdot_key    = qdot_high[indices]

# # Print result
# actual_freq = 1.0 / np.mean(np.diff(t_key)) if len(t_key)>1 else 0
# print(f"Output: {len(t_key)} samples @ {actual_freq:.2f} Hz "
#       f"with {latency_seconds*1000:.1f} ms latency")
# print(f"First key time : {t_key[0]:.4f} s (true first was {t_high[0]:.4f} s)")
# print(f"Last key time  : {t_key[-1]:.4f} s")

# # Evaluation: use ALL original high-rate points as truth
# t_eval = t_high
# q_true_eval = q_high

# # Build interpolators from low-rate keys
# interp_slerp    = quat_slerp.make_slerp_interpolator(t_key, q_key, qdot_key, fix_sign_swap)
# interp_hermite  = quat_slerp.make_hermite_interpolator(t_key, q_key, qdot_key, fix_sign_swap)
# interp_cubic    = quat_slerp.make_cubic_spline_interpolator(t_key, q_key, fix_sign_swap)
# # Evaluate
# q_slerp             = interp_slerp(t_eval)
# q_hermite           = interp_hermite(t_eval)
# q_cubicspline       = interp_cubic(t_eval)
# # Angular error in microradians
# # def angular_error_micro(q_est, q_ref):
# #     q_est = quat_slerp.normalize(q_est)
# #     q_ref = quat_slerp.normalize(q_ref)
# #     dot = (np.sum(q_est * q_ref, axis=1))
# #     dot = np.clip(dot, -1.0, 1.0)
# #         # For small angles, use stable approximation
# #     small = dot > 0.999999
# #     angle_rad = np.empty_like(dot)

# #     # Standard
# #     angle_rad[~small] = 2 * np.arccos(np.abs(dot[~small]))

# #     # Stable small-angle
# #     angle_rad[small] = 2 * np.sqrt(2 * (1 - np.abs(dot[small])))
# #     #angle_rad = 2 * np.arccos(dot)
# #     return angle_rad * 1e6

# err_slerp       = quat_slerp.quat_angle_error(q_slerp, q_true_eval)
# err_hermite     = quat_slerp.quat_angle_error(q_hermite, q_true_eval)
# err_cubicspline = quat_slerp.quat_angle_error(q_cubicspline, q_true_eval)
# # ====================== STATISTICS ======================
# print("\n" + "="*80)
# print("      REAL QUATERNION INTERPOLATION COMPARISON (vs True High-Rate Data)")
# print("="*80)
# print(f"Keyframes: {len(t_key)} @ ~{1/np.mean(np.diff(t_key)):.1f} Hz")
# print(f"Truth:     {len(t_high)} points @ ~{1/np.mean(np.diff(t_high)):.1f} Hz")
# print("Error metrics (µrad):")
# print("-"*80)
# print(f"{'Method':<12} {'Mean (µrad)':>12} {'RMS (µrad)':>12} {'Max (µrad)':>12} {'99th %ile':>12}")
# print("-"*80)
# for name, err in [('SLERP', err_slerp), ('Hermite+ω', err_hermite), ('CubicSpline', err_cubicspline)]:
#     print(f"{name:<12} {np.mean(err):12.2f} {np.sqrt(np.mean(err**2)):12.2f} {np.max(err):12.1f} {np.percentile(err, 99):12.1f}")
# print("="*80)
# # ====================== PLOTTING ======================
# fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)

# components = ['w', 'x', 'y', 'z']
# colors = {'true': 'black', 'slerp': 'tab:blue', 'hermite': 'tab:purple'}

# # Plot components
# for i in range(4):
#     ax = axes[i]
#     ax.plot(t_high, q_high[:, i], '.', color=colors['true'], markersize=3, alpha=0.7, label='True (high-rate)')
#     ax.plot(t_key,  q_key[:, i], 'o', color='red', markersize=6, label='Keyframes' if i==0 else None)
#     ax.plot(t_eval, q_slerp[:, i], '-', color=colors['slerp'], linewidth=1.5, label='SLERP')
#     ax.plot(t_eval, q_hermite[:, i], '-', color=colors['hermite'], linewidth=2.2, label='Hermite (w/ ω)')
#     ax.plot(t_eval, q_cubicspline[:, i], '--', color='tab:green', linewidth=1.5, label='Cubic Spline')
#     ax.set_ylabel(f'q[{components[i]}]')
#     ax.grid(True, alpha=0.3)
#     if i == 0:
#         ax.legend(fontsize=9)

# # Norm plot
# axes[4].plot(t_high, np.linalg.norm(q_high, axis=1), '.', color='black', alpha=0.7, label='True')
# axes[4].plot(t_eval, np.linalg.norm(q_slerp, axis=1), color=colors['slerp'], label='SLERP')
# axes[4].plot(t_eval, np.linalg.norm(q_hermite, axis=1), color=colors['hermite'], label='Hermite')
# axes[4].plot(t_eval, np.linalg.norm(q_cubicspline, axis=1), '--', color='tab:green', label='Cubic Spline')
# axes[4].axhline(1.0, color='red', linestyle='--', alpha=0.6)
# axes[4].set_ylabel('||q||')
# axes[4].legend(fontsize=9)
# axes[4].grid(True, alpha=0.3)

# # Error plot
# axes[5].plot(t_eval, err_slerp, color=colors['slerp'], linewidth=2, label=f'SLERP  | Max: {err_slerp.max():.1f} µrad')
# #axes[5].plot(t_eval, err_hermite, color=colors['hermite'], linewidth=2.5, label=f'Hermite | Max: {err_hermite.max():.1f} µrad')
# axes[5].plot(t_eval, err_cubicspline, '--', color='tab:green', linewidth=1.5, label=f'Cubic Spline | Max: {err_cubicspline.max():.1f} µrad')
# axes[5].set_ylabel('Angular Error [µrad]')
# axes[5].set_xlabel('Time [s]')
# axes[5].set_yscale('log')
# axes[5].grid(True, alpha=0.3)
# axes[5].legend(fontsize=10)

# plt.suptitle('Quaternion Interpolation: SLERP vs Hermite (with Angular Velocity) vs Cubicspline\n'
#              'Evaluated against original high-rate ground truth (no fake interpolation!)', fontsize=14, y=0.98)
# plt.tight_layout()
# plt.savefig('quaternion_interpolation_real_comparison.png', dpi=200, bbox_inches='tight')
# plt.show()

# ====================== PARAMETER SWEEP ======================
"""
    Complete Quaternion Interpolation Parameter Sweep
    Compares SLERP, Hermite, and Cubic Spline across multiple frequencies and latencies
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
import quaternion_slerp_squad as quat_slerp

# ====================== CONFIGURATION ======================
# File paths
datadir = os.path.join(os.path.dirname(__file__), 'output_data/tables/rocketlab_march_quatpred')
true_file = os.path.join(datadir, 'true_quat_rocketlab_march.csv')

# Parameter sweep settings
desired_freq_hz_list = [1, 2, 5, 10]
latency_seconds_list = [0, 1, 2, 3, 4]
fix_sign_swap = False

# Output settings
results_csv = 'interpolation_parameter_sweep_results.csv'
summary_plot = 'parameter_sweep_summary.png'

# ====================== LOAD DATA ======================
print("="*80)
print("LOADING DATA")
print("="*80)

df = pd.read_csv(true_file)
q_raw = df.iloc[:, 1:5].values.copy()

# Sign fix if enabled
if fix_sign_swap:
    for i in range(1, len(q_raw)):
        if np.dot(q_raw[i], q_raw[i-1]) < 0:
            q_raw[i:] = -q_raw[i:]
            break

df.iloc[:, 1:5] = q_raw

t_high = df.iloc[:, 0].values
q_high = df.iloc[:, 1:5].values
qdot_high = df.iloc[:, 5:9].values

# Diagnostics
freq_true = 1.0 / np.mean(np.diff(t_high))
print(f"True data frequency: {freq_true:.3f} Hz → dt = {1/freq_true*1000:.3f} ms")

omegas = np.array([quat_slerp.angular_velocity_from_qdot(q, qd) for q, qd in zip(q_high, qdot_high)])
print(f"Max angular rate: {np.max(np.linalg.norm(omegas, axis=1))*180/np.pi:.2f} deg/s")

# Normalize high-rate truth
q_high = quat_slerp.normalize(q_high)

# Evaluation setup (use ALL original high-rate points as truth)
t_eval = t_high
q_true_eval = q_high

# ====================== PARAMETER SWEEP ======================
print("\n" + "="*80)
print("RUNNING PARAMETER SWEEP")
print("="*80)

results = []

for desired_freq_hz in desired_freq_hz_list:
    for latency_seconds in latency_seconds_list:
        print(f"\n--- Testing: freq={desired_freq_hz} Hz, latency={latency_seconds}s ---")
        
        dt_output = 1.0 / desired_freq_hz
        
        # Generate output timestamps with latency
        t_start = t_high[0] + latency_seconds
        t_end   = t_high[-1]
        
        # Build regular timestamp grid
        t_key = np.arange(t_start, t_end + dt_output/2, dt_output)
        
        # Map each desired time to nearest true sample (left-side for causality)
        indices = np.searchsorted(t_high, t_key, side='right') - 1
        indices = np.clip(indices, 0, len(t_high)-1)
        
        # Final delayed + downsampled data
        t_key_actual = t_high[indices]
        q_key = q_high[indices]
        qdot_key = qdot_high[indices]
        
        # Build interpolators
        try:
            interp_slerp    = quat_slerp.make_slerp_interpolator(t_key_actual, q_key, qdot_key, fix_sign_swap)
            interp_hermite  = quat_slerp.make_hermite_interpolator(t_key_actual, q_key, qdot_key, fix_sign_swap)
            interp_cubic    = quat_slerp.make_cubic_spline_interpolator(t_key_actual, q_key, fix_sign_swap)
            
            # Evaluate
            q_slerp             = interp_slerp(t_eval)
            q_hermite           = interp_hermite(t_eval)
            q_cubicspline       = interp_cubic(t_eval)
            
            # Calculate errors
            err_slerp       = quat_slerp.quat_angle_error(q_slerp, q_true_eval)
            err_hermite     = quat_slerp.quat_angle_error(q_hermite, q_true_eval)
            err_cubicspline = quat_slerp.quat_angle_error(q_cubicspline, q_true_eval)
            
            # Store results
            results.append({
                'freq_hz': desired_freq_hz,
                'latency_s': latency_seconds,
                'n_keyframes': len(t_key_actual),
                'slerp_mean': np.mean(err_slerp),
                'slerp_rms': np.sqrt(np.mean(err_slerp**2)),
                'slerp_max': np.max(err_slerp),
                'slerp_99th': np.percentile(err_slerp, 99),
                'hermite_mean': np.mean(err_hermite),
                'hermite_rms': np.sqrt(np.mean(err_hermite**2)),
                'hermite_max': np.max(err_hermite),
                'hermite_99th': np.percentile(err_hermite, 99),
                'cubic_mean': np.mean(err_cubicspline),
                'cubic_rms': np.sqrt(np.mean(err_cubicspline**2)),
                'cubic_max': np.max(err_cubicspline),
                'cubic_99th': np.percentile(err_cubicspline, 99),
            })
            
            print(f"  ✓ Completed: {len(t_key_actual)} keyframes")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results.append({
                'freq_hz': desired_freq_hz,
                'latency_s': latency_seconds,
                'n_keyframes': 0,
                'slerp_mean': np.nan,
                'slerp_rms': np.nan,
                'slerp_max': np.nan,
                'slerp_99th': np.nan,
                'hermite_mean': np.nan,
                'hermite_rms': np.nan,
                'hermite_max': np.nan,
                'hermite_99th': np.nan,
                'cubic_mean': np.nan,
                'cubic_rms': np.nan,
                'cubic_max': np.nan,
                'cubic_99th': np.nan,
            })

# ====================== SAVE RESULTS ======================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

summary_df = pd.DataFrame(results)
summary_df.to_csv(results_csv, index=False)
print(f"✓ Results saved to: {results_csv}")

# ====================== DISPLAY SUMMARY TABLE ======================
print("\n" + "="*120)
print("PARAMETER SWEEP RESULTS SUMMARY")
print("="*120)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}' if not np.isnan(x) else 'N/A')

# Select key columns for display
display_cols = ['freq_hz', 'latency_s', 'n_keyframes', 
                'slerp_mean', 'hermite_mean', 'cubic_mean',
                'slerp_max', 'hermite_max', 'cubic_max']

print(summary_df[display_cols].to_string(index=False))

# ====================== FIND OPTIMAL CONFIGURATIONS ======================
print("\n" + "="*80)
print("OPTIMAL CONFIGURATIONS (lowest mean error per method)")
print("="*80)

for method in ['slerp', 'hermite', 'cubic']:
    best_idx = summary_df[f'{method}_mean'].idxmin()
    best_row = summary_df.loc[best_idx]
    print(f"\n{method.upper()}:")
    print(f"  Frequency: {best_row['freq_hz']} Hz")
    print(f"  Latency: {best_row['latency_s']} s")
    print(f"  Mean Error: {best_row[f'{method}_mean']:.2f} µrad")
    print(f"  Max Error: {best_row[f'{method}_max']:.2f} µrad")

# ====================== VISUALIZATION ======================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

methods = ['slerp',  'hermite', 'cubic']
method_labels = ['SLERP',  'Hermite', 'Cubic Spline']
colors = ['tab:blue', 'tab:purple', 'tab:green']

# Plot 1: Mean Error vs Latency (all methods)
ax = axes[0, 0]
for method, label, color in zip(methods, method_labels, colors):
    for freq in desired_freq_hz_list:
        subset = summary_df[summary_df['freq_hz'] == freq]
        ax.plot(subset['latency_s'], subset[f'{method}_mean'], 
                marker='o', label=f'{freq} Hz ({label})', color=color, linewidth=2)

ax.set_xlabel('Latency (seconds)', fontsize=12)
ax.set_ylabel('Mean Error (µrad)', fontsize=12)
ax.set_title('Mean Error vs Latency by Method & Frequency', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 2: Max Error vs Frequency (all methods)
ax = axes[0, 1]
for method, label, color in zip(methods, method_labels, colors):
    for latency in latency_seconds_list:
        subset = summary_df[summary_df['latency_s'] == latency]
        ax.plot(subset['freq_hz'], subset[f'{method}_max'], 
                marker='s', label=f'{latency}s ({label})', color=color, linewidth=2)

ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Max Error (µrad)', fontsize=12)
ax.set_title('Max Error vs Frequency by Method & Latency', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# Plot 3: Heatmap - SLERP Mean Error
ax = axes[1, 0]
pivot_slerp = summary_df.pivot(index='latency_s', columns='freq_hz', values='slerp_mean')
im = ax.imshow(pivot_slerp.values, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(len(desired_freq_hz_list)))
ax.set_yticks(range(len(latency_seconds_list)))
ax.set_xticklabels([f'{f} Hz' for f in desired_freq_hz_list])
ax.set_yticklabels([f'{l}s' for l in latency_seconds_list])
ax.set_xlabel('Frequency', fontsize=12)
ax.set_ylabel('Latency', fontsize=12)
ax.set_title('SLERP Mean Error Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mean Error (µrad)')

# Plot 4: Heatmap - Cubic Mean Error
ax = axes[1, 1]
pivot_cubic = summary_df.pivot(index='latency_s', columns='freq_hz', values='cubic_mean')
im = ax.imshow(pivot_cubic.values, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(len(desired_freq_hz_list)))
ax.set_yticks(range(len(latency_seconds_list)))
ax.set_xticklabels([f'{f} Hz' for f in desired_freq_hz_list])
ax.set_yticklabels([f'{l}s' for l in latency_seconds_list])
ax.set_xlabel('Frequency', fontsize=12)
ax.set_ylabel('Latency', fontsize=12)
ax.set_title('Cubic Mean Error Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mean Error (µrad)')

plt.suptitle('Quaternion Interpolation Parameter Sweep Analysis\n'
             f'Dataset: {len(t_high)} high-rate samples @ {freq_true:.1f} Hz', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(summary_plot, dpi=200, bbox_inches='tight')
print(f"✓ Plots saved to: {summary_plot}")

# ====================== ADDITIONAL ANALYSIS ======================
print("\n" + "="*80)
print("ADDITIONAL ANALYSIS")
print("="*80)

# Best overall configuration (lowest max error across all methods)
best_overall_idx = summary_df[['slerp_max', 'hermite_max', 'cubic_max']].min(axis=1).idxmin()
best_overall = summary_df.loc[best_overall_idx]
print(f"\nBest Overall Configuration (lowest max error):")
print(f"  Frequency: {best_overall['freq_hz']} Hz")
print(f"  Latency: {best_overall['latency_s']} s")
print(f"  Keyframes: {best_overall['n_keyframes']}")

# Method comparison at each frequency
print(f"\nMethod Comparison by Frequency:")
print("-"*80)
print(f"{'Freq (Hz)':<10} {'SLERP Mean':<15} {'Hermite Mean':<15} {'Cubic Mean':<15} {'Best Method':<15}")
print("-"*80)

for freq in desired_freq_hz_list:
    subset = summary_df[summary_df['freq_hz'] == freq]
    # Average across latencies for this frequency
    slerp_avg = subset['slerp_mean'].mean()
    hermite_avg = subset['hermite_mean'].mean()
    cubic_avg = subset['cubic_mean'].mean()
    
    best_method = min([('SLERP', slerp_avg), ('Hermite', hermite_avg), ('Cubic', cubic_avg)], 
                      key=lambda x: x[1])[0]
    
    print(f"{freq:<10} {slerp_avg:<15.2f} {hermite_avg:<15.2f} {cubic_avg:<15.2f} {best_method:<15}")

print("="*80)
print("PARAMETER SWEEP COMPLETE")
print("="*80)
print(f"\nTotal configurations tested: {len(results)}")
print(f"Successful: {summary_df['n_keyframes'].gt(0).sum()}")
print(f"Failed: {summary_df['n_keyframes'].eq(0).sum()}")
print(f"\nOutput files:")
print(f"  - {results_csv}")
print(f"  - {summary_plot}")