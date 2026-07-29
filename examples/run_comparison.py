"""
Complete comparison runner for quaternion interpolation methods.
This version properly integrates MEKF with vector and quaternion measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# Import your existing modules
import quaternion_slerp_squad as quat_slerp
from MEKF1 import (
    MultiplicativeEKF_HighOrder,
    AttitudeMeasurement,
    QuaternionMeasurement,
    QuaternionMath,
    MEKFComparator,
    angular_velocity_from_quaternion_derivative
)


# =============================================================================
# Method Runners
# =============================================================================

def run_mekf_method(q_key: np.ndarray, q_dot_key: np.ndarray,
                    t_key: np.ndarray, timestamps: np.ndarray,
                    noise_level: float = 0.0001,
                    use_vector_measurements: bool = False) -> np.ndarray:
    """
    Run MEKF with proper measurement model.
    
    Two measurement modes:
    1. Direct quaternion measurements (use_vector_measurements=False)
       - Simulates star tracker providing full attitude
       - Converts to 3D error representation for MEKF
    
    2. Vector observations (use_vector_measurements=True)
       - Simulates sun sensor / magnetometer
       - More physically realistic for many spacecraft
    
    Args:
        q_key: Sparse keyframe quaternions [K, 4] in [w,x,y,z] format
        q_dot_key: Sparse keyframe quaternion derivatives [K, 4]
        t_key: Keyframe timestamps [K]
        timestamps: Full-resolution query times [N]
        noise_level: Measurement noise level (rad for attitude)
        use_vector_measurements: Use vector obs instead of direct quat
    
    Returns:
        q_estimates: Estimated quaternions at all timestamps [N, 4]
    """
    
    # Initialize MEKF
    mekf = MultiplicativeEKF_HighOrder(
        q_init=q_key[0],
        estimate_bias=True,  # Enable bias estimation
        gyro_noise_std=1e-4,     # Realistic gyro ARW: ~0.01 deg/sqrt(hr)
        gyro_bias_std=1e-7,      # Realistic bias drift
        process_noise_scale=1.0,
        measurement_noise_scale=1.0
    )
    
    # Setup
    key_idx = 0
    n_keys = len(t_key)
    last_timestamp = timestamps[0]
    q_estimates = np.zeros((len(timestamps), 4))
    
    # Initialize angular velocity from first keyframe
    omega_current = angular_velocity_from_quaternion_derivative(
        q_key[0], q_dot_key[0]
    )
    
    # For vector measurements: define reference direction
    # (e.g., sun direction in inertial frame)
    sun_inertial = np.array([1.0, 0.0, 0.0])
    
    print(f"    MEKF mode: {'Vector measurements' if use_vector_measurements else 'Quaternion measurements'}")
    
    for i, t in enumerate(timestamps):
        dt = t - last_timestamp if i > 0 else 1e-3
        
        # =================================================================
        # PREDICTION STEP (dead-reckoning with gyro)
        # =================================================================
        # Add realistic gyro noise
        gyro_noise_std = 1e-4  # rad/s
        omega_noisy = omega_current + np.random.randn(3) * gyro_noise_std
        
        mekf.predict(omega_noisy, dt)
        
        # =================================================================
        # UPDATE STEP (when keyframe arrives)
        # =================================================================
        while key_idx < n_keys and t_key[key_idx] <= t:
            q_true_key = q_key[key_idx]
            
            if use_vector_measurements:
                # =========================================================
                # MODE 1: Vector Observation (Sun Sensor / Magnetometer)
                # =========================================================
                # Simulate sun sensor measuring sun direction in body frame
                A_true = QuaternionMath.to_rotation_matrix(q_true_key)
                sun_body_true = A_true @ sun_inertial
                
                # Add noise
                sun_body_meas = sun_body_true + np.random.randn(3) * noise_level
                sun_body_meas = sun_body_meas / np.linalg.norm(sun_body_meas)
                
                # Create measurement object
                measurement = AttitudeMeasurement(
                    body_vector=sun_body_meas,
                    reference_vector=sun_inertial,
                    covariance=np.eye(3) * noise_level**2
                )
                
                mekf.update_vector_measurement(measurement)
            
            else:
                # =========================================================
                # MODE 2: Direct Quaternion Measurement (Star Tracker)
                # =========================================================
                # Add small rotation noise (physically realistic)
                delta_theta = np.random.randn(3) * noise_level
                angle = np.linalg.norm(delta_theta)
                
                if angle < 1e-12:
                    delta_q = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    axis = delta_theta / angle
                    delta_q = QuaternionMath.from_axis_angle(axis, angle)
                
                # Apply noise: q_meas = δq ⊗ q_true
                q_meas = QuaternionMath.multiply(delta_q, q_true_key)
                q_meas = QuaternionMath.normalize(q_meas)
                
                # Create measurement object
                measurement = QuaternionMeasurement(
                    q_measured=q_meas,
                    covariance=np.eye(3) * noise_level**2
                )
                
                mekf.update_quaternion_measurement(measurement)
            
            # Update angular velocity reference from keyframe
            omega_current = angular_velocity_from_quaternion_derivative(
                q_key[key_idx], q_dot_key[key_idx]
            )
            
            key_idx += 1
        
        # Store estimate
        q_estimates[i] = mekf.q.copy()
        last_timestamp = t
    
    return q_estimates


def run_slerp_method(q_key, q_dot_key, t_key, t_query, fix_sign_swap=False):
    """Slerp interpolation (your existing implementation)."""
    return quat_slerp.make_slerp_interpolator(t_key, q_key, q_dot_key, fix_sign_swap)(t_query)


def run_cubic_spline_method(q_key, t_key, t_query, fix_sign_swap=False):
    """Cubic spline interpolation (your existing implementation)."""
    return quat_slerp.make_cubic_spline_interpolator(t_key, q_key, fix_sign_swap)(t_query)


def compute_errors(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Compute angular error for each sample."""
    return np.array([
        MEKFComparator.quaternion_error(q_true[i], q_est[i])
        for i in range(len(q_true))
    ])


# =============================================================================
# Post-loop: Summary and Visualization
# =============================================================================

def print_and_save_summary(results: list, save_path: str):
    """Print formatted summary table and save to file."""
    header = (
        f"{'Freq(Hz)':>10} {'Latency(s)':>12} {'Keys':>6} "
        f"{'Slerp RMS':>11} {'Spline RMS':>12} {'MEKF RMS':>11} {'Best':>10}"
    )
    sep = "-" * len(header)
    
    lines = [
        "=" * len(header),
        "QUATERNION METHOD COMPARISON — FULL SWEEP RESULTS",
        "=" * len(header),
        "",
        "All error values in microradians (µrad)",
        "MEKF uses same sparse delayed keyframes as Slerp/Spline (fair comparison)",
        "",
        header,
        sep,
    ]
    
    for r in results:
        rms = {'Slerp': r['slerp_rms'], 'Spline': r['spline_rms'], 'MEKF': r['mekf_rms']}
        best = min(rms, key=rms.get)
        lines.append(
            f"{r['freq_hz']:>10} {r['latency_s']:>12.1f} {r['n_keyframes']:>6} "
            f"{r['slerp_rms']*1e6:>11.2f} {r['spline_rms']*1e6:>12.2f} "
            f"{r['mekf_rms']*1e6:>11.2f} {best:>10}"
        )
    
    lines.append(sep)
    
    # Find best configurations
    best_slerp = min(results, key=lambda r: r['slerp_rms'])
    best_spline = min(results, key=lambda r: r['spline_rms'])
    best_mekf = min(results, key=lambda r: r['mekf_rms'])
    
    lines += [
        "",
        f"Best Slerp  : freq={best_slerp['freq_hz']} Hz, latency={best_slerp['latency_s']}s "
        f"→ RMS={best_slerp['slerp_rms']*1e6:.2f} µrad",
        f"Best Spline : freq={best_spline['freq_hz']} Hz, latency={best_spline['latency_s']}s "
        f"→ RMS={best_spline['spline_rms']*1e6:.2f} µrad",
        f"Best MEKF   : freq={best_mekf['freq_hz']} Hz, latency={best_mekf['latency_s']}s "
        f"→ RMS={best_mekf['mekf_rms']*1e6:.2f} µrad",
        "=" * len(header),
    ]
    
    text = "\n".join(lines)
    print(text)
    
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(f"\n✓ Summary saved → {save_path}")


def plot_sweep_results(results: list, freq_list: list, latency_list: list,
                       save_path: str):
    """Generate 4-panel comparison plot."""
    cmap = cm.get_cmap('tab10', len(freq_list))
    colors = {freq: cmap(i) for i, freq in enumerate(freq_list)}
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Quaternion Method Comparison — Fair Sweep (same keyframes for all methods)",
                 fontsize=13, fontweight='bold')
    
    # Panels 0-2: RMS vs latency for each method
    method_cfg = [
        ('slerp_rms', 'Slerp', axes[0]),
        ('spline_rms', 'Cubic Spline', axes[1]),
        ('mekf_rms', 'MEKF', axes[2]),
    ]
    
    for rms_key, title, ax in method_cfg:
        for freq in freq_list:
            subset = [r for r in results if r['freq_hz'] == freq]
            lats = [r['latency_s'] for r in subset]
            rms = [r[rms_key] * 1e6 for r in subset]
            ax.plot(lats, rms, marker='o', linewidth=2, markersize=5,
                    color=colors[freq], label=f"{freq} Hz")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Latency (s)')
        ax.set_ylabel('RMS Error (µrad)')
        ax.legend(title='Keyframe freq', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Panel 3: Grouped bar chart
    ax = axes[3]
    x = np.arange(len(latency_list))
    w = 0.25
    best = lambda key, lat: min(r[key] for r in results if r['latency_s'] == lat) * 1e6
    
    ax.bar(x - w, [best('slerp_rms', l) for l in latency_list], w,
           label='Best Slerp', color='steelblue', alpha=0.85)
    ax.bar(x, [best('spline_rms', l) for l in latency_list], w,
           label='Best Spline', color='darkorange', alpha=0.85)
    ax.bar(x + w, [best('mekf_rms', l) for l in latency_list], w,
           label='Best MEKF', color='forestgreen', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}s" for l in latency_list])
    ax.set_xlabel('Latency')
    ax.set_ylabel('Best RMS Error (µrad)')
    ax.set_title('Head-to-Head (best freq per method)', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Sweep plot saved → {save_path}")


def plot_heatmaps(results: list, freq_list: list, latency_list: list, save_path: str):
    """Generate heatmap comparison."""
    def build_grid(key):
        grid = np.zeros((len(freq_list), len(latency_list)))
        for r in results:
            grid[freq_list.index(r['freq_hz']),
                 latency_list.index(r['latency_s'])] = r[key] * 1e6
        return grid
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("RMS Error Heatmap (µrad) — Keyframe Freq × Latency",
                 fontsize=13, fontweight='bold')
    
    for ax, key, title in zip(axes,
                              ['slerp_rms', 'spline_rms', 'mekf_rms'],
                              ['Slerp', 'Cubic Spline', 'MEKF']):
        grid = build_grid(key)
        vmin, vmax = grid.min(), grid.max()
        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r',
                      origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label='RMS (µrad)')
        ax.set_xticks(range(len(latency_list)))
        ax.set_xticklabels([f"{l}s" for l in latency_list])
        ax.set_yticks(range(len(freq_list)))
        ax.set_yticklabels([f"{f} Hz" for f in freq_list])
        ax.set_xlabel('Latency')
        ax.set_ylabel('Keyframe Frequency')
        ax.set_title(title, fontsize=12)
        
        for fi in range(len(freq_list)):
            for li in range(len(latency_list)):
                ax.text(li, fi, f"{grid[fi, li]:.1f}",
                       ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved → {save_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Load ground truth data
    datadir = os.path.join(os.path.dirname(__file__), 'output_data/tables/custom_quatpred')
    true_file = os.path.join(datadir, 'true_quat_custom.csv')
    
    df = pd.read_csv(true_file)
    
    # Extract data (assuming CSV columns: time, qw, qx, qy, qz, qw_dot, qx_dot, qy_dot, qz_dot)
    timestamps = df.iloc[:, 0].values.copy()
    q_true = df.iloc[:, 1:5].values.copy()  # [qw, qx, qy, qz]
    q_dot_true = df.iloc[:, 5:9].values.copy()
    
    # Sweep parameters
    desired_freq_hz_list = [1, 2, 5, 10]
    latency_seconds_list = [0, 1, 2, 3, 4]
    fix_sign_swap = False
    noise_level = 0.001  # 1 mrad noise
    
    # Data info
    freq_true = 1.0 / np.mean(np.diff(timestamps))
    omegas = np.array([
        angular_velocity_from_quaternion_derivative(q, qd)
        for q, qd in zip(q_true, q_dot_true)
    ])
    
    print("=" * 70)
    print("QUATERNION ESTIMATION METHOD COMPARISON — SWEEP")
    print("=" * 70)
    print(f"Data:             {len(timestamps)} samples @ {freq_true:.3f} Hz")
    print(f"Max angular rate: {np.max(np.linalg.norm(omegas, axis=1))*180/np.pi:.2f} deg/s")
    print(f"Sweep:            {len(desired_freq_hz_list)} freqs × "
          f"{len(latency_seconds_list)} latencies = "
          f"{len(desired_freq_hz_list)*len(latency_seconds_list)} combinations")
    print("=" * 70 + "\n")
    
    # Main sweep loop
    results = []
    total_runs = len(desired_freq_hz_list) * len(latency_seconds_list)
    
    for run_idx, (desired_freq_hz, latency_seconds) in enumerate(
        [(f, l) for f in desired_freq_hz_list for l in latency_seconds_list], start=1
    ):
        print(f"  [{run_idx:2d}/{total_runs}] freq={desired_freq_hz:2d} Hz  "
              f"latency={latency_seconds}s", flush=True)
        
        # Build sparse keyframe grid (shared by all methods)
        dt_output = 1.0 / desired_freq_hz
        t_key = np.arange(timestamps[0] + latency_seconds,
                         timestamps[-1] + dt_output / 2,
                         dt_output)
        indices = np.clip(np.searchsorted(timestamps, t_key, side='right') - 1,
                         0, len(timestamps) - 1)
        t_key_actual = timestamps[indices]
        q_key = q_true[indices]
        qdot_key = q_dot_true[indices]
        
        # Fix random seed for reproducibility
        np.random.seed(42)
        
        # Run all three methods
        err_slerp = compute_errors(
            run_slerp_method(q_key, qdot_key, t_key_actual, timestamps, fix_sign_swap),
            q_true
        )
        
        err_spline = compute_errors(
            run_cubic_spline_method(q_key, t_key_actual, timestamps, fix_sign_swap),
            q_true
        )
        
        err_mekf = compute_errors(
            run_mekf_method(q_key, qdot_key, t_key_actual, timestamps, noise_level,
                           use_vector_measurements=True),  # Change to True for vector mode
            q_true
        )
        
        # Store results
        results.append({
            'freq_hz': desired_freq_hz,
            'latency_s': latency_seconds,
            'n_keyframes': len(t_key_actual),
            'slerp_mean': float(np.mean(err_slerp)),
            'slerp_rms': float(np.sqrt(np.mean(err_slerp**2))),
            'slerp_max': float(np.max(err_slerp)),
            'spline_mean': float(np.mean(err_spline)),
            'spline_rms': float(np.sqrt(np.mean(err_spline**2))),
            'spline_max': float(np.max(err_spline)),
            'mekf_mean': float(np.mean(err_mekf)),
            'mekf_rms': float(np.sqrt(np.mean(err_mekf**2))),
            'mekf_max': float(np.max(err_mekf)),
        })
        
        r = results[-1]
        print(f"    slerp={r['slerp_rms']*1e6:7.1f}  "
              f"spline={r['spline_rms']*1e6:7.1f}  "
              f"mekf={r['mekf_rms']*1e6:7.1f}  µrad")
    
    # Generate reports
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE — generating summary and plots")
    print("=" * 70 + "\n")
    
    print_and_save_summary(
        results,
        save_path=os.path.join(datadir, 'sweep_results.txt')
    )
    
    plot_sweep_results(
        results,
        desired_freq_hz_list, latency_seconds_list,
        save_path=os.path.join(datadir, 'sweep_comparison.png')
    )
    
    plot_heatmaps(
        results,
        desired_freq_hz_list, latency_seconds_list,
        save_path=os.path.join(datadir, 'sweep_heatmap.png')
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)