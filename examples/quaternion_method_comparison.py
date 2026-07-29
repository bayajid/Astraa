import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

import quaternion_slerp_squad as quat_slerp
from MEKF import (
    MultiplicativeEKF_HighOrder,
    AttitudeMeasurement,
    QuaternionMath,
    MEKFComparator,
)

# =============================================================================
# Method Runners
# =============================================================================

def run_mekf_method(q_key: np.ndarray, q_dot_key: np.ndarray,
                    t_key: np.ndarray, timestamps: np.ndarray,
                    noise_level: float = 0.0001) -> np.ndarray:

    mekf = MultiplicativeEKF_HighOrder(
        q_init=q_key[0],
        estimate_bias=False,
        process_noise_scale=1e-10,
        measurement_noise_scale=1e-8
    )

    key_idx = 0
    n_keys = len(t_key)

    last_q_dot = q_dot_key[0]
    last_timestamp = timestamps[0]

    q_estimates = np.zeros((len(timestamps), 4))

    for i, t in enumerate(timestamps):

        dt = t - last_timestamp if i > 0 else 1e-3

        # prediction step
        mekf.predict(last_q_dot, dt)

        while key_idx < n_keys and t_key[key_idx] <= t:

            q_true_key = q_key[key_idx]

            # --- small rotation noise ---
            delta_theta = np.random.randn(3) * noise_level
            angle = np.linalg.norm(delta_theta)

            if angle < 1e-12:
                delta_q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                axis = delta_theta / angle
                half = 0.5 * angle
                delta_q = np.concatenate(([np.cos(half)], axis * np.sin(half)))

            # multiplicative measurement
            q_meas = QuaternionMath.multiply(delta_q, q_true_key)
            q_meas = QuaternionMath.normalize(q_meas)

            # --- gyro noise added HERE ---
            gyro_noise_std = 1e-4
            qdot_true = q_dot_key[key_idx]
            qdot_meas = qdot_true + np.random.randn(4) * gyro_noise_std

            last_q_dot = q_dot_key[key_idx]

            mekf.update(q_meas, q_meas_cov=np.eye(3) * noise_level**2)

            key_idx += 1

        q_estimates[i] = mekf.q.copy()
        last_timestamp = t

    return q_estimates


def run_slerp_method(q_key, q_dot_key, t_key, t_query, fix_sign_swap=False):
    """Slerp interpolation from keyframe quaternions to query timestamps."""
    return quat_slerp.make_slerp_interpolator(t_key, q_key, q_dot_key, fix_sign_swap)(t_query)

def run_squad_method(q_key, q_dot_key, t_key, t_query, fix_sign_swap=False):
    """Slerp interpolation from keyframe quaternions to query timestamps."""
    return quat_slerp.make_squad_interpolator(t_key, q_key, q_dot_key, fix_sign_swap)(t_query)

def run_cubic_spline_method(q_key, t_key, t_query, fix_sign_swap=False):
    """Cubic spline interpolation from keyframe quaternions to query timestamps."""
    return quat_slerp.make_cubic_spline_interpolator(t_key, q_key, fix_sign_swap)(t_query)

def compute_errors(q_est: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    """Per-sample angular error (radians) between estimated and true quaternions."""
    return np.array([
        MEKFComparator.quaternion_error(q_true[i], q_est[i])
        for i in range(len(q_true))
    ])


# =============================================================================
# Post-loop: summary table print + save
# =============================================================================

def print_and_save_summary(results: list, save_path: str):
    """
    Print a formatted summary table to console and save as .txt.
    All error values reported in µrad. MEKF is now per-combination.
    """
    header = (
        f"{'Freq(Hz)':>10} {'Latency(s)':>12} {'Keys':>6} "
        f"{'Slerp RMS':>11}{'Squad RMS':>11} {'Spline RMS':>12} {'MEKF RMS':>11} {'Best':>10}"
    )
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "QUATERNION METHOD COMPARISON — FULL SWEEP RESULTS  (all values µrad)",
        "=" * len(header),
        "",
        "NOTE: MEKF receives the same sparse delayed keyframes as Slerp/Spline.",
        "      It dead-reckons between keyframes and corrects only when one arrives.",
        "",
        header,
        sep,
    ]

    for r in results:
        rms = {'Slerp': r['slerp_rms'],'Squad': r['squad_rms'], 'Spline': r['spline_rms'], 'MEKF': r['mekf_rms']}
        best = min(rms, key=rms.get)
        lines.append(
            f"{r['freq_hz']:>10} {r['latency_s']:>12.1f} {r['n_keyframes']:>6} "
            f"{r['slerp_rms']*1e6:>11.2f} {r['squad_rms']*1e6:>11.2f} {r['spline_rms']*1e6:>12.2f} "
            f"{r['mekf_rms']*1e6:>11.2f} {best:>10}"
        )

    lines.append(sep)

    best_slerp  = min(results, key=lambda r: r['slerp_rms'])
    best_squad  = min(results, key=lambda r: r['squad_rms'])
    best_spline = min(results, key=lambda r: r['spline_rms'])
    best_mekf   = min(results, key=lambda r: r['mekf_rms'])
    lines += [
        "",
        f"Best Slerp  : freq={best_slerp['freq_hz']} Hz, latency={best_slerp['latency_s']}s"
        f"  → RMS={best_slerp['slerp_rms']*1e6:.2f} µrad",
        f"Best Squad  : freq={best_squad['freq_hz']} Hz, latency={best_squad['latency_s']}s"
        f"  → RMS={best_squad['squad_rms']*1e6:.2f} µrad",        
        f"Best Spline : freq={best_spline['freq_hz']} Hz, latency={best_spline['latency_s']}s"
        f"  → RMS={best_spline['spline_rms']*1e6:.2f} µrad",
        f"Best MEKF   : freq={best_mekf['freq_hz']} Hz, latency={best_mekf['latency_s']}s"
        f"  → RMS={best_mekf['mekf_rms']*1e6:.2f} µrad",
        "=" * len(header),
    ]

    text = "\n".join(lines)
    print(text)
    with open(save_path, "w") as f:
        f.write(text + "\n")
    print(f"\n✓ Summary saved → {save_path}")


# =============================================================================
# Post-loop: plots
# =============================================================================

def plot_sweep_results(results: list, freq_list: list, latency_list: list,
                       save_path: str):
    """
    4-panel figure — all three methods on equal footing:
      Panels 0-2 : RMS vs latency for Slerp / Spline / MEKF,
                   one colored line per keyframe frequency
      Panel 3    : grouped bar — best RMS of each method per latency bucket
    """
    cmap   = cm.get_cmap('tab10', len(freq_list))
    colors = {freq: cmap(i) for i, freq in enumerate(freq_list)}

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Quaternion Method Comparison — Fair Sweep (same keyframes for all methods)",
                 fontsize=13, fontweight='bold')

    method_cfg = [
        ('slerp_rms',  'Slerp',        axes[0]),
        ('spline_rms', 'Cubic Spline', axes[1]),
        ('mekf_rms',   'MEKF',         axes[2]),
    ]

    for rms_key, title, ax in method_cfg:
        for freq in freq_list:
            subset = [r for r in results if r['freq_hz'] == freq]
            lats   = [r['latency_s']   for r in subset]
            rms    = [r[rms_key] * 1e6 for r in subset]
            ax.plot(lats, rms, marker='o', linewidth=2, markersize=5,
                    color=colors[freq], label=f"{freq} Hz")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Latency (s)')
        ax.set_ylabel('RMS Error (µrad)')
        ax.legend(title='Keyframe freq', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: grouped bar — best of each method per latency
    ax  = axes[3]
    x   = np.arange(len(latency_list))
    w   = 0.25
    best = lambda key, lat: min(r[key] for r in results if r['latency_s'] == lat) * 1e6

    ax.bar(x - w, [best('slerp_rms',  l) for l in latency_list], w,
           label='Best Slerp',  color='steelblue',   alpha=0.85)
    ax.bar(x,     [best('spline_rms', l) for l in latency_list], w,
           label='Best Spline', color='darkorange',  alpha=0.85)
    ax.bar(x + w, [best('mekf_rms',   l) for l in latency_list], w,
           label='Best MEKF',   color='forestgreen', alpha=0.85)

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
    print(f"✓ Sweep plot saved  → {save_path}")


def plot_heatmaps(results: list, freq_list: list, latency_list: list, save_path: str):
    """
    3-panel heatmap: Slerp / Spline / MEKF RMS (µrad) across freq × latency.
    Green = low error, red = high. Each cell annotated with its value.
    """
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
                               ['Slerp',     'Cubic Spline', 'MEKF']):
        grid = build_grid(key)
        vmin, vmax = grid.min(), grid.max()
        im   = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r',
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
    print(f"✓ Heatmap saved     → {save_path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    datadir   = os.path.join(os.path.dirname(__file__), 'output_data/tables/custom_quatpred')
    true_file = os.path.join(datadir, 'true_quat_custom.csv')

    df         = pd.read_csv(true_file)
    q_true     = df.iloc[:, 1:5].values.copy()
    q_dot_true = df.iloc[:, 5:9].values.copy()
    timestamps = df.iloc[:, 0].values.copy()

    desired_freq_hz_list = [1, 2, 5, 10]
    latency_seconds_list = [0, 1, 2, 3, 4]
    fix_sign_swap        = False
    noise_level          = 0.001

    freq_true = 1.0 / np.mean(np.diff(timestamps))
    omegas    = np.array([
        quat_slerp.angular_velocity_from_qdot(q, qd)
        for q, qd in zip(q_true, q_dot_true)
    ])
    print("=" * 60)
    print("QUATERNION ESTIMATION METHOD COMPARISON — SWEEP")
    print("=" * 60)
    print(f"Data:             {len(timestamps)} samples @ {freq_true:.3f} Hz")
    print(f"Max angular rate: {np.max(np.linalg.norm(omegas, axis=1))*180/np.pi:.2f} deg/s")
    print(f"Sweep:            {len(desired_freq_hz_list)} freqs × "
          f"{len(latency_seconds_list)} latencies = "
          f"{len(desired_freq_hz_list)*len(latency_seconds_list)} combinations")
    print("NOTE: All methods receive the same sparse delayed keyframes.")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Main sweep — MEKF now runs inside the loop on the same keyframes
    # ------------------------------------------------------------------
    results    = []
    total_runs = len(desired_freq_hz_list) * len(latency_seconds_list)

    for run_idx, (desired_freq_hz, latency_seconds) in enumerate(
        [(f, l) for f in desired_freq_hz_list for l in latency_seconds_list], start=1
    ):
        print(f"  [{run_idx:2d}/{total_runs}] freq={desired_freq_hz:2d} Hz  "
              f"latency={latency_seconds}s", end="  ", flush=True)

        # Build causal, downsampled keyframe grid — shared by all three methods
        dt_output    = 1.0 / desired_freq_hz
        t_key        = np.arange(timestamps[0] + latency_seconds,
                                  timestamps[-1] + dt_output / 2,
                                  dt_output)
        indices      = np.clip(np.searchsorted(timestamps, t_key, side='right') - 1,
                               0, len(timestamps) - 1)
        t_key_actual = timestamps[indices]
        q_key        = q_true[indices]
        qdot_key     = q_dot_true[indices]

        np.random.seed(42)   # same noise realisation for every method
        err_slerp  = compute_errors(
            run_slerp_method(q_key, qdot_key, t_key_actual, timestamps, fix_sign_swap),
            q_true
        )
        err_squad  = compute_errors(
            run_squad_method(q_key, qdot_key, t_key_actual, timestamps, fix_sign_swap),
            q_true
        )

        err_spline = compute_errors(
            run_cubic_spline_method(q_key, t_key_actual, timestamps, fix_sign_swap),
            q_true
        )
        err_mekf   = compute_errors(
            # MEKF gets exactly the same sparse keyframes — fair comparison
            run_mekf_method(q_key, qdot_key, t_key_actual, timestamps, noise_level),
            q_true
        )

        results.append({
            'freq_hz':       desired_freq_hz,
            'latency_s':     latency_seconds,
            'n_keyframes':   len(t_key_actual),
            # Slerp
            'slerp_mean':    float(np.mean(err_slerp)),
            'slerp_rms':     float(np.sqrt(np.mean(err_slerp**2))),
            'slerp_max':     float(np.max(err_slerp)),
            'slerp_errors':  err_slerp,
            # Squad
            'squad_mean':    float(np.mean(err_squad)),
            'squad_rms':     float(np.sqrt(np.mean(err_squad**2))),
            'squad_max':     float(np.max(err_squad)),
            'squad_errors':  err_squad,
            # Spline
            'spline_mean':   float(np.mean(err_spline)),
            'spline_rms':    float(np.sqrt(np.mean(err_spline**2))),
            'spline_max':    float(np.max(err_spline)),
            'spline_errors': err_spline,
            # MEKF (same keyframes)
            'mekf_mean':     float(np.mean(err_mekf)),
            'mekf_rms':      float(np.sqrt(np.mean(err_mekf**2))),
            'mekf_max':      float(np.max(err_mekf)),
            'mekf_errors':   err_mekf,
        })

        r = results[-1]
        print(f"slerp={r['slerp_rms']*1e6:7.1f}  "
              f"squad={r['squad_rms']*1e6:7.1f}  "
              f"spline={r['spline_rms']*1e6:7.1f}  "
              f"mekf={r['mekf_rms']*1e6:7.1f}  µrad")

    # ------------------------------------------------------------------
    # Post-loop: all reporting happens here
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE — generating summary and plots")
    print("=" * 60 + "\n")

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