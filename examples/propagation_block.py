#%%
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import os,sys
import pandas as pd
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import basic_tools.time_conversion as t_conv

## TODO: CHECK THE ORIGINAL SATDATA FILE TIMESTEP. USE 1Hz DATA

def propagate_with_updates(t_init,
                            prop_duration,
                            prop_timeout,
                            t_gps,
                            r_target,
                            v_target,
                            odeRK,
                            orbit_prop,
                            step=1,
                            update_times=None):

            updates = deque(update_times if update_times is not None else [])

            prop_trajectory = []
            true_trajectory = []
            t_all = []

            r_prop_1 = None
            v_prop_1 = None

            iteration = 0

            while t_init < prop_timeout:

                next_update = updates[0] if updates else None

                if next_update is not None and next_update <= t_init + prop_duration:
                    prop_end = next_update
                    update = True
                else:
                    prop_end = min(t_init + prop_duration, prop_timeout)
                    update = False

                indices = np.arange(t_init, prop_end + 1, step)
                if len(indices) == 0:
                    print("⚠️ Empty tspan, stopping")
                    break

                tspan = t_gps[indices]
                true_slice = r_target[indices]

                if iteration == 0 or update:
                    init_idx = t_init
                    v_sv_op = np.hstack([r_target[init_idx], v_target[init_idx]])
                else:
                    v_sv_op = np.hstack([r_prop_1[-1], v_prop_1[-1]])

                t, path = odeRK(orbit_prop, tspan, v_sv_op, substeps=30)

                r_prop_1 = path[:, :3]
                v_prop_1 = path[:, 3:6]

                print(f"Propagated indices {t_init}→{prop_end} | time {tspan[0]:.1f}→{tspan[-1]:.1f} ({tspan[-1]-tspan[0]:.1f}s) | steps={len(tspan)} | update={update}")

                prop_slice = r_prop_1

                assert prop_slice.shape[0] == true_slice.shape[0]

                prop_trajectory.append(prop_slice)
                true_trajectory.append(true_slice)
                t_all.append(t)

                # === FIXED ADVANCE LOGIC ===
                if update:
                    t_init = next_update
                    updates.popleft()
                else:
                    # After normal propagation, next start = last index we just propagated + 1
                    t_init = indices[-1]# prop_end + 1

                iteration += 1

            prop_trajectory = np.vstack(prop_trajectory) if prop_trajectory else np.array([])
            true_trajectory = np.vstack(true_trajectory) if true_trajectory else np.array([])
            t_all = np.hstack(t_all) if t_all else np.array([])

            return t_all, prop_trajectory, true_trajectory

def debug_timeline(t_init, prop_duration, updates, timeout):
   
    updates = deque(updates)
    timeline = []

    while t_init < timeout:
        next_update = updates[0] if updates else None

        if next_update is not None and next_update <= t_init + prop_duration:
            prop_end = next_update - 1
            timeline.append((t_init, prop_end, "UPDATE"))
            t_init = next_update
            updates.popleft()
        else:
            prop_end = min(t_init + prop_duration - 1, timeout)
            timeline.append((t_init, prop_end, "NO UPDATE"))
            t_init = prop_end + 1

    return timeline

def plot_timeline(timeline, updates, timeout):
    fig, ax = plt.subplots(figsize=(12, 2))

    # Plot propagation segments
    for (start, end, kind) in timeline:
        color = "tab:blue" if kind == "NO UPDATE" else "tab:orange"
        ax.hlines(1, start, end, linewidth=8, color=color)

        # Label segment
        ax.text((start + end)/2, 1.05, f"{start}-{end}",
                ha='center', va='bottom', fontsize=8)

    # Plot update lines
    for u in updates:
        ax.axvline(u, color='red', linestyle='--', alpha=0.7)
        ax.text(u, 0.85, f"U@{u}", rotation=90,
                ha='center', va='top', fontsize=8, color='red')

    # Formatting
    ax.set_ylim(0.7, 1.3)
    ax.set_xlim(0, timeout)
    ax.set_yticks([])
    ax.set_xlabel("Time [sec]")
    ax.set_title("Propagation Timeline with Updates")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=6, label='No Update'),
        Line2D([0], [0], color='tab:orange', lw=6, label='Pre-Update Propagation'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Update Arrival')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    #plt.show()

def animate_propagation(t_all, true_traj, prop_traj, updates):
    fig, ax = plt.subplots()

    ax.set_title("Propagation vs Truth (Animated)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    true_line, = ax.plot([], [], 'g-', label="True")
    prop_line, = ax.plot([], [], 'b--', label="Propagated")
    update_scatter = ax.scatter([], [], color='red', label="Updates")

    ax.legend()

    def init():
        ax.set_xlim(np.min(true_traj[:,0]), np.max(true_traj[:,0]))
        ax.set_ylim(np.min(true_traj[:,1]), np.max(true_traj[:,1]))
        return true_line, prop_line, update_scatter

    def update(frame):
        true_line.set_data(true_traj[:frame, 0], true_traj[:frame, 1])
        prop_line.set_data(prop_traj[:frame, 0], prop_traj[:frame, 1])

        # mark update points
        update_idx = [i for i, t in enumerate(t_all[:frame]) if t in updates]
        if update_idx:
            update_scatter.set_offsets(prop_traj[update_idx, :2])

        return true_line, prop_line, update_scatter

    ani = FuncAnimation(fig, update, frames=len(t_all),
                        init_func=init, interval=50, blit=True)

    plt.show()

def compute_update_jumps(t_all, true_traj, prop_traj, updates):
   
    error = np.linalg.norm(true_traj - prop_traj, axis=1)

    jumps = []

    for u in updates:
        idx = np.where(t_all == u)[0]
        if len(idx) == 0:
            continue

        i = idx[0]

        if i > 0:
            before = error[i-1]
            after = error[i]
            jumps.append((u, before, after, before - after))

    return jumps

def compute_pointing_error(t_rel, true_traj, prop_traj):
    # normalize vectors
    true_unit = true_traj / np.linalg.norm(true_traj, axis=1, keepdims=True)
    prop_unit = prop_traj / np.linalg.norm(prop_traj, axis=1, keepdims=True)

    # dot product
    dot = np.sum(true_unit * prop_unit, axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    # angle in radians → degrees
    angle = np.arccos(dot)

    plt.figure(figsize=(10, 5))
    plt.plot(t_rel, angle, label="Pointing Error", color='blue', linewidth=2)
    
    # Mark update times
    for u in updates:
        plt.axvline(u, color='red', linestyle='--', alpha=0.7, label='Update' if u == updates[0] else "")
    
    plt.title("Pointing Error vs Time (link distance = 100 km)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Pointing Error [µrad]")
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()

    return 1e6*(angle)  # µrad

def orbit_prop(t, vec_sv):
        # Constants
        mu = 398600.44  # km^3/s^2
        J2 = 1082.6267e-6
        R = 6378.1366  # km, Earth Radius
        
        r0 = vec_sv[:3]
        v0 = vec_sv[3:6]
        r_norm = max(np.linalg.norm(r0),1e-9)
        a1 = -(mu / r_norm**3) * r0
        
        # J2 Perturbations
        const = -3 * J2 * mu * R**2 / (2 * r_norm**5)
        #x, y, z = r0
        ai = const *r0[0]* (5 * r0[2]**2 / r_norm**2 - 1)
        aj = const *r0[1]* (5 * r0[2]**2 / r_norm**2 - 1)
        ak = const *r0[2]* (5 * r0[2]**2 / r_norm**2 - 3)
        acc = np.array([ai, aj, ak])
        a = a1 + acc
        
        return np.hstack([v0, a])

def odeRK(forbit, tspan, x0, substeps=20):
    """Improved RK4 with internal sub-stepping for stability"""
    N = len(tspan)
    n = len(x0)
    x0 = x0.reshape(-1, 1)
    x = np.zeros((N, n))
    x[0, :] = x0.flatten()
    w = x0.flatten()

    for i in range(N-1):
        h_outer = tspan[i+1] - tspan[i]          # usually 1.0 s
        h = h_outer / substeps                   # smaller internal step

        t = tspan[i]
        for _ in range(substeps):
            K1 = h * forbit(t, w)
            K2 = h * forbit(t + h/2, w + K1/2)
            K3 = h * forbit(t + h/2, w + K2/2)
            K4 = h * forbit(t + h, w + K3)
            w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
            t += h

        x[i+1, :] = w

    return tspan, x

def plot_error(t_rel, true_traj, prop_traj, updates, link_distance_m=100000.0):                                               

    """
    Plot pointing error in micro-radians
    link_distance: distance to the target satellite in meters (default 100 km)
    """
    
    # Position difference in meters
    delta_r = (true_traj - prop_traj)*1e3   # convert km to m
    pos_error_m = np.linalg.norm(delta_r, axis=1)
    
    # Pointing error in radians → micro-radians
    pointing_error_urad = 1e6 * np.arctan2(pos_error_m, link_distance_m)

    
    plt.figure(figsize=(10, 5))
    plt.plot(t_rel, pointing_error_urad, label="Pointing Error", color='blue', linewidth=2)
    
    # Mark update times
    for u in updates:
        plt.axvline(u, color='red', linestyle='--', alpha=0.7, label='Update' if u == updates[0] else "")
    
    plt.title(f"Pointing Error vs Time (link distance = {link_distance_m*1e-3} km)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Pointing Error [µrad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig,ax = plt.subplots(4,1, sharex=True, figsize=(8,6))
    ax[0].plot(t_rel, delta_r[:,0], label="True X", color='green')
    ax[1].plot(t_rel, delta_r[:,1], label="True Y", color='orange')
    ax[2].plot(t_rel, delta_r[:,2], label="True Z", color='purple')
    ax[3].plot(t_rel, pos_error_m, label="$\delta", color='black', alpha=0.5)
    for u in updates:
        for a in ax:
            a.axvline(u, color='red', linestyle='--', alpha=0.7)
    ax[0].set_title("Position Error Components vs Time")
    ax[2].set_xlabel("Time [sec]")
    ax[0].set_ylabel("Error X [m]")
    ax[1].set_ylabel("Error Y [m]")
    ax[2].set_ylabel("Error Z [m]")
    ax[3].set_ylabel("$\delta [m]")
    ax[1].grid(True)
    ax[0].grid(True)
    ax[2].grid(True)
    ax[3].grid(True)
    plt.tight_layout()

    return pointing_error_urad
                
t_init = 0
# prop_duration = 10
# prop_timeout = 100                         
update_step = 10
update_no = 3
link_distance_m = 250e3
latency = 0
updates = t_init + np.array([update_step * (i + 1) for i in range(update_no)])

target_delay = latency# int(input("\nTARGET initial ephemeris delay in sec [max. 50]? (If no delay, put 0): "))
if target_delay > 50:
    print("Error: Target delay exceeds maximum of 50 seconds.")
    exit(1)

# Target initial conditions
t_init = latency # target_delay

## Propagation parameters
#update_no = int(input("\nHow many ephemeris updates will be received? : "))
start_point = 0
prop_duration = 30
prop_timeout = start_point + 100
print(f"SDA limiting propagation timeout (considering HOST position shift by {start_point}): {prop_timeout} sec")

# Load your satellite data
data_dir = r"/home/bkhan/Documents/Git/astropynaric/examples/output_data/pointing_error"
file_path = os.path.join(data_dir,'state_history.dat')
sat_data = pd.read_csv(file_path, sep='\t', header = None, comment='#')
r_target = sat_data.iloc[:,13:16].values*1e-3 # in km 
v_target = sat_data.iloc[:,16:19].values*1e-3 # in km/s
t_gps = t_conv.j2000_to_gps(sat_data.iloc[:,0].values)

# Convert updates to GPS time for proper alignment
updates_gps = t_gps[updates.astype(int)]

# Run propagation
t_all, prop_traj, true_traj = propagate_with_updates(
    t_init=t_init,
    prop_duration=prop_duration,
    prop_timeout=prop_timeout,
    t_gps=t_gps,
    r_target=r_target,
    v_target=v_target,
    odeRK=odeRK,
    orbit_prop=orbit_prop,
    update_times=updates
)

# Compute error jumps
jumps = compute_update_jumps(t_all, true_traj, prop_traj, updates)  # use index-based updates
for u, before, after, improvement in jumps:
    print(f"Update @ {u:.1f}: error {before:.6f} → {after:.6f} (Δ = {improvement:.6f})")

# Plot error
t_rel = t_all - t_all[0]
updates_rel = np.array(updates)   # since updates are indices starting from 0


pointing_error = plot_error(t_rel, true_traj, prop_traj, updates_rel, link_distance_m)


# Optional: print some statistics
print(f"Max pointing error: {np.max(pointing_error):.2f} µrad")
print(f"Mean pointing error: {np.mean(pointing_error):.2f} µrad")
# print(f" Pointing error: {np.max(PE):.2f} µrad")

#plot_error(t_rel, true_traj, prop_traj, updates_rel,link_distance_m)
# Plot pointing error instead of position error
#pointing_error = plot_error(t_rel, true_traj, prop_traj, updates_rel, link_distance_m)

# Optional: print some statistics
#print(f"Max pointing error: {pointing_error.max():.2f} µrad")
#print(f"Mean pointing error: {pointing_error.mean():.2f} µrad")
plt.show()
# %%
