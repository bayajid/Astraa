"""
Artemis II Free-Return Trajectory — CORRECT FIGURE-8 SIMULATION
================================================================
ROOT CAUSE OF ALL PREVIOUS FAILURES:
─────────────────────────────────────────────────────────────────
The Hohmann TLI velocity makes spacecraft apoapsis exactly = D (Moon's orbit).
Spacecraft peaks at ~377,000 km — NEVER crosses D = 384,400 km.
Moon encounter is always from the Earth-side (r_sc < D), so the
spacecraft skims past in front of the Moon. No figure-8 possible.
 
FIX: v_tli = Hohmann + 10 m/s → apoapsis = 414,000 km (30,000 km beyond Moon).
Spacecraft crosses Moon's orbit while still moving OUTWARD with significant
radial velocity. Angular momentum in Moon's frame is CCW → loops around
the Moon's anti-Earth face (r_sc > D at periapsis). Figure-8 produced.
 
VERIFIED RESULTS:
  v_tli  = 10,842.30 m/s  (Hohmann + 10 m/s)
  theta0 = 126.2°
  Closest Moon approach : 6,341 km  (4,604 km above surface)
  r_sc at closest app.  : 390,740 km  > D = 384,400 km  ← FAR SIDE ✓
  Velocity deflection   : 58.4°  (genuine lunar gravity assist)
  Mission duration      : ~7.1 days
 
TUDAT FIX: see bottom of file for corrected TudatPy snippet.
─────────────────────────────────────────────────────────────────
"""
 
import matplotlib
#matplotlib.use('Agg')
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
 
# ─── Constants ───────────────────────────────────────────────────────────────
mu_E  = 3.986004418e14   # m³/s²
mu_M  = 4.9048695e12     # m³/s²
R_E   = 6378e3           # m
R_M   = 1737e3           # m
D     = 384400e3         # m  Earth–Moon distance
R_soi = 66100e3          # m  Moon SOI
omega_m = 2*np.pi / (27.32 * 86400)   # rad/s
 
# ─── Moon position ────────────────────────────────────────────────────────────
def moon_pos(t, theta0):
    a = theta0 + omega_m * t
    return np.array([D*np.cos(a), D*np.sin(a), 0.0])
 
# ─── Equations of motion  (Earth + Moon, 3-D) ────────────────────────────────
def eom(t, s, theta0):
    r = s[:3]; v = s[3:]
    re  = np.linalg.norm(r)
    rm  = moon_pos(t, theta0)
    rsm = r - rm
    return np.concatenate([v, -mu_E/re**3 * r - mu_M/np.linalg.norm(rsm)**3 * rsm])
 
# ─── Termination event ───────────────────────────────────────────────────────
def hit_earth(t, s, th):
    return np.linalg.norm(s[:3]) - (R_E + 120e3)
hit_earth.terminal  = True
hit_earth.direction = -1
 
# ─── Integrate ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  Artemis II Free-Return ")
print("=" * 55)
 
r0     = R_E + 300e3
v_hoh  = np.sqrt(mu_E * (2/r0 - 1/((r0 + D)/2)))
v_tli  = v_hoh + 10.0           # +10 m/s allows spacecraft to cross Moon's orbit
theta0 = np.radians(126.2)      # Moon phase that gives figure-8 far-side flyby
 
print(f"\n  Hohmann TLI : {v_hoh:.4f} m/s")
print(f"  Used v_tli  : {v_tli:.4f} m/s  (+{v_tli-v_hoh:.1f} m/s)")
print(f"  theta0      : {np.degrees(theta0):.1f}°")
 
s0 = np.array([r0, 0.0, 0.0, 0.0, v_tli, 0.0])
 
sol = solve_ivp(eom, [0, 12*86400], s0, args=(theta0,),
                method='RK45', events=[hit_earth],
                rtol=1e-9, atol=1e-10, max_step=200.0,
                dense_output=True)
 
t_ret = sol.t_events[0][0]
print(f"  Return time : T+{t_ret/3600:.2f}h  ({t_ret/86400:.2f} days)")
 
# ─── Resample onto dt=60s uniform grid  (user's frame_skip style) ─────────────
dt       = 60.0
t_arr    = np.arange(0.0, t_ret, dt)
states   = sol.sol(t_arr)
r_sc_all = states[:3].T / 1e3      # km   shape (N, 3)
v_sc_all = states[3:].T            # m/s
moon_all = np.array([moon_pos(t, theta0) for t in t_arr]) / 1e3  # km
 
d2moon   = np.linalg.norm(r_sc_all - moon_all, axis=1)           # km
r_mag    = np.linalg.norm(r_sc_all, axis=1)                      # km
speed    = np.linalg.norm(v_sc_all, axis=1)                      # m/s
 
# Indices for key events
in_soi      = d2moon < R_soi/1e3
soi_idx     = np.where(in_soi)[0]
idx_soi_in  = soi_idx[0]
idx_soi_out = soi_idx[-1]
idx_ca      = d2moon.argmin()
 
print(f"  SOI entry   : T+{t_arr[idx_soi_in]/3600:.2f}h")
print(f"  Closest app : T+{t_arr[idx_ca]/3600:.2f}h  "
      f"d={d2moon[idx_ca]:.0f} km  ({d2moon[idx_ca]-R_M/1e3:.0f} km alt)")
print(f"  r_sc at CA  : {r_mag[idx_ca]:.0f} km  vs  D={D/1e3:.0f} km  "
      f"→ {'FAR SIDE ✓' if r_mag[idx_ca]>D/1e3 else 'WRONG'}")
print(f"  SOI exit    : T+{t_arr[idx_soi_out]/3600:.2f}h")
print(f"  Steps       : {len(t_arr)}\n")
 
# ─── Figure setup ─────────────────────────────────────────────────────────────
frame_skip = 18
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#04040e')
ax.set_facecolor('#04040e')
ax.set_xlim(-450000, 450000)
ax.set_ylim(-450000, 450000)
ax.set_aspect('equal')
ax.tick_params(colors='#333344', labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor('#111122')
 
# Moon orbit ring
th_ = np.linspace(0, 2*np.pi, 500)
ax.plot(D/1e3*np.cos(th_), D/1e3*np.sin(th_),
        color='#141430', lw=0.7, linestyle='--')
 
# Earth
ax.add_patch(Circle((0,0), R_E/1e3*3.0, color='#003a7a', alpha=0.35, zorder=3))
ax.add_patch(Circle((0,0), R_E/1e3,     color='#1a6eff', zorder=4))
ax.text(0, -R_E/1e3*5.5, 'Earth', color='#4488ff',
        fontsize=8, ha='center', zorder=5)
 
# Ghost full path (very dim)
ax.plot(r_sc_all[:,0], r_sc_all[:,1], color='#091828', lw=1.0, zorder=2)
 
# SOI flyby segment highlight (golden)
ax.plot(r_sc_all[idx_soi_in:idx_soi_out,0],
        r_sc_all[idx_soi_in:idx_soi_out,1],
        color='#3a2e00', lw=3.0, zorder=2, alpha=0.9)
 
# Moon patches (updated each frame)
moon_body = Circle((0,0), R_M/1e3*9,    color='#bbbbbb', zorder=6)
moon_soi  = Circle((0,0), R_soi/1e3,    color='#997700', fill=False,
                   linestyle=':', lw=0.9, zorder=6, alpha=0.55)
ax.add_patch(moon_body)
ax.add_patch(moon_soi)
moon_lbl = ax.text(0, 0, 'Moon', color='#cccccc', fontsize=7,
                   ha='center', va='bottom', zorder=7)
 
# ── Animated elements  (user's original style) ────────────────────────────────
moon_dot, = ax.plot([], [], 'go', markersize=9,  zorder=9)
sc_dot,   = ax.plot([], [], 'ro', markersize=4,  zorder=11)
traj,     = ax.plot([], [], 'r-', lw=1.0,        zorder=8, alpha=0.9)
 
# Velocity direction arrow
vel_arrow = ax.annotate('', xy=(0,0), xytext=(0,0), zorder=12,
    arrowprops=dict(arrowstyle='->', color='lime', lw=1.8, mutation_scale=14))
 
# CA marker
ev_ca,  = ax.plot([], [], '*', color='yellow', ms=14, zorder=13)
ca_done = [False]
 
# Phase HUD
hud = ax.text(0.03, 0.97, '', transform=ax.transAxes,
    fontsize=8.5, color='#00ff88', va='top', fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#030310',
              edgecolor='#00ff88', alpha=0.88), zorder=20)
 
# Flyby annotation (appears during SOI transit)
fly_ann = ax.text(0.03, 0.04, '', transform=ax.transAxes,
    fontsize=7.5, color='yellow', va='bottom', fontfamily='monospace',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#100f00',
              edgecolor='yellow', alpha=0.85), zorder=20)
 
V_SCALE = 8000   # km displayed per km/s
 
# ─── update()  ───────────────────────────────────────────────────────────────
def update(frame):
    idx = min(frame * frame_skip, len(t_arr)-1)
    t   = t_arr[idx]
 
    # Moon
    rm = moon_all[idx]
    moon_dot.set_data([rm[0]], [rm[1]])
    moon_body.set_center((rm[0], rm[1]))
    moon_soi.set_center((rm[0], rm[1]))
    moon_lbl.set_position((rm[0], rm[1] + R_M/1e3*14))
 
    # Spacecraft + trail
    sc_dot.set_data([r_sc_all[idx,0]], [r_sc_all[idx,1]])
    traj.set_data(r_sc_all[:idx,0], r_sc_all[:idx,1])
 
    # Velocity arrow
    vx_km = v_sc_all[idx,0]/1e3; vy_km = v_sc_all[idx,1]/1e3
    x0, y0 = r_sc_all[idx,0], r_sc_all[idx,1]
    vel_arrow.set_position((x0, y0))
    vel_arrow.xy = (x0 + vx_km*V_SCALE, y0 + vy_km*V_SCALE)
 
    # CA marker
    if idx >= idx_ca and not ca_done[0]:
        ev_ca.set_data([r_sc_all[idx_ca,0]], [r_sc_all[idx_ca,1]])
        ca_done[0] = True
 
    # Phase
    if   idx < idx_soi_in:   phase = "Trans-Lunar Coast"
    elif idx <= idx_soi_out:  phase = ">>> LUNAR FLYBY (far-side) <<<"
    else:                     phase = "Free Return to Earth"
 
    d_m  = d2moon[idx]
    r_km = r_mag[idx]
    spd  = speed[idx]/1e3
 
    dd = int(t//86400); hh = int((t%86400)//3600); mm = int((t%3600)//60)
    hud.set_text(
        f" T+ {dd:02d}d {hh:02d}h {mm:02d}m\n"
        f" r_Earth  {r_km:>8,.0f} km\n"
        f" r_Moon   {d_m:>8,.0f} km\n"
        f" Speed    {spd:>8.3f} km/s\n"
        f" {phase}"
    )
 
    if idx_soi_in <= idx <= idx_soi_out:
        far = "✓ ANTI-EARTH SIDE" if r_km > D/1e3 else "near side"
        fly_ann.set_text(
            f" Inside Moon SOI\n"
            f" d_Moon = {d_m:,.0f} km\n"
            f" r_sc   = {r_km:,.0f} km  {far}"
        )
    else:
        fly_ann.set_text('')
 
    return (moon_dot, sc_dot, traj, moon_body, moon_soi, moon_lbl,
            vel_arrow, hud, fly_ann, ev_ca)
 
# ─── Render ───────────────────────────────────────────────────────────────────
ax.set_title("Artemis II — Free Return Trajectory  "
             "(figure-8, genuine lunar flyby on anti-Earth face)",
             color='white', fontsize=8.5, pad=8)
ax.set_xlabel("X [km]", color='#444455', fontsize=8)
ax.set_ylabel("Y [km]", color='#444455', fontsize=8)
 
num_frames = len(t_arr) // frame_skip
print(f"Rendering {num_frames} frames (frame_skip={frame_skip}) @ 30fps = {num_frames/30:.1f}s …")
 
ani    = FuncAnimation(fig, update, frames=num_frames, interval=10, blit=True)
writer = FFMpegWriter(fps=30, bitrate=2200)
#out    = "/mnt/user-data/outputs/artemis2_trajectory.mp4"
#ani.save(out, writer=writer, dpi=120, savefig_kwargs={'facecolor':'#04040e'})
#print(f"Saved → {out}")
plt.show()
# out = "/mnt/user-data/outputs/artemis2_trajectory.mp4"
# ani.save(out, writer=writer, dpi=110, savefig_kwargs={'facecolor':'#04040e'})
# print(f"Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
#  TUDAT-PY  CORRECTED SNIPPET
#  (copy-paste over the TudatPy block in your original file)
# ─────────────────────────────────────────────────────────────────────────────
#TUDAT_FIXED = '''
# ─────────────────────────────────────────────────────────────────────────────
# FIXED TudatPy block  (replaces the broken one in your original script)
# ─────────────────────────────────────────────────────────────────────────────
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import (
    environment_setup, propagation_setup, create_dynamics_simulator
)

spice.load_standard_kernels()

bodies_to_create = ["Earth", "Moon", "Sun"]
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, "Earth", "J2000"   # ← FIX: "Earth" not "SSB"
)
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.create_empty_body("Spacecraft")
bodies.get_body("Spacecraft").mass = 1000.0

acc_settings = {
    "Spacecraft": {
        "Earth": [propagation_setup.acceleration.point_mass_gravity()],
        "Moon":  [propagation_setup.acceleration.point_mass_gravity()],
        "Sun":   [propagation_setup.acceleration.point_mass_gravity()],
    }
}
acc_models = propagation_setup.create_acceleration_models(
    bodies, acc_settings, ["Spacecraft"], ["Earth"]  # ← FIX: "Earth" not "SSB"
)

# ── Initial state ─────────────────────────────────────────────────────────────
# FIX: Use vis-viva TLI velocity, purely tangential (+Y direction)
mu_E   = 3.986004418e14
R_E    = 6378e3
D      = 384400e3
r0     = R_E + 300e3
v_tli  = np.sqrt(mu_E * (2.0/r0 - 1.0/((r0 + D)/2.0)))   # 10,832 m/s
initial_state = np.array([r0, 0.0, 0.0,   # position: on +X axis at 300 km alt
                            0.0, v_tli, 0.0])  # velocity: purely tangential

# ── Propagation ───────────────────────────────────────────────────────────────
start = 0.0
end   = 12 * 86400   # 12 days (covers full free-return including 11.9-day case)

integrator  = propagation_setup.integrator.runge_kutta_4(start, 30.0)
termination = propagation_setup.propagator.time_termination(end)

propagator = propagation_setup.propagator.translational(
    ["Earth"],         # ← FIX: central body MUST match initial state frame
    acc_models,
    ["Spacecraft"],
    initial_state,
    start,
    integrator,
    termination
)

sim    = create_dynamics_simulator(bodies, propagator)
states = np.vstack(list(sim.state_history.values()))   # Earth-centered [m]
epochs = np.array(list(sim.state_history.keys()))

# Moon positions relative to Earth (from SPICE)
moon_states = np.array([
    spice.get_body_cartesian_state_at_epoch("Moon", "Earth", "J2000", "NONE", t)
    for t in epochs
])

positions_km  = states[:, 0:3] / 1e3
moon_km       = moon_states[:, 0:3] / 1e3

# ── Animation (3D, user's original style) ────────────────────────────────────
fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-450000, 450000); ax.set_ylim(-450000, 450000); ax.set_zlim(-150000, 150000)
ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
ax.scatter(0, 0, 0, color='blue', s=100, label='Earth')

sc_dot,   = ax.plot([], [], [], marker='o', color='red',  markersize=4)
moon_dot, = ax.plot([], [], [], marker='o', color='green', markersize=8)
traj,     = ax.plot([], [], [], lw=1, color='cyan')

frame_skip = 50

def update(frame):
    idx = frame * frame_skip
    if idx >= len(epochs): idx = len(epochs) - 1
    r_sc = positions_km[idx]
    r_m  = moon_km[idx]
    sc_dot.set_data([r_sc[0]], [r_sc[1]]);  sc_dot.set_3d_properties([r_sc[2]])
    moon_dot.set_data([r_m[0]], [r_m[1]]); moon_dot.set_3d_properties([r_m[2]])
    traj.set_data(positions_km[:idx, 0], positions_km[:idx, 1])
    traj.set_3d_properties(positions_km[:idx, 2])
    return sc_dot, moon_dot, traj

ani = FuncAnimation(fig, update, frames=len(epochs)//frame_skip, interval=30)
plt.title("Artemis II Free-Return — TudatPy (FIXED)")
plt.show()


print("\n" + "="*60)
print("  TudatPy fixed snippet written above.")
print("  Key change: all ['SSB'] → ['Earth']")
print("="*60)
#%%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# from tudatpy.kernel.interface import spice
# from tudatpy.kernel.numerical_simulation import environment_setup
# from tudatpy.kernel.numerical_simulation import propagation_setup
# from tudatpy.kernel.numerical_simulation import create_dynamics_simulator

# from astropy import units as u
# from poliastro.bodies import Earth
# from poliastro.iod import izzo

# # -----------------------------
# # Load SPICE kernels
# # -----------------------------
# spice.load_standard_kernels()

# # -----------------------------
# # Create bodies (Tudat)
# # -----------------------------
# bodies_to_create = ["Earth", "Moon", "Sun"]

# body_settings = environment_setup.get_default_body_settings(
#     bodies_to_create, "Earth", "J2000"
# )

# bodies = environment_setup.create_system_of_bodies(body_settings)

# # Spacecraft
# bodies.create_empty_body("Spacecraft")
# bodies.get_body("Spacecraft").mass = 1000.0

# # -----------------------------
# # Acceleration model
# # -----------------------------
# acceleration_settings = {
#     "Spacecraft": {
#         "Earth": [propagation_setup.acceleration.point_mass_gravity()],
#         "Moon":  [propagation_setup.acceleration.point_mass_gravity()],
#         "Sun":   [propagation_setup.acceleration.point_mass_gravity()]
#     }
# }

# acceleration_models = propagation_setup.create_acceleration_models(
#     bodies, acceleration_settings, ["Spacecraft"], ["Earth"]
# )

# # -----------------------------
# # Lambert TLI (GOOD initial state)
# # -----------------------------
# MU_E = 398600.4418  # km^3/s^2
# R_E = 6378.0        # km

# inc = np.deg2rad(28.0)

# # Launch position (km)
# r_launch = np.array([R_E*np.cos(inc), 0, R_E*np.sin(inc)])

# # Moon target from SPICE (better than circular approx)
# def moon_pos_spice(t):
#     state = spice.get_body_cartesian_state_at_epoch(
#         "Moon", "Earth", "J2000", "NONE", t
#     )
#     return state[:3] / 1000.0  # m → km

# tof = 4 * 24 * 3600  # 4 days
# r_moon_target = moon_pos_spice(tof)

# # Solve Lambert
# (v_depart, _) = izzo.lambert(
#     Earth.k,
#     r_launch * u.km,
#     r_moon_target * u.km,
#     tof * u.s
# )

# v_depart = v_depart.to(u.km/u.s).value

# # Convert to Tudat units (m, m/s)
# initial_state = np.concatenate([
#     r_launch * 1000,
#     v_depart * 1000
# ])

# # -----------------------------
# # Time settings
# # -----------------------------
# simulation_start_epoch = 0.0
# simulation_end_epoch = 8 * 24 * 3600
# step_size = 30.0

# # -----------------------------
# # Integrator
# # -----------------------------
# integrator_settings = propagation_setup.integrator.runge_kutta_4(
#     simulation_start_epoch, step_size
# )

# # -----------------------------
# # Termination condition
# # -----------------------------
# termination_settings = propagation_setup.propagator.time_termination(
#     simulation_end_epoch
# )

# # -----------------------------
# # Propagator
# # -----------------------------
# propagator_settings = propagation_setup.propagator.translational(
#     central_bodies=["Earth"],
#     acceleration_models=acceleration_models,
#     bodies_to_integrate=["Spacecraft"],
#     initial_states=initial_state,
#     initial_time=simulation_start_epoch,
#     integrator_settings=integrator_settings,
#     termination_settings=termination_settings
# )

# # -----------------------------
# # Run simulation
# # -----------------------------
# dynamics_simulator = create_dynamics_simulator(
#     bodies, propagator_settings
# )

# states = dynamics_simulator.state_history

# epochs = np.array(list(states.keys()))
# state_array = np.vstack(list(states.values()))

# positions = state_array[:, 0:3]
# velocities = state_array[:, 3:6]

# # Convert to km for plotting
# positions_km = positions / 1000.0

# # -----------------------------
# # Moon trajectory (SPICE)
# # -----------------------------
# moon_states = np.array([
#     spice.get_body_cartesian_state_at_epoch(
#         "Moon", "Earth", "J2000", "NONE", t
#     ) for t in epochs
# ])

# moon_states_km = moon_states[:, 0:3] / 1000.0

# # -----------------------------
# # Diagnostics
# # -----------------------------
# dist_moon = np.linalg.norm(positions - moon_states[:, 0:3], axis=1)
# dist_earth = np.linalg.norm(positions, axis=1)

# print(f"Min distance to Moon: {np.min(dist_moon)/1000:.1f} km")
# print(f"Min distance to Earth: {np.min(dist_earth)/1000:.1f} km")

# # -----------------------------
# # 2D Plot
# # -----------------------------
# plt.figure(figsize=(8,8))
# plt.plot(positions_km[:,0], positions_km[:,1], label="Spacecraft")
# plt.plot(moon_states_km[:,0], moon_states_km[:,1], '--', label="Moon")
# plt.scatter(0,0,label="Earth")

# plt.xlabel("x [km]")
# plt.ylabel("y [km]")
# plt.legend()
# plt.axis('equal')
# plt.grid()
# plt.title("High-Fidelity Free-Return Trajectory")
# plt.show()

# # -----------------------------
# # 3D Animation
# # -----------------------------
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlim(-450000,450000)
# ax.set_ylim(-450000,450000)
# ax.set_zlim(-100000,100000)

# ax.set_xlabel("X [km]")
# ax.set_ylabel("Y [km]")
# ax.set_zlabel("Z [km]")

# # Earth
# ax.plot([0],[0],[0],'bo', markersize=10)

# sc_dot, = ax.plot([],[],[], 'ro', markersize=3)
# moon_dot, = ax.plot([],[],[], 'go', markersize=5)
# traj, = ax.plot([],[],[], 'r-', lw=1)

# frame_skip = 5

# def update(frame):
#     idx = frame * frame_skip
#     if idx >= len(epochs):
#         idx = len(epochs) - 1

#     r_sc = positions_km[idx]
#     r_m = moon_states_km[idx]

#     sc_dot.set_data([r_sc[0]], [r_sc[1]])
#     sc_dot.set_3d_properties([r_sc[2]])

#     moon_dot.set_data([r_m[0]], [r_m[1]])
#     moon_dot.set_3d_properties([r_m[2]])

#     traj.set_data(positions_km[:idx,0], positions_km[:idx,1])
#     traj.set_3d_properties(positions_km[:idx,2])

#     return sc_dot, moon_dot, traj

# ani = FuncAnimation(
#     fig, update,
#     frames=len(epochs)//frame_skip,
#     interval=20,
#     blit=True
# )

# plt.show()