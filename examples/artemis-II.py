# import numpy as np
# from matplotlib import pyplot as plt

# from tudatpy.kernel import constants
# from tudatpy.kernel.interface import spice
# from tudatpy.kernel.numerical_simulation import environment_setup
# from tudatpy.kernel.numerical_simulation import propagation_setup
# from tudatpy.kernel.numerical_simulation import create_dynamics_simulator
# from tudatpy.kernel.astro import element_conversion

# # -----------------------------
# # Load SPICE kernels
# # -----------------------------
# spice.load_standard_kernels()

# # -----------------------------
# # Create bodies
# # -----------------------------
# bodies_to_create = ["Earth", "Moon", "Sun"]

# body_settings = environment_setup.get_default_body_settings(
#     bodies_to_create,
#     "Earth",
#     "J2000"
# )

# bodies = environment_setup.create_system_of_bodies(body_settings)

# # Add spacecraft
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
# # Initial state (TLI-like)
# # -----------------------------
# mu_E = bodies.get_body("Earth").gravitational_parameter

# r0 = 6678e3  # LEO radius
# v0 = 10850   # TLI velocity (tune this!)

# # Position along x-axis, velocity mostly tangential
# # initial_state = np.array([
# #     r0, 0, 0,
# #     0, v0, 0
# # ])
# initial_state = [r0, 0, 0, 200, 10800, 0]
# # -----------------------------
# # Time settings
# # -----------------------------
# simulation_start_epoch = 0.0
# simulation_end_epoch = 8 * 24 * 3600  # 8 days
# step_size=30.0
# # -----------------------------
# # Integrator
# # -----------------------------
# integrator_settings = propagation_setup.integrator.runge_kutta_4(
#     simulation_start_epoch, step_size)
# # integrator_settings = propagation_setup.integrator.runge_kutta_4(
# #                 simulation_start_epoch, fixed_step_size
# #             )

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

# # Convert to arrays
# epochs = np.array(list(states.keys()))
# state_array = np.vstack(list(states.values()))

# # -----------------------------
# # Extract positions
# # -----------------------------
# x = state_array[:,0]
# y = state_array[:,1]
# z = state_array[:,2]

# # -----------------------------
# # Moon position for reference
# # -----------------------------
# moon_states = [
#     spice.get_body_cartesian_state_at_epoch(
#         target_body_name="Moon",
#         observer_body_name="Earth",
#         reference_frame_name="J2000",
#         aberration_corrections="NONE",
#         ephemeris_time=t
#     ) for t in epochs
# ]

# moon_states = np.array(moon_states)

# # -----------------------------
# # Plot trajectory
# # -----------------------------
# plt.figure(figsize=(8,8))

# plt.plot(x/1e6, y/1e6, label="Spacecraft")
# plt.plot(moon_states[:,0]/1e6, moon_states[:,1]/1e6, '--', label="Moon")

# plt.scatter(0,0,label="Earth")

# plt.xlabel("x [Mm]")
# plt.ylabel("y [Mm]")
# plt.legend()
# plt.axis('equal')
# plt.title("Free-Return Trajectory (High-Fidelity)")
# plt.grid()

# plt.show()

# # -----------------------------
# # Basic checks
# # -----------------------------
# dist_to_moon = np.linalg.norm(
#     state_array[:,0:3] - moon_states[:,0:3], axis=1
# )

# dist_to_earth = np.linalg.norm(state_array[:,0:3], axis=1)

# print(f"Min distance to Moon: {np.min(dist_to_moon)/1e3:.1f} km")
# print(f"Min distance to Earth (return): {np.min(dist_to_earth)/1e3:.1f} km")

# #%%
# import numpy as np
# from tudatpy.kernel.interface import spice
# from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, create_dynamics_simulator
# import numpy as np
# import matplotlib.pyplot as plt


# # Constants
# mu_E = 3.986004418e14
# mu_M = 4.9048695e12
# D = 384400e3
# R_s = 66100e3
# R_m = 1737e3
# omega_m = 2*np.pi / (27.32*86400)

# # -----------------------------
# # SLIDE 1: Departure
# # -----------------------------
# def departure(r0, v0, phi0):
#     E = v0**2 / 2 - mu_E / r0
#     h = r0 * v0 * np.cos(phi0)
#     return E, h

# # -----------------------------
# # SLIDE 1: At Moon SOI
# # -----------------------------
# def at_moon_soi(E, h, lambda1):
#     r1 = np.sqrt(D**2 + R_s**2 - 2*D*R_s*np.cos(lambda1))
#     v1 = np.sqrt(2*(E + mu_E/r1))
#     phi1 = np.arccos(h / (r1*v1))
#     return r1, v1, phi1

# # -----------------------------
# # SLIDE 2: Orbital elements + TOF
# # -----------------------------
# def orbital_elements(E, h):
#     p = h**2 / mu_E
#     a = -mu_E / (2*E)
#     e = np.sqrt(1 - p/a)
#     return p, a, e

# def true_anomaly(p, e, r):
#     return np.arccos((p - r)/(r*e))

# def eccentric_anomaly(e, f):
#     return np.arccos((e + np.cos(f)) / (1 + e*np.cos(f)))

# def time_of_flight(a, e, E0, E1):
#     return np.sqrt(a**3/mu_E) * ((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))

# # -----------------------------
# # SLIDE 3: Moon-relative velocity
# # -----------------------------
# def moon_relative(v1, phi1, gamma1):
#     v_m = D * omega_m
#     v2 = np.sqrt(v1**2 + v_m**2 - 2*v1*v_m*np.cos(phi1 - gamma1))
#     return v2, v_m

# # -----------------------------
# # SLIDE 4: Flyby hyperbola
# # -----------------------------
# def flyby(r2, v2, alpha):
#     E_m = v2**2 / 2 - mu_M / r2
#     h_m = r2 * v2 * np.sin(alpha)

#     p_m = h_m**2 / mu_M
#     e_m = np.sqrt(1 + (2*E_m*h_m**2)/(mu_M**2))

#     r_mp = p_m / (1 + e_m)

#     return E_m, h_m, e_m, r_mp

# # -----------------------------
# # MASTER FUNCTION
# # -----------------------------
# def compute_free_return():

#     r0 = 6678e3
#     v0 = 10900.0
#     phi0 = np.deg2rad(5)

#     lambda1 = np.deg2rad(30)
#     gamma1 = np.deg2rad(10)
#     alpha = np.deg2rad(30)

#     # Slide 1
#     E, h = departure(r0, v0, phi0)

#     # Slide 2
#     p, a, e = orbital_elements(E, h)

#     f0 = true_anomaly(p, e, r0)

#     # Moon SOI
#     r1, v1, phi1 = at_moon_soi(E, h, lambda1)

#     f1 = true_anomaly(p, e, r1)

#     E0 = eccentric_anomaly(e, f0)
#     E1 = eccentric_anomaly(e, f1)

#     tof = time_of_flight(a, e, E0, E1)

#     # Slide 3
#     v2, v_m = moon_relative(v1, phi1, gamma1)

#     # Slide 4
#     E_m, h_m, e_m, r_mp = flyby(R_s, v2, alpha)

#     print("\n--- ANALYTICAL RESULTS ---")
#     print(f"TOF to Moon: {tof/3600:.2f} hr")
#     print(f"Moon-relative velocity: {v2:.1f} m/s")
#     print(f"Hyperbola eccentricity: {e_m:.3f}")
#     print(f"Perilune altitude: {(r_mp - R_m)/1e3:.1f} km")

#     return v0


# v0_solution = compute_free_return()

# spice.load_standard_kernels()

# bodies_to_create = ["Earth", "Moon", "Sun"]

# body_settings = environment_setup.get_default_body_settings(
#     bodies_to_create, "SSB", "J2000"
# )

# bodies = environment_setup.create_system_of_bodies(body_settings)

# bodies.create_empty_body("Spacecraft")
# bodies.get_body("Spacecraft").mass = 1000.0

# acc_settings = {
#     "Spacecraft": {
#         "Earth": [propagation_setup.acceleration.point_mass_gravity()],
#         "Moon":  [propagation_setup.acceleration.point_mass_gravity()],
#         "Sun":   [propagation_setup.acceleration.point_mass_gravity()]
#     }
# }

# acc_models = propagation_setup.create_acceleration_models(
#     bodies, acc_settings, ["Spacecraft"], ["SSB"]
# )

# # Initial state (use analytical v0)
# r0 = 6678e3
# initial_state = np.array([r0, 0, 0, 0, v0_solution, 0])

# start = 0.0
# end = 8 * 86400

# integrator_settings = propagation_setup.integrator.runge_kutta_4(start, 30.0)

# termination = propagation_setup.propagator.time_termination(end)

# propagator = propagation_setup.propagator.translational(
#     ["SSB"],
#     acc_models,
#     ["Spacecraft"],
#     initial_state,
#     start,
#     integrator_settings,
#     termination
# )

# sim = create_dynamics_simulator(bodies, propagator)

# states = np.vstack(list(sim.state_history.values()))

# # Plot
# plt.figure(figsize=(8,8))
# plt.plot(states[:,0]/1e6, states[:,1]/1e6, label="Trajectory")
# plt.scatter(0,0,label="Earth")
# plt.axis('equal')
# plt.legend()
# plt.grid()
# plt.title("High-Fidelity Free Return")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from matplotlib.animation import FuncAnimation

# from tudatpy.kernel.interface import spice
# from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, create_dynamics_simulator

# # -----------------------------
# # Load SPICE
# # -----------------------------
# spice.load_standard_kernels()

# # -----------------------------
# # Create environment
# # -----------------------------
# bodies_to_create = ["Earth", "Moon", "Sun"]

# body_settings = environment_setup.get_default_body_settings(
#     bodies_to_create, "SSB", "J2000"
# )

# bodies = environment_setup.create_system_of_bodies(body_settings)

# bodies.create_empty_body("Spacecraft")
# bodies.get_body("Spacecraft").mass = 1000.0

# acc_settings = {
#     "Spacecraft": {
#         "Earth": [propagation_setup.acceleration.point_mass_gravity()],
#         "Moon":  [propagation_setup.acceleration.point_mass_gravity()],
#         "Sun":   [propagation_setup.acceleration.point_mass_gravity()]
#     }
# }

# acc_models = propagation_setup.create_acceleration_models(
#     bodies, acc_settings, ["Spacecraft"], ["SSB"]
# )

# # -----------------------------
# # Simulation function
# # -----------------------------
# def simulate(v0, vy_offset):

#     r0 = 6678e3

#     initial_state = np.array([
#         r0, 0, 0,
#         vy_offset, v0, 0
#     ])

#     start = 0.0
#     end = 8 * 86400

#     integrator = propagation_setup.integrator.runge_kutta_4(start, 60.0)

#     termination = propagation_setup.propagator.time_termination(end)

#     propagator = propagation_setup.propagator.translational(
#         ["SSB"],
#         acc_models,
#         ["Spacecraft"],
#         initial_state,
#         start,
#         integrator,
#         termination
#     )

#     sim = create_dynamics_simulator(bodies, propagator)

#     states = np.vstack(list(sim.state_history.values()))
#     epochs = np.array(list(sim.state_history.keys()))

#     # Moon states
#     moon_states = np.array([
#         spice.get_body_cartesian_state_at_epoch(
#             "Moon", "Earth", "J2000", "NONE", t
#         ) for t in epochs
#     ])

#     r_sc = states[:,0:3]
#     r_m = moon_states[:,0:3]

#     dist_moon = np.linalg.norm(r_sc - r_m, axis=1)
#     dist_earth = np.linalg.norm(r_sc, axis=1)

#     return dist_moon, dist_earth, states, epochs, moon_states


# # -----------------------------
# # Objective (automatic targeting)
# # -----------------------------
# def objective(x):

#     v0, vy = x

#     dist_moon, dist_earth, _, _, _ = simulate(v0, vy)

#     min_moon = np.min(dist_moon)
#     final_earth = dist_earth[-1]

#     # Want:
#     # - close Moon flyby (~50,000 km)
#     # - return to Earth (small final distance)

#     cost = (min_moon - 5e7)**2 + (final_earth - 7e6)**2

#     print(f"v0={v0:.1f}, vy={vy:.1f}, cost={cost:.2e}")

#     return cost


# # -----------------------------
# # Solve (single-shot)
# # -----------------------------
# x0 = [10850, 100]  # initial guess

# result = minimize(objective, x0, method='Nelder-Mead',
#                   options={'maxiter': 30})

# v0_opt, vy_opt = result.x

# print("\n=== OPTIMIZED ===")
# print(f"v0 = {v0_opt:.2f} m/s")
# print(f"vy offset = {vy_opt:.2f} m/s")


# # -----------------------------
# # Final simulation
# # -----------------------------
# dist_moon, dist_earth, states, epochs, moon_states = simulate(v0_opt, vy_opt)

# positions_km = states[:,0:3] / 1000
# moon_states_km = moon_states[:,0:3] / 1000


# # -----------------------------
# # 3D Animation (FIXED)
# # -----------------------------
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlim(-450000,450000)
# ax.set_ylim(-450000,450000)
# ax.set_zlim(-150000,150000)

# ax.set_xlabel("X [km]")
# ax.set_ylabel("Y [km]")
# ax.set_zlabel("Z [km]")

# # Earth
# ax.scatter(0,0,0)

# sc_dot, = ax.plot([],[],[], marker='o')
# moon_dot, = ax.plot([],[],[], marker='o')
# traj, = ax.plot([],[],[], lw=1)

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
#     fig,
#     update,
#     frames=len(epochs)//frame_skip,
#     interval=30,
#     blit=False
# )

# plt.title("Free-Return Trajectory (Auto-Solved)")
# plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.animation import FuncAnimation

from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, create_dynamics_simulator

# -----------------------------
# Constants
# -----------------------------
mu_E = 3.986004418e14
mu_M = 4.9048695e12
D = 384400e3
R_s = 66100e3
R_m = 1737e3
omega_m = 2*np.pi / (27.32*86400)
R_E = 6378.0e3           # m

def departure(r0, v0, phi0):
        E = v0**2 / 2 - mu_E / r0
        h = r0 * v0 * np.cos(phi0)
        return E, h
    
def at_moon_soi(E, h, lambda1):
    r1 = np.sqrt(D**2 + R_s**2 - 2*D*R_s*np.cos(lambda1))
    v1_sq = 2*(E + mu_E/r1)
    if v1_sq < 0:
        return np.array([1e10]*3)  # still large penalty, but physically meaningful
    v1 = np.sqrt(v1_sq)    
    phi1 = np.arccos(np.clip(h/(r1*v1), -1.0, 1.0))
    
    return r1, v1, phi1

def safe_arccos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))

def true_to_eccentric(f, e):
    return 2*np.arctan2(np.sqrt(1-e)*np.sin(f/2),
                        np.sqrt(1+e)*np.cos(f/2))
        
def orbital_elements(E, h):
    p = h**2 / mu_E
    a = -mu_E / (2*E)
    e = np.sqrt(1 - p/a)
    return p, a, e

def true_anomaly(p, e, r):
    return safe_arccos((p - r)/(r*e))

def eccentric_anomaly(e, f):
    if e < 1:
        # elliptic
        return true_to_eccentric(f, e)
    else:        
        # hyperbolic
         return 2*np.arctanh(np.sqrt((e-1)/(e+1)) * np.tan(f/2))
    #return np.arccos((e + np.cos(f)) / (1 + e*np.cos(f)))

def time_of_flight(a, e, E0, E1):
    if e < 1:
        return np.sqrt(a**3/mu_E)*((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))
    else :
        return np.sqrt((-a)**3/mu_E) * ((e*np.sinh(E1)-E1) - (e*np.sinh(E0)-E0))
    #     return np.sqrt((-a)**3/mu_E) * ((e*np.sinh(E1)-E1) - (e*np.sinh(E0)-E0))
    # return np.sqrt(a**3/mu_E) * ((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))

def moon_relative(v1, phi1, gamma1, delta):
    v_m = D * omega_m
    # v2 = np.sqrt(v1**2 + v_m**2 - 2*v1*v_m*np.cos(phi1 - gamma1))
    # return v2, v_m
    v_in = np.array([v1*np.cos(phi1) - v_m*np.cos(gamma1),
                     v1*np.sin(phi1) - v_m*np.sin(gamma1),
                     0.0])
    axis = np.array([0,0,1])
    v_out = rotate_vector(v_in, axis, delta)
    v2 = np.linalg.norm(v_out)
    return v2, v_m

def flyby(r2, v2, alpha):
    E_m = v2**2 / 2 - mu_M / r2
    h_m = r2 * v2 * np.sin(alpha)

    p_m = h_m**2 / mu_M
    e_m = np.sqrt(1 + (2*E_m*h_m**2)/(mu_M**2))

    r_mp = p_m / (1 + e_m)

    return E_m, h_m, e_m, r_mp
# -----------------------------
# Flyby vector rotation
# -----------------------------
def rotate_vector(v_rel, axis, angle):
    """Rodrigues rotation"""
    k = axis/np.linalg.norm(axis)
    v_rot = (v_rel*np.cos(angle) +
             np.cross(k, v_rel)*np.sin(angle) +
             k*np.dot(k, v_rel)*(1 - np.cos(angle)))
    return v_rot

def safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def true_to_eccentric(f, e):
    return 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(f/2),
        np.sqrt(1 + e) * np.cos(f/2)
    )

def full_model(x):
    """
    x = [v0, lambda1, delta]
    v0      : initial velocity at LEO (m/s)
    lambda1 : Moon encounter angle (rad)
    delta   : hyperbolic flyby deflection angle (rad)
    """
    v0, lambda1, delta = x

    # -----------------------------
    # Constants
    # -----------------------------
    r0 = 6678e3        # initial circular orbit radius
    phi0 = np.deg2rad(5)  # launch flight path angle
    gamma1 = np.deg2rad(10)  # Moon velocity direction
    alpha = np.deg2rad(30)   # flyby plane inclination
    target_perilune = R_m + 100e3
    target_tof = 3.5*24*3600  # seconds
        
    E,h = departure(r0, v0, phi0)
    r1, v1, phi1 = at_moon_soi(E, h, lambda1)
    p,a,e = orbital_elements(E, h)

    f0 = true_anomaly(p, e, r0)
    f1 = true_anomaly(p, e, r1)

    E0 = eccentric_anomaly(e, f0)
    E1 = eccentric_anomaly(e, f1)

    tof = time_of_flight(a, e, E0, E1)
    v2, v_m = moon_relative(v1, phi1, gamma1, delta)
    E_m, h_m, e_m, r_mp = flyby(R_s, v2, alpha)

def full_model(x):
    """
    x = [v0, lambda1, delta]
    v0      : initial velocity at LEO (m/s)
    lambda1 : Moon encounter angle (rad)
    delta   : hyperbolic flyby deflection angle (rad)
    """
    v0, lambda1, delta = x

    # -----------------------------
    # Constants
    # -----------------------------
    r0 = 6678e3        # initial circular orbit radius
    phi0 = np.deg2rad(5)  # launch flight path angle
    gamma1 = np.deg2rad(10)  # Moon velocity direction
    alpha = np.deg2rad(30)   # flyby plane inclination
    target_perilune = R_m + 100e3
    target_tof = 3.5*24*3600  # seconds

    # -----------------------------
    # Step 1: Initial orbit
    # -----------------------------
    E = v0**2/2 - mu_E/r0
    h = r0 * v0 * np.cos(phi0)

    p = h**2 / mu_E
    a = -mu_E / (2*E)
    e = np.sqrt(1 - p/a)

    # -----------------------------
    # Step 2: Moon encounter
    # -----------------------------
    r1 = np.sqrt(D**2 + R_s**2 - 2*D*R_s*np.cos(lambda1))
    # v1 = np.sqrt(2*(E + mu_E/r1))
    v1_sq = 2*(E + mu_E/r1)
    if v1_sq < 0:
        return np.array([1e10]*3)  # still large penalty, but physically meaningful
    v1 = np.sqrt(v1_sq)
    # phi1 = np.arccos(h / (r1*v1))
    phi1 = np.arccos(np.clip(h/(r1*v1), -1.0, 1.0))

    # -----------------------------
    # Step 3: True & eccentric anomalies
    # -----------------------------
    
    f0 = safe_arccos((p - r0)/(r0*e))
    f1 = safe_arccos((p - r1)/(r1*e))
    # E0 = true_to_eccentric(f0, e)
    # E1 = true_to_eccentric(f1, e)

    if e < 1:
        # elliptic
        E0 = true_to_eccentric(f0, e)
        E1 = true_to_eccentric(f1, e)
        tof = np.sqrt(a**3/mu_E)*((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))
    else:
        # hyperbolic
        F0 = 2*np.arctanh(np.sqrt((e-1)/(e+1)) * np.tan(f0/2))
        F1 = 2*np.arctanh(np.sqrt((e-1)/(e+1)) * np.tan(f1/2))
        tof = np.sqrt((-a)**3/mu_E) * ((e*np.sinh(F1)-F1) - (e*np.sinh(F0)-F0))

    #tof = np.sqrt(a**3/mu_E) * ((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))

    # -----------------------------
    # Step 4: Moon hyperbolic flyby (vector-based)
    # -----------------------------
    # Velocity magnitude relative to Moon
    v_m = D * omega_m
       # Incoming Moon-relative vector (2D in orbital plane)
    v_in = np.array([v1*np.cos(phi1) - v_m*np.cos(gamma1),
                     v1*np.sin(phi1) - v_m*np.sin(gamma1),
                     0.0])
    axis = np.array([0,0,1])
    v_out = rotate_vector(v_in, axis, delta)

    v_rel = np.linalg.norm(v_out) #np.sqrt(v1**2 + v_m**2 - 2*v1*v_m*np.cos(phi1 - gamma1))

    # Hyperbolic flyby parameters
    E_m = v_rel**2/2 - mu_M/R_s
    h_m = R_s * v_rel * np.sin(alpha)
    p_m = h_m**2 / mu_M
    e_m = np.sqrt(1 + (2*E_m*h_m**2)/(mu_M**2))
    r_mp = p_m / (1 + e_m)

    # Apply delta rotation for deflection
    # Post-flyby velocity vector rotated by delta in flyby plane
    # This is the "delta" you can tune in optimizer
    v_post = v_rel * np.array([np.cos(delta), np.sin(delta), 0])
    v_post_mag = np.linalg.norm(v_post)

    # -----------------------------
    # Step 5: Check Earth return
    # -----------------------------
    # Simple check: make sure post-flyby periapsis < 2*R_E
    # Penalize escaping trajectories
    r_aft = p_m / (1 + e_m)  # periapsis after flyby
    earth_penalty = 0.0
    if r_aft > 2*R_E:
        earth_penalty = 1e7 * (r_aft - 2*R_E)**2

    # -----------------------------
    # Residuals
    # -----------------------------
    res1 = r_mp - target_perilune      # Moon flyby altitude
    res2 = tof - target_tof            # time-of-flight
    res3 = earth_penalty               # Earth return check
    

    if np.isnan(res1) or np.isnan(res2) or np.isnan(res3):
        return np.array([1e10,1e10,1e10])  # large penalty
    else:
        return np.array([res1, res2, res3])

def full_model_1(x):

    v0, lambda1 = x

    r0 = 6678e3
    phi0 = np.deg2rad(5)
    gamma1 = np.deg2rad(10)
    alpha = np.deg2rad(30)

    # Slide 1
    
    
    E = v0**2/2 - mu_E/r0
    h = r0 * v0 * np.cos(phi0)

    # Slide 2
    p = h**2 / mu_E
    a = -mu_E / (2*E)
    e = np.sqrt(1 - p/a)

    r1 = np.sqrt(D**2 + R_s**2 - 2*D*R_s*np.cos(lambda1))
    v1 = np.sqrt(2*(E + mu_E/r1))
    phi1 = np.arccos(h/(r1*v1))

    

    # anomalies
    f0 = safe_arccos((p - r0)/(r0*e))
    f1 = safe_arccos((p - r1)/(r1*e))

    # E0 = safe_arccos((e + np.cos(f0))/(1 + e*np.cos(f0)))
    # E1 = safe_arccos((e + np.cos(f1))/(1 + e*np.cos(f1)))

    E0 = true_to_eccentric(f0, e)
    E1 = true_to_eccentric(f1, e)

    tof = np.sqrt(a**3/mu_E)*((E1 - e*np.sin(E1)) - (E0 - e*np.sin(E0)))

    # Moon-relative
    v_m = D * omega_m
    v2 = np.sqrt(v1**2 + v_m**2 - 2*v1*v_m*np.cos(phi1 - gamma1))

    # Flyby
    E_m = v2**2/2 - mu_M/R_s
    h_m = R_s * v2 * np.sin(alpha)

    p_m = h_m**2 / mu_M
    e_m = np.sqrt(1 + (2*E_m*h_m**2)/(mu_M**2))
    r_mp = p_m/(1 + e_m)

    # -----------------------------
    # Residuals (constraints)
    # -----------------------------
    target_perilune = R_m + 100e3

    res1 = r_mp - target_perilune      # altitude constraint
    res2 = tof - 3.5*24*3600          # timing (~3.5 days)

    return np.array([res1, res2])


# -----------------------------
# Solve analytically
# -----------------------------
inc = np.deg2rad(28.0)          # inclination
alt_perigee = 185.0 *1e3             # m
alt_apogee = 1800.0 *1e3             # m
r_p = R_E + alt_perigee # m
r_a = R_E + alt_apogee # m
a = (r_p + r_a)/2
e = (r_a - r_p)/(r_a + r_p)
v_perigee = np.sqrt(mu_E * (1 + e)/(a*(1 - e)))

a_TLI = (r_p + D)/2
v_TLI = np.sqrt(mu_E * (2/r_p - 1/a_TLI))

x0 = [v_TLI, np.deg2rad(30), np.deg2rad(10)]  # initial guess for [v0, lambda1, delta]
sol = least_squares(full_model, x0, xtol=1e-6)

# v0_sol, lambda_sol = sol.x
v0_sol, lambda_sol, delta_sol = sol.x  # if full_model has 3 inputs now

print("\n=== ANALYTICAL SOLUTION ===")
print(f"v0 = {v0_sol:.2f} m/s")
print(f"lambda1 = {np.rad2deg(lambda_sol):.2f} deg")

# -----------------------------
# TudatPy propagation
# -----------------------------
spice.load_standard_kernels()

bodies_to_create = ["Earth","Moon","Sun"]
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, "SSB", "J2000"
)

bodies = environment_setup.create_system_of_bodies(body_settings)

bodies.create_empty_body("Spacecraft")
bodies.get_body("Spacecraft").mass = 1000.0

acc_settings = {
    "Spacecraft": {
        "Earth":[propagation_setup.acceleration.point_mass_gravity()],
        "Moon":[propagation_setup.acceleration.point_mass_gravity()],
        "Sun":[propagation_setup.acceleration.point_mass_gravity()]
    }
}

acc_models = propagation_setup.create_acceleration_models(
    bodies, acc_settings, ["Spacecraft"], ["SSB"]
)

r0 = 6678e3
initial_state = np.array([r0,0,0, 0,v0_sol,0])

start = 0.0
end = 8*86400

integrator = propagation_setup.integrator.runge_kutta_4(start, 30.0)
termination = propagation_setup.propagator.time_termination(end)

propagator = propagation_setup.propagator.translational(
    ["SSB"],
    acc_models,
    ["Spacecraft"],
    initial_state,
    start,
    integrator,
    termination
)

sim = create_dynamics_simulator(bodies, propagator)

states = np.vstack(list(sim.state_history.values()))
epochs = np.array(list(sim.state_history.keys()))

moon_states = np.array([
    spice.get_body_cartesian_state_at_epoch(
        "Moon","Earth","J2000","NONE",t
    ) for t in epochs
])

positions_km = states[:,0:3]/1000
moon_states_km = moon_states[:,0:3]/1000

# -----------------------------
# 3D Animation
# -----------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-450000,450000)
ax.set_ylim(-450000,450000)
ax.set_zlim(-150000,150000)

ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_zlabel("Z [km]")

ax.scatter(0,0,0)

sc_dot, = ax.plot([],[],[], marker='o')
moon_dot, = ax.plot([],[],[], marker='o')
traj, = ax.plot([],[],[], lw=1)

frame_skip = 5

def update(frame):
    idx = frame*frame_skip
    if idx >= len(epochs):
        idx = len(epochs)-1

    r_sc = positions_km[idx]
    r_m = moon_states_km[idx]

    sc_dot.set_data([r_sc[0]],[r_sc[1]])
    sc_dot.set_3d_properties([r_sc[2]])

    moon_dot.set_data([r_m[0]],[r_m[1]])
    moon_dot.set_3d_properties([r_m[2]])

    traj.set_data(positions_km[:idx,0],positions_km[:idx,1])
    traj.set_3d_properties(positions_km[:idx,2])

    return sc_dot, moon_dot, traj

ani = FuncAnimation(fig, update,
                    frames=len(epochs)//frame_skip,
                    interval=30)

plt.title("Robust Free-Return Trajectory")
plt.show()