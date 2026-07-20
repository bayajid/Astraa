# Load standard modules
from cgi import print_arguments
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
# Add parent directory to paths
path_cwd = os.getcwd()
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import datetime as dt
import json
import tudat_tools.simulation_utilities as util
from tudat_tools.data_processing.data_saving_utilities import dict2txt
# import plotting_functions.plotting_basic as bplt
# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.data import save2txt
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

# Path variables
save_parent_dir = "simulation_output"
save_append = "leo_meo_srpcheck"
change_eph = 0

"""
NAIF's `SPICE` kernels are first loaded, so that the position of various bodies such as the Earth can be make known to `tudatpy`.
Then, the start and end simulation epochs are setups. In this case, the start epoch is set to `0`, corresponding to the 1st of January 2000.
The times should be specified in seconds since J2000.
Please refer to the API documentation of the `time_conversion module` [here](https://tudatpy.readthedocs.io/en/latest/time_conversion.html) for more information on this.
"""
hp_setting = 1 # High Precision simulation setting
# Load spice kernels
spice.load_standard_kernels()

# Load initial conditions

# Simulation time
sim_time = 3600*24*31
# sim_time = 3600
sim_start = 0
simulation_name = f'leo_meo_srpcheck'
if hp_setting:
    save_append = f'{save_append}_high_precision'
    add_srp = 1
else:
    save_append = f'{save_append}_medium_precision'
    add_srp = 0


# Set Drag/SRP reference area, coefficients, mass
mass = 400 # kg
reference_area = 3 # m^3
drag_coefficient = 2.0
radiation_pressure_coefficient = 1.2
# Drag coefficient for 3U cubesat at 20 deg inclination
# source https://www.researchgate.net/figure/Drag-and-lift-coefficient-for-a-3U-Cubesat-using-empirical-method-at-150-km_fig3_266852694
if hp_setting:
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, radiation_pressure_coefficient, ['Earth']
    )
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0, 0]
    )
## Create acceleration model class
acceleration_models_available = util.select_acceleration_model()

## Set Simulation Parameters
# Set simulation start and end epochs
# Start time : 2022, January, 12:00:00 

simulation_start_epoch = 22 * 365.25 * 24 * 3600 + sim_start #  x 365.25 days/yr * 24 hrs/day * 3600 sec/hr [seconds since J2000]
simulation_end_epoch = simulation_start_epoch + sim_time
# Set simulation labels
# simulation_name = f'Leo_globalMeo_equator{sim_time:.2f}h'
simulation_run_date = dt.datetime.now()
if hp_setting:
    acceleration_setting_earth = 7
    acceleration_setting_3rdbody = [0,1] # Add Moon and Sun PMG accelerations
else:
    acceleration_setting_earth = 6
    acceleration_setting_3rdbody = [0,1] # Add Moon and Sun PMG accelerations
acceleration_label, acceleration_settings_sat = acceleration_models_available.select_model(setting_earth = acceleration_setting_earth,
                                                                                setting_other_bodies = acceleration_setting_3rdbody,
                                                                                add_srp = add_srp)
# Conditional statements
plot_orbit_3d = 1
save_outputs = 1
output_kepler_states = 0
full_dep_var = 1 # add acceleration norms to dependent variables

output_sun_vector = 1
sim_params = {}
sim_params['name'] = simulation_name
sim_params['t_start'] = simulation_start_epoch
sim_params['t_end'] = simulation_end_epoch


n_sats_total = 4
bodies_to_create = [body for body in acceleration_settings_sat.keys()]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
# global_frame_origin = "SSB"
global_frame_orientation = "J2000"
# global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)
earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
earth_average_radius = spice.get_average_radius("Earth")
sat_names = [
    'leo_polar',
    'leo_incl',
    'leo_eq',
    'meo_eq',
]
        # a = {ic_const_all[ii,0]/1e3:.0f} km
        # e =  {ic_const_all[ii,1]}
        # i = {np.rad2deg(ic_const_all[ii,2]):.1f} deg
        # w =  {np.rad2deg(ic_const_all[ii,3]):.1f} deg
        # RAAN = {np.rad2deg(ic_const_all[ii,4]):.1f} deg
        # theta =  {np.rad2deg(ic_const_all[ii,5]):.1f} deg
ic_sats = np.array(
    [[7378e3, 0.0001, np.deg2rad(89), np.deg2rad(0.1), 0, np.deg2rad(0)],
    [7378e3, 0.0001, np.deg2rad(55), np.deg2rad(0.1), 0, np.deg2rad(0)],
    [7378e3, 0.0001, np.deg2rad(0), np.deg2rad(0.1), 0, np.deg2rad(0)],
    [13892e3+6378e3, 0.0001, np.deg2rad(0), np.deg2rad(0.1), 0, np.deg2rad(0)]]
)
ic_const_all = np.zeros((4,6))
for ii, row in enumerate(ic_const_all):
    ic_const_all[ii,:] = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis= ic_sats[ii,0],
    eccentricity=ic_sats[ii,1],
    inclination=ic_sats[ii,2],
    argument_of_periapsis=ic_sats[ii,3],
    longitude_of_ascending_node=ic_sats[ii,4],
    true_anomaly=ic_sats[ii,5])
initial_states = np.concatenate(ic_const_all)


### Setup constellation generation inputs
bodies_to_propagate = sat_names
acceleration_settings = {}
for sat_name in sat_names:
    bodies.create_empty_body(sat_name)
    
    if hp_setting:
        # Add SRP interface
        environment_setup.add_radiation_pressure_interface(
            bodies, sat_name, radiation_pressure_settings
        )
        # Add Drag interface
        environment_setup.add_aerodynamic_coefficient_interface(
            bodies, sat_name, aero_coefficient_settings)
        bodies.get(sat_name).mass = mass
    acceleration_settings[sat_name] = acceleration_settings_sat

central_bodies = ["Earth" for ii in bodies_to_propagate]

ic_concatenated = initial_states

# Create acceleration models 
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

# Create numerical integrator settings
fixed_step_size = 60
integration_setting = "rk4"
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_step_size
)
# Save used settings in simulation metadata output
sim_params['generation_date'] = str(simulation_run_date)
sim_params['accelerations_setting'] = acceleration_label
sim_params['integrator'] = integration_setting
sim_params['time_step'] = fixed_step_size
sim_params['constellation'] = {}
sim_params['total_sats'] = n_sats_total

## Execute simulation

"""
With these commands, we execute the simulation and retrieve the output.
"""
if full_dep_var:
    dep_var_to_save = [
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type, 'leo_polar', 'Sun'),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type, 'leo_incl', 'Sun'),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type, 'leo_eq', 'Sun'),
        propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type, 'meo_eq', 'Sun'),
        ]        
else:
    dep_var_to_save = []
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    ic_concatenated,
    termination_condition,
    output_variables = dep_var_to_save
)
simulation_start_time = dt.datetime.now()
# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings, 
    print_dependent_variable_data = False, 
    print_state_data = False)

function_evaluations = int(list(dynamics_simulator.cumulative_number_of_function_evaluations)[-1])
states = dynamics_simulator.state_history
states_array = result2array(states)
dependent_variables = dynamics_simulator.dependent_variable_history

dependent_variables = result2array(dependent_variables)

simulation_end_time = dt.datetime.now()
simulation_run_time =  (simulation_end_time - simulation_start_time).total_seconds()

sim_params['simulation_run_time'] = simulation_run_time
sim_params['n_function_evaluations'] = function_evaluations
sim_params['sat_names'] = bodies_to_propagate
sim_params['r_index'] = {}
for ii, sat in enumerate(sat_names):
    sim_params['r_index'][sat] = [6*ii+1, 6*ii+2, 6*ii+3]
print(f'''
    Simulation done in {simulation_run_time} s
    \n3d orbit plot setting: {bool(plot_orbit_3d)}
    simulation output saving setting: {bool(save_outputs)}
    function evaluations: {function_evaluations:d}
    ''')
if save_outputs:
    print(f'Saving results to {save_parent_dir}/{save_append}')
    output_path = fr'orbital_simulations/{save_append}/{simulation_name}'
    save2txt(states, 'state_history.dat', output_path)
    dict2txt(sim_params, 'simulation_parameters', output_path)    
    save2txt(dynamics_simulator.dependent_variable_history, 'dependent_variables.dat', output_path)

if plot_orbit_3d:
    # prepare to plot
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Constellation trajectory around Earth')

    for ii in range(n_sats_total):
        ax.scatter(states_array[:, 6*ii+1],
                states_array[:, 6*ii+2],
                states_array[:, 6*ii+3],
                label = sat_names[ii])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    plt.show()    
    # bplt.savefig(fig, name = '3d_sim_plot', save_folder = f'{output_path}')