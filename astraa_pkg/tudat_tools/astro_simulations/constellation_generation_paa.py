### Code to generate and simulate a LEO Polar, LEO Inclined + MEO equatorial constellation
# carried over from MSc Thesis work. Requires TUDATpy
# Date: April 12, 2023 
# Load standard modules
from cgi import print_arguments
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
# Add parent directory to paths
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import datetime as dt
import json
import simulation_utilities as util
from data_processing.data_saving_utilities import dict2txt
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
save_append = "initial_constellation"
"""
NAIF's `SPICE` kernels are first loaded, so that the position of various bodies such as the Earth can be make known to `tudatpy`.
Then, the start and end simulation epochs are setups. In this case, the start epoch is set to `0`, corresponding to the 1st of January 2000.
The times should be specified in seconds since J2000.
Please refer to the API documentation of the `time_conversion module` [here](https://tudatpy.readthedocs.io/en/latest/time_conversion.html) for more information on this.
"""

# Load spice kernels
spice.load_standard_kernels()

## Create acceleration model class
acceleration_models = util.select_acceleration_model()

## Set Simulation Parameters
# whether to simulate only limited satellites
shells_simulated = 'all' 
# shells_simulated = 'leo_i'
basic_acceleration = 0 # turn off MOST perturbation
# Set simulation start and end epochs
# Start time : 2022, January, 12:00:00 
# sim_time = 24 # hours (MEO shell orbital period = 7.9 hrs)
sim_time = 8
simulation_start_epoch = 22 * 365.25 * 24 * 3600 # 22 years x 365.25 days/yr * 24 hrs/day * 3600 sec/hr
simulation_end_epoch = simulation_start_epoch + sim_time * 3600
# Set simulation labels
simulation_name = f'Leo_globalMeo_equator{sim_time:.2f}h'
simulation_run_date = dt.datetime.now()
# Acceleration model selection
if basic_acceleration:
    acceleration_setting_earth = 3
    acceleration_setting_3rdbody = [] # Add Moon and Sun PMG accelerations
else:
    acceleration_setting_earth = 6
    acceleration_setting_3rdbody = [0,1] # Add Moon and Sun PMG accelerations

acceleration_label, acceleration_settings_sat = acceleration_models.select_model(setting_earth = acceleration_setting_earth,
                                                                                setting_other_bodies = acceleration_setting_3rdbody)
                                                                                

# Conditional statements
plot_orbit_3d = 1
save_outputs = 1
output_kepler_states = 0
debug_mode = 0


## Set Constellation parameters
eccentricity = 0.0001 # ToDo Setup eccentricity in constellation params
phasing_param = 3 # anything above 0 for collision avoidance

wdp_leo_polar = [89, 13, 14, phasing_param]  # incl [deg], Nr_planes [-], Nr_sats/plane [-], rel_phasing, total_sats
wdp_leo_polar.append(int(wdp_leo_polar[1]*wdp_leo_polar[2]))
h_leo_polar = 1000 # km
id_leo_polar = 'leo_polar'
wdp_leo_incl = [53, 13, 14, phasing_param] # incl [deg], Nr_planes [-], Nr_sats/plane [-], rel_phasing, total_sats
wdp_leo_incl.append(int(wdp_leo_incl[1]*wdp_leo_incl[2]))
h_leo_incl = 1000 # km
id_leo_incl = 'leo_incl'
wdp_meo = [0, 1, 5, phasing_param] # incl [deg], Nr_planes [-], Nr_sats/plane [-], rel_phasing, total_sats
wdp_meo.append(int(wdp_meo[1]*wdp_meo[2]))
h_meo = 13892 # km
id_meo = 'meo'


shell_keys = ['id', 'h', 'wdp', 'd_omega, w']
shell_parameters_leo_polar = util.summarize_shell_params(wdp_leo_polar,
                                                   h_leo_polar,
                                                   id_leo_polar)
shell_parameters_leo_incl = util.summarize_shell_params(wdp_leo_incl,
                                                   h_leo_incl,
                                                   id_leo_incl)
shell_parameters_meo = util.summarize_shell_params(wdp_meo,
                                                   h_meo,
                                                   id_meo)

const_params = {}
if shells_simulated == 'all':
    const_params[1] = shell_parameters_leo_polar
    const_params[2] = shell_parameters_leo_incl
    const_params[3] = shell_parameters_meo
elif shells_simulated == 'leo_i':
    # LIMIT nr of satellites simulates
    wdp_leo_incl = [53, 1, 14, 0] # incl [deg], Nr_planes [-], Nr_sats/plane [-], rel_phasing, total_sats
    wdp_leo_incl.append(int(wdp_leo_incl[1]*wdp_leo_incl[2]))
    shell_parameters_leo_incl = util.summarize_shell_params(wdp_leo_incl,
                                                   h_leo_incl,
                                                   id_leo_incl)
    const_params[1] = shell_parameters_leo_incl

sim_params = {}
sim_params['name'] = simulation_name
sim_params['t_start'] = simulation_start_epoch
sim_params['t_end'] = simulation_end_epoch

n_sats_total = 0
for key in const_params:
    n_sats_total += const_params[key]['wdp'][4]

bodies_to_create = [body for body in acceleration_settings_sat.keys()]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies (in this case only Earth)
bodies = environment_setup.create_system_of_bodies(body_settings)

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
earth_average_radius = spice.get_average_radius("Earth")

### Setup constellation generation inputs
ic_const_all, central_bodies, bodies_to_propagate, bodies, acceleration_settings = util.setup_constellations(
            const_params,
            bodies,
            acceleration_settings_sat,
            earth_gravitational_parameter,
            earth_average_radius,
            eccentricity
        )

# Create acceleration models 
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)
if debug_mode:
    for ii, row in enumerate(ic_const_all):
        print(f'''
        NUMBER : {ii}, {bodies_to_propagate[ii]}
        a = {ic_const_all[ii,0]/1e3:.0f} km
        e =  {ic_const_all[ii,1]}
        i = {np.rad2deg(ic_const_all[ii,2]):.1f} deg
        w =  {np.rad2deg(ic_const_all[ii,3]):.1f} deg
        RAAN = {np.rad2deg(ic_const_all[ii,4]):.1f} deg
        theta =  {np.rad2deg(ic_const_all[ii,5]):.1f} deg
         -----------
        ''')

for ii, row in enumerate(ic_const_all):
    ic_const_all[ii,:] = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis= ic_const_all[ii,0],
    eccentricity=ic_const_all[ii,1],
    inclination=ic_const_all[ii,2],
    argument_of_periapsis=ic_const_all[ii,3],
    longitude_of_ascending_node=ic_const_all[ii,4],
    true_anomaly=ic_const_all[ii,5])
initial_states = np.concatenate(ic_const_all)

# Create numerical integrator settings
fixed_step_size = 10.0
integration_setting = "rk4"
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_step_size
)

# Save used settings in simulation metadata output
sim_params['generation_date'] = str(simulation_run_date)
sim_params['accelerations_setting'] = acceleration_label
sim_params['ref_frame'] = global_frame_orientation
sim_params['frame_origin'] = global_frame_origin
sim_params['integrator'] = integration_setting
sim_params['time_step'] = fixed_step_size
sim_params['constellation'] = {}
sim_params['total_sats'] = n_sats_total
sim_params['constellation'] = const_params

## Execute simulation
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
"""
With these commands, we execute the simulation and retrieve the output.
"""
if output_kepler_states:
    dep_var_to_save = [
        propagation_setup.dependent_variable.keplerian_state( "sat_leo_polar_4_4", "Earth" ),
        propagation_setup.dependent_variable.keplerian_state( "sat_leo_polar_4_5", "Earth" ),
        propagation_setup.dependent_variable.keplerian_state( "sat_leo_incl_4_4", "Earth" ),
        propagation_setup.dependent_variable.keplerian_state( "sat_leo_incl_4_5", "Earth" ),
        propagation_setup.dependent_variable.keplerian_state( "sat_leo_incl_4_3", "Earth" )
        ]
else:
    dep_var_to_save = []
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
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
if output_kepler_states:
    dependent_variables = dynamics_simulator.dependent_variable_history
    dependent_variables_array = result2array(dependent_variables)

simulation_end_time = dt.datetime.now()
simulation_run_time =  (simulation_end_time - simulation_start_time).total_seconds()

sim_params['simulation_run_time'] = simulation_run_time
sim_params['n_function_evaluations'] = function_evaluations
sim_params['sat_names'] = bodies_to_propagate
print(f'''
    Simulation done in {simulation_run_time} s
    \n3d orbit plot setting: {bool(plot_orbit_3d)}
    simulation output saving setting: {bool(save_outputs)}
    function evaluations: {function_evaluations:d}
    ''')
if save_outputs:
    output_path = f'{current_dir}/{save_parent_dir}/{save_append}/{simulation_name}'
    save2txt(states, 'state_history.dat', output_path)
    dict2txt(sim_params, 'simulation_parameters', output_path)
    if output_kepler_states:
        save2txt(dependent_variables, 'dependent_variables.dat', output_path)
#%%
if plot_orbit_3d:
    # prepare to plot
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Constellation trajectory around Earth')

    for ii in range(n_sats_total):
        ax.scatter(states_array[0, 6*ii+1],
                states_array[0, 6*ii+2],
                states_array[0, 6*ii+3])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()    
