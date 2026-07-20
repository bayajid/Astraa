## Here, useful functions will be saved modular simulation purposes
# such as functions to define the acceleration settings, generate simulation inputs
import numpy as np 
from tudatpy.kernel.numerical_simulation import propagation_setup

class select_acceleration_model():
    def __init__(self) -> None:
        self.earth_gravity_models = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.spherical_harmonic_gravity(2,0),
            propagation_setup.acceleration.spherical_harmonic_gravity(2,2),
            propagation_setup.acceleration.spherical_harmonic_gravity(4,4),
            propagation_setup.acceleration.spherical_harmonic_gravity(8,8),
            propagation_setup.acceleration.spherical_harmonic_gravity(16,16),
            propagation_setup.acceleration.spherical_harmonic_gravity(32,32),
            propagation_setup.acceleration.spherical_harmonic_gravity(64,64),
            
                                     ]
        self.labels_earth = [
            'Earth, PMG',
            'Earth, SH 2x0',
            'Earth, SH 2x2',
            'Earth, SH 4x4',
            'Earth, SH 8x8',
            'Earth, SH 16x16',
            'Earth, SH 32x32',
            'Earth, SH 64x64',
        ]
        self.third_body_gravity = np.array((
            'Moon',
            'Sun',
            'Venus',
            'Mercury',
            'Mars',
            'Jupiter',
            'Saturn',
            'Uranus',
            'Neptune'            
        ))
        self.labels_third_body_gravity = [
            'Moon',
            'Sun',
            'Venus',
            'Mercury',
            'Mars',
            'Jupiter',
            'Saturn',
            'Uranus',
            'Neptune'               
        ]      

    def select_model(self, setting_earth:int, 
                                  setting_other_bodies:list = None, 
                                  add_srp:bool = 0,
                                  print_cond:bool = 1):
        """Function to select acceleration models for numerical proapgation

        Args:
            setting_earth (int): index to select EGM level (from PMG to SH64x64)
            setting_other_bodies (list, optional): List of bodies to include as PMG (starting with Moon-Sun-etc). Defaults to None.
            print_cond (bool, optional): Print condition to state which accelerations are chosen. Defaults to 1.

        Returns:
            accelerations_label - string: label of all chosen accelerations
            acceleration_settings_sat - dict: acceleration settings
        """        
        
        acc_earth = self.earth_gravity_models[setting_earth]
        if print_cond:
            print(f'EGM: {str(acc_earth)}')
        label_earth = self.labels_earth[setting_earth]
        label_other_bodies = '' # acceleration label for 3rd bodies
        
        if setting_other_bodies != None:
            acc_other_bodies = self.third_body_gravity[setting_other_bodies]
        else:
            acc_other_bodies = None
        # Make accelerations dict
        acceleration_settings_sat = dict(Earth = [acc_earth])
        
        if type(acc_other_bodies)!= None:
            if print_cond:
                print(f'3rd body accelerations: {acc_other_bodies}')
            label_other_bodies = 'PMG: '
            for ii, body in enumerate(acc_other_bodies):
                acceleration_settings_sat[body] = [propagation_setup.acceleration.point_mass_gravity()]
                if ii != 0:
                    label_other_bodies = label_other_bodies + ','    
                label_other_bodies = label_other_bodies + f' {body[:3]}'
        
        if add_srp:
            if len(acceleration_settings_sat['Sun']) != 0:
                acceleration_settings_sat['Sun'].append(
                    propagation_setup.acceleration.cannonball_radiation_pressure()
                )
            else:
                acceleration_settings_sat['Sun'] = [
                    propagation_setup.acceleration.cannonball_radiation_pressure()
                ]

        # Make accelerations label
        acceleration_label = f'{label_earth}, {label_other_bodies}'
        
        return acceleration_label, acceleration_settings_sat
        
        
    
    # def print_all_models(self):
    #     return None
    # return acceleration_label, acceleration_settings_sat

def summarize_shell_params(wdp, 
                           h, 
                           id,
                           d_omega = None,
                           w = 0,
                           shell_keys = ['id', 'h', 'wdp', 'd_omega', 'w']):
    """Function to generate the shell parameters dictionary

    Args:
        wdp (list): list of ALTERED Walker Delta Patterns [0 inclination [deg], 
        1 total nr planes, 2 sats per plane, 3 phasing parameter, 4 total nr of sats]
        h (float): altitude [km]
        id (string): shell naming identifier
        d_omega (float, optional): change in RAAN between sats in parallel planes. Defaults to None.
        w (float, optional): argument of perigee. Defaults to 0.
        shel_keys (list, optional): keys to define dict. Defaults to ['id', 'h', 'wdp', 'd_omega, w'].

    Returns:
        dict: shell orbital parameters summarized in dictionary
    """    
    shell_parameters = dict.fromkeys(shell_keys)
    
    if d_omega == None: # If change in RAAN not defined, calc uniform
        d_omega = 360 / wdp[1] # [deg] 360 div. by number of planes 
    shell_parameters[shell_keys[0]] = id
    shell_parameters[shell_keys[1]] = h
    shell_parameters[shell_keys[2]] = wdp
    shell_parameters[shell_keys[3]] = d_omega
    shell_parameters[shell_keys[4]] = w
    
    return shell_parameters
def string_list2array(string_list):
    # function to process the IC lists in the pandas dataframes, which get loaded incorrectly as strings
    split_list = string_list.split('\n')
    split_0 = split_list[0].split(' ')
    split_1 = split_list[1].split(' ')
    numbers_0 = [float(num) for num in split_0 if num != '']
    numbers_1 = [float(num) for num in split_1 if num != '']
    concat_list = np.concatenate((numbers_0, numbers_1))
    return concat_list
    
def setup_constellations(const_params,
                         bodies,
                         acceleration_settings_sat,
                         earth_gravitational_parameter,
                        earth_average_radius,
                        eccentricity,
                        ta_0 = 0,
                        constellation_shape = 'uniform'):
    """Function to generate IC for every satellite constellation shell,
    as well as the satellite names and required lists for propagation,
    such as bodies_to_prop, central_bodies, acceleration_settings, etc.
    KP 12-04-2023: Added ii_phasing in loop to add relative phasing between two LEO shells
    Altered WDP: [0 inclination [deg], 1 total nr planes, 2 sats per plane, 3 phasing parameter, 4 total nr of sats]
    KP 09-08-2023 Added constellation_shape. uniform -> angular separation between planes only considers
    ascending node. Eg 360/n_planes -> RAAN gap. Aug 21 - added ta_0, initial True anomaly offset
    to better match specific constellation setups

        Mode 'split' leads to consideration of both nodes, ie. 360/(2*n_planes) -> RAAN gap
    
    Args:
        const_params (dict): dict with each key containing the parameters of a single
        constellation shell
        bodies (list): TUDAT body object 
        acceleration_settings_sat (dict): dictionary with the acceleration setings for each sat
        earth_average_radius [m]

    Returns:
        tuple:
        0 ic_const_all, 
        1 central_bodies, 
        2 bodies_to_propagate, 
        3 bodies, 
        4  acceleration_settings
    """    
    if constellation_shape == 'uniform':
        draan_factor = 1
    elif constellation_shape == 'split':
        draan_factor = 2
    sat_names = []
    ii_phasing = 0
    for mm, key in enumerate(const_params.keys()):
        shell_dict = const_params[key]
        if 'leo' in shell_dict['id']:
            ii_phasing +=1
        incl = shell_dict['wdp'][0]
        nr_planes = shell_dict['wdp'][1]
        nr_spp = shell_dict['wdp'][2]
        phasing_param = shell_dict['wdp'][3]
        nr_sats_total = shell_dict['wdp'][4]
        ic_const = np.array(np.zeros((nr_sats_total, 6))) # nr_sats_total x 6 (xyz, kep. params)
        ic_const[:,0] = shell_dict['h']*1e3 + earth_average_radius # semi-major axis
        ic_const[:,1] = eccentricity 
        ic_const[:,2] = np.deg2rad(incl) # inclination [rad]
        ic_const[:,3] = np.deg2rad(shell_dict['w'])  # argument of periapsis
        
        ii = 0 # satellite index
        rel_phasing = phasing_param * 360 / nr_sats_total
        #!! phasing_param = separation_wanted / nr_sats*360 !!
        for pp in range(nr_planes):  # Iterate over orbital planes
            d_raan_pp = shell_dict['d_omega'] / draan_factor * pp
            d_theta = 360 / nr_spp
            rel_phase_pp = (rel_phasing * pp)*ii_phasing # KP 12-04-23 Add phasing between LEO shells
            for jj in range(nr_spp):  # iterate over sats in plane
                ic_const[ii,4] = np.deg2rad(d_raan_pp) # right ascension of the ascending node
                ic_const[ii,5] = np.deg2rad(ta_0 + rel_phase_pp + d_theta * jj) # true anomaly
                sat_name = f"sat_{shell_dict['id']}_{pp}_{jj}" # sat_const_planeindex_satindex
                sat_names.append(sat_name)
                ii += 1
        if key == 1:
            ic_const_all = ic_const
        else:
            ic_const_all = np.vstack((ic_const_all, ic_const))
    
    acceleration_settings = {}
    # Add empty bodies to body list and accelerations to each sat
    for sat_name in sat_names:
        bodies.create_empty_body(sat_name)
        acceleration_settings[sat_name] = acceleration_settings_sat
    
    # Add central bodies for each satellite
    central_bodies = ["Earth" for ii in sat_names]
    
    # Propagate all satellites
    bodies_to_propagate = sat_names
    
    return ic_const_all, central_bodies, bodies_to_propagate, bodies, acceleration_settings

# def choose_acceleration_settings()
if __name__ == "__main__":
    # Load standard modules
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    import sys
    # Load tudatpy modules
    from tudatpy.kernel.interface import spice
    from tudatpy.data import save2txt
    from tudatpy.kernel import numerical_simulation
    from tudatpy.kernel.numerical_simulation import environment_setup
    from tudatpy.kernel.numerical_simulation import propagation_setup
    from tudatpy.kernel.astro import element_conversion
    from tudatpy.kernel import constants
    from tudatpy.util import result2array
    # Create default body settings for "Earth"
    if 0:
        spice.load_standard_kernels()
        bodies_to_create = ["Earth"]

        # Create default body settings for bodies_to_cre+ate, with "Earth"/"J2000" as the global frame origin and orientation
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

        # Create system of bodies (in this case only Earth)
        bodies = environment_setup.create_system_of_bodies(body_settings)

        earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
        earth_average_radius = spice.get_average_radius("Earth")
        earth_average_radius = 6378e3
        eccentricity = 0.01
        
        wdp_starlink53 = [53, 50*32, 32, 2] # phasing parameter guessed
        h_starlink53 = 1140 # km
        id_starlink53 = 'starlink53'
        wdp_starlink74 = [74, 50*8, 8, 2] # phasing parameter guessed
        h_starlink74 = 1130 # km
        id_starlink74 = 'starlink74'
        shell_keys = ['id', 'h', 'wdp', 'd_omega, w']
        
        
        shell_parameters_sl53 = summarize_shell_params(wdp_starlink53, 
                                                    h_starlink53,
                                                    id_starlink53)
        shell_parameters_sl74 = summarize_shell_params(wdp_starlink74, 
                                                    h_starlink74,
                                                    id_starlink74)
        
        const_params = {}
        const_params[1] = shell_parameters_sl53
        const_params[2] = shell_parameters_sl74
        
        acceleration_settings_sat= dict(
            Earth=[propagation_setup.acceleration.point_mass_gravity()]
        )
        
        if 1:
            ic_const_all, central_bodies, bodies_to_propagate, bodies, acceleration_settings = setup_constellations(
                const_params,
                bodies,
                acceleration_settings_sat
            )
        else:
                
            sat_names = [] # list of satellite names
            
            for key in const_params.keys():
                shell_dict = const_params[key]
                ic_const = np.array(np.zeros((shell_dict['wdp'][1], 6)))
                ic_const[:,0] = shell_dict['h']*1e3 + earth_average_radius # semi-major axis
                ic_const[:,1] = eccentricity 
                ic_const[:,2] = np.deg2rad(shell_dict['wdp'][0]) # inclination
                ic_const[:,3] = np.deg2rad(shell_dict['w'])  # argument of periapsis
                
                ii = 0 # satellite index
                rel_phasing = shell_dict['wdp'][3] * 360 / shell_dict['wdp'][1]
                for pp in range(shell_dict['wdp'][2]):  # Iterate over orbital planes
                    d_raan_pp = shell_dict['d_omega'] * pp
                    rel_phase_pp = rel_phasing * pp
                    for jj in range(int(shell_dict['wdp'][1] / shell_dict['wdp'][2])): 
                        ic_const[ii,4] = np.deg2rad(d_raan_pp) # right ascension of the ascending node
                        ic_const[ii,5] = np.deg2rad(rel_phase_pp + shell_dict['d_omega'] * jj) # true anomaly
                        sat_name = f"sat_{shell_dict['id']}_{pp}_{jj}" # sat_const_planeindex_satindex
                        sat_names.append(sat_name)
                        ii += 1
                if key == 1:
                    ic_const_all = ic_const
                else:
                    ic_const_all = np.vstack((ic_const_all, ic_const))
            
            acceleration_settings = {}
            
            # # Add empty bodies to body list and accelerations to each sat
            # for sat_name in sat_names:
            #     bodies.create_empty_body(sat_name)
            #     acceleration_settings[sat_name] = acceleration_settings_sat
            # # Add central bodies for each satellite
            # central_bodies = ["Earth" for ii in sat_names]
