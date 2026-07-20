# TUDAT-supported class to perform astronomic rotations/conversions, which 
# are normally too tediosu to implement
from tudatpy.kernel.interface import spice
from tudatpy.data import save2txt
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup, environment
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation.environment_setup import rotation_model
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.astro import time_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

import os
import sys
import numpy as np
import datetime as dt
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
## Own imports
import basic_tools.time_conversion as t_conv
import attitude_tools.rotations as rot
import basic_tools.vector_operations as vec
# Add parent directory to paths

from astronomy_tools.constants import mu_e 

class tudat_predictor:
    """Class to perform KF prediction step with different dynamical model choices
    Additionally expanded to perform (carried over from thesis)
    - Propagations
        - full orbit propagations
    - Conversions:
        - ECI-ECEF-RSW rotations        
        - TODO Interface Az/El calculation
    - Error in RSW rotation to ECI and subsequent propagation for some Initial Condition
    - IC error implementations and rotations
    """    
    def __init__(self, integrator = 'rk4', convention = 'IAU2006', sat_name = 'GRACE-FO', t_offset = None, dep_var = []):
        spice.load_standard_kernels()
        bodies_to_create = ['Earth', 'Sun', 'Moon']
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"

        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

        bodies = environment_setup.create_system_of_bodies(body_settings)

        ### Create Sat host body
        bodies.create_empty_body(sat_name)

        if convention == 'IAU2006':
            convention_used = environment_setup.rotation_model.IAUConventions.iau_2006
        elif convention == 'IAU2000A':
            convention_used = environment_setup.rotation_model.IAUConventions.iau_2000_a
        elif convention == 'IAU2000B':
            convention_used = environment_setup.rotation_model.IAUConventions.iau_2000_b

            print(f'----ITRF to ICRF rotation convention: {convention}. Propagating satellite: {sat_name}----')
            rotation_model_settings =  environment_setup.rotation_model.gcrs_to_itrs(convention_used, 'GCRS')
            environment_setup.add_rotation_model(bodies, 'Earth', rotation_model_settings)
        earth_rotation_model = bodies.get_body("Earth").rotation_model

        bodies_to_propagate = [sat_name]
        central_bodies = ["Earth" for ii in bodies_to_propagate]
        
        self.sat_name = sat_name
        self.integrator = integrator

        ## Set time-offset
        if type(t_offset) == type(None):            
            self.dt_2j2000 = t_conv.dt_gps2j2000tt() 
        else:
            self.dt_2j2000 = t_offset
        self.dep_var = dep_var
        self.acceleration_label = ''
        self.acceleration_settings = None
        self.earth_rotation_model = earth_rotation_model
        ## NEEDED FOR PROPAGATION
        self.central_bodies = central_bodies 
        self.bodies_to_propagate = bodies_to_propagate 
        self.bodies = bodies
    
    def set_time(self, t_input, t_zone = 'UTC'):
        # Method to give a time input and its t_zone (UTC, GPS_seconds, J2000)
        # set current time attribute in t_j2000
        if type(t_input) == dt.datetime:
            t_gws = t_conv.utc2gws(t_input)
            t_j2000 = t_gws + self.dt_2j2000
        else:
            t_j2000 = t_input
        self.time = t_j2000
    
    def rotate_eci2ecef(self, X_eci, t_j2000 = None):
        """
        Rotate a state vector from Earth-Centered Inertial (ECI) frame to
        Earth-Centered Earth-Fixed (ECEF) frame.

        This function converts a 6-element state vector [position, velocity]
        in the ECI (GCRS) frame to the ECEF (ITRS) frame, taking into account
        the Earth's rotation. The velocity is correctly adjusted using
        the angular velocity of the Earth.

        Parameters
        ----------
        X_eci : array_like, shape (6,)
            State vector in ECI frame: [x, y, z, vx, vy, vz] in meters and meters/second.
        t_j2000 : float or None, optional
            Time since J2000 epoch in seconds. If None, `self.time` is used.

        Returns
        -------
        X_ecef : ndarray, shape (6,)
            State vector in ECEF frame: [x, y, z, vx, vy, vz] in meters and meters/second.

        Notes
        -----
        Velocity transformation uses:
            v_ecef = R * v_eci - ω × r_ecef
        where R is the ECI→ECEF rotation matrix, ω is Earth's angular velocity,
        and r_ecef is the rotated position vector.
        """
        if type(t_j2000) == type(None):
            t_j2000 = self.time
        rot_eci2ecef = environment.GcrsToItrsRotationModel.inertial_to_body_fixed_rotation(self.earth_rotation_model, t_j2000)
        om_earth = self.earth_rotation_model.angular_velocity_in_inertial_frame(t_j2000)
        r_ecef = rot_eci2ecef@ X_eci[:3]
        v_ecef = rot_eci2ecef@ X_eci[3:] - np.cross(om_earth, r_ecef) 
        X_ecef = np.hstack((r_ecef, v_ecef))
        return X_ecef
    
    def rotate_ecef2eci(self, X_ecef, t_j2000 = None):
        """
        Rotate a state vector from Earth-Centered Earth-Fixed (ECEF) frame to
        Earth-Centered Inertial (ECI) frame.

        This function converts a 6-element state vector [position, velocity]
        in the ECEF (ITRS) frame to the ECI (GCRS) frame, taking into account
        the Earth's rotation. The velocity is correctly adjusted using
        the angular velocity of the Earth.

        Parameters
        ----------
        X_ecef : array_like, shape (6,)
            State vector in ECEF frame: [x, y, z, vx, vy, vz] in meters and meters/second.
        t_j2000 : float or None, optional
            Time since J2000 epoch in seconds. If None, `self.time` is used.

        Returns
        -------
        X_eci : ndarray, shape (6,)
            State vector in ECI frame: [x, y, z, vx, vy, vz] in meters and meters/second.

        Notes
        -----
        Velocity transformation uses:
            v_eci = R * v_ecef + ω × r_eci
        where R is the ECEF→ECI rotation matrix, ω is Earth's angular velocity,
        and r_eci is the rotated position vector.
        """
        if type(t_j2000) == type(None):
            t_j2000 = self.time
        rot_ecef2eci = environment.GcrsToItrsRotationModel.body_fixed_to_inertial_rotation(self.earth_rotation_model, t_j2000)
        om_earth = self.earth_rotation_model.angular_velocity_in_inertial_frame(t_j2000)
        r_eci = rot_ecef2eci@ X_ecef[:3]
        v_eci = rot_ecef2eci@ X_ecef[3:] + np.cross(om_earth, r_eci)
        X_eci = np.hstack((r_eci, v_eci))
        return X_eci
    
    def convert_lla2ecef(self, lla):
        # function to rotate a lat/lon/h [deg, deg, km] to xyz ECEF
        R_E = 6378e3
        long = lla[1]
        lat = lla[0]
        spherical_state = np.array([R_E + lla[2]*1e3, np.deg2rad(lla[0]), np.deg2rad(lla[1]),
                                                                                 0, 0, 0])
        # ECEF
        cartesian_state = element_conversion.spherical_to_cartesian( spherical_state )
        return cartesian_state
    
    def rot_ecef2eci(self):
        rot_ecef2eci = environment.GcrsToItrsRotationModel.body_fixed_to_inertial_rotation(self.earth_rotation_model, self.time)
        return rot_ecef2eci
    
    def rot_eci2ecef(self):
        rot_eci2ecef = environment.GcrsToItrsRotationModel.inertial_to_body_fixed_rotation(self.earth_rotation_model, self.time)
        return rot_eci2ecef
    
    def convert_cart2kepler(self, input_cart):
        # 0 Semi-major axis (except if eccentricity = 1.0, then represents semilatus rectum)
        # 1 Eccentricity
        # 2 Inclination
        # 3 Argument of periapsis
        # 4 Longitude of ascending node
        # 5 True anomaly
        output_kep = element_conversion.cartesian_to_keplerian(input_cart, gravitational_parameter = mu_e)
    
        return output_kep
    
    def convert_kepler2cart(self, input_kepler):
        # 0 Semi-major axis (except if eccentricity = 1.0, then represents semilatus rectum)
        # 1 Eccentricity
        # 2 Inclination
        # 3 Argument of periapsis
        # 4 Longitude of ascending node
        # 5 True anomaly
        output_cart = element_conversion.keplerian_to_cartesian(input_kepler, gravitational_parameter = mu_e)
    
        return output_cart
    
    def rot_ecef2ned(self, lla):
        # Function to return the rotation matrix from ECEF to NED
        # inputs - latitude, longitude [deg]

        Rot_ecef2ned = rot.rot_basic(-(90 + lla[0]), 2) @ rot.rot_basic(lla[1], 3)
        
        return Rot_ecef2ned
    
    def calc_rotrsweci(self, r_h, v_h):
        # function to calculate and return the rotation matrix
        # from ECI to RSW at the host's position and velocity
        # inputs:
        # r_h - host sat pos [m] in ECI
        # v_h - host sat vel [m/s] in ECI
        # outputs:
        # ROT_RSWtoECI. Use: r_rsw = np.matmul(ROT_RSWfromECI, r_eci)
        R = vec.norm_vector(r_h) # radial direction unit vector (point at zenith)    
        rv_cross = np.cross(r_h, v_h)
        W = vec.norm_vector(rv_cross) # cross-track component, orthogonal to velocity and Radial
        S = np.cross(W, R)
        ROT_RSWfromECI = np.zeros((3,3))
        ROT_RSWfromECI[0,:] = R
        ROT_RSWfromECI[1,:] = S
        ROT_RSWfromECI[2,:] = W
        return ROT_RSWfromECI