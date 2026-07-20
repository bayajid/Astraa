#%% IMPORTS
import numpy as np
import os, sys
sys.path.insert(1, os.getcwd()[:os.getcwd().index('astropynaric')+13])
import astronomy_tools.constants as const
def lla_2_ecef(lla):
    # convert geodetic coordinates to ECEF [deg deg m]
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2

    lat_deg = lla[0]
    lon_deg = lla[1]
    # Convert degrees to radians
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    X = (N + lla[2]) * np.cos(lat) * np.cos(lon)
    Y = (N + lla[2]) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + lla[2]) * np.sin(lat)
    return np.array([X, Y, Z])
def get_kepler_param(tud_converter, states, calc_arglat = 1, reverse = 0):
    """function to convert cartesian pos/vel [m,m/s] to kepler parameters
    0 Semi-major axis (except if eccentricity = 1.0, then represents semilatus rectum)
    1 Eccentricity
    2 Inclination
    3 Argument of periapsis [rad]
    4 Longitude of ascending node
    5 True anomaly [rad]

    Args:
        tud_converter (_type_): _description_
        states (_type_): x y z v_x/y/z
        calc_arglat - whether to compute argument of latitude
        reverse - whether the arg of latitude must be reversed to the descending node 
            (necessary to compare N and S bound satellites in boundary seam)
    Reutnrs:
        0 kep states [rad]
        1 arg_lat [rad]
    """    
    kepler_outputs = np.zeros(states.shape)
    for ii, s_ii in enumerate(states):
        kepler_states_ii = tud_converter.convert_cart2kepler(s_ii)
        kepler_outputs[ii,:] = kepler_states_ii
    if calc_arglat:
        arg_lat = np.unwrap(kepler_outputs[:,5] + kepler_outputs[:,3]) % (2*np.pi)
        # arg_lat = kepler_outputs[:,5] + kepler_outputs[:,3]
        if reverse:
            arg_lat = np.pi - arg_lat
            arg_lat = [ii + np.pi*2 if ii < 0 else ii for ii in arg_lat]
    else:
        arg_lat=0
    return kepler_outputs, arg_lat