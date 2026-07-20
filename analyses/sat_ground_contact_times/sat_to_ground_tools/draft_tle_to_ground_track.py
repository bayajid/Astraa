#%%
import numpy as np
import matplotlib.pyplot as plt
# import poliastro as poli
import os
import io
import json
import pandas as pd
import xml.etree.ElementTree as ET
import importlib
from warnings import warn
from astropy.coordinates import TEME, GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy import units as u
# from poliastro.ephem import Ephem
# from poliastro.frames import Planes
# import httpx
from sgp4 import exporter, omm
from sgp4.api import Satrec
import skyfield.sgp4lib as sgp4lib
import skyfield
# from poliastro.util import time_range
from astropy import units as u
from astropy.time import Time
import gt_calc as gt_tools
# import attitutude_tools. as att
## Functions taken from https://docs.poliastro.space/en/stable/examples/Loading%20OMM%20and%20TLE%20satellite%20data.html
def _generate_url(catalog_number, international_designator, name):
    params = {
        "CATNR": catalog_number,
        "INTDES": international_designator,
        "NAME": name,
    }
    param_names = [
        param_name
        for param_name, param_value in params.items()
        if param_value is not None
    ]
    if len(param_names) != 1:
        raise ValueError(
            "Specify exactly one of catalog_number, international_designator, or name"
        )
    param_name = param_names[0]
    param_value = params[param_name]
    url = (
        "https://celestrak.org/NORAD/elements/gp.php?"
        f"{param_name}={param_value}"
        "&FORMAT=XML"
    )
    return url


def _segments_from_query(url):
    response = httpx.get(url)
    response.raise_for_status()

    if response.text == "No GP data found":
        raise ValueError(
            f"Query '{url}' did not return any results, try a different one"
        )
    tree = ET.parse(io.StringIO(response.text))
    root = tree.getroot()

    yield from omm.parse_xml(io.StringIO(response.text))


def load_gp_from_celestrak(
    *, catalog_number=None, international_designator=None, name=None
):
    """Load general perturbations orbital data from Celestrak.

    Returns
    -------
    Satrec
        Orbital data from specified object.

    Notes
    -----
    This uses the OMM XML format from Celestrak as described in [1]_.

    References
    ----------
    .. [1] Kelso, T.S. "A New Way to Obtain GP Data (aka TLEs)"
       https://celestrak.org/NORAD/documentation/gp-data-formats.php

    """
    # Assemble query, raise an error if malformed
    url = _generate_url(catalog_number, international_designator, name)

    # Make API call, raise an error if data is malformed
    for segment in _segments_from_query(url):
        # Initialize and return Satrec object
        sat = Satrec()
        omm.initialize(sat, segment)

        yield sat
def print_sat(sat, name):
    """Prints Satrec object in convenient form."""
    sat_dict = exporter.export_omm(sat, name)
    print(json.dumps(sat_dict, indent=2))
    return sat_dict
def ephem_from_gp(sat, times):
    errors, rs, vs = sat.sgp4_array(times.jd1, times.jd2)
    if not (errors == 0).all():
        warn(
            "Some objects could not be propagated, "
            "proceeding with the rest",
            stacklevel=2,
        )
        rs = rs[errors == 0]
        vs = vs[errors == 0]
        times = times[errors == 0]

    cart_teme = CartesianRepresentation(
        rs << u.km,
        xyz_axis=-1,
        differentials=CartesianDifferential(
            vs << (u.km / u.s),
            xyz_axis=-1,
        ),
    )
    cart_gcrs = (
        TEME(cart_teme, obstime=times)
        .transform_to(GCRS(obstime=times))
        .cartesian
    )

    return Ephem(cart_gcrs, times, plane=Planes.EARTH_EQUATOR)

def propagate_tle_teme(tle_file, jd_start, jd_end, dt):
    """Function to propagate a tle_file object using the SGP4 propagator

    Args:
        tle_file (object): TLE object
        jd_start (float): prop start time in Julian Date [UTC]
        jd_end (float): prop end time in Julian Date [UTC]
        dt (float/int): time-step [s] between output steps
    Returns:
        state_output - t; r; v [JD, m, m/s] in True-Equator Mean-Equinox (TEME) frame
    """    
    t_vec_used = np.arange(jd_start, jd_end, dt / 86400)
    nrows = len(t_vec_used)

    state_output = np.zeros((nrows, 7)) # t, r, v [m,m/s]
    for ii, jd_ii in enumerate(t_vec_used):
        error, r, v = tle_file.sgp4(jd_ii, 0)        
        r_ii = np.array(r)*1e3
        v_ii = np.array(v)*1e3
        # TEME outputs
        state_output[ii,0] = jd_ii
        state_output[ii,1:4] = r_ii
        state_output[ii,4:] = v_ii
    return state_output
def rotate_teme2ecefgt(states_teme):
    """fUNCTION to rotate TEME states [t_jd_utc, r, v] to
    ECEF states [t_jd_utc, r, v] and also provide the ground track
    [long, lat, altitutde] using Astropy's built in transformations

    Args:
        states_teme (array): TEME states [JD, m, m/s]

    Returns:
        states_ecef, gt_data: ECEF states [JD, m, m/s], gt_data- time, long, lat, alt [JD, deg, deg, m]
    """
    # make placeholders
    states_ecef = np.zeros(states_teme.shape)
    gt_data = np.zeros((states_teme.shape[0], 4))

    # extract Juilian Date time vector (UTC)
    t_jd_all = states_teme[:,0]

    for ii, t_jd in enumerate(t_jd_all):
        # Setup TEME state object in Astropy
        t_ap = Time(t_jd, format = 'jd')
        r_teme = states_teme[ii,1:4]
        v_teme = states_teme[ii,4:]
        r_teme_ap = CartesianRepresentation(r_teme*u.m)
        v_teme_ap = CartesianDifferential(v_teme*u.m/u.s)
        state_teme = TEME(r_teme_ap.with_differentials(v_teme_ap), obstime=t_ap)
        # transform to International Terrestrial Reference System (ECEF)
        state_ecef = state_teme.transform_to(ITRS(obstime = t_ap))     
        r_ecef = state_ecef.cartesian.get_xyz().value # m
        v_ecef = state_ecef.velocity.get_d_xyz().value*1e3 # m/s
        loc_state = state_ecef.earth_location.geodetic
        
        # store
        states_ecef[ii,0] = t_jd
        states_ecef[ii,1:4] = r_ecef
        states_ecef[ii,4:] = v_ecef

        gt_data[ii,0] = t_jd
        gt_data[ii,1] = loc_state.lon.value
        gt_data[ii,2] = loc_state.lat.value
        gt_data[ii,3] = loc_state.height.value
    return states_ecef, gt_data
def get_heading_angle(v_ecef, long_data, lat_data):
    # function to rotate from ECEF to East North Up and compute
    # the heading angle using the velocity vector
    # heading_angle - heading angle wrt North [deg]
    
    v_enu_all = np.zeros(v_ecef.shape)
    heading_angles = np.zeros(v_ecef.shape[0])

    for ii, v_ecef_ii in enumerate(v_ecef):
        lat = lat_data[ii]
        long = long_data[ii]

        rot_1 = att.rot_basic(90 - lat, rot_ax = 1)
        rot_2 = att.rot_basic(90 + long, rot_ax = 3)

        # rotate that idiot velocity
        rot_comb = rot_1 @ rot_2
        v_enu_ii = rot_comb @ v_ecef_ii # get rotated, idiot

        # compute heading angle
        # heading_rad = np.arctan2(-v_enu_ii[0], v_enu_ii[1])
        heading_rad = np.arctan(-v_enu_ii[0]/v_enu_ii[1])
        heading_angle = np.rad2deg(heading_rad)
        # store
        v_enu_all[ii,:] = v_enu_ii
        heading_angles[ii] = heading_angle

    heading_angles = heading_angles + 360
    for ii, angle in enumerate(heading_angles):
        if  angle > 360:
            heading_angles[ii] = heading_angles[ii] - 360
    return heading_angles, v_enu_all

if __name__ == '__main__':
    ## Configure Host & Target S/C sat1Name = "STARLINK-3776"
    iss_number = 8709
    iss_tle = list(load_gp_from_celestrak(catalog_number = 25544    
    ))
    # print_sat(iss_tle[0], 'ISS')
    # print_sat(iss_tle[10], 'ISS')
    tle_chosen = iss_tle[0]
    tle_metadata = print_sat(tle_chosen, 'ISS')
    t_tle = tle_metadata['EPOCH']
    # 1 day prop
    t_start = Time(t_tle, format = 'isot', scale = 'utc')
    t_0 = t_start.jd
    t_end = t_start.jd + 2 # 6 hrs
    dt = 10
    #%%
    # state_output = propagate_tle_teme(tle_chosen, t_0, t_end, dt)[3600:3700]
    i0 = 590
    i0 = i0 *9.7
    
    state_output = propagate_tle_teme(tle_chosen, t_0, t_end, dt)[int(i0):int(i0)+590]

    # rotate to GTRF example
    if 0: # single rotation
        ii_rot = 0
        t_jd = state_output[ii_rot,0]
        r_teme = state_output[ii_rot,1:4]
        v_teme = state_output[ii_rot,4:]

        t_ap = Time(t_jd, format = 'jd')
        r_teme_ap = CartesianRepresentation(r_teme*u.m)
        v_teme_ap = CartesianDifferential(v_teme*u.m/u.s)
        state_teme = TEME(r_teme_ap.with_differentials(v_teme_ap), obstime=t_ap)
        state_ecef = state_teme.transform_to(ITRS(obstime = t_ap))
        loc_state = state_ecef.earth_location.geodetic
        print(loc_state)
    states_ecef, gt_data = rotate_teme2ecefgt(state_output)
    gt_data[:,1] = gt_data[:,1] + 180 # make [0:360]
    heading_angles, v_enu_all = get_heading_angle(states_ecef[:,4:], gt_data[:,1], gt_data[:,2])
    heading_angles = gt_tools.calc_heading_fromgt(gt_data[:,0], gt_data[:,1], gt_data[:,2])
    #%%
    if 1:
        # recalculate heaind angle...analytically?

        t_vec_jd = state_output[:,0] - state_output[0,0]
        t_vec_used = heading_angles[:,[0]] 
        t_vec_used = t_vec_used - t_vec_used[0]
        df_gt = pd.DataFrame(data = np.hstack((t_vec_used, gt_data[:-1,[1,2]], heading_angles[:,[1]])), columns = ['t_jd', 'long', 'lat', 'heading'])
        df_gt.to_csv(f'verification/gt_tle.csv', index = False
        )
        import tle_iss_verification
        importlib.reload(tle_iss_verification)
        if 0:
            f, axs = plt.subplots(3, figsize = (16,10))
            ii = 0

            ax = axs[ii]
            for jj in range(3):
                ax.plot(t_vec_jd, state_output[:,jj+1], label = 'xyz'[jj])
            ax.set_ylabel('Cart pos ECEF [m]')
            ax.grid()
            ax.legend()

            ii = 1
            ax = axs[ii]
            for jj in range(3):
                ax.plot(t_vec_jd, v_enu_all[:,jj], label = 'ENU'[jj])
            ax.set_ylabel('ENU velocity [m/s]')
            ax.grid()
            ax.legend()

            ii = 2
            ax = axs[ii]
            
            ax.plot(t_vec_jd, heading_angles, label = 'Heading')
            ax.plot(t_vec_jd, gt_data[:,1], label = 'Long')
            ax.plot(t_vec_jd, gt_data[:,2], label = 'Lat')
            ax.set_ylabel('Angles [deg]')
            ax.grid()
            ax.legend()
    # plots
    if 0:
        f, ax = plt.subplots()
        t_vec_jd = state_output[:,0] - state_output[0,0]
        for ii in range(3):
            ax.plot(t_vec_jd, state_output[:,ii+1], label = 'xyz'[ii])
        ax.plot(t_vec_jd, np.linalg.norm(state_output[:,1:4],axis=1), label = 'orb. radius')
        ax.set_xlabel('JD [days since start]')
        ax.set_ylabel('TEME Cart. Pos [m]')
        ax.legend()
        ax.grid()
    if 0:
        f, ax = plt.subplots()
        t_vec_jd = state_output[:,0] - state_output[0,0]
        head_angles = [get_heading_angle(state_output[ii,4:], gt_data[ii,1], gt_data[ii,2])[0] for ii, state in enumerate(state_output)]
        
        ax.plot(t_vec_jd, head_angles, label = 'heading')        
        ax.plot(t_vec_jd, gt_data[:,1], label = 'long')
        ax.plot(t_vec_jd, gt_data[:,2], label = 'lat')
        ax.set_xlabel('JD [days since start]')
        ax.set_ylabel('Angles [deg]')
        ax.legend()
        ax.grid()
    # assert error == 0
    # CartesianRepresentation(rs << u.km, xyz_axis=-1)
    # CartesianDifferential(vs << (u.km / u.s), xyz_axis=-1)