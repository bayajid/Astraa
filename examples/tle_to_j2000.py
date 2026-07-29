'''
From Succulent
'''
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy import units as u
from astropy.time import Time
from sgp4.api import Satrec
import os
import numpy as np
import time
import pandas as pd
import datetime
from tqdm import tqdm

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
    for ii, jd_ii in tqdm(enumerate(t_vec_used), total=len(t_vec_used), desc="Propagatine TLE in TEME:"):
        _, r, v = tle_file.sgp4(jd_ii, 0)        
        r_ii = np.array(r)*1e3
        v_ii = np.array(v)*1e3
        # TEME outputs
        state_output[ii,0] = jd_ii
        state_output[ii,1:4] = r_ii
        state_output[ii,4:] = v_ii
    return state_output

def rotate_teme2eci(states_teme, t_gps_0):
    """fUNCTION to rotate TEME states [t_jd_utc, r, v] to
    J2000 states [t_jd_utc, r, v] 
    Args:
        states_teme (array): TEME states [JD, m, m/s]
        t_gps_0 (float): GPS time at t_0 [s]
    Returns:
        states_j2000: ECI states [t_gps [s], m, m/s]
    """
    # make placeholders
    states_ecef = np.zeros(states_teme.shape)

    # extract Juilian Date time vector (UTC)
    t_jd_all = states_teme[:,0]
    dt_jd_s = (t_jd_all - t_jd_all[0])*86400 # [s]
    
    for ii, t_jd in tqdm(enumerate(t_jd_all),total=len(t_jd_all), desc="Rotating TEME to J2000:"):
        # Setup TEME state object in Astropy
        t_ap = Time(t_jd, format = 'jd')
        r_teme = states_teme[ii,1:4]
        v_teme = states_teme[ii,4:]
        r_teme_ap = CartesianRepresentation(r_teme*u.m)
        v_teme_ap = CartesianDifferential(v_teme*u.m/u.s)
        state_teme = TEME(r_teme_ap.with_differentials(v_teme_ap), obstime=t_ap)
        # transform to International Terrestrial Reference System (ECEF)
        state_ecef = state_teme.transform_to(GCRS(obstime = t_ap))     
        r_ecef = state_ecef.cartesian.get_xyz().value # m
        v_ecef = state_ecef.velocity.get_d_xyz().value*1e3 # m/s
        
        # store
        states_ecef[ii,0] = t_gps_0 + dt_jd_s[ii]
        states_ecef[ii,1:4] = r_ecef
        states_ecef[ii,4:] = v_ecef

    return states_ecef

def propagate_and_rotate_tle(tle_row_1, tle_row_2, save_name='tle_orbit', prop_length=1/24, dt=60, output_path=r'inputs/tle_orbits', force_prop = 0, 
                             t_gps_prop_start = None):
    """script to load and propagate a TLE file to GCRF/J2000 frame

    Args:
        tle_path (path): path to text file containing TLE data [sat name; line 1; line 2]
        prop_length (int, optional): Length of propagation [days]. Defaults to 1/24 day
        dt (int, optional): propagation time-step [s]. Defaults to 60.
    """    
    
    # create full file-name
    date_start = datetime.datetime.now().strftime("%y_%m_%d")
    save_name_full = f'{output_path}/{save_name}_{date_start}.csv'
    #Check if TLE file is already propagated
    if os.path.exists(save_name_full):
        print(f'TLE file already propagated for date: {date_start}')
        print(f'Loading TLE file from {save_name_full}')
        df = pd.read_csv(save_name_full)
        propagated_orbit_eci = df.to_numpy()
        return propagated_orbit_eci, save_name_full

        if not force_prop:
            print(f'If you wish to propagate the TLE regardless; set force_propagation = 1, currently: force_propagation = {force_prop}')
            return save_name_full
        else:
            print(f'Propagating the TLE regardless. To use already propagated data, set force_propagation = 0. Currently force_propagation = {force_prop}')
    else:

        print(f'TLE propagation for date: {date_start} for {prop_length*24} hours')
        
        
        # with open(tle_path, 'r') as tle_all:
        #     tle_file = tle_all.readlines()
        #     for ii, row in enumerate(tle_file):
        #         row = row.strip('\n')
        #         tle_file[ii] = row
        # tle_row_1 = tle_file[1]
        # tle_row_2 = tle_file[2]
        tle_to_prop = Satrec.twoline2rv(tle_row_1, tle_row_2)
        
        # Get time of start
        if t_gps_prop_start is None:
            t_start = Time.now()
        else:
            t_start = Time(t_gps_prop_start, format='gps')
        t_gps_0 = t_start.gps
        t_jd_0 = t_start.jd
        t_jd_end = t_start.jd + prop_length

        print('Propagating TLE')
        t_pre_prop = time.perf_counter()
        propagated_orbit_teme = propagate_tle_teme(tle_to_prop, t_jd_0, t_jd_end, dt)    
        dt_prop = time.perf_counter() - t_pre_prop
        
        print(f'TLE propagated in {dt_prop:1f} s')
        propagated_orbit_eci = rotate_teme2eci(propagated_orbit_teme, t_gps_0 = t_gps_0)
        dt_rot = time.perf_counter() - t_pre_prop - dt_prop
        print(f'TLE rotated to J2000 in {dt_rot:.1f} s')
        
        output_file = pd.DataFrame(data=propagated_orbit_eci, columns=['t_gps_s', 'r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z'])
        output_file.to_csv(save_name_full, index=False)
        print(f'TLE orbit successfully saved to {save_name_full}')
        return propagated_orbit_eci,save_name_full
if __name__ == '__main__':
    tle_path = r'examples/input_data/sat/ISS_tle.txt'
    # propagate
    propagate_and_rotate_tle(tle_path, 'tle_input_example', prop_length=1, dt=60, output_path=r'examples/output_data/tables')