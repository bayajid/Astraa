# import numpy as np
# import pandas as pd
# from astropy.time import Time
# from astropy.coordinates import get_body_barycentric_posvel
# import astropy.units as u
# import json

# # --- Load configuration ---
# with open("config.json", "r") as f:
#     config = json.load(f)

# start_time_str = config["start_time"]
# end_time_str   = config["end_time"]
# timezone       = config["timezone"]
# sampling_rate_hz = int(config.get("sampling_rate_hz", 1))
# scale_distance_km = float(config.get("scale_distance_km", 700))  # LEO distance

# # --- Create time array ---
# times_local = pd.date_range(start=start_time_str,
#                             end=end_time_str,
#                             freq=f"{int(1000/sampling_rate_hz)}L",
#                             tz=timezone)
# times_utc = times_local.tz_convert("UTC")
# astropy_times = Time(times_utc.to_pydatetime())

# # --- Prepare arrays ---
# x_list, y_list, z_list = [], [], []
# vx_list, vy_list, vz_list = [], [], []
# gps_seconds_list = []

# angular_uncertainties_rad = []

# for t in astropy_times:
#     # Get Sun position and velocity in GCRS
    
#     # Sun barycentric
#     sun_pos, sun_vel = get_body_barycentric_posvel("sun", t)

#     # Earth barycentric
#     earth_pos, earth_vel = get_body_barycentric_posvel("earth", t)

#     # Convert to Earth-centered inertial (GCRS ~ ECI)
#     eci_pos = (sun_pos - earth_pos).xyz.to(u.km).value
#     eci_vel = (sun_vel - earth_vel).xyz.to(u.km/u.s).value

#     # --- Create unit vector and scale to LEO ---
#     norm_pos = np.linalg.norm(eci_pos)
#     unit_vec = eci_pos / norm_pos
#     pos_scaled = unit_vec * scale_distance_km
#     vel_scaled = eci_vel / norm_pos * scale_distance_km  # scale velocity proportionally

#     # --- Store results ---
#     x_list.append(pos_scaled[0])
#     y_list.append(pos_scaled[1])
#     z_list.append(pos_scaled[2])
#     vx_list.append(vel_scaled[0])
#     vy_list.append(vel_scaled[1])
#     vz_list.append(vel_scaled[2])
#     gps_seconds_list.append(t.gps)

# # --- Save CSV ---
# df = pd.DataFrame({
#     "time_gps_s": gps_seconds_list,
#     "x_km": x_list,
#     "y_km": y_list,
#     "z_km": z_list,
#     "vx_km_s": vx_list,
#     "vy_km_s": vy_list,
#     "vz_km_s": vz_list
# })

# df.to_csv("sun_positions_scaled_gps.csv", index=False)
# print("Saved scaled Sun positions & velocities to sun_positions_scaled_gps.csv")


# from astropy.coordinates import solar_system_ephemeris
# solar_system_ephemeris.set('de440')  


#%%
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_body, solar_system_ephemeris
import warnings
from astropy.coordinates import EarthLocation, AltAz, CartesianRepresentation, GCRS, ITRS
from astropy import units as u

import numpy as np

def cartesian_to_az_el(x, y, z, degrees=True):
    """
    Convert Cartesian coordinates (x, y, z) to azimuth and elevation.
    
    Parameters:
        x, y, z : float or array-like
            Cartesian coordinates.
        degrees : bool
            If True, return angles in degrees. If False, return in radians.
    
    Returns:
        az : float or ndarray
            Azimuth angle (0 to 360 degrees, or 0 to 2*pi radians).
        el : float or ndarray
            Elevation angle (-90 to 90 degrees, or -pi/2 to pi/2 radians).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    r_xy = np.hypot(x, y)  # Distance in XY-plane
    az = np.arctan2(y, x)  # Azimuth angle (radians)
    el = np.arctan2(z, r_xy)  # Elevation angle (radians)

    # Normalize azimuth to [0, 2*pi)
    az = np.mod(az, 2 * np.pi)

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    return az, el

def calc_ae(los, official_convention = 1, wrap =1):
    # function to calculate the Azimuth and Elevation [rad]
    # for a given LOS vector in the Global/LCT frame
    # az = np.arctan2(-los[2], los[1]) # rad
    # az = np.arctan((los[1]/ los[0])) # rad
    # el = np.arctan(los[2]/np.linalg.norm(los[:2])) # rad
    if not official_convention:
        az = np.arctan((los[1]/ los[0])) # rad
        el = np.arctan(-los[2]/np.linalg.norm(los[:2])) # rad
    else:
        # Sep 7 - Betul tells me to remove this
        # if los[0] > 0:
        #     az = np.arctan(los[1] / los[0])
        # else:
        #     az = np.pi + np.arctan(los[1] / los[0])        
        az = np.arctan2(los[1], los[0]) # rad
        el = np.arcsin(los[2] / np.linalg.norm(los))
        if wrap:
            if az > np.pi:
                az = az - 2 * np.pi
    return [az, el]

def get_sun_position_eci(time, ephemeris_type='builtin', verbose=True):
    """
    Get Sun position and velocity in Earth-Centered Inertial (ECI/GCRS) coordinates
    using different ephemeris methods.
    
    Parameters:
    -----------
    time : astropy.time.Time or str or float
        Time for calculation (can be Time object, ISO string, or JD)
    ephemeris_type : str
        Ephemeris to use: 'builtin', 'jpl' (DE430), 'de440', or 'de440s'
    verbose : bool
        Print information about ephemeris being used
        
    Returns:
    --------
    dict : Dictionary containing position, velocity, and metadata
    """
    
    # Convert input to Time object if needed
    if not isinstance(time, Time):
        if isinstance(time, (int, float)):
            # Interpret numeric input as Julian Date
            t = Time(time, format='jd')
        else:
            # Let astropy parse strings or datetime-like inputs
            t = Time(time)
    else:
        t = time
    
    # Dictionary to store results
    result = {
        'time': t,
        'ephemeris': ephemeris_type,
        'position_km': None,
        'velocity_km_s': None,
        'accuracy_estimate': None,
        'valid_range': None
    }
    
    # Set ephemeris and get accuracy information
    original_ephemeris = solar_system_ephemeris.get()
    
    try:
        if ephemeris_type == 'builtin':
            solar_system_ephemeris.set('builtin')
            result['accuracy_estimate'] = '~1-10 arcseconds'
            result['valid_range'] = 'All times (limited accuracy)'
            
        elif ephemeris_type == 'jpl':
            try:
                solar_system_ephemeris.set('jpl')  # Uses DE430 by default
                result['accuracy_estimate'] = '~0.1-1 arcseconds'
                result['valid_range'] = '1550-2650 CE'
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not set JPL ephemeris: {e}")
                    print("Install jplephem: pip install jplephem")
                # Fall back to builtin
                solar_system_ephemeris.set('builtin')
                result['ephemeris'] = 'builtin (fallback)'
                result['accuracy_estimate'] = '~1-10 arcseconds'
                
        elif ephemeris_type == 'de440':
            try:
                solar_system_ephemeris.set('de440')
                result['accuracy_estimate'] = '~0.01-0.1 arcseconds'
                result['valid_range'] = '1550-2650 CE'
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not set DE440 ephemeris: {e}")
                    print("Install jplephem: pip install jplephem")
                    print("DE440 files may need to be downloaded automatically")
                # Fall back to builtin
                solar_system_ephemeris.set('builtin')
                result['ephemeris'] = 'builtin (fallback)'
                result['accuracy_estimate'] = '~1-10 arcseconds'
                
        elif ephemeris_type == 'de440s':
            try:
                solar_system_ephemeris.set('de440s')
                result['accuracy_estimate'] = '~0.01-0.1 arcseconds'
                result['valid_range'] = '1950-2050 CE (~10MB file)'
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not set DE440s ephemeris: {e}")
                    print("Install jplephem: pip install jplephem")
                # Fall back to builtin
                solar_system_ephemeris.set('builtin')
                result['ephemeris'] = 'builtin (fallback)'
                result['accuracy_estimate'] = '~1-10 arcseconds'
        else:
            raise ValueError(f"Unknown ephemeris type: {ephemeris_type}")
        
        if verbose:
            print(f"Using ephemeris: {solar_system_ephemeris.get()}")
            print(f"Accuracy estimate: {result['accuracy_estimate']}")
            print(f"Valid range: {result['valid_range']}")
        
        # Get Sun position in GCRS (Earth-centered inertial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress ephemeris warnings for cleaner output
            sun_gcrs = get_body("sun", t)
        
        # Extract Cartesian coordinates
        sun_cartesian = sun_gcrs.cartesian
        
        # Position in km
        result['position_km'] = sun_cartesian.xyz.to(u.km).value
        
        # Velocity in km/s (if available)
        try:
            if hasattr(sun_cartesian, 'differentials') and 's' in sun_cartesian.differentials:
                result['velocity_km_s'] = sun_cartesian.differentials['s'].d_xyz.to(u.km/u.s).value
            else:
                # Calculate numerical derivative for velocity if not available
                dt = 1 * u.s  # 1 second offset
                t_plus = t + dt
                sun_gcrs_plus = get_body("sun", t_plus)
                sun_cart_plus = sun_gcrs_plus.cartesian
                
                pos_plus = sun_cart_plus.xyz.to(u.km)
                pos_current = sun_cartesian.xyz.to(u.km)
                result['velocity_km_s'] = ((pos_plus - pos_current) / dt).to(u.km/u.s).value
                
        except Exception as e:
            if verbose:
                print(f"Could not calculate velocity: {e}")
            result['velocity_km_s'] = None
    
    finally:
        # Restore original ephemeris setting
        solar_system_ephemeris.set(original_ephemeris)
    
    return result

def compare_ephemeris_methods(time, verbose=True):
    """
    Compare Sun positions using different ephemeris methods.
    
    Parameters:
    -----------
    time : astropy.time.Time or str or float
        Time for calculation
    verbose : bool
        Print detailed comparison
    
    Returns:
    --------
    dict : Dictionary with results from all methods
    """
    
    methods = ['builtin', 'jpl', 'de440', 'de440s']
    results = {}
    
    if verbose:
        print(f"\n=== Sun Position Comparison at {time} ===\n")
    
    for method in methods:
        if verbose:
            print(f"\n--- {method.upper()} Method ---")
        
        try:
            results[method] = get_sun_position_eci(time, method, verbose=verbose)
            
            if verbose and results[method]['position_km'] is not None:
                pos = results[method]['position_km']
                distance = np.linalg.norm(pos)
                print(f"Position [km]: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
                print(f"Distance: {distance:.1f} km ({distance/1.496e8:.6f} AU)")
                
                if results[method]['velocity_km_s'] is not None:
                    vel = results[method]['velocity_km_s']
                    speed = np.linalg.norm(vel)
                    print(f"Velocity [km/s]: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
                    # print(f"Speed: {speed:.3f} km/s")
                    
        except Exception as e:
            if verbose:
                print(f"Failed to get {method} ephemeris: {e}")
            results[method] = None
    
    # Calculate differences between methods
    if verbose and len([r for r in results.values() if r is not None]) > 1:
        print(f"\n--- Position Differences ---")
        builtin_pos = results.get('builtin', {}).get('position_km')
        
        for method in ['jpl', 'de440', 'de440s']:
            if results.get(method) and results[method]['position_km'] is not None and builtin_pos is not None:
                diff = results[method]['position_km'] - builtin_pos
                diff_magnitude = np.linalg.norm(diff)
                print(f"{method.upper()} vs BUILTIN: {diff_magnitude:.1f} km difference")
    
    return results

# Example usage
if __name__ == "__main__":
    # Example 1: Single ephemeris method
    print("=== Example 1: Using DE440 ephemeris ===")
    # t = Time('2025-09-23T12:00:00')
    now = Time.now()        # Current time in UTC
    gps_time_obj = Time(now.gps, format='gps')
    #t = Time(t_gps, format='gps', scale='utc')
    t = gps_time_obj
    print(t)
    
    result = get_sun_position_eci(t, ephemeris_type='de440', verbose=True)
    
    if result['position_km'] is not None:
        print(f"\nSun position (ECI/GCRS): {result['position_km']} km")
        #az, el = cartesian_to_az_el(result['position_km'][0]*1e3,result['position_km'][1]*1e3,result['position_km'][2]*1e3 )
        az1, el1 = calc_ae(result['position_km']*1e3)
        
        #print(f"Az: {az}\t El: {el}")
        print(f"Az1: {np.degrees(az1)}\t El1: {np.degrees(el1)}")
        if result['velocity_km_s'] is not None:
            print(f"Sun velocity (ECI/GCRS): {result['velocity_km_s']} km/s")
    
    # Example 2: Compare all methods
    print("\n" + "="*60)
    print("=== Example 2: Compare all ephemeris methods ===")
    
    comparison = compare_ephemeris_methods(t, verbose=True)
    
    # Example 3: Different time formats
    print("\n" + "="*60)
    print("=== Example 3: Different time formats ===")
    
    # Using Julian Date
    result_jd = get_sun_position_eci(2460310.0, 'de440s', verbose=False)
    print(f"Using JD 2460310.0: Distance = {np.linalg.norm(result_jd['position_km']):.1f} km")
    
    # Using ISO string
    result_iso = get_sun_position_eci('2024-06-21T12:00:00', 'de440s', verbose=False)
    print(f"Using ISO string: Distance = {np.linalg.norm(result_iso['position_km']):.1f} km")
# %%
