# ==================================================
# File: capella_states.py
# Author: Bayajid Khan
# Created: 2026-03-27
# Description: Process Capella satellite state data + TLE generation
# ==================================================

import numpy as np
import pandas as pd
import math
from datetime import datetime

# Constants
MU = 398600.4418          # km³/s²
R_EARTH = 6378.137        # km
F = 1 / 298.257223563     # WGS84 flattening

def quat_to_rotmat(q):
    """Convert quaternion [qx, qy, qz, qw] to rotation matrix"""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

def ecef_to_eci(r_ecef, v_ecef, q):
    """Convert ECEF to ECI using quaternion"""
    R = quat_to_rotmat(q)
    r_eci = R @ r_ecef
    v_eci = R @ v_ecef
    return r_eci, v_eci

import numpy as np

MU = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km

# def orbital_elements(r, v):
#     """Compute classical orbital elements from state vector (r, v in km, km/s)"""

#     r = np.array(r, dtype=float)
#     v = np.array(v, dtype=float)

#     r_norm = np.linalg.norm(r)
#     v_norm = np.linalg.norm(v)

#     # Angular momentum
#     h = np.cross(r, v)
#     h_norm = np.linalg.norm(h)

#     # Inclination
#     i = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))

#     # Node vector
#     k = np.array([0.0, 0.0, 1.0])
#     n = np.cross(k, h)
#     n_norm = np.linalg.norm(n)

#     # Eccentricity vector
#     e_vec = (np.cross(v, h) / MU) - (r / r_norm)
#     e = np.linalg.norm(e_vec)

#     # Semi-major axis
#     a = 1 / ((2 / r_norm) - (v_norm**2 / MU))

#     # -----------------------
#     # RAAN
#     # -----------------------
#     if n_norm > 1e-10:
#         raan = np.arccos(np.clip(n[0] / n_norm, -1.0, 1.0))
#         if n[1] < 0:
#             raan = 2 * np.pi - raan
#     else:
#         raan = 0.0  # undefined → set to 0

#     # -----------------------
#     # Argument of Perigee
#     # -----------------------
#     if n_norm > 1e-10 and e > 1e-10:
#         argp = np.arccos(np.clip(np.dot(n, e_vec) / (n_norm * e), -1.0, 1.0))
#         if e_vec[2] < 0:
#             argp = 2 * np.pi - argp
#     else:
#         argp = 0.0  # undefined for circular/equatorial

#     # -----------------------
#     # True Anomaly
#     # -----------------------
#     if e > 1e-10:
#         nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0))
#         if np.dot(r, v) < 0:
#             nu = 2 * np.pi - nu
#     else:
#         # circular orbit → use angle from node vector instead
#         if n_norm > 1e-10:
#             nu = np.arccos(np.clip(np.dot(n, r) / (n_norm * r_norm), -1.0, 1.0))
#             if r[2] < 0:
#                 nu = 2 * np.pi - nu
#         else:
#             nu = 0.0  # fully undefined case

#     # -----------------------
#     # Perigee / Apogee
#     # -----------------------
#     rp = a * (1 - e)
#     ra = a * (1 + e)

#     return {
#         "a_km": a,
#         "ecc": e,
#         "inc_deg": np.degrees(i),
#         "raan_deg": np.degrees(raan),
#         "argp_deg": np.degrees(argp),
#         "true_anomaly_deg": np.degrees(nu),
#         "perigee_alt_km": rp - R_EARTH,
#         "apogee_alt_km": ra - R_EARTH
#     }

import numpy as np

MU = 398600.4418 # km^3/s^2
R_EARTH = 6378.137 # km

def orbital_elements(r, v):
    """Compute classical orbital elements from state vector"""
    r = np.array(r, dtype=float)
    v = np.array(v, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    # Angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    
    # Inclination
    i = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))
    
    # Node vector
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    n_norm = np.linalg.norm(n)
    
    # Eccentricity vector
    e_vec = (np.cross(v, h) / MU) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    a = 1 / ((2 / r_norm) - (v_norm**2 / MU))
    
    # -----------------------
    # RAAN
    # -----------------------
    if n_norm > 1e-10:
        raan = np.arccos(np.clip(n[0] / n_norm, -1.0, 1.0))
        if n[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0
    
    # -----------------------
    # Argument of Perigee
    # -----------------------
    if n_norm > 1e-10 and e > 1e-10:
        arg_perigee = np.arccos(np.clip(np.dot(n, e_vec) / (n_norm * e), -1.0, 1.0))
        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0.0
    
    # -----------------------
    # True Anomaly   <--- This was the main bug
    # -----------------------
    if e > 1e-10:
        nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_norm), -1.0, 1.0))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    else:
        # Circular orbit case - fixed version
        if n_norm > 1e-10:
            nu = np.arccos(np.clip(np.dot(n, r) / (n_norm * r_norm), -1.0, 1.0))
            # Better quadrant check using velocity direction
            if np.dot(np.cross(n, r), v) < 0:   # use cross product for correct direction
                nu = 2 * np.pi - nu
        else:
            # Equatorial circular - use atan2
            nu = np.arctan2(r[1], r[0])
            if nu < 0:
                nu += 2 * np.pi
    
    # -----------------------
    # Perigee / Apogee
    # -----------------------
    rp = a * (1 - e)
    ra = a * (1 + e)
    
    return {
        "a_km": a,
        "ecc": e,
        "inc_deg": np.degrees(i),
        "raan_deg": np.degrees(raan),
        "argp_deg": np.degrees(arg_perigee),
        "true_anomaly_deg": np.degrees(nu),
        "perigee_alt_km": rp - R_EARTH,
        "apogee_alt_km": ra - R_EARTH
    }

def keplerian_to_tle(a_km, ecc, inc_deg, raan_deg, argp_deg, mean_anomaly_deg,
                     tle_epoch, sat_name="CAPELLA", sat_num=99999, rev_num=0):
    """Generate valid TLE from Keplerian elements"""
    
    # Mean motion (revolutions per day)
    period_sec = 2 * np.pi * np.sqrt(a_km**3 / MU)
    n = 86400.0 / period_sec                     # rev/day

    # Epoch formatting
    # epoch =datetime.strptime(epoch_utc, "%Y-%m-%d %H:%M:%S")
    # year = epoch.year % 100
    # day_frac = epoch_utc#epoch.timetuple().tm_yday + (epoch.hour + epoch.minute/60.0 + epoch.second/3600.0)/24.0

    # Eccentricity must be 7 digits without decimal point (e.g., 0012345 for 0.0012345)
    ecc_str = f"{int(round(ecc * 1_000_0000)):07d}"

    line0 = sat_name
    # line1 = f"1 {sat_num:05d}U 00000A   {year:02d}{day_frac:012.8f}  .00000000  00000-0  00000-0 0  9999"
    line1 = f"1 {sat_num:05d}U 00000A   {tle_epoch}  .00000000  00000-0  00000-0 0  9999"
    line2 = (f"2 {sat_num:05d} {inc_deg:8.4f} {raan_deg:8.4f} {ecc_str} "
             f"{argp_deg:8.4f} {mean_anomaly_deg:8.4f} {n:11.8f}{rev_num:5d}0")

    print("\n=== Generated TLE ===")
    print(line0)
    print(line1)
    print(line2)
    
    return line0, line1, line2

# ====================== Ground Station Functions ======================
def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e2 = F * (2 - F)
    N = R_EARTH / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])

def ecef_to_enu(r_sat, r_gs, lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    dx = r_sat - r_gs
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    return R @ dx

def process_pass(csv_file, lat, lon, alt_km, elev_mask_deg=20):
    df = pd.read_csv(csv_file)
    r_gs = geodetic_to_ecef(lat, lon, alt_km)
    results = []

    for _, row in df.iterrows():
        r_sat = np.array([row["GPS_POSITION_ECEF[0]"],
                          row["GPS_POSITION_ECEF[1]"],
                          row["GPS_POSITION_ECEF[2]"]])

        rho_vec = r_sat - r_gs
        rho = np.linalg.norm(rho_vec)

        enu = ecef_to_enu(r_sat, r_gs, lat, lon)
        east, north, up = enu
        horiz = np.sqrt(east**2 + north**2)
        elev = np.degrees(np.arctan2(up, horiz))
        az = np.degrees(np.arctan2(east, north)) % 360

        visible = (elev >= elev_mask_deg)

        results.append({
            "time": row["time"],
            "range_km": rho,
            "elevation_deg": elev,
            "azimuth_deg": az,
            "visible": visible
        })

    df_out = pd.DataFrame(results)

    # Statistics
    print("\n=== ALL DATA ===")
    print(f"Min range: {df_out['range_km'].min():.2f} km")
    print(f"Max range: {df_out['range_km'].max():.2f} km")

    visible_df = df_out[df_out["visible"]]
    if len(visible_df) > 0:
        print(f"\n=== VISIBLE PASS (> {elev_mask_deg}°) ===")
        print(f"Min range: {visible_df['range_km'].min():.2f} km")
        print(f"Max range: {visible_df['range_km'].max():.2f} km")
        closest = visible_df.loc[visible_df["range_km"].idxmin()]
        print("\nClosest approach:")
        print(closest)

    return df_out

def slant_range(R, H, E_deg):
    E = math.radians(E_deg)
    term = (R / (R + H)) * math.cos(E)
    term = max(-1.0, min(1.0, term))
    angle = E + math.asin(term)
    sr = math.sqrt(R**2 + (R + H)**2 - 2 * R * (R + H) * math.sin(angle))
    return sr

def ecef_to_eci(r_ecef, v_ecef, q_ecef_to_eci):
    """Convert position and velocity from ECEF to ECI using quaternion"""
    R = quat_to_rotmat(q_ecef_to_eci)
    r_eci = R @ r_ecef
    v_eci = R @ v_ecef
    return r_eci, v_eci

def orbital_elements(r, v):
    """Compute classical orbital elements from state vector"""
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    # Angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    # Inclination
    i = np.arccos(h[2] / h_norm)
    # Node vector
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    n_norm = np.linalg.norm(n)
    # Eccentricity vector
    e_vec = (np.cross(v, h) / MU) - (r / r_norm)
    e = np.linalg.norm(e_vec)
    # Semi-major axis
    a = 1 / ((2 / r_norm) - (v_norm**2 / MU))
    # RAAN
    if n_norm != 0:
        raan = np.arccos(n[0] / n_norm)
    if n[1] < 0:
        raan = 2 * np.pi - raan
    else:
        raan = 0
    # Argument of perigee
    if n_norm != 0 and e > 1e-8:
        arg_perigee = np.arccos(np.dot(n, e_vec) / (n_norm * e))
    if e_vec[2] < 0:
        arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0
    # True anomaly
    if e > 1e-8:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
    if np.dot(r, v) < 0:
        nu = 2 * np.pi - nu
    else:
        nu = 0

    # Eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))
    
    # Ensure E is positive
    if E < 0:
        E += 2*np.pi
    
    # Mean anomaly
    M = E - e * np.sin(E)

    # Apogee / Perigee
    rp = a * (1 - e)
    ra = a * (1 + e)
    return {
        "a_km": a,
        "ecc": e,
        "inc_deg": np.degrees(i),
        "raan_deg": np.degrees(raan),
        "argp_deg": np.degrees(arg_perigee),
        "true_anomaly_deg": np.degrees(nu),
        "mean_anomaly_deg": np.degrees(M),
        "perigee_alt_km": rp - R_EARTH,
        "apogee_alt_km": ra - R_EARTH
    }

def iso_to_tle_epoch(iso_str):
    dt = pd.to_datetime(iso_str, utc=True)

    year = dt.year % 100
    day_of_year = dt.dayofyear
    frac_day = (
        dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6
    ) / 86400

    return f"{year:02d}{day_of_year:03d}.{frac_day:.8f}".replace("0.", "")

def process_csv(file_path):
    df = pd.read_csv(file_path)
    results = []
    for _, row in df.iterrows():
        r_ecef = np.array([
            row["GPS_POSITION_ECEF[0]"],
            row["GPS_POSITION_ECEF[1]"],
            row["GPS_POSITION_ECEF[2]"]])
        v_ecef = np.array([
            row["GPS_VELOCITY_ECEF[0]"],
            row["GPS_VELOCITY_ECEF[1]"],
            row["GPS_VELOCITY_ECEF[2]"]])
        q = np.array([
            row["REFS_Q_ECEF_WRT_ECI[0]"],
            row["REFS_Q_ECEF_WRT_ECI[1]"],
            row["REFS_Q_ECEF_WRT_ECI[2]"],
            row["REFS_Q_ECEF_WRT_ECI[3]"]])

    r_eci, v_eci = ecef_to_eci(r_ecef, v_ecef, q)
    elems = orbital_elements(r_eci, v_eci)
    elems["time"] = row["time"]

    elems['epoch_TLE'] = iso_to_tle_epoch(row["time"])  # Store original time for TLE epoch
    results.append(elems)
    return pd.DataFrame(results)

# ========================= MAIN =========================
if __name__ == "__main__":
    file_path = "/home/bkhan/Downloads/Position_Attitude_capella.csv"

    # Process orbital elements
    print("Processing orbital elements...")
    # You can add process_csv here if needed

    # Ground station analysis
    lat = 20.7463667
    lon = -156.4317222
    alt_km = 0.85

    df = process_pass(file_path, lat, lon, alt_km, elev_mask_deg=20)
    df_out = process_csv(file_path)

    # Example: Generate TLE from mean orbit (Replace with your actual averaged values)
    print("\n=== Example TLE Generation ===")
    for _, row in df_out.iterrows():
        keplerian_to_tle(
            a_km=row['a_km'],
            ecc=row['ecc'],
            inc_deg=row['inc_deg'],
            raan_deg=row['raan_deg'],
            argp_deg=row['argp_deg'],
            mean_anomaly_deg=row['mean_anomaly_deg'],
            tle_epoch=row['epoch_TLE'],
            sat_name="CAPELLA-1"
        )

    print("\n=== Orbital Elements (first rows) ===")
    print(df_out.head())
    print("\n=== Mean Orbit (averaged) ===")
    print(df_out.mean(numeric_only=True))
    sat_mean_altitude_km = np.mean((df_out['apogee_alt_km']+df_out['perigee_alt_km'])/2)
    max_range_km = slant_range(R_EARTH, sat_mean_altitude_km, 20)
    min_range_km = slant_range(R_EARTH, sat_mean_altitude_km, 85)
    print(f"Slant range at elevation {20}°: {max_range_km:.2f} km")
    print(f"Slant range at elevation {85}°: {min_range_km:.2f} km")