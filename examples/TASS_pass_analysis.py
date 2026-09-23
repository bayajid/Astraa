# ==================================================
# File: TASS_pass_analysis.py
# Author: Bayajid Khan
# Created: 2026-09-15
# Description: 
# WORKING--> AZ/EL form TMTC matches 
# analytical and geometeric calculation
# PAA range is reasonable
# ==================================================
#%%
import json, csv
import pandas as pd
import folium
import os,sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from astropy import units as u
from astropy.time import Time
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from basic_tools import vector_operations as vec
from pointing_calculations import conversion_pointing as pt_conv
from astropy.coordinates import (SkyCoord, GCRS, ITRS, CartesianRepresentation,FK5,
        EarthLocation,CartesianRepresentation,PrecessedGeocentric,CartesianDifferential)
from scipy.interpolate import CubicHermiteSpline

# File paths
parent_dir = Path(__file__).parent.parent.parent.resolve()
datadir = os.path.join(parent_dir,'analyses','TAS_data')

CSV_FILE =  os.path.join(datadir,"tas_pass_report_analysis_2026_09_02_131953_to_2026_09_02_134251_1788355193000_1788356571000.csv")
JSON_FILE = os.path.join(datadir,"2026-09-02-06.json")

def interp_target_state_on_grid(commands_df, t_grid):
    """
    Build a target state array (N,6) = [x,y,z,vx,vy,vz] on t_grid (telemetry
    gps_seconds), via per-axis cubic Hermite interpolation of commands_df's
    sparse (pos, vel) samples. Uses commanded velocity as the derivative
    constraint -> interpolated velocity is consistent with interpolated
    position, unlike nearest-neighbor hold.
    """
    t_cmd = commands_df["gps_time"].to_numpy()
    order = np.argsort(t_cmd)
    t_cmd = t_cmd[order]

    # in-range mask: never extrapolate target ephemeris
    in_range = (t_grid >= t_cmd[0]) & (t_grid <= t_cmd[-1])
    t_eval = t_grid[in_range]

    cols = ["X_POS", "Y_POS", "Z_POS", "X_VEL", "Y_VEL", "Z_VEL"]
    pos_cols, vel_cols = cols[:3], cols[3:]

    target = np.empty((t_eval.size, 6))
    for i, (pc, vc) in enumerate(zip(pos_cols, vel_cols)):
        p = commands_df[pc].to_numpy()[order]
        v = commands_df[vc].to_numpy()[order]
        spline = CubicHermiteSpline(t_cmd, p, v)
        target[:, i]     = spline(t_eval)     # interpolated position
        target[:, i + 3] = spline(t_eval, 1)  # derivative -> consistent velocity

    return target, in_range
def rotate_vector_by_quaternion(v, q):
    """
    Rotate 3-vector(s) by Hamilton quaternion(s).
    
    Parameters
    ----------
    v : array-like, shape (3,) or (N, 3)
        Vector(s) to rotate
    q : array-like, shape (4,) or (N, 4)
        Quaternion(s) in [q0, q1, q2, q3] (scalar first)
        
    Returns
    -------
    rotated : ndarray, same shape as v
    """
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)

    # Normalize quaternion(s)
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(q_norm == 0):
        raise ValueError("Quaternion must be non-zero.")
    q = q / q_norm

    # Handle single vs batch
    single_v = v.ndim == 1
    single_q = q.ndim == 1

    if single_v:
        v = v[None, :]
    if single_q:
        q = q[None, :]

    # Broadcast if one is single and the other is batched
    if q.shape[0] == 1 and v.shape[0] > 1:
        q = np.repeat(q, v.shape[0], axis=0)
    elif v.shape[0] == 1 and q.shape[0] > 1:
        v = np.repeat(v, q.shape[0], axis=0)
    
    w = q[:, 0:1]
    qv = q[:, 1:4]

    # ECI -> Body = q^{-1} * v * q
    # Compared with the active q*v*q^{-1} rotation,
    # the cross-product term changes sign.
    rotated = (
        (2.0 * w**2 - 1.0) * v
        - 2.0 * w * np.cross(qv, v)
        + 2.0 * qv * np.sum(qv * v, axis=1, keepdims=True)
    )

    if single_v and single_q:
        return rotated[0]
    
    return rotated

def ecef_to_eci(x, y, z, vx, vy, vz,timestamps):
    """
    Transform position from ECEF (ITRS) to ECI (GCRS ≈ J2000).

    Parameters
    ----------
    x, y, z : float or array-like
        ECEF coordinates [meters]
    time_utc : datetime.datetime or astropy.time.Time
        UTC time of the observation

    Returns
    -------
    x_eci, y_eci, z_eci : float or ndarray
        ECI coordinates [meters]
    """
    
    
    t = Time(timestamps, format="gps")
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vz = np.asarray(vz)

    print("=== INSIDE FUNCTION ===")
    print("x[0], y[0], z[0] =", x[0], y[0], z[0])
    print("Norm inside =", np.linalg.norm([x[0], y[0], z[0]]))

    # Position + velocity in ITRS (ECEF)
    position = CartesianRepresentation(x * u.m, y * u.m, z * u.m)
    velocity = CartesianDifferential(vx * u.m/u.s, vy * u.m/u.s, vz * u.m/u.s)

    itrs = ITRS(position.with_differentials(velocity), obstime=t)

    # Transform to GCRS (ECI)
    gcrs = itrs.transform_to(GCRS(obstime=t))

    pos = gcrs.cartesian
    vel = pos.differentials["s"]

    print("=== DEBUG ===")
    print("Type of x:", type(x))
    print("Shape of x:", np.shape(x))
    print("First 3 values of x, y, z:")
    print(x[:3] if hasattr(x, '__len__') else x)
    print(y[:3] if hasattr(y, '__len__') else y)
    print(z[:3] if hasattr(z, '__len__') else z)


    print("\nFirst timestamp:", timestamps.iloc[0])

    print("Type of timestamps:", type(timestamps))
    print("format used: gps")

    return (
        pos.x.to_value(u.m),
        pos.y.to_value(u.m),
        pos.z.to_value(u.m),
        vel.d_x.to_value(u.m/u.s),
        vel.d_y.to_value(u.m/u.s),
        vel.d_z.to_value(u.m/u.s),
    )

def eci_to_latlon_gps_seconds(x_eci, y_eci, z_eci, gps_seconds, unit=u.km):
    """
    Convert ECI (J2000) → lat, lon, alt
    using GPS seconds (seconds since 1980-01-06)
    """
    # Astropy Time from GPS seconds
    t = Time(gps_seconds, format='gps')
    x_eci = np.asarray(x_eci)
    y_eci = np.asarray(y_eci)
    z_eci = np.asarray(z_eci)
    gps_seconds = np.asarray(gps_seconds)

    eci = PrecessedGeocentric(
        CartesianRepresentation(
            x_eci * unit,
            y_eci * unit,
            z_eci * unit,
        ),
        equinox=Time("J2000"),
        obstime=t,
    )

    # ECI -> Earth-fixed ITRS.
    itrs = eci.transform_to(ITRS(obstime=t))

    # Convert geocentric ITRS coordinates to geodetic coordinates.
    loc = itrs.earth_location

    lat = loc.lat.to_value(u.deg)
    lon = loc.lon.to_value(u.deg)
    alt = loc.height.to_value(unit)

    return lat, lon, alt

def los2azel(los, degrees=False):
    """
    Convert a line-of-sight (LOS) vector in Cartesian coordinates to
    spherical azimuth/elevation, with elevation measured positive upward
    from the local horizontal plane (i.e., toward zenith).

    Assumes a local topocentric East-North-Up (ENU) frame:
        x = East, y = North, z = Up
    Azimuth is measured clockwise from North (standard compass/tracking
    convention: North = 0, East = +90 deg).

    Parameters
    ----------
    los : array_like, shape (3,) or (N, 3)
        LOS vector(s) [x, y, z] in a local ENU frame. Units are arbitrary
        (only direction matters) -- does not need to be normalized.
    degrees : bool, optional
        If True, return az/el in degrees. Default False (radians).

    Returns
    -------
    az : float or ndarray, shape (N,)
        Azimuth, clockwise from North, range (-pi, pi] / (-180, 180].
    el : float or ndarray, shape (N,)
        Elevation above local horizontal, range [-pi/2, pi/2] / [-90, 90].
        Positive = above horizon (up), negative = below horizon.
    """
    los = np.asarray(los, dtype=float)
    single = (los.ndim == 1)
    if single:
        los = los[np.newaxis, :]

    x, y, z = los[:, 0], los[:, 1], los[:, 2]
    r = np.linalg.norm(los, axis=1)

    az = np.arctan2(y,x)          # clockwise from North (y), toward East (x)
    el = np.arcsin(z / r)          # positive = up

    if degrees:
        az = np.degrees(az)
        el = np.degrees(el)

    if single:
        return np.array([az[0], el[0]])
    return np.column_stack((az, el))

def quaternion_to_dcm(qc, qx, qy, qz):
    """
    Convert quaternion to Direction Cosine Matrix (DCM).
    Quaternion convention: q = [qc, qx, qy, qz] (scalar-first)
    DCM transforms from ECI -> Body frame
    """
    # Normalize quaternion (ensure unit quaternion)
    q_norm = np.sqrt(qc**2 + qx**2 + qy**2 + qz**2)
    qc = qc / q_norm
    qx = qx / q_norm
    qy = qy / q_norm
    qz = qz / q_norm
    
    # DCM (Body <- ECI) for scalar-first quaternion
    # This rotates a vector from ECI to Body frame
    dcm = np.array([
        [qc**2 + qx**2 - qy**2 - qz**2,  2*(qx*qy + qc*qz),        2*(qx*qz - qc*qy)],
        [2*(qx*qy - qc*qz),              qc**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz + qc*qx)],
        [2*(qx*qz + qc*qy),              2*(qy*qz - qc*qx),        qc**2 - qx**2 - qy**2 + qz**2]
    ])
    return dcm

def calc_paa(host, target, attitude,
             mounting_offset,official_convention = 1):
    
    c = 299792458*1e-3  # km/s
    r_t = target[:,[0,1,2]]
    r_h = host[:,[0,1,2]]

    v_t = target[:,[3,4,5]]
    v_h = host[:,[3,4,5]]
    # LOS in ECI
    los = r_t - r_h
    slant = np.linalg.norm(los, axis = 1)
    # tangential host velocity wrt LOS
    dt_paa = slant / c
    dt_paa = dt_paa.reshape((attitude.shape[0], 1))

    v_rel = v_t - v_h
    v_rel_tangential = vec.get_tangential_comp(v_rel, los) 

    ### target pos at tx and rx
    # target offset by relative tangential velocity   
    
    los_tx_rel = r_t + v_rel_tangential * dt_paa - r_h
    los_rx_rel = r_t - v_rel_tangential * dt_paa - r_h
    
    
    # rotate using quaternions with unit-tested method
    los_rx_bf_tan = rotate_vector_by_quaternion(los_rx_rel, attitude)
    los_tx_bf_tan = rotate_vector_by_quaternion(los_tx_rel, attitude)
    los_bf = rotate_vector_by_quaternion(los, attitude)

    
    # rotation from bf to lct (assuming quat is given from ECI to LCT, so nothing done here)
    los_rx_lct_tan = rotate_vector_by_quaternion(los_rx_bf_tan, mounting_offset)
    los_tx_lct_tan = rotate_vector_by_quaternion(los_tx_bf_tan, mounting_offset)
    los_lct =  rotate_vector_by_quaternion(los_bf, mounting_offset)
    
    ae_rx_lct_tan   = los2azel(los_rx_lct_tan )
    ae_tx_lct_tan   = los2azel(los_tx_lct_tan)
    ae_lct          = los2azel(los_lct)

    ## Robust computation of 2-way Point ahead angles [urad]
    PAA_full_vrel = (ae_tx_lct_tan - ae_rx_lct_tan) # LCT-frame

    pt_angles = np.hstack((los_lct, ae_lct, slant.reshape((slant.shape[0], 1)), PAA_full_vrel))

    return pt_angles

def point_ahead_angle(host,target ):
    """
    Returns the Point Ahead Angle vector (radians) in the same frame.
    """

    c = 299792458*1e-3 # km/s
    r_t = target[:,[0,1,2]]
    r_h = host[:,[0,1,2]]

    v_t = target[:,[3,4,5]]
    v_h = host[:,[3,4,5]]

    R =  r_t - r_h 
    R_norm = np.linalg.norm(R, axis=1, keepdims=True)
    R_hat = R / R_norm

    v_rel = v_t - v_h 
    v_perp = np.cross(R_hat, np.cross(v_rel, R_hat))   # or v_rel - (v_rel·R_hat)*R_hat

    theta_paa = (2.0 / c) * v_perp
    return theta_paa

def analytical_paa_lct(host, target, attitude, mounting_offset,
                       official_convention=1):

    paa_analytical = point_ahead_angle(host, target)

    # ECI -> body frame
    paa_analytical_bf = rotate_vector_by_quaternion(paa_analytical,attitude)
    # body -> LCT frame
    paa_analytical_lct = rotate_vector_by_quaternion(paa_analytical_bf,mounting_offset)

    # LOS
    r_t = target[:, [0, 1, 2]]
    r_h = host[:, [0, 1, 2]]
    los = r_t - r_h

    los_bf = rotate_vector_by_quaternion(los, attitude)
    los_lct = rotate_vector_by_quaternion(los_bf, mounting_offset)

    # Nominal az/el
    # ae_lct = pt_conv.conv_los2ae(los_lct,official_convention)
    ae_lct = los2azel(los_lct)

    az = ae_lct[:, 0]
    el = ae_lct[:, 1]

    # Local azimuth direction
    e_az = np.column_stack((-np.sin(az), np.cos(az), np.zeros_like(az)))

    # Local elevation direction
    e_el = np.column_stack((
        -np.sin(el) * np.cos(az),
        -np.sin(el) * np.sin(az),
         np.cos(el)
    ))

    # Convert angular vector to delta az/el
    delta_az = np.sum(paa_analytical_lct * e_az,axis=1) / np.cos(el)
    delta_el = np.sum(paa_analytical_lct * e_el,axis=1)

    # rad -> urad
    delta_az *= 1e6
    delta_el *= 1e6

    paa_analytical_az_el = np.column_stack((delta_az,delta_el))

    return paa_analytical_az_el, az ,el



#%%
TELEMETRY_COLUMNS = [
    "timestamp",

    'CPA Pointing :: OH1_AZIMUTH',
    'CPA Pointing :: OH1_ELEVATION',

    "Fast steering mirrors evolution :: OH1_PAA_TIP",
    "Fast steering mirrors evolution :: OH1_PAA_TILT",

    "Point Ahead Assembly movement :: OH1_PAA_TIP",
    "Point Ahead Assembly movement :: OH1_PAA_TILT",

    "Quaternions ECI to body :: bus_quat_eci_to_body_scalar",
    "Quaternions ECI to body :: bus_quat_eci_to_body_x",
    "Quaternions ECI to body :: bus_quat_eci_to_body_y",
    "Quaternions ECI to body :: bus_quat_eci_to_body_z",

    "ECEF position :: bus_pos_ecef_m_x",
    "ECEF position :: bus_pos_ecef_m_y",
    "ECEF position :: bus_pos_ecef_m_z",

    "GPS time :: bus_gps_time_weeks",
    "GPS time :: bus_gps_time_time_of_week_ms",
    "Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p0",
    "Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p1",
    "Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p0",
    "Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p1",

    'Velocities :: bus_vel_ecef_m_per_s_x',
    'Velocities :: bus_vel_ecef_m_per_s_y',
    'Velocities :: bus_vel_ecef_m_per_s_z',
]

COMMAND_COLUMNS = [
    "command_time",
    "gps_time",
    "X_POS",
    "Y_POS",
    "Z_POS",
    "X_VEL",
    "Y_VEL",
    "Z_VEL",
    "TIME_SEC",
    "TIME_MSEC",
]

# 1. Read telemetry CSV
telemetry_full_df = pd.read_csv(CSV_FILE)

missing_telemetry = [
    col for col in TELEMETRY_COLUMNS
    if col not in telemetry_full_df.columns]

if missing_telemetry:
    raise ValueError("Missing telemetry columns:\n"+ "\n".join(missing_telemetry))

telemetry_df = telemetry_full_df[TELEMETRY_COLUMNS].copy()
telemetry_df["timestamp"] = pd.to_datetime(telemetry_df["timestamp"],utc=True,errors="coerce")
telemetry_df  = telemetry_df[telemetry_df["timestamp"].notna()] 
telemetry_df = telemetry_df.drop_duplicates(subset=["timestamp"], keep="first")
telemetry_df = telemetry_df.reset_index(drop=True)

# 2. Read command JSON
with open(JSON_FILE, "r", encoding="utf-8") as f:
    command_data = json.load(f)

command_rows = []
for command in command_data.get("commands", []):
    mnemonic = command.get("cmdMnemonic", {})
    values = mnemonic.get("values", {})

    row = {
        "command_uid": command.get("uid"),
        "command_time": pd.to_datetime(
            command.get("utcTime"),
            utc=True,
            errors="coerce"
        ),
        "gps_time": command.get("gpsTime"),
        "delay": command.get("delay"),
        "status": command.get("status"),
        "target_id": command.get("targetId"),

        "mnemonic": mnemonic.get("mnemonic"),
        "st": mnemonic.get("st"),
        "sst": mnemonic.get("sst"),

        "MOD_ID": values.get("MOD_ID"),
        "OH_ID": values.get("OH_ID"),
        "FORMAT": values.get("FORMAT"),

        "X_POS": values.get("X_POS"),
        "Y_POS": values.get("Y_POS"),
        "Z_POS": values.get("Z_POS"),

        "X_VEL": values.get("X_VEL"),
        "Y_VEL": values.get("Y_VEL"),
        "Z_VEL": values.get("Z_VEL"),

        "TIME_SEC": values.get("TIME_SEC"),
        "TIME_MSEC": values.get("TIME_MSEC"),
    }

    command_rows.append(row)

commands_df = pd.DataFrame(command_rows)
commands_df["command_time"] = pd.to_datetime(commands_df["command_time"],utc=True,errors="coerce")

numeric_columns = [
    "gps_time",
    "delay",
    "st",
    "sst",
    "X_POS",
    "Y_POS",
    "Z_POS",
    "X_VEL",
    "Y_VEL",
    "Z_VEL",
    "TIME_SEC",
    "TIME_MSEC",
]

for column in numeric_columns:
    commands_df[column] = pd.to_numeric(commands_df[column],errors="coerce")

telemetry_df = telemetry_df.sort_values("timestamp").reset_index(drop=True)
commands_df = commands_df.sort_values("command_time").reset_index(drop=True)
telemetry_df["gps_seconds"] = (
    telemetry_df["GPS time :: bus_gps_time_weeks"] * 7 * 86400
    + telemetry_df["GPS time :: bus_gps_time_time_of_week_ms"] / 1000.0)

# ============================================================
# Display DataFrames
# scaling telemetry with command df
telemetry_df = telemetry_df.iloc[1085:19081]
# print("===== TELEMETRY DATAFRAME =====")
commands_df = commands_df.iloc[:-1]

Elevation =telemetry_df['CPA Pointing :: OH1_ELEVATION'].dropna().to_numpy()

x_eci,y_eci,z_eci,vx_eci,vy_eci,vz_eci = ecef_to_eci(
                                        telemetry_df["ECEF position :: bus_pos_ecef_m_x"].dropna(),
                                        telemetry_df["ECEF position :: bus_pos_ecef_m_y"].dropna(),
                                        telemetry_df["ECEF position :: bus_pos_ecef_m_z"].dropna(),
                                        telemetry_df["Velocities :: bus_vel_ecef_m_per_s_x"].dropna(),
                                        telemetry_df["Velocities :: bus_vel_ecef_m_per_s_y"].dropna(),
                                        telemetry_df["Velocities :: bus_vel_ecef_m_per_s_z"].dropna(),
                                        telemetry_df["gps_seconds"].dropna()
                                        )
# Convert the ECI values to Km
x_eci *= 1e-3
y_eci *= 1e-3
z_eci *= 1e-3
vx_eci*= 1e-3
vy_eci*= 1e-3
vz_eci*= 1e-3

lat, long, alt = eci_to_latlon_gps_seconds(
                    commands_df["X_POS"], commands_df["Y_POS"],
                    commands_df["Z_POS"], commands_df['gps_time'], unit=u.km)
commands_df['lat'] = lat
commands_df['lon'] = long
commands_df['alt'] = alt

#%% # ============================================================
#     PAA Calculation
attitude = np.column_stack((telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_scalar"].dropna(),
            telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_x"].dropna(),
            telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_y"].dropna(),
            telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_z"].dropna()))

mounting_offset = np.array([-0.00138716, 0.70605231, -0.00164009, 0.70815643])
host            = np.column_stack((x_eci, y_eci, z_eci,vx_eci, vy_eci, vz_eci))
target          = np.column_stack((commands_df["X_POS"],commands_df["Y_POS"],commands_df["Z_POS"],
                                commands_df["X_VEL"],commands_df["Y_VEL"],commands_df["Z_VEL"]))

mounting_offset = np.array([-0.00138716, 0.70605231, -0.00164009, 0.70815643])

pt_angles = calc_paa(host, target, attitude, mounting_offset,official_convention = 1)
paa_analytical = 1e6*point_ahead_angle(host,target)
paa_geometric =  1e6* pt_angles[:, 6:] # µrads
print(f"\nPAA: {paa_geometric} µrad")
print(f"\nPAA Analytical 3D: {paa_analytical}µrad")

paa_analytical_az_el, az_lct, el_lct = analytical_paa_lct(host,target,
                    attitude,mounting_offset,official_convention=1)

# ============================================================
##------------------Save data to csv--------------------------
header = "Hx[km], Hy[km], Hz[km], Hvx[km/s], Hvy[km/s], Hvz[km/s], Tx[km], Ty[km], Tz[km], Tvx[km/s], Tvy[km/s], Tvz[km/s], qc, q1, q2, q3, paa_tip[µrad], paa_tilt[µrad], az_deg, el_deg, t_gps"
data = np.hstack([host, target, attitude, paa_geometric, np.rad2deg(az_lct).reshape(-1,1), np.rad2deg(el_lct).reshape(-1,1), telemetry_df["gps_seconds"].dropna().to_numpy().reshape(-1,1)])
np.savetxt(os.path.join(datadir, 'output.csv'), data, header=header,delimiter=",", comments ="")

print("Analytical PAA [dAz, dEl] µrad:")
print(paa_analytical_az_el[:3])

print("Geometric PAA [dAz, dEl] µrad:")
print(paa_geometric)

# %%# ============================================================
#  -----------PLOTS-----------------------------
fig,ax = plt.subplots(2,1, sharex=True)
plt.suptitle('PAA calculated')
ax[0].scatter(telemetry_df["gps_seconds"].dropna(),paa_analytical_az_el[:,0], marker = '.',s= 5,label = 'paa_analytical_tip')
ax[0].scatter(telemetry_df["gps_seconds"].dropna(),1e6 * pt_angles[:, 6],     marker = '+',s= 20,alpha = 0.05,label = 'paa_geometric_tip')
ax[1].scatter(telemetry_df["gps_seconds"].dropna(),paa_analytical_az_el[:,1], marker = '.',s= 5,label = 'paa_analytical_tilt')
ax[1].scatter(telemetry_df["gps_seconds"].dropna(),1e6 * pt_angles[:, 7],     marker = '+',s= 20,alpha = 0.05,label = 'paa_geometric_tilt')

for i in range(2):
    ax[i].legend()
    ax[i].grid()
    ax[i].set_ylabel('µrad')

fig,ax = plt.subplots(3,1, sharex=True)
plt.suptitle('TM data')
ax[0].plot(telemetry_df["gps_seconds"].dropna(),1e-3*telemetry_df["ECEF position :: bus_pos_ecef_m_x"].dropna(), label = 'bus_pos_ecef_km_x')
ax[1].plot(telemetry_df["gps_seconds"].dropna(),1e-3*telemetry_df["ECEF position :: bus_pos_ecef_m_y"].dropna(), label = 'bus_pos_ecef_km_y')
ax[2].plot(telemetry_df["gps_seconds"].dropna(),1e-3*telemetry_df["ECEF position :: bus_pos_ecef_m_z"].dropna(), label = 'bus_pos_ecef_km_z')

for i in range (3):
    ax[i].legend()
    ax[i].grid()
fig,ax = plt.subplots(3,1, sharex=True)
ax[0].plot(telemetry_df["gps_seconds"].dropna(),x_eci, label = 'bus_pos_eci_km_x')
ax[1].plot(telemetry_df["gps_seconds"].dropna(),y_eci, label = 'bus_pos_eci_km_y')
ax[2].plot(telemetry_df["gps_seconds"].dropna(),z_eci, label = 'bus_pos_eci_km_z')
for i in range (3):
    ax[i].legend()
    ax[i].grid()

fig,ax = plt.subplots(3,1, sharex=True)
plt.suptitle('TM data')
ax[0].plot(telemetry_df["gps_seconds"].dropna(),commands_df["X_POS"].dropna(),label = 'target_X_POS')
ax[1].plot(telemetry_df["gps_seconds"].dropna(),commands_df["Y_POS"].dropna(),label = 'target_Y_POS')
ax[2].plot(telemetry_df["gps_seconds"].dropna(),commands_df["Z_POS"].dropna(),label = 'target_Z_POS')
for i in range (3):
    ax[i].legend()
    ax[i].grid()

fig,ax = plt.subplots(2,1) 
plt.suptitle('TM data')
ax[0].plot(telemetry_df["gps_seconds"].dropna(), label = 'time_csv')
ax[1].plot(commands_df['gps_time'].dropna(), label = 'time_json')
for i in range (2):
    ax[i].legend()
    ax[i].grid()

fig,ax=  plt.subplots(4,1, sharex=True)
plt.suptitle('TM data')
ax[0].plot(telemetry_df["gps_seconds"].dropna(),telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_scalar"].dropna(), label = 'q_c')
ax[1].plot(telemetry_df["gps_seconds"].dropna(),telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_x"].dropna(), label = 'q_x')
ax[2].plot(telemetry_df["gps_seconds"].dropna(),telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_y"].dropna(), label = 'q_z')
ax[3].plot(telemetry_df["gps_seconds"].dropna(),telemetry_df["Quaternions ECI to body :: bus_quat_eci_to_body_z"].dropna(), label = 'q_z')
for i in range (4):
    ax[i].legend()
    ax[i].grid()

fig,ax=  plt.subplots(3,1, sharex=True)
plt.suptitle('TM data')
ax[0].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p0"].dropna(), label = 'PRED_AZ')
ax[1].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p1"].dropna(), label = 'PRED_EL')
ax[0].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p0"].dropna() , label = 'ACTUAL_AZ')
ax[1].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p1"].dropna() , label = 'ACTUAL_EL')
ax[2].plot(telemetry_df['Point Ahead Assembly movement :: OH1_PAA_TIP'].dropna() , label = 'paa_tip')
ax[2].plot(telemetry_df['Point Ahead Assembly movement :: OH1_PAA_TILT'].dropna(), label = 'paa_tilt')
for i in range (3):
    ax[i].legend()
    ax[i].grid()

fig,ax=  plt.subplots(4,1, sharex=False)
plt.suptitle('Calculated AZ/EL(missing PMG) vs TM data')
ax[2].scatter(telemetry_df["gps_seconds"].dropna(),np.rad2deg(pt_angles[:,4]), marker= 'o', alpha = 0.05,label = 'el_geometric')
ax[2].scatter(telemetry_df["gps_seconds"].dropna(),np.rad2deg(el_lct),         marker= '.',s = 10,label = 'el_analytical')
ax[0].scatter(telemetry_df["gps_seconds"].dropna(),np.rad2deg(pt_angles[:,3]), marker= 'o', alpha = 0.05,label = 'az_geometric')
ax[0].scatter(telemetry_df["gps_seconds"].dropna(),np.rad2deg(az_lct),         marker= '.',s = 10,label = 'az_analytical')
ax[1].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p0"].dropna(), label = 'PRED_AZ_TM')
ax[3].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_PRED_AZ_EL_p1"].dropna(), label = 'PRED_EL_TM')
# ax[0].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p0"].dropna() , label = 'ACTUAL_AZ')
# ax[1].plot(telemetry_df["Line of Sight predicted VS actual :: OH1_LOS_AZ_EL_p1"].dropna() , label = 'ACTUAL_EL')
for i in range (4):
    ax[i].legend()
    ax[i].grid()

# %%

paa_corrected = pt_angles[:, 6:].copy()
#paa_corrected[:, 0] /= np.cos(np.deg2rad(Elevation))

print(f"\nPAA_Corrected_tip:  {paa_corrected[:, 0]}")  # corrected Δaz
print(f"\nPAA_Corrected_tilt: {paa_corrected[:, 1]}")  # Δel



#%%  ##-----------------------OGS PLOT------------------------------------
# Remove invalid points
track = commands_df[
    np.isfinite(commands_df["lat"]) &
    np.isfinite(commands_df["lon"])
].copy()

# Center map on the middle of the track
center_lat = track["lat"].mean()
center_lon = track["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=3,
    tiles="OpenStreetMap",
)

# Ground track
coordinates = track[["lat", "lon"]].values.tolist()

folium.PolyLine(
    coordinates,
    color="red",
    weight=3,
    opacity=0.8,
).add_to(m)

# Start point
folium.Marker(
    [track.iloc[0]["lat"], track.iloc[0]["lon"]],
    popup=(
        f"START<br>"
        f"Lat: {track.iloc[0]['lat']:.6f}<br>"
        f"Lon: {track.iloc[0]['lon']:.6f}<br>"
        f"Alt: {track.iloc[0]['alt']:.2f} km"
    ),
    icon=folium.Icon(color="green", icon="play"),
).add_to(m)

# End point
folium.Marker(
    [track.iloc[-1]["lat"], track.iloc[-1]["lon"]],
    popup=(
        f"END<br>"
        f"Lat: {track.iloc[-1]['lat']:.6f}<br>"
        f"Lon: {track.iloc[-1]['lon']:.6f}<br>"
        f"Alt: {track.iloc[-1]['alt']:.2f} km"
    ),
    icon=folium.Icon(color="red", icon="stop"),
).add_to(m)
m




#%%
















#%% -----EXTRA Suff------------------------------------------------------
x = commands_df["gps_time"].dropna().to_numpy()
y = telemetry_df["gps_seconds"].dropna().to_numpy()
# Sort telemetry values for fast nearest-neighbor lookup
y_sorted = np.sort(y)

# Find closest telemetry value for each gps_time
idx = np.searchsorted(y_sorted, x)
idx = np.clip(idx, 1, len(y_sorted) - 1)

left = y_sorted[idx - 1]
right = y_sorted[idx]

nearest = np.where(
    np.abs(x - left) <= np.abs(x - right),
    left,
    right
)

difference = np.abs(x - nearest)

# First match within your tolerance
tolerance = 1.0  # seconds

matches = np.where(difference <= tolerance)[0]

if len(matches):
    i = matches[0]
    print("gps_time:      ", x[i])
    print("gps_seconds:   ", nearest[i])
    print("difference:    ", difference[i])
else:
    print("No match within", tolerance, "seconds")
    print("Closest difference:", difference.min())
plt.figure(figsize=(10, 10))

# All values from each dataset
plt.scatter(x, x, s=1, alpha=1 , label="gps_time", color='C0')
plt.scatter(y, y, s=20, alpha=0.05, label="gps_seconds", color='C1')

# Perfect common-value line
lims = [min(x.min(), y.min()), max(x.max(), y.max())]
#lt.plot(lims, lims, "k--", alpha=0.5)

plt.xlabel("Time")
plt.ylabel("Time")
plt.legend()
plt.grid()

# %%
