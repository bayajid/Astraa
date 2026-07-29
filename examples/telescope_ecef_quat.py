import numpy as np
from scipy.spatial.transform import Rotation as R
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import matplotlib.pyplot as plt

def utc2gps(utc_dt):
    """
    Input: UTC datetime
    """
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

    # Use current known leap seconds (update manually when new leap seconds announced)
    leap_seconds = 18

    gps_seconds = (utc_dt - gps_epoch).total_seconds() + leap_seconds

    return gps_seconds

def normalize(v):
    return v / np.linalg.norm(v)

def site_ecef(lat, lon, alt):
    # WGS-84
    a = 6378137.0
    e2 = 6.69437999014e-3

    sl = np.sin(lat)
    cl = np.cos(lat)

    N = a / np.sqrt(1 - e2 * sl**2)

    x = (N + alt) * cl * np.cos(lon)
    y = (N + alt) * cl * np.sin(lon)
    z = (N * (1 - e2) + alt) * sl

    return np.array([x, y, z])

def ecef_to_nwu_matrix(lat, lon):
    sl, cl = np.sin(lat), np.cos(lat)
    sb, cb = np.sin(lon), np.cos(lon)

    e = np.array([-sb, cb, 0])
    n = np.array([-sl*cb, -sl*sb, cl])
    u = np.array([cl*cb, cl*sb, sl])

    # NWU
    return np.vstack((n, -e, u))

def telescope_quaternion_nwu(lat, lon, az, el):
    # ECEF → NWU
    R_ecef_nwu = ecef_to_nwu_matrix(lat, lon)

    # Pointing direction in NWU
    d = np.array([
        np.cos(el) * np.cos(az),   # North
        -np.cos(el) * np.sin(az),  # West
        np.sin(el)                 # Up
    ])
    z = normalize(d)

    # Telescope frame axes
    up = np.array([0, 0, 1])
    x = normalize(np.cross(z, up))
    y = np.cross(z, x)

    R_nwu_tel = np.vstack((x, y, z))

    # Full rotation
    R_ecef_tel = R_nwu_tel @ R_ecef_nwu

    # Quaternion scalar-first
    q = R.from_matrix(R_ecef_tel).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def sat_ecef_from_tle(sat, t):
    jd, fr = jday(
        t.year, t.month, t.day,
        t.hour, t.minute, t.second + t.microsecond*1e-6
    )
    e, r_teme, v_teme = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError("SGP4 propagation error")

    r_teme = np.array(r_teme)  # km
    # -----------------------------
    # Convert TEME to ECEF using GMST
    # -----------------------------
    T = (jd + fr - 2451545.0) / 36525.0  # Julian centuries since J2000
    # IAU 2000 simplified GMST in seconds
    gmst_sec = 67310.54841 + (876600*3600 + 8640184.812866)*T + 0.093104*T**2 - 6.2e-6*T**3
    gmst_sec = gmst_sec % 86400.0  # wrap to 0–86400 sec
    gmst_rad = 2*np.pi * (gmst_sec / 86400.0)  # convert to radians

    c, s = np.cos(gmst_rad), np.sin(gmst_rad)
    R_eci_ecef = np.array([
        [ c,  s, 0],
        [-s,  c, 0],
        [ 0,  0, 1]
    ])

    r_ecef = R_eci_ecef @ r_teme  # km → still km
    return r_ecef * 1000.0        # convert to meters


def az_el_from_ecef(site_ecef, sat_ecef, lat, lon):
    R_ecef_nwu = ecef_to_nwu_matrix(lat, lon)

    rho = sat_ecef - site_ecef
    rho_nwu = R_ecef_nwu @ rho
    rho_nwu = normalize(rho_nwu)

    north, west, up = rho_nwu

    az = np.arctan2(-west, north) % (2*np.pi)
    el = np.arcsin(up)

    return az, el



def angular_velocity(q1, q2, dt):
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    r = r2 * r1.inv()
    rotvec = r.as_rotvec()
    return rotvec / dt

def quaternion_rate(q, omega):
    """
    q      : quaternion [w, x, y, z]
    omega  : angular velocity [wx, wy, wz] in body frame (rad/s)
    """
    w, x, y, z = q
    wx, wy, wz = omega

    qdot = 0.5 * np.array([
        -x*wx - y*wy - z*wz,
         w*wx - z*wy + y*wz,
         z*wx + w*wy - x*wz,
        -y*wx + x*wy + w*wz
    ])
    return qdot

def track_satellite(
    tle1, tle2,
    lat, lon, alt,
    start_time,
    duration_sec=300,
    dt=0.5
):
    sat = Satrec.twoline2rv(tle1, tle2)
    site = site_ecef(lat, lon, alt)

    t = start_time
    data = []

    while (t - start_time).total_seconds() <= duration_sec:
        sat_ecef = sat_ecef_from_tle(sat, t)
        az, el = az_el_from_ecef(site, sat_ecef, lat, lon)
        q = telescope_quaternion_nwu(lat, lon, az, el)

        data.append((t, az, el, q))
        t += timedelta(seconds=dt)

    return data

# ===============================================================
# Example
if __name__ == "__main__":

    
    outputdir = os.path.join(os.getcwd(),'output_data')

    # Example TLE for ISS (ZARYA)
    tle_line1 = "1 25544U 98067A   20344.54791667  .00016717  00000-0  10270-3 0  9003"
    tle_line2 = "2 25544  51.6442 348.7415 0002187  85.0996 325.0603 15.49315339257116"

    # Observer location: Example - Mynaric HQ Neuaubing, Munich,DE    
    lat = np.deg2rad(48.137017)    # radians
    lon = np.deg2rad(11.419067)  # radians
    alt = 567.5               # meters

    # Tracking parameters
    start_time = datetime(2025, 12, 16, 12, 0, 0, tzinfo=timezone.utc)
    duration_sec = 600          # seconds
    dt = 1.0                    # seconds

    tracking_data = track_satellite(
        tle_line1, tle_line2,
        lat, lon, alt,
        start_time,
        duration_sec,
        dt
    )
    rows = []

    for i, (t, az, el, q) in enumerate(tracking_data):
        gps_time = utc2gps(t)

        # Angular velocity
        if i == 0:
            omega = np.zeros(3)
            qdot = np.zeros(4)
        else:
            q_prev = tracking_data[i-1][3]
            dt = (t - tracking_data[i-1][0]).total_seconds()
            omega = angular_velocity(q_prev, q, dt)
            qdot = quaternion_rate(q, omega)

        rows.append({
            "gps_time": gps_time,
            "az_rad": az,
            "el_rad": el,
            "q0": q[0],
            "q1": q[1],
            "q2": q[2],
            "q3": q[3],
            "wx_rad_s": omega[0],
            "wy_rad_s": omega[1],
            "wz_rad_s": omega[2],
            "qdot0": qdot[0],
            "qdot1": qdot[1],
            "qdot2": qdot[2],
            "qdot3": qdot[3],
        })
        df = pd.DataFrame(rows)

    csv_filename = os.path.join(outputdir, "telescope_tracking.csv")
    df.to_csv(csv_filename, index=False, float_format="%.9f")

    print(f"Saved tracking data to {csv_filename}")

    
    plt.figure(figsize=(10,6))
    plt.plot(df['gps_time'], df['el_rad']*180/np.pi, label="Elevation (deg)")
    plt.plot(df['gps_time'], df['az_rad']*180/np.pi, label="Azimuth (deg)")
    plt.xlabel("GPS Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Telescope Tracking Angles")
    plt.legend()
    plt.grid(True)
    plt.show()

#-----------------------------------------------------------------------------------
    
#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz
import astropy.units as u
from scipy.spatial.transform import Rotation as R

# -------------------------
# Observer location
# -------------------------
lat_deg = 35.247
lon_deg = -116.793
alt_m = 1000.0
location = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=alt_m*u.m)

# -------------------------
# Tracking parameters
# -------------------------
start_time = datetime.utcnow().replace(tzinfo=timezone.utc)
duration_sec = 600   # 10 minutes
dt_sec = 1           # seconds
times = [start_time + timedelta(seconds=i) for i in range(0, duration_sec, int(dt_sec))]

# -------------------------
# GPS time helper
# -------------------------
GPS_UTC_OFFSET = 18.0  # seconds

def utc2gps(t_utc):
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    return (t_utc - gps_epoch).total_seconds() + GPS_UTC_OFFSET

# -------------------------
# NWU telescope quaternion
# -------------------------
def normalize(v):
    return v / np.linalg.norm(v)

def telescope_quaternion_nwu(lat, lon, az, el):
    # Compute telescope pointing vector in NWU
    z = np.array([np.cos(el)*np.cos(az), -np.cos(el)*np.sin(az), np.sin(el)])
    z = normalize(z)

    # NWU-consistent frame (x=East-like, y=North-like, z=boresight)
    up = np.array([0, 0, 1])
    x = np.cross(up, z)
    x = normalize(x)
    y = np.cross(z, x)
    R_nwu_tel = np.vstack((x, y, z))

    # ECEF -> NWU basis at observer
    sl, cl = np.sin(lat), np.cos(lat)
    sb, cb = np.sin(lon), np.cos(lon)
    n = np.array([-sl*cb, -sl*sb, cl])
    w = np.array([-sb, cb, 0])
    u = np.array([cl*cb, cl*sb, sl])
    R_ecef_nwu = np.vstack((n, -w, u))

    # Rotation from ECEF -> telescope
    R_ecef_tel = R_nwu_tel @ R_ecef_nwu
    q = R.from_matrix(R_ecef_tel).as_quat()  # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]])  # scalar-first

# -------------------------
# Angular velocity and quaternion rate
# -------------------------
def angular_velocity(q_prev, q_curr, dt):
    r1 = R.from_quat([q_prev[1], q_prev[2], q_prev[3], q_prev[0]])
    r2 = R.from_quat([q_curr[1], q_curr[2], q_curr[3], q_curr[0]])
    r_rel = r2 * r1.inv()
    rotvec = r_rel.as_rotvec()
    return rotvec / dt

def quaternion_rate(q, omega):
    w, x, y, z = q
    wx, wy, wz = omega
    qdot = 0.5 * np.array([
        -x*wx - y*wy - z*wz,
         w*wx - z*wy + y*wz,
         z*wx + w*wy - x*wz,
        -y*wx + x*wy + w*wz
    ])
    return qdot

# -------------------------
# Build DataFrame
# -------------------------
rows = []

for i, t in enumerate(times):
    astropy_time = Time(t)
    altaz = AltAz(obstime=astropy_time, location=location)
    sun_altaz = get_sun(astropy_time).transform_to(altaz)

    az = sun_altaz.az.rad
    el = sun_altaz.alt.rad

    q = telescope_quaternion_nwu(np.deg2rad(lat_deg), np.deg2rad(lon_deg), az, el)

    if i == 0:
        omega = np.zeros(3)
        qdot = np.zeros(4)
    else:
        dt_step = (t - times[i-1]).total_seconds()
        omega = angular_velocity(q_prev, q, dt_step)
        qdot = quaternion_rate(q, omega)

    rows.append({
        "gps_time": utc2gps(t),
        "az_rad": az,
        "el_rad": el,
        "q0": q[0], "q1": q[1], "q2": q[2], "q3": q[3],
        "wx_rad_s": omega[0], "wy_rad_s": omega[1], "wz_rad_s": omega[2],
        "qdot0": qdot[0], "qdot1": qdot[1], "qdot2": qdot[2], "qdot3": qdot[3]
    })

    q_prev = q

df = pd.DataFrame(rows)
csv_filename = "sun_tracking_nwu.csv"
df.to_csv(csv_filename, index=False, float_format="%.9f")

print(f"Saved Sun tracking data to {csv_filename}")
