"""
Compute PAA from a TLE using Skyfield with TEME->ITRS conversion built-in.

Requirements:
    pip install skyfield sgp4 numpy

Inputs:
    tle_line1, tle_line2 : strings (TLE)
    ogs_lat_deg, ogs_lon_deg, ogs_alt_m : OGS geodetic (deg, deg, meters)
    dt_utc : a datetime.datetime (UTC) or None => now()
Returns:
    dict with slant_range_m, paa_rad, paa_arcsec, az_rad, el_rad, az_offset_rad, el_offset_rad, etc.
"""


#%%

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timezone
from astropy.coordinates import (
    EarthLocation, GCRS, ITRS,
    CartesianRepresentation
)
from astropy.time import Time
import astropy.units as u
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.framelib import itrs

# constants
c = 299_792_458.0            # m/s
omega_earth = 7.2921150e-5   # rad/s (approx)
# WGS84 ellipsoid constants for geodetic->ECEF
a_wgs84 = 6378137.0          # semi-major axis (m)
f_wgs84 = 1.0 / 298.257223563
e2_wgs84 = f_wgs84 * (2 - f_wgs84)

# constants
Re = 6378137.0          # Earth radius [m]
mu = 3.986004418e14     # Earth GM [m^3/s^2]
c = 299792458.0         # speed of light [m/s]



# # === Helper functions ===
# def geodetic_to_ecef(lat_rad, lon_rad, alt_m):
#     r = Re + alt_m
#     x = r * np.cos(lat_rad) * np.cos(lon_rad)
#     y = r * np.cos(lat_rad) * np.sin(lon_rad)
#     z = r * np.sin(lat_rad)
#     return np.array([x, y, z])

# def enu_rotation(lat_rad, lon_rad):
#     slat, clat = np.sin(lat_rad), np.cos(lat_rad)
#     slon, clon = np.sin(lon_rad), np.cos(lon_rad)
#     # North-East-Up to ECEF rotation
#     return np.array([
#         [-slat*clon, -slat*slon,  clat],
#         [-slon,       clon,       0.0 ],
#         [ clat*clon,  clat*slon,  slat]
#     ])

# def ecef_to_azel(u_ecef, lat_rad, lon_rad):
#     R = enu_rotation(lat_rad, lon_rad)
#     enu = R.T @ u_ecef
#     n, e, u = enu
#     az = np.arctan2(e, n)       # from north toward east
#     el = np.arcsin(np.clip(u, -1, 1))
#     return az, el

# # === Main function ===
# def compute_paa_from_tle(tle_line1, tle_line2, ogs_lat_deg, ogs_lon_deg, ogs_alt_m=0.0, dt_utc=None):
#     # Parse TLE
#     sat = Satrec.twoline2rv(tle_line1, tle_line2)

#     # Observation time (UTC now if not provided)
#     if dt_utc is None:
#         dt_utc = datetime.now(timezone.utc)
#     jd, fr = jday(dt_utc.year, dt_utc.month, dt_utc.day,
#                   dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond*1e-6)

#     # Propagate TLE
#     e, r, v = sat.sgp4(jd, fr)
#     if e != 0:
#         raise RuntimeError(f"SGP4 propagation error code {e}")
#     r_sat = np.array(r) * 1000.0  # km -> m
#     v_sat = np.array(v) * 1000.0  # km/s -> m/s

#     # OGS position in ECEF
#     lat_rad = np.deg2rad(ogs_lat_deg)
#     lon_rad = np.deg2rad(ogs_lon_deg)
#     r_ogs = geodetic_to_ecef(lat_rad, lon_rad, ogs_alt_m)

#     # Relative vectors
#     los_vec = r_sat - r_ogs
#     R = np.linalg.norm(los_vec)     # slant range (m)
#     u = los_vec / R                 # line of sight unit

#     # OGS velocity due to Earth rotation
#     omega_earth = 7.2921150e-5  # rad/s
#     v_ogs = omega_earth * np.array([-r_ogs[1], r_ogs[0], 0.0])  # cross(omega, r)

#     v_rel = v_sat - v_ogs

#     # Transverse velocity
#     v_par = np.dot(v_rel, u) * u
#     v_perp_vec = v_rel - v_par
#     v_perp = np.linalg.norm(v_perp_vec)

#     # Point-ahead vector
#     alpha_vec = v_perp_vec / c
#     u_point = (u + alpha_vec)
#     u_point /= np.linalg.norm(u_point)

#     # Convert to az/el
#     az, el = ecef_to_azel(u, lat_rad, lon_rad)
#     az_p, el_p = ecef_to_azel(u_point, lat_rad, lon_rad)

#     return {
#         "slant_range_km": R*1e-3,
#         "paa_rad": np.linalg.norm(alpha_vec),
#         "paa_arcsec": np.linalg.norm(alpha_vec) * 206265,
#         "az_deg": np.rad2deg(az),
#         "el_deg": np.rad2deg(el),
#         "az_offset_rad": az_p - az,
#         "el_offset_rad": el_p - el
#     }

###--------------------------------------------------------------------------------##
#%%


def geodetic_to_ecef_wgs84(lat_rad, lon_rad, h_m):
    """Convert geodetic (rad,rad,m) to ECEF (m) using WGS84."""
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * np.cos(lon_rad)
    y = (N + h_m) * cos_lat * np.sin(lon_rad)
    z = (N * (1 - e2_wgs84) + h_m) * sin_lat
    return np.array([x, y, z])

def enu_from_ecef(u_ecef, lat_rad, lon_rad):
    """
    Rotate an ECEF unit vector into local ENU coordinates (north, east, up).
    Note: az will be computed from north->east convention.
    """
    slat = np.sin(lat_rad); clat = np.cos(lat_rad)
    slon = np.sin(lon_rad); clon = np.cos(lon_rad)
    # rotation matrix from ENU to ECEF (same as in earlier code)
    R = np.array([
        [-slat*clon, -slat*slon,  clat],
        [-slon,       clon,       0.0 ],
        [ clat*clon,  clat*slon,  slat]
    ])
    enu = R.T @ u_ecef   # [north, east, up]
    return enu

def azel_from_enu(enu):
    north, east, up = enu
    az = np.arctan2(east, north)        # from North toward East (radians)
    el = np.arcsin(np.clip(up, -1.0, 1.0))
    return az, el

def compute_paa_from_tle_skyfield(tle_line1, tle_line2,
                                  ogs_lat_deg, ogs_lon_deg, ogs_alt_m=0.0,
                                  dt_utc: datetime = None):
    # 1) time
    if dt_utc is None:
        dt_utc = datetime.now(timezone.utc)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    ts = load.timescale()
    t = ts.from_datetime(dt_utc)

    # 2) satellite object (Skyfield uses SGP4 internally)
    sat = EarthSatellite(tle_line1, tle_line2, 'sat', ts)

    # 3) get geocentric position/velocity in ITRS (Earth-fixed) frame
    #    use frame_xyz_and_velocity(itrs) to get position & velocity in ITRS
    #    returns Distance, Velocity-like objects (use .m, .m_per_s or .km, .km_per_s)
    geocentric = sat.at(t)
    pos_itrs, vel_itrs = geocentric.frame_xyz_and_velocity(itrs)  # ITRS/ECEF
    r_sat_ecef = pos_itrs.m       # meters, shape (3,)
    v_sat_ecef = vel_itrs.m_per_s # m/s, shape (3,)

    # 4) compute OGS ECEF position + velocity (rigid Earth rotation)
    lat_rad = np.deg2rad(ogs_lat_deg)
    lon_rad = np.deg2rad(ogs_lon_deg)
    r_ogs_ecef = geodetic_to_ecef_wgs84(lat_rad, lon_rad, ogs_alt_m)  # meters
    # OGS velocity due to Earth rotation: v = omega x r
    # omega vector (0, 0, omega_earth) in ECEF, so v = [ -omega * y, omega * x, 0 ]
    v_ogs_ecef = omega_earth * np.array([-r_ogs_ecef[1], r_ogs_ecef[0], 0.0])

    # 5) relative geometry
    los_vec = r_sat_ecef - r_ogs_ecef
    R = np.linalg.norm(los_vec)
    if R == 0:
        raise ValueError("satellite and OGS positions coincide (!)")

    u_los = los_vec / R

    # 6) relative velocity and transverse component
    v_rel = v_sat_ecef - v_ogs_ecef
    v_par = np.dot(v_rel, u_los) * u_los
    v_perp_vec = v_rel - v_par
    v_perp = np.linalg.norm(v_perp_vec)

    # 7) point-ahead vector (small angle approx)
    alpha_vec = v_perp_vec / c             # radians vector
    theta = np.linalg.norm(alpha_vec)      # magnitude (rad)
    theta_arcsec = theta * 206265.0

    # 8) compute pointing LOS unit after applying PAA
    u_point = u_los + alpha_vec
    u_point /= np.linalg.norm(u_point)

    # 9) convert LOS and pointed LOS into az/el at OGS (north->east az convention)
    enu_los = enu_from_ecef(u_los, lat_rad, lon_rad)
    az_los, el_los = azel_from_enu(enu_los)

    enu_point = enu_from_ecef(u_point, lat_rad, lon_rad)
    az_point, el_point = azel_from_enu(enu_point)

    # normalize az diffs to (-pi, pi]
    d_az = az_point - az_los
    d_az = (d_az + np.pi) % (2*np.pi) - np.pi
    d_el = el_point - el_los

    return {
        'time_utc': dt_utc.isoformat(),
        'r_sat_ecef_m': r_sat_ecef,
        'v_sat_ecef_m_s': v_sat_ecef,
        'r_ogs_ecef_m': r_ogs_ecef,
        'v_ogs_ecef_m_s': v_ogs_ecef,
        'slant_range_m': R,
        'v_perp_m_s': v_perp,
        'paa_rad': theta,
        'paa_arcsec': theta_arcsec,
        'az_rad': az_los,
        'el_rad': el_los,
        'az_offset_rad': d_az,
        'el_offset_rad': d_el,
        'notes': 'computed using Skyfield TEME->ITRS via frame_xyz_and_velocity(itrs). OGS velocity = Earth rotation only.'
    }

# ---------------------------Example usage

if __name__ == "__main__":
    # sample TLE (ISS as example; replace with your TLE)
    tle1 = "1 25544U 98067A   24250.51782528  .00009752  00000-0  18303-3 0  9991"
    tle2 = "2 25544  51.6421  35.5850 0006657  73.1340  42.9114 15.50037988432284"

    res = compute_paa_from_tle_skyfield(tle1, tle2, ogs_lat_deg=48.1351, ogs_lon_deg=11.5820, ogs_alt_m=0.0)
    print("*"*80)
    print("Computing paa from TLE using skyfield")
    print("*"*80)
    print("\ntime:", res['time_utc'])
    print("slant range (km):", res['slant_range_m'] / 1000.0)
    print("PAA (arcsec):", res['paa_arcsec'])
    print("Az (deg):", np.rad2deg(res['az_rad']), "El (deg):", np.rad2deg(res['el_rad']))
    print("Az offset (arcsec):", np.rad2deg(res['az_offset_rad']) * 3600.0)
    print("El offset (arcsec):", np.rad2deg(res['el_offset_rad']) * 3600.0)


    # ogs_lat, ogs_lon, ogs_alt = 48.1351, 11.5820, 0.0  # Munich
    # res = compute_paa_from_tle(tle1, tle2, ogs_lat, ogs_lon, ogs_alt)
    # print("*"*80)
    # print("Computing paa from TLE using manual SGP4 + ECEF conversion")
    # print("*"*80)
    # for k,v in res.items():
    #     print(f"{k}: {v}")
# %%


# def ground_station_state_eci(lat_deg, lon_deg, alt_m, time):
#     loc = EarthLocation(lat=lat_deg*u.deg,
#                         lon=lon_deg*u.deg,
#                         height=alt_m*u.m)
#     gcrs = loc.get_gcrs(obstime=time)
#     r = gcrs.cartesian.xyz.to(u.m).value
#     v = gcrs.velocity.d_xyz.to(u.m/u.s).value
#     return r, v


# def los_unit_eci(az_rad, el_rad, lat_deg, lon_deg, time):

#     loc = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg)

#     # Define LOS in AltAz frame
#     from astropy.coordinates import AltAz

#     altaz = AltAz(
#         az=az_rad * u.rad,
#         alt=el_rad * u.rad,
#         location=loc,
#         obstime=time
#     )

#     gcrs = altaz.transform_to(GCRS(obstime=time))

#     los = gcrs.cartesian.xyz.value
#     return los / np.linalg.norm(los)

# def intersect_ray_sphere(r0, u_vec, R):
#     # solve |r0 + t u| = R
#     b = 2.0 * np.dot(r0, u_vec)
#     c = np.dot(r0, r0) - R**2
#     disc = b*b - 4*c
#     if disc < 0:
#         return None
#     t = (-b + np.sqrt(disc)) / 2.0
#     return r0 + t*u_vec


# def satellite_state_guess(r_sat):
#     r = np.linalg.norm(r_sat)
#     r_hat = r_sat / r

#     v_mag = np.sqrt(mu / r)

#     z_hat = np.array([0.0, 0.0, 1.0])
#     v_dir = np.cross(z_hat, r_hat)
#     if np.linalg.norm(v_dir) < 1e-8:
#         v_dir = np.cross(r_hat, np.array([1.0, 0.0, 0.0]))

#     v_dir /= np.linalg.norm(v_dir)
#     return v_mag * v_dir


# def point_ahead_from_az_el(
#     az_rad, el_rad,
#     lat_deg, lon_deg, alt_m,
#     time,
#     h_assumed=600e3
# ):
#     # ground station ECI state
#     r_gs, v_gs = ground_station_state_eci(
#         lat_deg, lon_deg, alt_m, time
#     )

#     # LOS unit vector in ECI
#     u_los = los_unit_eci(
#         az_rad, el_rad, lat_deg, lon_deg, time
#     )

#     # satellite position guess
#     r_sat = intersect_ray_sphere(
#         r_gs, u_los, Re + h_assumed
#     )
#     if r_sat is None:
#         return {"error": "no intersection with assumed orbit sphere"}

#     # satellite velocity guess
#     v_sat = satellite_state_guess(r_sat)

#     # relative transverse velocity
#     v_rel = v_sat - v_gs
#     v_par = np.dot(v_rel, u_los) * u_los
#     v_perp_vec = v_rel - v_par

#     # point-ahead (aberration)
#     alpha_vec = v_perp_vec / c
#     u_point = u_los + alpha_vec
#     u_point /= np.linalg.norm(u_point)

#     # convert corrected LOS back to ENU
#     loc = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg)
#     itrs = GCRS(
#         CartesianRepresentation(u_point * u.one),
#         obstime=time
#     ).transform_to(ITRS(obstime=time, location=loc))

#     enu = itrs.cartesian.xyz.value
#     north, east, up = enu

#     az_point = np.arctan2(east, north)
#     el_point = np.arcsin(up)

#     delta_az = (az_point - az_rad + np.pi) % (2*np.pi) - np.pi
#     delta_el = el_point - el_rad

#     return {
#         "delta_az_rad": delta_az,
#         "delta_el_rad": delta_el,
#         "theta_rad": np.linalg.norm(alpha_vec),
#         "v_perp": np.linalg.norm(v_perp_vec)
#     }

# if __name__ == "__main__":
#     # ---- example scenario ----
#     lat_deg = 52.0          # ground station latitude
#     lon_deg = 5.0           # ground station longitude
#     alt_m = 50.0            # station altitude [m]

#     az_deg = 135.0          # azimuth (deg, from north toward east)
#     el_deg = 30.0           # elevation (deg)

#     h_assumed = 600e3       # assumed satellite altitude [m]

#     time = Time("2026-02-16T17:00:00", scale="utc")

#     # convert to radians
#     az_rad = np.deg2rad(az_deg)
#     el_rad = np.deg2rad(el_deg)

#     # ---- compute point-ahead ----
#     result = point_ahead_from_az_el(
#         az_rad, el_rad,
#         lat_deg, lon_deg, alt_m,
#         time,
#         h_assumed=h_assumed
#     )

#     if "error" in result:
#         print("Error:", result["error"])
#     else:
#         print("Point-ahead result:")
#         print(f"  Δaz  = {np.rad2deg(result['delta_az_rad'])*1e6:.3f} µdeg")
#         print(f"  Δel  = {np.rad2deg(result['delta_el_rad'])*1e6:.3f} µdeg")
#         print(f"  θ    = {result['theta_rad']*1e6:.3f} µrad")
#         print(f"  v⊥   = {result['v_perp']/1e3:.3f} km/s")



# %%
