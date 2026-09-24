#%%
import numpy as np
import erfa, math
import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import EarthLocation
from datetime import datetime, timezone, timedelta
import valladopy.astro.time.frame_conversions as vallado
#from fc import IAU80Array, ecef2eci

ERA0 = 0.7790572732640
ERA_RATE = 1.00273781191135448
TWO_PI = 2.0 * math.pi
DJ00 = 2451545.0
Re = 6378137.0


def coord_pos_lla_eme2000(epochTime, station, iau80arr,
                           xp=0.0, yp=0.0,
                           lod=0.0, ddpsi=0.0, ddeps=0.0):

    # WGS84 geodetic position
    loc = EarthLocation.from_geodetic(
        lon=station.longitude * u.deg,
        lat=station.latitude * u.deg,
        height=station.altitude_km * 1000.0 * u.m,
        ellipsoid="WGS84",
    )

    # ECEF / ITRS position
    itrs = loc.get_itrs(obstime=epochTime)

def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def calc_eci2ecef(epochTime):
    ut1  = (epochTime- 18.0 -0.0450)/86400.0
    
    d2r = np.pi / 180

    # Precession: GM2000 -> MOD
    t = (ut1 - 0.5) / 36525
    zeta  = t * (0.6406161 + t * (0.0000839 + 0.0000050*t))
    z     = t * (0.6406161 + t * (0.0003041 + 0.0000051*t))
    theta = t * (0.5567530 - t * (0.0001185 + 0.0000116*t))

    m1 = Rz(-np.pi/2 - z*d2r) @ Rx(theta*d2r) @ Rz(np.pi/2 - zeta*d2r)

    # Nutation: MOD -> TOD
    t = ut1 - 0.5
    a = (125.0 - 0.05295*t) * d2r
    b = (200.9 + 1.97129*t) * d2r

    dpsi = (-0.0048*np.sin(a) - 0.0004*np.sin(b)) * d2r
    deps = ( 0.0026*np.cos(a) + 0.0002*np.cos(b)) * d2r

    eps = 23.439291 * d2r
    dmu = dpsi * np.cos(eps)
    dnu = dpsi * np.sin(eps)

    m2 = Rz(-dmu) @ Rx(-deps) @ Ry(dnu)

    # Earth rotation: TOD -> PEF
    g = 99.96779469 + ut1 * (360.985647366286 + 0.29079e-12*ut1)
    m3 = Rz(g*d2r + dmu)

    return m3 @ m2 @ m1

def ecef_to_eme2000(r_ecef, t):
    """
    ITRS/ECEF -> J2000/EME2000
    r_ecef : [x, y, z] in meters
    t      : astropy.time.Time
    Returns:        r_eme2000 : [x, y, z] meters
    """

    # IAU 1976 precession matrix:
    # J2000 mean equator/equinox -> mean equator/equinox of date
    P = erfa.pmat76(t.tt.jd1, t.tt.jd2)

    # IAU 1980 nutation:
    # mean equator/equinox of date -> true equator/equinox of date
    N = erfa.nut80(t.tt.jd1, t.tt.jd2)

    # Greenwich apparent sidereal time
    # This is the Earth rotation angle used by the EME2000
    # conventional transformation.
    gast = erfa.gst94(t.ut1.jd1, t.ut1.jd2)

    # ECEF -> true equator/equinox of date
    R = erfa.rz(gast, np.eye(3))

    # True-of-date -> J2000/EME2000    #
    # P maps J2000 -> mean-of-date
    # N maps mean-of-date -> true-of-date    #
    # Therefore inverse is:
    C = P.T @ N.T @ R

    # Polar motion: ITRS -> TIRS correction
    W = erfa.pom00(xp, yp, 0.0)

    # Complete ITRS -> EME2000
    C = P.T @ N.T @ R @ W

    return C @ np.asarray(r_ecef)

def era_smart(jd1, jd2):
    d1 = jd1 - DJ00
    d2 = jd2

    N = np.floor(d1 + d2)
    f = (d1 - N) + d2

    x = ERA0 + ERA_RATE * N + ERA_RATE * f

    return float(np.mod(TWO_PI * x, TWO_PI))

def gps_seconds_now():
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    # Current GPS-UTC offset
    gps_utc = 18.0
    return (now - gps_epoch).total_seconds() + gps_utc

def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def gps_to_time(gps_week, gps_tow):
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    # GPS = TAI - 19 s
    return Time(gps_epoch + timedelta(seconds=gps_week*604800 + gps_tow + 19), scale="tai")

def ecef_to_eci_full(r_ecef, t, xp, yp):
    # GCRS -> ITRS matrix
    # transpose gives ITRS -> GCRS
    C_itrs_to_gcrs = erfa.c2t06a(t.tt.jd1, t.tt.jd2, t.ut1.jd1, t.ut1.jd2, xp, yp).T
    
    # GCRS -> J2000/EME2000
    rb, rp, rbp = erfa.bp06(2400000.5, 51544.5)
    C_gcrs_to_j2000 = rb.T

    C = C_gcrs_to_j2000 @ C_itrs_to_gcrs
    return C @ r_ecef, C

def ecef_to_eci_R(r_ecef, t):
    era = erfa.era00(t.ut1.jd1, t.ut1.jd2)

    C = R3(era).T
    return C @ r_ecef, C, era

def ecef_to_eci_R_smart(r_ecef, t):
    era = era_smart(t.ut1.jd1,t.ut1.jd2)

    C = R3(era).T

    return C @ r_ecef, C, era

def old_ecef_to_eci_j2000(r_ecef, gps_seconds):
    """Accurately converts ECEF (ITRS) coordinates into ECI (GCRS / J2000) frame.

    Parameters:
      x_ecef, y_ecef, z_ecef : Coordinates in ECEF (km or m)
      gps_seconds            : Continuous GPS seconds since 1980-01-06 00:00:00
      dut1                   : (UT1 - UTC) in seconds (from IERS Bulletin A;
      default: -0.0450)
      utc_to_gps             : Leap seconds between GPST and UTC (default: 18.0)

    Returns:
      (x_eci, y_eci, z_eci) in the J2000/GCRS frame.
    """
   
    dut1=-0.0450
    utc_to_gps=18.0
    
    x_ecef, y_ecef, z_ecef = r_ecef
    # -------------------------------------------------------------------------
    # 1. TIME BASES (TT for Precession/Nutation, UT1 for Earth Rotation)
    # -------------------------------------------------------------------------
    # TT = GPST + 51.184 s
    tt_days_since_j2000 = (gps_seconds + 51.184) / 86400.0 - 7300.5
    T = tt_days_since_j2000 / 36525.0  # Julian centuries of TT since J2000

    # UT1 = GPST - UtcToGpsTime + DUT1
    ut1_days_since_j2000 = (gps_seconds - utc_to_gps + dut1) / 86400.0 - 7300.5

    # -------------------------------------------------------------------------
    # 2. STEP 1: ROTATE ECEF -> CIRS (Earth Rotation Angle)
    # -------------------------------------------------------------------------
    era = (2.0* math.pi* (0.7790572732640 + 1.00273781191135448 * ut1_days_since_j2000)
    ) % (2.0 * math.pi)
    cos_era = math.cos(era)
    sin_era = math.sin(era)

    # R3(ERA)^T * r_ecef
    x_cirs = cos_era * x_ecef - sin_era * y_ecef
    y_cirs = sin_era * x_ecef + cos_era * y_ecef
    z_cirs = z_ecef

    # -------------------------------------------------------------------------
    # 3. STEP 2: IAU 2006/2000A BIAS-PRECESSION-NUTATION (CIO coordinates X, Y, s)
    # -------------------------------------------------------------------------
    arcsec2rad = math.pi / (180.0 * 3600.0)

    # Fundamental Delaunay solar/lunar arguments (in radians)
    l = ((485868.249036 + 1717915923.2178 * T) % 1296000.0) * arcsec2rad  # Moon anomaly
    lp = ((1287104.793048 + 129596581.0481 * T) % 1296000.0) * arcsec2rad  # Sun anomaly
    F = ((335779.526232 + 1739527262.8478 * T) % 1296000.0) * arcsec2rad  # Moon arg of lat
    D = ((1072260.703692 + 1602961601.2090 * T) % 1296000.0) * arcsec2rad  # Sun-Moon elong
    Om = ((450160.398036 - 6962890.5431 * T) % 1296000.0) * arcsec2rad  # Moon node

    # IAU 2006 Precession polynomials for CIP unit vector (arcseconds)
    X_poly = -0.016617 + 2004.191898 * T - 0.4297829 * T**2 - 0.19861834 * T**3
    Y_poly = -0.006951 - 0.025896 * T - 22.4072747 * T**2 + 0.00190059 * T**3

    # Primary IAU 2000A Nutation series terms (arcseconds)
    dX = (
        -6844.318 * math.sin(Om)
        - 523.908 * math.sin(2 * F - 2 * D + 2 * Om)
        + 90.552 * math.sin(2 * Om)
        + 82.168 * math.sin(2 * F + 2 * Om)
        + 58.707 * math.sin(l)
        + 20.099 * math.sin(2 * F - 2 * D + Om)
        - 17.500 * math.sin(2 * F)
        - 14.765 * math.sin(l + 2 * F + 2 * Om)
    ) * 1e-3

    dY = (
        9205.236 * math.cos(Om)
        + 573.033 * math.cos(2 * F - 2 * D + 2 * Om)
        - 97.845 * math.cos(2 * Om)
        - 89.615 * math.cos(2 * F + 2 * Om)
        + 22.438 * math.cos(2 * F - 2 * D + Om)
        + 20.070 * math.cos(2 * F)
        + 16.144 * math.cos(l + 2 * F + 2 * Om)
    ) * 1e-3

    X = (X_poly + dX) * arcsec2rad
    Y = (Y_poly + dY) * arcsec2rad

    # CIO locator s (radians)
    s = (-0.000094 + 0.003808 * T) * arcsec2rad - (X * Y) / 2.0

    # -------------------------------------------------------------------------
    # 4. ROTATE CIRS -> GCRS / J2000 (Exact IAU CIO BPN matrix)
    # -------------------------------------------------------------------------
    d2 = X**2 + Y**2
    cos_d = math.sqrt(max(0.0, 1.0 - d2))
    a = 1.0 / (1.0 + cos_d)

    sin_s = math.sin(s)
    cos_s = math.cos(s)

    # Base intermediate matrix
    P11, P12, P13 = 1.0 - a * X**2, -a * X * Y, X
    P21, P22, P23 = -a * X * Y, 1.0 - a * Y**2, Y
    P31, P32, P33 = -X, -Y, cos_d

    # Full Q matrix components (Q = P * R3(s))
    Q11 = P11 * cos_s + P12 * sin_s
    Q12 = -P11 * sin_s + P12 * cos_s
    Q13 = P13

    Q21 = P21 * cos_s + P22 * sin_s
    Q22 = -P21 * sin_s + P22 * cos_s
    Q23 = P23

    Q31 = P31 * cos_s + P32 * sin_s
    Q32 = -P31 * sin_s + P32 * cos_s
    Q33 = P33

    # Multiply Q * r_cirs
    x_eci = Q11 * x_cirs + Q12 * y_cirs + Q13 * z_cirs
    y_eci = Q21 * x_cirs + Q22 * y_cirs + Q23 * z_cirs
    z_eci = Q31 * x_cirs + Q32 * y_cirs + Q33 * z_cirs

    return (x_eci, y_eci, z_eci)

def ecef_to_eci_j2000(r_ecef, gps_seconds):
    """Accurately converts ECEF (ITRS) coordinates into ECI (GCRS / J2000) frame.

    Numerically stabilized for high precision over long operational timelines.

    Parameters:
      r_ecef      : Tuple or list of (x, y, z) in ECEF (m or km)
      gps_seconds : Continuous GPS seconds since 1980-01-06 00:00:00
      dut1        : (UT1 - UTC) in seconds (default: -0.0450)
      utc_to_gps  : Leap seconds between GPST and UTC (default: 18.0)

    Returns:
      (x_eci, y_eci, z_eci) in the J2000 / GCRS frame.
    """
    dut1=-0.0450
    utc_to_gps=18.0

    x_ecef, y_ecef, z_ecef = r_ecef

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    ARCSEC2RAD = math.pi / 648000.0
    TWO_PI = 2.0 * math.pi
    ONE_OVER_ARCSEC_TURN = 1.0 / 1296000.0

    # -------------------------------------------------------------------------
    # 1. Time Bases (Stabilized Integer / Fractional Day Split)
    # -------------------------------------------------------------------------
    # Days from GPS Epoch (1980-01-06 00:00:00) to J2000 (2000-01-01 12:00:00) = 7300.5
    tt_total_sec = gps_seconds + 51.184
    ut1_total_sec = gps_seconds - utc_to_gps + dut1

    tt_days = (tt_total_sec / 86400.0) - 7300.5
    T = tt_days / 36525.0  # Julian centuries of TT since J2000

    ut1_days = (ut1_total_sec / 86400.0) - 7300.5
    d_int = math.floor(ut1_days)
    d_frac = ut1_days - d_int

    # -------------------------------------------------------------------------
    # 2. Earth Rotation Angle (ERA) & ECEF -> CIRS Rotation
    # -------------------------------------------------------------------------
    # Decompose integer day cycles to retain sub-millimeter angular precision
    era_turns = 0.7790572732640 + d_frac + 0.00273781191135448 * ut1_days
    era = TWO_PI * (era_turns - math.floor(era_turns))

    cos_era = math.cos(era)
    sin_era = math.sin(era)

    # CIRS coordinates: R3(-ERA) * r_ecef
    x_cirs = cos_era * x_ecef - sin_era * y_ecef
    y_cirs = sin_era * x_ecef + cos_era * y_ecef
    z_cirs = z_ecef

    # -------------------------------------------------------------------------
    # 3. Delaunay Arguments (Pre-reduced modulo 1 full turn)
    # -------------------------------------------------------------------------
    l = TWO_PI * (((485868.249036 + (1717915923.2178 % 1296000.0) * T) * ONE_OVER_ARCSEC_TURN) % 1.0)
    lp = TWO_PI * (((1287104.793048 + (129596581.0481 % 1296000.0) * T) * ONE_OVER_ARCSEC_TURN) % 1.0)
    F = TWO_PI * (((335779.526232 + (1739527262.8478 % 1296000.0) * T) * ONE_OVER_ARCSEC_TURN) % 1.0)
    D = TWO_PI * (((1072260.703692 + (1602961601.2090 % 1296000.0) * T) * ONE_OVER_ARCSEC_TURN) % 1.0)
    Om = TWO_PI * (((450160.398036 - (6962890.5431 % 1296000.0) * T) * ONE_OVER_ARCSEC_TURN) % 1.0)

    # -------------------------------------------------------------------------
    # 4. CIP Coordinates (X, Y) and CIO Locator (s)
    # -------------------------------------------------------------------------
    # IAU 2006 Precession polynomials (arcseconds)
    X_poly = -0.016617 + 2004.191898 * T - 0.4297829 * T**2 - 0.19861834 * T**3
    Y_poly = -0.006951 - 0.025896 * T - 22.4072747 * T**2 + 0.00190059 * T**3

    # Primary IAU 2000A Nutation series (arcseconds)
    dX = (
        -6844.318 * math.sin(Om)
        - 523.908 * math.sin(2 * F - 2 * D + 2 * Om)
        + 90.552 * math.sin(2 * Om)
        + 82.168 * math.sin(2 * F + 2 * Om)
        + 58.707 * math.sin(l)
        + 20.099 * math.sin(2 * F - 2 * D + Om)
        - 17.500 * math.sin(2 * F)
        - 14.765 * math.sin(l + 2 * F + 2 * Om)
    ) * 1e-3

    dY = (
        9205.236 * math.cos(Om)
        + 573.033 * math.cos(2 * F - 2 * D + 2 * Om)
        - 97.845 * math.cos(2 * Om)
        - 89.615 * math.cos(2 * F + 2 * Om)
        + 22.438 * math.cos(2 * F - 2 * D + Om)
        + 20.070 * math.cos(2 * F)
        + 16.144 * math.cos(l + 2 * F + 2 * Om)
    ) * 1e-3

    X = (X_poly + dX) * ARCSEC2RAD
    Y = (Y_poly + dY) * ARCSEC2RAD

    # CIO locator s (radians)
    s = (-0.000094 + 0.003808 * T) * ARCSEC2RAD - (X * Y) * 0.5

    # -------------------------------------------------------------------------
    # 5. Rotate CIRS -> GCRS / J2000 (Exact CIO Transformation Matrix)
    # -------------------------------------------------------------------------
    d2 = X**2 + Y**2
    cos_d = math.sqrt(max(0.0, 1.0 - d2))
    a = 1.0 / (1.0 + cos_d)

    cos_s = math.cos(s)
    sin_s = math.sin(s)

    # Simplified direct terms for matrix Q = P * R3(s)
    Q11 = (1.0 - a * X**2) * cos_s - a * X * Y * sin_s
    Q12 = -(1.0 - a * X**2) * sin_s - a * X * Y * cos_s
    Q13 = X

    Q21 = -a * X * Y * cos_s + (1.0 - a * Y**2) * sin_s
    Q22 = a * X * Y * sin_s + (1.0 - a * Y**2) * cos_s
    Q23 = Y

    Q31 = -X * cos_s - Y * sin_s
    Q32 = X * sin_s - Y * cos_s
    Q33 = cos_d

    # Multiply Q * r_cirs
    x_eci = Q11 * x_cirs + Q12 * y_cirs + Q13 * z_cirs
    y_eci = Q21 * x_cirs + Q22 * y_cirs + Q23 * z_cirs
    z_eci = Q31 * x_cirs + Q32 * y_cirs

    return (x_eci, y_eci, z_eci)

def angle_error(a, b):
    """Smallest signed angular difference."""
    return (a - b + math.pi) % TWO_PI - math.pi

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ============================================================
    # MONTE CARLO: FULL ECEF->ECI vs J2000 vs R-only
    # ============================================================

    N = 10000
    rng = np.random.default_rng(42)

    # -------------------------
    # Time / EOP
    # -------------------------
    gps_seconds = gps_seconds_now()
    gps_week = int(gps_seconds // 604800)
    gps_tow = gps_seconds - gps_week * 604800
    t = gps_to_time(gps_week, gps_tow)

    eop = iers.IERS_Auto.open()
    dut1 = t.get_delta_ut1_utc().to_value(u.s)
    xp, yp = eop.pm_xy(t)
    xp, yp = xp.to_value(u.rad), yp.to_value(u.rad)

    print("UTC :", t.utc.isot)
    print("UT1 :", t.ut1.isot)
    print(f"DUT1 = {dut1:.6f} s")
    print(f"xp   = {xp:.6e} rad")
    print(f"yp   = {yp:.6e} rad")

    # -------------------------
    # Random ECEF positions
    # -------------------------
    direction = rng.normal(size=(N, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]

    altitude = rng.uniform(400e3, 10000e3, N)
    radius = Re + altitude
    r_ecef = direction * radius[:, None]

    # -------------------------
    # Allocate
    # -------------------------
    r_full  = np.zeros((N, 3))
    r_j2000 = np.zeros((N, 3))
    r_R     = np.zeros((N, 3))
    r_Rs    = np.zeros((N, 3))

    era_R  = np.zeros(N)
    era_Rs = np.zeros(N)

    # -------------------------
    # Transform
    # -------------------------
    for i in range(N):
        r_full[i], _ = ecef_to_eci_full(r_ecef[i], t, xp, yp)
        r_j2000[i] = ecef_to_eci_j2000(r_ecef[i], t.gps)
        r_R[i], _, era_R[i] = ecef_to_eci_R(r_ecef[i], t)
        r_Rs[i], _, era_Rs[i] = ecef_to_eci_R_smart(r_ecef[i], t)

    # ============================================================
    # ACTUAL RELATIVE ROTATION: FULL vs R
    # ============================================================

    # R-only
    _, C_R, _ = ecef_to_eci_R(r_ecef[0], t)

    # ITRS -> GCRS
    C_gcrs = erfa.c2t06a(
    t.tt.jd1, t.tt.jd2,
    t.ut1.jd1, t.ut1.jd2,
    xp, yp
    ).T

    # GCRS -> J2000 frame-bias only
    rb, rp, rbp = erfa.bp06(2400000.5, 51544.5)

    def rot_angle(A, B):
        C = A @ B.T
        return math.acos(
            np.clip((np.trace(C) - 1.0) / 2.0, -1.0, 1.0)
        )

    print("\nR vs GCRS:")
    a = rot_angle(C_gcrs, C_R)
    print(f"{np.degrees(a):.6f} deg")
    print(f"{np.degrees(a)*3600:.3f} arcsec")

    print("\nR vs GCRS+bias:")
    C_bias = rb.T @ C_gcrs
    a = rot_angle(C_bias, C_R)
    print(f"{np.degrees(a):.6f} deg")
    print(f"{np.degrees(a)*3600:.3f} arcsec")
    # ============================================================
    # DIFFERENCES
    # ============================================================

    # Full ERFA QRW vs R-only
    d_full_R = r_full - r_R

    # J2000 implementation vs R-only
    d_j2000_R = r_j2000 - r_R

    # Full ERFA vs J2000 implementation
    d_full_j2000 = r_full - r_j2000

    # Smart ERA vs normal ERFA ERA
    d_R_Rs = r_R - r_Rs

    # -------------------------
    # Norms
    # -------------------------
    e_full_R = np.linalg.norm(d_full_R, axis=1)
    e_j2000_R = np.linalg.norm(d_j2000_R, axis=1)
    e_full_j2000 = np.linalg.norm(d_full_j2000, axis=1)
    e_R_Rs = np.linalg.norm(d_R_Rs, axis=1)


    # ============================================================
    # STATISTICS
    # ============================================================

    def report(name, e):
        print(f"\n{name}")
        print("-" * 55)
        print(f"Min     = {np.min(e):.6f} m")
        print(f"Mean    = {np.mean(e):.6f} m")
        print(f"Median  = {np.median(e):.6f} m")
        print(f"Std     = {np.std(e):.6f} m")
        print(f"Max     = {np.max(e):.6f} m")
        print(f"RMS     = {np.sqrt(np.mean(e**2)):.6f} m")
        print(f"P90     = {np.percentile(e,90):.6f} m")
        print(f"P95     = {np.percentile(e,95):.6f} m")
        print(f"P99     = {np.percentile(e,99):.6f} m")
        print(f"P99.9   = {np.percentile(e,99.9):.6f} m")


    print("\n==============================================")
    print("ECEF -> ECI MONTE CARLO")
    print("==============================================")
    print(f"N = {N}")

    report("FULL ERFA QRW  vs  R-only", e_full_R)
    report("J2000         vs  R-only", e_j2000_R)
    report("FULL ERFA     vs  J2000", e_full_j2000)
    report("ERFA ERA R    vs  SMART ERA R", e_R_Rs)


    # ============================================================
    # ERA ANGULAR DIFFERENCE
    # ============================================================

    d_era = np.array([angle_error(a, b)
        for a, b in zip(era_R, era_Rs)])

    print("\n==============================================")
    print("ERA ONLY")
    print("==============================================")
    print(f"Max ERA error = {np.max(np.abs(d_era)):.6e} rad")
    print(f"Max ERA error = "
        f"{np.degrees(np.max(np.abs(d_era))) * 3600:.9e} arcsec")
    
    # ============================================================
    # PLOTS
    # ============================================================

    # -------------------------
    # 1. Histograms
    # -------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    plots = [
        (e_full_R,      "Full ERFA QRW − R-only"),
        (e_j2000_R,     "J2000 − R-only"),
        (e_full_j2000,  "Full ERFA QRW − J2000"),
        (e_R_Rs,        "ERFA ERA R − Smart ERA R"),
    ]

    for a, (e, title) in zip(ax.flat, plots):
        a.hist(e, bins=100, edgecolor="none")
        a.set_xlabel("Position difference [m]")
        a.set_ylabel("Count")
        a.set_title(title)
        a.grid(True, alpha=0.3)

    plt.tight_layout()


    # -------------------------
    # 2. Error vs altitude
    # -------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    plots = [
        (e_full_R,      "Full ERFA QRW − R-only"),
        (e_j2000_R,     "J2000 − R-only"),
        (e_full_j2000,  "Full ERFA QRW − J2000"),
        (e_R_Rs,        "ERFA ERA R − Smart ERA R"),
    ]

    for a, (e, title) in zip(ax.flat, plots):
        a.scatter(
            altitude / 1000.0,
            e,
            s=3,
            alpha=0.25
        )
        a.set_xlabel("Altitude [km]")
        a.set_ylabel("Position difference [m]")
        a.set_title(title)
        a.grid(True, alpha=0.3)

    plt.tight_layout()


    # -------------------------
    # 3. Component errors
    # -------------------------
    fig, ax = plt.subplots(3, 4, figsize=(16, 10))

    datasets = [
        d_full_R,
        d_j2000_R,
        d_full_j2000,
        d_R_Rs,
    ]

    titles = [
        "Full ERFA − R",
        "J2000 − R",
        "Full ERFA − J2000",
        "ERFA ERA − Smart ERA",
    ]

    components = ["ΔX [m]", "ΔY [m]", "ΔZ [m]"]

    for col, (d, title) in enumerate(zip(datasets, titles)):
        for row in range(3):
            ax[row, col].hist(
                d[:, row],
                bins=100,
                edgecolor="none"
            )
            ax[row, col].set_xlabel(components[row])
            ax[row, col].set_ylabel("Count")
            ax[row, col].grid(True, alpha=0.3)

            if row == 0:
                ax[row, col].set_title(title)

    plt.tight_layout()


    # -------------------------
    # 4. ERA numerical error
    # -------------------------
    plt.figure(figsize=(10, 5))

    plt.hist(
        np.abs(d_era),
        bins=100,
        edgecolor="none"
    )

    plt.xlabel("|ERFA ERA − Smart ERA| [rad]")
    plt.ylabel("Count")
    plt.title("ERA Numerical Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # -------------------------
    # 5. ERA error → position error
    # -------------------------
    era_position_error = np.abs(d_era) * radius

    plt.figure(figsize=(10, 5))

    plt.hist(
        era_position_error * 1000.0,
        bins=100,
        edgecolor="none"
    )

    plt.xlabel("Position difference [mm]")
    plt.ylabel("Count")
    plt.title("Position Error Caused by ERA Numerical Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # -------------------------
    # 6. Direct comparison
    # -------------------------
    plt.figure(figsize=(10, 6))

    plt.scatter(
        altitude / 1000,
        e_full_R,
        s=3,
        alpha=0.25,
        label="Full ERFA − R"
    )

    plt.scatter(
        altitude / 1000,
        e_j2000_R,
        s=3,
        alpha=0.25,
        label="J2000 − R"
    )

    plt.scatter(
        altitude / 1000,
        e_full_j2000,
        s=3,
        alpha=0.25,
        label="Full ERFA − J2000"
    )

    plt.xlabel("Altitude [km]")
    plt.ylabel("Position difference [m]")
    plt.title("ECEF → ECI Transformation Differences")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


    plt.show()

    # for p in [50, 90, 95, 99, 99.9]:
    #     print(f"P{p:<4}   = {np.percentile(error, p):.6f} m")

    # # ------------------------------------------------------------
    # # Plot histogram
    # # ------------------------------------------------------------
    # plt.figure(figsize=(9, 5))
    # plt.hist(error, bins=100, edgecolor="none")
    # plt.xlabel("Position difference |QRW - R| [m]")
    # plt.ylabel("Number of samples")
    # plt.title("Monte Carlo: ECEF → ECI Full QRW vs R-only")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # # plt.figure(figsize=(9, 5))
    # # plt.hist(error_smart, bins=100, edgecolor="none")
    # # plt.xlabel("Position difference |QRW - R| [m]")
    # # plt.ylabel("Number of samples")
    # # plt.title("Monte Carlo: ECEF → ECI Full QRW vs R-only_Smart")
    # # plt.grid(True, alpha=0.3)
    # # plt.tight_layout()

    # plt.figure(figsize=(9, 5))
    # plt.hist(error_j2000, bins=100, edgecolor="none")
    # plt.xlabel("Position difference |QRW - R| [m]")
    # plt.ylabel("Number of samples")
    # plt.title("Monte Carlo: ECEF → ECI Full QRW vs R-only _J2000")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # # ------------------------------------------------------------
    # # Error vs altitude
    # # ------------------------------------------------------------
    # plt.figure(figsize=(9, 5))
    # plt.scatter(altitude / 1000.0,error,s=3,alpha=0.25    )
    # plt.xlabel("Altitude [km]")
    # plt.ylabel("|QRW - R| [m]")
    # plt.title("Transformation Error vs Altitude")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # plt.figure(figsize=(9, 5))
    # plt.scatter(altitude / 1000.0,error_smart,s=3,alpha=0.25    )
    # plt.xlabel("Altitude [km]")
    # plt.ylabel("|QRW - R| [m]")
    # plt.title("Transformation Error vs Altitude_Smart")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # plt.figure(figsize=(9, 5))
    # plt.scatter(altitude / 1000.0,error_j2000,s=3,alpha=0.25    )
    # plt.xlabel("Altitude [km]")
    # plt.ylabel("|QRW - R| [m]")
    # plt.title("Transformation Error vs Altitude_J2000")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # ------------------------------------------------------------
    # XYZ error distributions
    # ------------------------------------------------------------
    # fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # labels = ["ΔX [m]", "ΔY [m]", "ΔZ [m]"]

    # for i in range(3):
    #     ax[i].hist(diff[:, i], bins=100)
    #     ax[i].set_xlabel(labels[i])
    #     ax[i].set_ylabel("Count")
    #     ax[i].grid(True, alpha=0.3)

    # fig.suptitle("Component Errors: Full QRW − R-only")
    # plt.tight_layout()
    
    # fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    # labels = ["ΔX [m]", "ΔY [m]", "ΔZ [m]"]

    # for i in range(3):
    #     ax[i].hist(diff_j2000[:, i], bins=100)
    #     ax[i].set_xlabel(labels[i])
    #     ax[i].set_ylabel("Count")
    #     ax[i].grid(True, alpha=0.3)
    
    # fig.suptitle("Component Errors: Full QRW − R-only")
    # plt.tight_layout()

    # plt.show()

#%%

import math
import numpy as np
import erfa
import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
ERA0 = 0.7790572732640
ERA_RATE = 1.00273781191135448
TWO_PI = 2.0 * math.pi
DJ00 = 2451545.0
Re = 6378137.0

# ============================================================
# ERA implementations
# ============================================================
def era_full(jd):
    """Original calculation using one float64 JD."""
    theta = TWO_PI * (ERA0 + ERA_RATE * (jd - DJ00))
    return theta % TWO_PI

def era_smart(jd):
    """Split elapsed days into integer + fractional part."""
    D = jd - DJ00
    N = math.floor(D)
    f = D - N

    x = ERA0 + ERA_RATE * N + ERA_RATE * f
    return (TWO_PI * x) % TWO_PI

def era_erfa(jd1, jd2):
    """ERFA/SOFA reference."""
    return erfa.era00(jd1, jd2)

def angle_error(a, b):
    """Smallest signed angular difference."""
    return (a - b + math.pi) % TWO_PI - math.pi


# ============================================================
# GPS -> Astropy Time
# ============================================================
def gps_seconds_now():
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - epoch).total_seconds() + 18.0


def gps_to_time(gps_week, gps_tow):
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    return Time(epoch + timedelta(seconds=gps_week * 604800 + gps_tow + 19.0), scale="tai")


# ============================================================
# Rotation
# ============================================================
def R3(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


# ============================================================
# Main Monte Carlo
# ============================================================
if __name__ == "__main__":

    N = 10000
    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Current GPS epoch
    # --------------------------------------------------------
    gps_now = gps_seconds_now()

    # Random epochs spanning 2000 -> 2200.
    # This makes the numerical behavior of the JD calculation
    # visible over a much larger absolute-JD range.
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    t1 = Time("2200-01-01T00:00:00", scale="utc")

    jd0 = t0.jd
    jd1 = t1.jd

    jd_utc = rng.uniform(jd0, jd1, N)

    # Use Astropy to obtain proper UT1 for each epoch.
    t = Time(jd_utc, format="jd", scale="utc")

    dut1 = t.get_delta_ut1_utc().to_value(u.s)

    # Convert UTC JD -> UT1 JD.
    jd_ut1 = t.ut1.jd

    # Also keep ERFA's two-part representation.
    ut1_jd1 = np.asarray(t.ut1.jd1)
    ut1_jd2 = np.asarray(t.ut1.jd2)

    # --------------------------------------------------------
    # ERA calculations
    # --------------------------------------------------------
    era_full_v = np.empty(N)
    era_smart_v = np.empty(N)
    era_erfa_v = np.empty(N)

    for i in range(N):
        era_full_v[i] = era_full(jd_ut1[i])
        era_smart_v[i] = era_smart(jd_ut1[i])
        era_erfa_v[i] = era_erfa(ut1_jd1[i],ut1_jd2[i])

    # --------------------------------------------------------
    # ERA errors
    # --------------------------------------------------------
    err_full = np.array([angle_error(a, b)for a, b in zip(era_full_v, era_erfa_v)])
    err_smart = np.array([angle_error(a, b)for a, b in zip(era_smart_v, era_erfa_v) ])

    # --------------------------------------------------------
    # Random satellite positions
    # --------------------------------------------------------
    direction = rng.normal(size=(N, 3))
    direction /= np.linalg.norm( direction, axis=1)[:, None]

    altitude = rng.uniform(400e3,10000e3,N)
    radius = Re + altitude

    # Position error caused ONLY by ERA error.
    pos_err_full = np.abs(err_full) * radius
    pos_err_smart = np.abs(err_smart) * radius

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    def stats(name, err_rad, err_m):
        print(f"\n{name}")
        print("-" * 50)
        print(    f"ERA max       = "    f"{np.max(np.abs(err_rad)):.6e} rad")
        print(f"ERA RMS       = "f"{np.sqrt(np.mean(err_rad**2)):.6e} rad")
        print(f"ERA max       = "f"{np.degrees(np.max(np.abs(err_rad)))*3600:.9f} arcsec"        )
        print(f"Position max  = "f"{np.max(err_m)*1000:.9f} mm"        )
        print(f"Position RMS  = "f"{np.sqrt(np.mean(err_m**2))*1000:.9f} mm"        )
        print(f"Position mean = "f"{np.mean(err_m)*1000:.9f} mm"        )
        print(f"Position P99  = "f"{np.percentile(err_m,99)*1000:.9f} mm"        )


    print("\n==============================================")
    print("ERA NUMERICAL PRECISION MONTE CARLO")
    print("==============================================")
    print(f"N             = {N}")
    print("Epoch range    = 2026-01-01 -> 2200-01-01")
    print(f"Altitude      = 400 km -> 10000 km")

    stats("FULL JD vs ERFA",err_full,pos_err_full    )

    stats("SMART JD vs ERFA",err_smart,pos_err_smart    )

    # --------------------------------------------------------
    # Direct smart-vs-full comparison
    # --------------------------------------------------------
    err_smart_full = np.array([angle_error(a, b)
        for a, b in zip(era_smart_v,era_full_v)])

    print("\nSMART vs FULL")
    print("-" * 50)
    print(f"Max ERA difference = "f"{np.max(np.abs(err_smart_full)):.6e} rad"    )

    # --------------------------------------------------------
    # Improvement factor
    # --------------------------------------------------------
    rms_full = np.sqrt(np.mean(err_full**2))
    rms_smart = np.sqrt(np.mean(err_smart**2))

    max_full = np.max(np.abs(err_full))
    max_smart = np.max(np.abs(err_smart))

    print("\n==============================================")
    print("IMPROVEMENT")
    print("==============================================")

    print(f"RMS improvement = "f"{rms_full / rms_smart:.3f} x")
    print(f"Max improvement = "f"{max_full / max_smart:.3f} x"    )

    # --------------------------------------------------------
    # Plot ERA error vs epoch
    # --------------------------------------------------------
    years = 2026.0 + (jd_ut1 - jd_ut1.min()) / 365.25

    plt.figure(figsize=(10, 5))
    plt.scatter(years,np.abs(err_full) * 1e9,s=4,alpha=0.3,label="Full JD"    )
    plt.scatter(years,np.abs(err_smart) * 1e9,s=4,alpha=0.3,label="Smart JD"    )
    plt.yscale("log")
    plt.xlabel("Epoch [year]")
    plt.ylabel("|ERA error| [nrad]")
    plt.title("ERA Numerical Error vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Position error comparison
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.hist(pos_err_full * 1000,bins=100,alpha=0.6,label="Full JD"    )

    plt.hist(pos_err_smart * 1000,bins=100,alpha=0.6,label="Smart JD"    )

    plt.xlabel("Position error [mm]")
    plt.ylabel("Count")
    plt.title("Position Error Caused by ERA Floating-Point Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --------------------------------------------------------
    # Error ratio
    # --------------------------------------------------------
    ratio = (np.abs(err_full) /np.maximum(np.abs(err_smart), 1e-30)    )
    plt.figure(figsize=(10, 5))
    plt.scatter(years,ratio,s=4,alpha=0.3    )

    plt.yscale("log")
    plt.xlabel("Epoch [year]")
    plt.ylabel("|Full error| / |Smart error|")
    plt.title("Numerical Improvement: Full JD vs Smart JD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
# %%
