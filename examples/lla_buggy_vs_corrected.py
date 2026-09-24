#%%
"""
coord_pos_lla: buggy (pure XML translation) vs corrected, for direct comparison.

BUGGY version = exactly what system_15624 computes, confirmed against raw
XML, with ALL THREE confirmed issues left in place:
  Bug 1: Latitude/Longitude used directly in sin()/cos() -- no
         DegreesToRadians block found anywhere in the chain, despite the
         station constants being stored in degrees.
  Bug 2: Earth Rotation Angle computed as Mod(frac, 2*pi) where frac is a
         dimensionless ROTATION COUNT, not a fractional rotation. Correct
         ERA requires Mod(frac, 1) FIRST, then x2*pi. Confirmed via raw
         XML: Constant7 (SID 15634, the Mod block's modulus input) =
         "2*pi", wired straight into Sin/Cos with no intervening
         x2*pi conversion.
  Bug 3: epochTime is GPS seconds (seconds since the GPS epoch,
         1980-01-06). Sun's and Moon's time chains both correctly convert
         this to days-since-J2000 via
             days = epochTime/86400 + julianDateGps - julianDateForCenturies
         (julianDateGps = 2444244.5, JD of the GPS epoch; julianDateForCenturies
         = 2451545.0, JD of J2000). system_15624's own JDUT1 (Divide1)
         computes ONLY epochTime/86400 -- confirmed by symbol-tracing:
         julianDateGps and julianDateForCenturies do not appear anywhere
         in the LLA chain's free symbols, and system_15176 (the LLA
         wrapper) passes its epochTime Inport straight through unmodified
         -- same raw top-level signal Sun/Moon receive. This leaves the
         LLA subsystem's day-count short by exactly
         (julianDateForCenturies - julianDateGps) = 7300.5 days relative
         to what Sun/Moon compute from the same input.

CORRECTED version = same algorithm, all three issues fixed, nothing else
changed.
"""
import math
from dataclasses import dataclass
from astropy import units as u
from astropy.coordinates import EarthLocation, ITRS, GCRS, CIRS, FK5
from astropy.time import Time


@dataclass
class GroundStation:
    latitude: float
    longitude: float
    altitude_km: float


MUNICH = GroundStation(latitude=48.137827, longitude=11.418762, altitude_km=0.536)
LONG_BEACH = GroundStation(latitude=33.913385, longitude=-118.334742, altitude_km=10)
MAHIA = GroundStation(latitude=-39.261410, longitude=177.865325, altitude_km=30)

def coord_pos_lla_astropy(epochTime, station, return_ECEF=None):

    Altitude = station.altitude_km * 1e3 # in meters
    Latitude = station.latitude     # FIX 1
    Longitude = station.longitude   # FIX 1

    loc = EarthLocation.from_geodetic(
    lon=Longitude * u.deg,
    lat=Latitude * u.deg,
    height=Altitude * u.m,
    ellipsoid="WGS84",
    )
    itrs = loc.get_itrs(obstime=epochTime)


    if return_ECEF:
        result_x, result_y, result_z = itrs.cartesian.xyz.to_value(u.m)
    else:
        # 2. Transform to Inertial frame (ECI / GCRS) at observation time    
        # gcrs_pos = itrs.transform_to(GCRS(obstime=epochTime)) 
        # cirs_pos = itrs.transform_to(CIRS(obstime=epochTime)) 
        j2000_pos = loc.get_gcrs(obstime=epochTime)  # J2000 ECI    

        # result_x, result_y, result_z = itrs.cartesian.xyz.to_value(u.m)
        # result_x, result_y, result_z = gcrs_pos.cartesian.xyz.to_value(u.m)
        # result_x, result_y, result_z = cirs_pos.cartesian.xyz.to_value(u.m)
        result_x, result_y, result_z = j2000_pos.cartesian.xyz.to_value(u.m)

    return (result_x*1e-3, result_y*1e-3, result_z*1e-3)

# ======================================================================
# BUGGY (pure XML translation -- exactly what system_15624 computes)
# ======================================================================
def coord_pos_lla_buggy(epochTime, station, GpsToTerrestrialTime, UtcToGpsTime,
                         earthFlattening, earthRadius):
    """Literal translation of system_15624, outport 'coordPos_E'.
    All four confirmed issues left exactly as wired. epochTime: GPS
    seconds. Returns ECI-frame (x, y, z)."""
    Altitude = station.altitude_km
    Latitude = station.latitude      # BUG 1: still in degrees here
    Longitude = station.longitude    # BUG 1: still in degrees here

    x0 = math.sin(Longitude)
    # BUGs 2+3+4 combined, exactly as the model computes them:
    x1 = (0.0000116057617119369731*GpsToTerrestrialTime
          - 0.0000116057617119369731*UtcToGpsTime
          + 0.0000116057617119369731*epochTime
          + 0.779057273263999983) % (2*math.pi)
    x2 = math.sin(x1)
    x3 = math.cos(Longitude)
    x4 = math.cos(x1)
    x5 = math.sin(Latitude)
    x6 = math.sqrt(-earthFlattening*x5**2*(2 - earthFlattening) + 1)
    x7 = (Altitude + earthRadius/x6)*math.cos(Latitude)
    result_x = x7*(x0*x2 + x3*x4)
    result_y = x7*(x0*x4 - x2*x3)
    result_z = x5*(Altitude + earthRadius*x6)
    return (result_x, result_y, result_z)

# ======================================================================
# CORRECTED ECEF(same algorithm, all four issues fixed)
# ======================================================================
def coord_pos_lla_corrected_ecef(epochTime, station, GpsToTerrestrialTime,UtcToGpsTime,
                             earthFlattening, earthRadius,
                             dut1=-0.045,
                             julianDateGps=2444244.5, julianDateForCenturies=2451545.0):
    """Same algorithm as coord_pos_lla_buggy, with:
    Fix 1: Latitude/Longitude converted degrees -> radians before use.
    Fix 2: ERA correctly computed as (frac mod 1) * 2*pi, instead of
           frac mod 2*pi.
    Fix 3: epochTime (GPS seconds) correctly referenced to J2000 via
           julianDateGps/julianDateForCenturies.
    Fix 4: Tu computed in UT1, not TT -- GPS -> UTC (subtract
           UtcToGpsTime) -> UT1 (add dut1). GpsToTerrestrialTime (the
           TT-GPS offset) does NOT belong in this chain at all; ERA has
           no TT dependence. dut1 (UT1-UTC, from IERS Bulletin A)
           defaults to 0.0 -- per the user's own note, omitting it
           introduces < 0.9s of error, negligible for most applications
           but should be supplied for high-precision comparison against
           a real Simulink/GPS-truth run.
    Returns ECI-frame (x, y, z)."""
    Altitude = station.altitude_km
    Latitude = math.radians(station.latitude)     # FIX 1
    Longitude = math.radians(station.longitude)   # FIX 1

    x0 = math.sin(Longitude)

    # FIX 3 + FIX 4: GPS -> UTC -> UT1 -> days since J2000 UT1 (Tu)
    Tu = ((epochTime - UtcToGpsTime + dut1) / 86400.0
          + julianDateGps - julianDateForCenturies)
    frac = 0.7790572732640 + 1.00273781191135448 * Tu
    # FIX 2: reduce to fractional rotation FIRST, then convert to radians
    era_fraction = frac % 1.0
    x1 = era_fraction * 2 * math.pi

    x2 = math.sin(x1)
    x3 = math.cos(Longitude)
    x4 = math.cos(x1)
    x5 = math.sin(Latitude)
    x6 = math.sqrt(-earthFlattening*x5**2*(2 - earthFlattening) + 1)
    x7 = (Altitude + earthRadius/x6)*math.cos(Latitude)

    N = earthRadius / x6
    e2 = earthFlattening * (2.0 - earthFlattening)

    result_x = x7*x3# (x0*x2 + x3*x4)
    result_y = x7* x0#(x0*x4 - x2*x3)
    result_z = x5*(Altitude +  N * (1.0 - e2))
    return (result_x, result_y, result_z)

     
# ======================================================================
# CORRECTED ECI(same algorithm, all three issues fixed)
# ======================================================================
def coord_pos_lla_corrected_eci(epochTime, station, GpsToTerrestrialTime, UtcToGpsTime,
                             earthFlattening, earthRadius,
                             julianDateGps=2444244.5, julianDateForCenturies=2451545.0):
    """Same algorithm as coord_pos_lla_buggy, with:
    Fix 1: Latitude/Longitude converted degrees -> radians before use.
    Fix 2: ERA correctly computed as (frac mod 1) * 2*pi, instead of
           frac mod 2*pi.
    Fix 3: epochTime (GPS seconds) correctly referenced to J2000 via
           julianDateGps/julianDateForCenturies, matching the conversion
           Sun/Moon's own time chains already use.
    Returns ECI-frame (x, y, z)."""
    Altitude = station.altitude_km
    Latitude = math.radians(station.latitude)     # FIX 1
    Longitude = math.radians(station.longitude)   # FIX 1

    # 1. Earth Rotation Angle (ERA)
    dut1 = -0.0450  # Seconds (IERS Bulletin A)
    Tu = ((epochTime - UtcToGpsTime + dut1) / 86400.0
        + julianDateGps - julianDateForCenturies)
    # era = (2.0 * math.pi * (0.7790572732640 + 1.00273781191135448 * Tu)) % (2.0 * math.pi)

    frac = 0.7790572732640 + 1.00273781191135448 * Tu
    # FIX 2: reduce to fractional rotation FIRST, then convert to radians
    era_fraction = frac % 1.0
    era = era_fraction * 2 * math.pi

    # 2. Ellipsoid Parameters
    e2 = earthFlattening * (2.0 - earthFlattening)
    sin_lat = math.sin(Latitude)
    cos_lat = math.cos(Latitude)
    x6 = math.sqrt(1.0 - e2 * sin_lat**2)
    N = earthRadius / x6

    # 3. Trigonometric terms
    sin_lon = math.sin(Longitude)
    cos_lon = math.cos(Longitude)
    sin_era = math.sin(era)
    cos_era = math.cos(era)

    # 4. Corrected ECI Coordinates
    x7 = (Altitude + N) * cos_lat

    result_x = x7 * (cos_lon * cos_era - sin_lon * sin_era)
    result_y = x7 * (sin_lon * cos_era + cos_lon * sin_era)
    result_z = sin_lat * (Altitude + N * (1.0 - e2))

    return (result_x, result_y, result_z)

    ###---
    x, y, s = erfa.xys06a(tt_jd1, tt_jd2)

    # Apply celestial pole offsets if provided
    x = x + dx
    y = y + dy

    # GCRS-to-CIRS matrix from X, Y, s
    # This is Q_c2i in the SOFA convention
    rc2i = erfa.c2ixys(x, y, s)

    # ------------------------------------------------------------------
    # Step 2: Earth Rotation Angle (ERA) — TIRS rotation
    #         R = R₃(ERA) rotates CIRS → TIRS
    # ------------------------------------------------------------------
    era = erfa.era00(ut1_jd1, ut1_jd2)

    # ------------------------------------------------------------------
    # Step 3: Polar motion matrix W (TIRS → ITRS)
    #         Uses TIO locator s' (sp)
    # ------------------------------------------------------------------
    sp = erfa.sp00(tt_jd1, tt_jd2)   # TIO locator s'
    rpom = erfa.pom00(xp, yp, sp)    # TIRS-to-ITRS matrix

    ##----

    
if __name__ == "__main__":
    common = dict(GpsToTerrestrialTime=51.184,
                  UtcToGpsTime=18.0,
                  earthFlattening=0.0033528106647474805, 
                  earthRadius=6378.137)
    now = Time.now()
    epochTime = now.gps # GPS seconds
    print(f"\nTime: {now} UTC\t GPSTime: {epochTime}")

    frame_ECEF = 1

    if frame_ECEF:
        print("\nECEF FRAME")
    else:
        print("\nECI FRAME")


    for name, station in [("MUNICH", MUNICH), ("LONG_BEACH", LONG_BEACH), ("MAHIA", MAHIA)]:
        buggy = coord_pos_lla_buggy(epochTime, station, **common)        
        astropy =  coord_pos_lla_astropy(now, station,return_ECEF=frame_ECEF)
        if frame_ECEF:
            fixed = coord_pos_lla_corrected_ecef(epochTime, station, **common)
        else:
            fixed = coord_pos_lla_corrected_eci(epochTime, station, **common)
        diff_b = [abs(a - b) for a, b in zip(astropy, buggy )]
        diff_f = [abs(a - b) for a, b in zip(astropy, fixed)]
        r_buggy = math.sqrt(sum(v**2 for v in buggy))
        r_fixed = math.sqrt(sum(v**2 for v in fixed))
        r_astropy =math.sqrt(sum(v**2 for v in astropy))
        print(f"=== {name} ===")
        print("  buggy      :", buggy, " |r| =", r_buggy, "km")
        print("  fixed      :", fixed, " |r| =", r_fixed, "km")
        print("  astropy    :", astropy, " |r| =", r_astropy, "km")
        print("  diff_buggy (km):", diff_b, " max =", max(diff_b))
        print("  diff_fixed (km):", diff_f, " max =", max(diff_f))
        print()