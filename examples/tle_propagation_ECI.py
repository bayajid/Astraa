# tle_propagation_ECI_with_gps_save.py
import numpy as np
import pandas as pd
from sgp4.api import Satrec
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm



def propagate_tle_to_gcrs_gps(t_start_iso: str, duration_s: float = 7200, step_s: int = 300):
    """
    Propagates ISS from t_start_iso (UTC) and returns DataFrame with GPS seconds.
    """
    # Reference GPS time at J2000 (2000-01-01 12:00:00 TAI = GPS epoch + 19s leap)
    # But astropy handles GPS scale correctly
    t_ref_gps = Time('2000-01-01T12:00:00', scale='tai')  # GPS zero point

    # Propagation times
    t0_utc = Time(t_start_iso, scale='utc')
    times_utc = t0_utc + np.arange(0, duration_s + step_s, step_s) * u.s

    # GPS time in seconds since GPS epoch
    times_gps = times_utc.gps    # This is the correct GPS seconds (float)
    t_gps_0 = times_gps[0]       # First epoch in GPS seconds

    N = len(times_utc)
    r_gcrs = np.zeros((N, 3))
    v_gcrs = np.zeros((N, 3))

    print(f"Propagating {N} points from {t0_utc.iso} (GPS week second {t_gps_0:.1f})")

    for i, t in tqdm (enumerate(times_utc),total=len(times_utc), desc="Propagating TLE → GCRS(ECI)"):
        
        err, r_teme, v_teme = sat.sgp4(t.jd1, t.jd2)    # Propagating TLE in TEME frame
        if err != 0:
            raise RuntimeError(f"SGP4 propagation error {err} at time {t.iso}")
        pos = CartesianRepresentation(r_teme * u.km)
        vel = CartesianDifferential(v_teme * u.km / u.s)
        teme_state = TEME(pos.with_differentials(vel), obstime=t)

        gcrs_state = teme_state.transform_to(GCRS(obstime=t, obsgeoloc=[0, 0, 0]*u.m))

        r_gcrs[i] = gcrs_state.cartesian.xyz.to(u.km).value
        v_gcrs[i] = gcrs_state.velocity.d_xyz.to(u.km/u.s).value

    # Build final array: [t_gps_s, rx, ry, rz, vx, vy, vz]
    propagated_orbit_eci = np.column_stack([
        times_gps,      # GPS seconds (float)
        r_gcrs,         # km
        v_gcrs          # km/s
    ])

    # Create DataFrame
    df = pd.DataFrame(
        data=propagated_orbit_eci,
        columns=['t_gps_s', 'r_x', 'r_y', 'r_z', 'v_x', 'v_y', 'v_z']
    )

    return df

# =========================== RUN & SAVE ===========================
if __name__ == "__main__":
    # Your desired propagation
    # Real ISS TLE — valid as of 26 Nov 2024
    line1 = "1 25544U 98067A   24331.60256481  .00001287  00000-0  27841-4 0  9995"
    line2 = "2 25544  51.6416 176.9512 0007419  68.3456 291.7892 15.48901234 12349"

    sat = Satrec.twoline2rv(line1, line2)

    t0_utc = Time.now()
    df = propagate_tle_to_gcrs_gps(
        t_start_iso=t0_utc.iso, 
        duration_s=2*3600,      # seconds
        step_s=5                # seconds
    )

    # Quick sanity check
    speeds = np.sqrt((df[['v_x','v_y','v_z']]**2).sum(axis=1))
    altitudes = np.sqrt((df[['r_x','r_y','r_z']]**2).sum(axis=1)) - 6378.137

    print("\nSanity check:")
    print(f"Mean speed   : {speeds.mean():.5f} km/s ")
    print(f"Mean altitude: {altitudes.mean():.0f} km")
    print(f"Data shape   : {df.shape}")
    print(f"First GPS time: {df['t_gps_s'].iloc[0]:.3f} s")
    print(f"Last GPS time : {df['t_gps_s'].iloc[-1]:.3f} s")

    # Save to files
    df.to_csv(f"propagated_orbit_{t0_utc}.csv", index=False)
    print(f"\nSaved to: propagated_orbit_{t0_utc}.csv")

    