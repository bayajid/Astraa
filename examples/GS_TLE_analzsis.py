from skyfield.api import EarthSatellite, Topos, load
import numpy as np

# Load timescale
ts = load.timescale()

# TLE lines
tle_line1 = "1 58299U 23174AV  26128.14824074  .00000000  00000-0 -20133-1 0    03"
tle_line2 = "2 58299  97.3891 211.3621 0005459 295.6990  28.9975 15.34104634    02"

# Create satellite object
satellite = EarthSatellite(tle_line1, tle_line2, "Satellite", ts)

# Ground station location (Buffalo, NY)
latitude_deg = 43.194896
longitude_deg = -76.422318
altitude_m = 0.176551 * 1000  # convert km to m

gs = Topos(latitude_degrees=latitude_deg,
           longitude_degrees=longitude_deg,
           elevation_m=altitude_m)

# Time array: next 10 minutes, 1-second intervals
t0 = ts.now()
times = ts.utc(t0.utc_datetime().year, t0.utc_datetime().month, t0.utc_datetime().day,
               t0.utc_datetime().hour, t0.utc_datetime().minute,
               np.arange(t0.utc_datetime().second, t0.utc_datetime().second+600, 1))

# Compute satellite position relative to ground station
difference = satellite - gs
topocentric = difference.at(times)

# Correct way: get alt/az from .altaz()
alt, az, distance = topocentric.altaz()

# Convert to radians for rate calculations
az_rad = az.radians
alt_rad = alt.radians

# Time step (1 second)
dt = 1.0

# Compute gimbal angular rates (rad/s)
az_rate = np.gradient(az_rad, dt)
alt_rate = np.gradient(alt_rad, dt)

# Convert to degrees per second
az_rate_deg = np.degrees(az_rate)
alt_rate_deg = np.degrees(alt_rate)

# Max angular rates required
print(f"Max azimuth rate [deg/s]: {np.max(np.abs(az_rate_deg)):.2f}")
print(f"Max elevation rate [deg/s]: {np.max(np.abs(alt_rate_deg)):.2f}")