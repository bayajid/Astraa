#%%
import numpy as np
import matplotlib.pyplot as plt
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime
from datetime import datetime, timedelta
from astropy.time import Time, TimeDelta
import astropy.units as u 
from skyfield.api import load, wgs84


# Example TLE for ISS (replace with real one if you want)
tle1 = (
    "1 25544U 98067A   20264.59097222  .00001264  00000-0  29621-4 0  9993",
    "2 25544  51.6432  21.2054 0001472 105.6180  32.4986 15.49106529243553"
)
sat = Satrec.twoline2rv(*tle1)
eph = load('de421.bsp')
sun, moon, earth = eph['sun'], eph['moon'], eph['earth']


ts = load.timescale()
station = wgs84.latlon(48.0, 11.0)  # example: Munich
# Munich location (hardcoded for now, can be made configurable later)
munich = eph['earth'] + wgs84.latlon(+48.13743, +11.57549)


times = ts.utc(2025, 9, 26, range(0,24*60))  # sample every minute
e = munich.at(times)
# alt, az, distance = (sat - munich).at(times).altaz()
alt, az, distance = (sat - munich).at(times).altaz()
zenith_index = alt.degrees.argmax()
zenith_time = times[zenith_index].utc_datetime()

# Define "zenith epoch" (TLE epoch shifted to zenith)
#zenith_epoch = sat_epoch_datetime(sat)  # datetime of TLE epoch
#zenith_time = Time(zenith_epoch)

# For comparison, pretend we only had a TLE 12h earlier
early_time = zenith_time - 0.5  # 0.5 days = 12h

# Propagate positions +/- 1h around zenith
dt_range = np.linspace(-1, 1, 201)  # hours
errors_zenith_epoch = []
errors_early_epoch = []

for dt in dt_range:
    t = zenith_time + dt * u.hour
    jd, fr = t.jd, t.jd % 1.0

    # True position: we'll take "zenith epoch" propagation as the reference
    e, r_true, v_true = sat.sgp4(jd, fr)

    # Propagate with TLE at zenith (perfectly aligned)
    e, r_zen, v_zen = sat.sgp4(jd, fr)
    # Along-track error is ~0 (this is reference)

    # Propagate with TLE shifted by 12h (simulate stale TLE)
    sat_shifted = Satrec.twoline2rv(*tle1)
    sat_shifted.jdsatepoch = early_time.jd  # fudge: move epoch back 12h
    e, r_early, v_early = sat_shifted.sgp4(jd, fr)

    # Compute error magnitude (km)
    err_zen = np.linalg.norm(np.array(r_zen) - np.array(r_true))
    err_early = np.linalg.norm(np.array(r_early) - np.array(r_true))

    errors_zenith_epoch.append(err_zen)
    errors_early_epoch.append(err_early)

# Plot
plt.figure(figsize=(8,5))
plt.plot(dt_range, errors_zenith_epoch, label="Epoch at Zenith (ideal)")
plt.plot(dt_range, errors_early_epoch, label="Epoch 12h earlier", linestyle="--")
plt.xlabel("Time from zenith (hours)")
plt.ylabel("Position error (km)")
plt.title("SGP4 position error vs. time from epoch")
plt.legend()
plt.grid(True)
plt.show()

# %%
