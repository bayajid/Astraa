from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time

# Your function
def ground_station_eci(lat_deg, lon_deg, alt_m, time):
    """
    Returns position and velocity of ground station in GCRS (ECI)
    """
    gs_location = EarthLocation(lat=lat_deg * u.deg,
                                lon=lon_deg * u.deg,
                                height=alt_m * u.m
                            )
    gs_gcrs = gs_location.get_gcrs(obstime=time)

    r_eci = gs_gcrs.cartesian.xyz.to(u.m).value
    v_eci = gs_gcrs.velocity.d_xyz.to(u.m/u.s).value

    return r_eci, v_eci

# The if __name__ == "__main__": block
if __name__ == "__main__":
    # Example test values
    lat_deg = 40.7128  # Example latitude (New York)
    lon_deg = -74.0060  # Example longitude (New York)
    alt_m = 10  # Example altitude (10 meters)
    
    # Example time (current UTC time)
    time = Time.now()

    # Call the function
    position, velocity = ground_station_eci(lat_deg, lon_deg, alt_m, time)

    # Print the results
    print("Position (ECI):", position)
    print("Velocity (ECI):", velocity)

