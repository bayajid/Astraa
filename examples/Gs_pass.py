"""
Simple Ground Station to Satellite Tracking Simulation
"""

from skyfield.api import load, wgs84, EarthSatellite
import requests

class GroundStationTracker:
    def __init__(self, lat, lon, elevation=0, name="GS"):
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.name = name
        self.ts = load.timescale()
        self.ground_station = wgs84.latlon(lat, lon, elevation)
    
    def get_satellite_tle(self, norad_id):
        """Fetch satellite TLE by NORAD ID"""
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=tle"
        response = requests.get(url, timeout=10)
        lines = response.text.strip().split('\n')
        
        if len(lines) >= 3:
            return lines[0].strip(), lines[1].strip(), lines[2].strip()
        return None
    
    def get_current_position(self, satellite):
        """Get current satellite position"""
        t = self.ts.now()
        
        # Satellite geocentric position
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        
        # Position relative to ground station
        difference = satellite - self.ground_station
        topocentric = difference.at(t)
        alt, az, distance = topocentric.altaz()
        
        # Velocity magnitude
        vel = geocentric.velocity.km_per_s
        velocity_mag = (vel[0]**2 + vel[1]**2 + vel[2]**2)**0.5
        
        return {
            'latitude': subpoint.latitude.degrees,
            'longitude': subpoint.longitude.degrees,
            'altitude_km': subpoint.elevation.km,
            'azimuth': az.degrees,
            'elevation': alt.degrees,
            'range_km': distance.km,
            'velocity_km_s': velocity_mag,
            'visible': alt.degrees > 0
        }
    
    def get_passes(self, satellite, hours=24):
        """Get satellite passes over ground station"""
        t0 = self.ts.now()
        t1 = self.ts.tt_jd(t0.tt + hours/24.0)
        
        t, events = satellite.find_events(self.ground_station, t0, t1, altitude_degrees=0.0)
        
        passes = []
        current_pass = {}
        
        for ti, event in zip(t, events):
            if event == 0:  # Rise
                current_pass = {'rise_time': ti.utc_iso()}
            elif event == 1:  # Max elevation
                difference = satellite - self.ground_station
                topocentric = difference.at(ti)
                alt, az, dist = topocentric.altaz()
                
                current_pass['max_elevation'] = alt.degrees
                current_pass['azimuth'] = az.degrees
                current_pass['max_time'] = ti.utc_iso()
                current_pass['range_km'] = dist.km
            elif event == 2:  # Set
                if 'max_elevation' in current_pass:
                    current_pass['set_time'] = ti.utc_iso()
                    passes.append(current_pass)
                current_pass = {}
        
        return passes
    
    def calculate_doppler(self, satellite, freq_mhz):
        """Calculate Doppler shift"""
        t0 = self.ts.now()
        t1 = self.ts.tt_jd(t0.tt + 1.0/86400.0)  # 1 second later
        
        difference = satellite - self.ground_station
        
        _, _, d0 = difference.at(t0).altaz()
        _, _, d1 = difference.at(t1).altaz()
        
        range_rate = d1.km - d0.km  # km/s
        c = 299792.458  # speed of light km/s
        
        doppler_shift = freq_mhz * (range_rate / c)
        
        return {
            'doppler_shift_khz': doppler_shift * 1000,
            'adjusted_freq_mhz': freq_mhz + doppler_shift
        }


# Example usage
if __name__ == "__main__":
    # Create ground station (Munich, Germany)
    gs = GroundStationTracker(lat=48.1351, lon=11.5820, elevation=519, name="Munich")
    
    # Get ISS TLE
    print("Fetching ISS data...")
    tle_data = gs.get_satellite_tle(25544)  # ISS
    
    if tle_data:
        name, line1, line2 = tle_data
        iss = EarthSatellite(line1, line2, name, gs.ts)
        
        print(f"\nTracking: {name}")
        print(f"Ground Station: {gs.name} ({gs.lat}°, {gs.lon}°)\n")
        
        # Current position
        print("=== Current Position ===")
        pos = gs.get_current_position(iss)
        print(f"Satellite: {pos['latitude']:.4f}°, {pos['longitude']:.4f}°")
        print(f"Altitude: {pos['altitude_km']:.2f} km")
        print(f"Azimuth: {pos['azimuth']:.2f}°")
        print(f"Elevation: {pos['elevation']:.2f}°")
        print(f"Range: {pos['range_km']:.2f} km")
        print(f"Velocity: {pos['velocity_km_s']:.2f} km/s")
        print(f"Visible: {pos['visible']}")
        
        # Doppler shift
        print("\n=== Doppler Shift (145.800 MHz) ===")
        doppler = gs.calculate_doppler(iss, 145.800)
        print(f"Shift: {doppler['doppler_shift_khz']:.3f} kHz")
        print(f"RX Frequency: {doppler['adjusted_freq_mhz']:.6f} MHz")
        
        # Next passes
        print("\n=== Next Passes (24 hours) ===")
        passes = gs.get_passes(iss, hours=24)
        for i, p in enumerate(passes[:5], 1):
            print(f"\nPass {i}:")
            print(f"  Rise: {p['rise_time']}")
            print(f"  Max: {p['max_elevation']:.1f}° at {p['max_time']}")
            print(f"  Set: {p['set_time']}")
            print(f"  Range: {p['range_km']:.0f} km")
    else:
        print("Failed to fetch satellite data")