# # Code to produce plots and numeric table for gimbal elevation measured from local HORIZON
# # Assumption: gimbal (0 deg elevation) = local horizon. Positive elevation is upwards, negative is downwards.
# # For LOS between co-planar equal-altitude satellites the elevation simplifies to E = -Delta/2 (degrees).
# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# import math
# import pandas as pd

# Re = 6371.0  # Earth radius [km]
# altitudes = [400, 800, 2000]  # km
# deltas = np.linspace(0.0, 180.0, 1801)  # degrees
# deltas_rad = np.radians(deltas)

# results = []

# # Plot elevation vs along-track separation (elevation measured from local horizon)
# plt.figure(figsize=(10,6))
# for h in altitudes:
#     Rs = Re + h
#     # analytic elevation in degrees
#     elev_deg = 0.5 * deltas  # E = -Delta/2 (deg)
#     # visibility mask
#     delta_max_rad = 2 * math.acos(Re / Rs) if Rs > Re else 0.0
#     delta_max_deg = math.degrees(delta_max_rad)
#     visible = deltas <= delta_max_deg
#     # plt.plot(deltas, elev_deg, linestyle = 'dashdot',alpha = 0.5,label=f'{h} km (all Δ)')
#     # highlight visible region
#     plt.plot(deltas[visible], elev_deg[visible], linestyle = 'dotted',alpha = 0.5)
#     # collect numeric results
#     max_downward = -np.min(elev_deg[visible]) if visible.any() else 0.0  # positive downward angle required
#     results.append({
#         'altitude_km': h,
#         'R_s_km': Rs,
#         'delta_max_deg': round(delta_max_deg, 3),
#         'max_downward_tilt_deg': round(max_downward, 3)
#     })

# plt.axhline(0, linewidth=1)
# plt.xlabel('Along-track separation Δ (deg)')
# plt.ylabel('Elevation from local horizon (deg)')
# plt.title('Required gimbal elevation (0° = local horizon). Negative = point DOWN')
# plt.legend()
# plt.grid(True)
# #plt.xlim(0, 180)
# #plt.ylim(-100, 5)


# # Show table of key numbers
# df_res = pd.DataFrame(results)
# df_res_display = df_res[['altitude_km', 'delta_max_deg', 'max_downward_tilt_deg']]
# df_res_display.index = np.arange(1, len(df_res_display)+1)

# # import caas_jupyter_tools as tools; tools.display_dataframe_to_user("Gimbal Requirements Summary", df_res_display)

# # Convert results into DataFrame and print to console
# df_res = pd.DataFrame(results)
# print("\nGimbal Requirements Summary:")
# print(df_res.to_string(index=False))

# # Also produce a focused plot zoomed to visible region only for clarity
# plt.figure(figsize=(10,5))
# for h in altitudes:
#     Rs = Re + h
#     delta_max_rad = 2 * math.acos(Re / Rs) if Rs > Re else 0.0
#     delta_max_deg = math.degrees(delta_max_rad)
#     mask = deltas <= delta_max_deg
#     elev_deg = 0.5 * deltas
#     plt.plot(deltas[mask], elev_deg[mask], linestyle = '--',alpha = 0.5,label=f'{h} km (visible)')
# plt.axhline(0, linewidth=1)
# plt.xlabel('Along-track separation Δ (deg)')
# plt.ylabel('Elevation from local horizon (deg)')
# plt.title('Visible-region gimbal elevation')
# plt.legend()
# plt.grid(True)
# #plt.ylim(-40, 5)
# plt.xlim(0,  max(df_res_display['delta_max_deg']) + 5)
# plt.show()

# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Constants
# EARTH_RADIUS = 6371.0  # km
# LEO_ALTITUDE = 550.0   # km (typical LEO altitude)
# ORBIT_RADIUS = EARTH_RADIUS + LEO_ALTITUDE

# # Munich coordinates
# MUNICH_LAT = 48.1351  # degrees
# MUNICH_LON = 11.5820  # degrees

# def deg_to_rad(degrees):
#     return degrees * np.pi / 180

# def rad_to_deg(radians):
#     return radians * 180 / np.pi

# def geodetic_to_ecef(lat, lon, alt=0):
#     """Convert geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed)"""
#     lat_rad = deg_to_rad(lat)
#     lon_rad = deg_to_rad(lon)
    
#     x = (EARTH_RADIUS + alt) * np.cos(lat_rad) * np.cos(lon_rad)
#     y = (EARTH_RADIUS + alt) * np.cos(lat_rad) * np.sin(lon_rad)
#     z = (EARTH_RADIUS + alt) * np.sin(lat_rad)
    
#     return np.array([x, y, z])

# def satellite_position(time_hours, inclination=98.0, raan=0.0):
#     """Generate satellite position in ECEF coordinates"""
#     # Orbital period for LEO at 550km altitude (approximately 95.5 minutes)
#     orbital_period = 2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / 398600.4418)  # seconds
#     orbital_period_hours = orbital_period / 3600
    
#     # Mean anomaly
#     mean_anomaly = 2 * np.pi * time_hours / orbital_period_hours
    
#     # Simplified circular orbit (ignoring eccentricity)
#     inclination_rad = deg_to_rad(inclination)
#     raan_rad = deg_to_rad(raan)
    
#     # Position in orbital plane
#     x_orbit = ORBIT_RADIUS * np.cos(mean_anomaly)
#     y_orbit = ORBIT_RADIUS * np.sin(mean_anomaly)
#     z_orbit = 0
    
#     # Rotation matrices for inclination and RAAN
#     # Rotation around x-axis (inclination)
#     R_inc = np.array([
#         [1, 0, 0],
#         [0, np.cos(inclination_rad), -np.sin(inclination_rad)],
#         [0, np.sin(inclination_rad), np.cos(inclination_rad)]
#     ])
    
#     # Rotation around z-axis (RAAN)
#     R_raan = np.array([
#         [np.cos(raan_rad), -np.sin(raan_rad), 0],
#         [np.sin(raan_rad), np.cos(raan_rad), 0],
#         [0, 0, 1]
#     ])
    
#     # Combined rotation
#     R = R_raan @ R_inc
    
#     # Apply rotations
#     pos_orbit = np.array([x_orbit, y_orbit, z_orbit])
#     pos_ecef = R @ pos_orbit
    
#     return pos_ecef

# def calculate_satellite_local_frame(sat_pos):
#     """Calculate satellite's local coordinate frame (nadir, along-track, cross-track)"""
#     # Nadir direction (towards Earth center)
#     nadir = -sat_pos / np.linalg.norm(sat_pos)
    
#     # Cross-track direction (normal to orbital plane, roughly north)
#     # For simplified calculation, assume cross-track is in z-direction
#     cross_track = np.array([0, 0, 1])
#     cross_track = cross_track - np.dot(cross_track, nadir) * nadir
#     cross_track = cross_track / np.linalg.norm(cross_track)
    
#     # Along-track direction (velocity direction)
#     along_track = np.cross(cross_track, nadir)
    
#     return nadir, along_track, cross_track

# def calculate_gimbal_angles(sat_pos, ground_pos):
#     """Calculate gimbal angles needed to point from satellite to ground station"""
#     # Vector from satellite to ground station
#     los_vector = ground_pos - sat_pos
#     los_unit = los_vector / np.linalg.norm(los_vector)
    
#     # Get satellite local frame
#     nadir, along_track, cross_track = calculate_satellite_local_frame(sat_pos)
    
#     # Local horizon plane normal is the nadir direction
#     # Project LOS vector onto the local horizontal plane
#     los_horizontal = los_unit - np.dot(los_unit, nadir) * nadir
    
#     # If the horizontal component is too small, the target is nearly at zenith/nadir
#     if np.linalg.norm(los_horizontal) < 1e-6:
#         azimuth = 0
#         elevation = 90 if np.dot(los_unit, nadir) < 0 else -90
#     else:
#         los_horizontal = los_horizontal / np.linalg.norm(los_horizontal)
        
#         # Elevation angle (angle from horizontal plane)
#         elevation = rad_to_deg(np.arcsin(np.dot(los_unit, -nadir)))
        
#         # Azimuth angle in local frame
#         cos_az = np.dot(los_horizontal, along_track)
#         sin_az = np.dot(los_horizontal, cross_track)
#         azimuth = rad_to_deg(np.arctan2(sin_az, cos_az))
    
#     return elevation, azimuth

# def is_satellite_visible(sat_pos, ground_pos, min_elevation=10.0):
#     """Check if satellite is visible from ground station"""
#     # Vector from ground to satellite
#     sat_vector = sat_pos - ground_pos
    
#     # Local vertical at ground station
#     local_vertical = ground_pos / np.linalg.norm(ground_pos)
    
#     # Elevation angle from ground station perspective
#     elevation = rad_to_deg(np.arcsin(np.dot(sat_vector, local_vertical) / np.linalg.norm(sat_vector)))
    
#     return elevation >= min_elevation

# # Simulation parameters
# simulation_hours = 24  # 24 hour simulation
# time_steps = 1440  # 1 minute intervals
# time_array = np.linspace(0, simulation_hours, time_steps)

# # Ground station position
# munich_ecef = geodetic_to_ecef(MUNICH_LAT, MUNICH_LON)

# # Storage for results
# elevations = []
# azimuths = []
# visible_times = []
# distances = []

# print("Calculating satellite passes over Munich...")
# print(f"Ground station: Munich ({MUNICH_LAT:.2f}°N, {MUNICH_LON:.2f}°E)")
# print(f"Satellite altitude: {LEO_ALTITUDE} km")
# print(f"Orbital radius: {ORBIT_RADIUS} km")

# # Simulate multiple orbital passes
# for t in time_array:
#     sat_pos = satellite_position(t)
    
#     # Check if satellite is visible
#     if is_satellite_visible(sat_pos, munich_ecef):
#         elevation, azimuth = calculate_gimbal_angles(sat_pos, munich_ecef)
#         distance = np.linalg.norm(sat_pos - munich_ecef)
        
#         elevations.append(elevation)
#         azimuths.append(azimuth)
#         visible_times.append(t)
#         distances.append(distance)

# elevations = np.array(elevations)
# azimuths = np.array(azimuths)
# visible_times = np.array(visible_times)
# distances = np.array(distances)

# # Calculate statistics
# if len(elevations) > 0:
#     avg_elevation = np.mean(elevations)
#     median_elevation = np.median(elevations)
#     std_elevation = np.std(elevations)
#     max_elevation = np.max(elevations)
#     min_elevation = np.min(elevations)
    
#     print(f"\nPayload Gimbal Elevation Angle Statistics:")
#     print(f"Average elevation: {avg_elevation:.1f}°")
#     print(f"Median elevation: {median_elevation:.1f}°")
#     print(f"Standard deviation: {std_elevation:.1f}°")
#     print(f"Maximum elevation: {max_elevation:.1f}°")
#     print(f"Minimum elevation: {min_elevation:.1f}°")
#     print(f"Number of visible points: {len(elevations)}")
    
#     # Create plots
#     fig = plt.figure(figsize=(15, 12))
    
#     # Plot 1: Elevation angle over time
#     plt.subplot(2, 3, 1)
#     plt.plot(visible_times, elevations, 'b-', alpha=0.7)
#     plt.axhline(y=avg_elevation, color='r', linestyle='--', label=f'Average: {avg_elevation:.1f}°')
#     plt.xlabel('Time (hours)')
#     plt.ylabel('Gimbal Elevation Angle (degrees)')
#     plt.title('Payload Gimbal Elevation Angle vs Time')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # Plot 2: Elevation angle histogram
#     plt.subplot(2, 3, 2)
#     plt.hist(elevations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.axvline(x=avg_elevation, color='r', linestyle='--', label=f'Average: {avg_elevation:.1f}°')
#     plt.axvline(x=median_elevation, color='g', linestyle='--', label=f'Median: {median_elevation:.1f}°')
#     plt.xlabel('Gimbal Elevation Angle (degrees)')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Gimbal Elevation Angles')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Plot 3: Azimuth vs Elevation (polar plot)
#     plt.subplot(2, 3, 3, projection='polar')
#     scatter = plt.scatter(deg_to_rad(azimuths), elevations, c=distances, 
#                          cmap='viridis', alpha=0.6, s=20)
#     plt.ylabel('Elevation Angle (degrees)')
#     plt.title('Gimbal Pointing Direction\n(Azimuth vs Elevation)')
#     cbar = plt.colorbar(scatter, shrink=0.8)
#     cbar.set_label('Distance (km)')
    
#     # Plot 4: Distance vs Elevation
#     plt.subplot(2, 3, 4)
#     plt.scatter(distances, elevations, alpha=0.6, c='orange', s=20)
#     plt.xlabel('Distance to Ground Station (km)')
#     plt.ylabel('Gimbal Elevation Angle (degrees)')
#     plt.title('Distance vs Gimbal Elevation Angle')
#     plt.grid(True, alpha=0.3)
    
#     # Plot 5: Ground track visualization
#     plt.subplot(2, 3, 5)
#     # Simple 2D projection of satellite positions
#     sat_positions = []
#     for t in time_array[::10]:  # Sample every 10 points for clarity
#         sat_pos = satellite_position(t)
#         sat_positions.append(sat_pos)
    
#     sat_positions = np.array(sat_positions)
#     # Convert to lat/lon for plotting
#     sat_lats = []
#     sat_lons = []
#     for pos in sat_positions:
#         lat = rad_to_deg(np.arcsin(pos[2] / np.linalg.norm(pos)))
#         lon = rad_to_deg(np.arctan2(pos[1], pos[0]))
#         sat_lats.append(lat)
#         sat_lons.append(lon)
    
#     plt.plot(sat_lons, sat_lats, 'b-', alpha=0.7, linewidth=1, label='Satellite ground track')
#     plt.plot(MUNICH_LON, MUNICH_LAT, 'ro', markersize=8, label='Munich')
#     plt.xlabel('Longitude (degrees)')
#     plt.ylabel('Latitude (degrees)')
#     plt.title('Satellite Ground Track')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # Plot 6: Elevation angle statistics summary
#     plt.subplot(2, 3, 6)
#     stats_data = [min_elevation, avg_elevation - std_elevation, avg_elevation, 
#                   avg_elevation + std_elevation, max_elevation]
#     stats_labels = ['Min', 'Avg-σ', 'Average', 'Avg+σ', 'Max']
#     colors = ['red', 'orange', 'green', 'orange', 'red']
    
#     bars = plt.bar(stats_labels, stats_data, color=colors, alpha=0.7)
#     plt.ylabel('Elevation Angle (degrees)')
#     plt.title('Elevation Angle Statistics Summary')
#     plt.grid(True, alpha=0.3)
    
#     # Add value labels on bars
#     for bar, value in zip(bars, stats_data):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
#                 f'{value:.1f}°', ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Additional analysis
#     print(f"\nAdditional Analysis:")
#     print(f"Visibility percentage: {len(elevations)/len(time_array)*100:.1f}% of simulation time")
#     print(f"Average distance to ground station: {np.mean(distances):.0f} km")
#     print(f"Range of distances: {np.min(distances):.0f} - {np.max(distances):.0f} km")
    
#     # Analyze elevation angle ranges
#     low_elev = np.sum(elevations < 30)
#     med_elev = np.sum((elevations >= 30) & (elevations < 60))
#     high_elev = np.sum(elevations >= 60)
    
#     print(f"\nElevation angle distribution:")
#     print(f"Low angles (< 30°): {low_elev} points ({low_elev/len(elevations)*100:.1f}%)")
#     print(f"Medium angles (30-60°): {med_elev} points ({med_elev/len(elevations)*100:.1f}%)")
#     print(f"High angles (> 60°): {high_elev} points ({high_elev/len(elevations)*100:.1f}%)")
    
# else:
#     print("No visible satellite passes found during simulation period.")

# print(f"\nNote: This analysis assumes:")
# print(f"- Circular LEO orbit at {LEO_ALTITUDE} km altitude")
# print(f"- 98° inclination (typical sun-synchronous orbit)")
# print(f"- Payload gimbal nominal LOS along local horizon")
# print(f"- Minimum elevation angle of 10° for visibility")
# print(f"- Earth is treated as a perfect sphere")




# #######################-------------------------------------------------------###################
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Professional orbital mechanics libraries
# try:
#     from skyfield.api import load, Topos, EarthSatellite
#     from skyfield.sgp4lib import EarthSatellite as SGP4Satellite
#     SKYFIELD_AVAILABLE = True
#     print("Using Skyfield for professional orbital mechanics calculations")
# except ImportError:
#     SKYFIELD_AVAILABLE = False
#     print("Skyfield not available. Install with: pip install skyfield")
#     print("Run: pip install skyfield numpy matplotlib")

# # Physical constants
# EARTH_RADIUS = 6371.0  # km
# LEO_ALTITUDE = 550.0   # km

# # Munich coordinates
# MUNICH_LAT = 48.1351  # degrees N
# MUNICH_LON = 11.5820  # degrees E
# MUNICH_ALT = 0.520    # km above sea level

# class LEOGimbalAnalyzer:
#     """Professional LEO satellite gimbal angle analyzer using Skyfield"""
    
#     def __init__(self):
#         if SKYFIELD_AVAILABLE:
#             # Load timescale for accurate time calculations
#             self.ts = load.timescale()
#             print("Skyfield timescale loaded successfully")
#         else:
#             raise ImportError("Skyfield is required for accurate calculations")
    
#     def create_realistic_tle(self, altitude_km=550, inclination_deg=98.0, epoch_year=2024):
#         """
#         Create realistic TLE for LEO satellite
        
#         Args:
#             altitude_km: Satellite altitude in km
#             inclination_deg: Orbital inclination in degrees
#             epoch_year: Epoch year for TLE
        
#         Returns:
#             TLE lines as strings
#         """
#         # Calculate mean motion from altitude
#         semi_major_axis = EARTH_RADIUS + altitude_km  # km
#         mu_earth = 398600.4418  # km³/s²
#         period_minutes = 2 * np.pi * np.sqrt((semi_major_axis)**3 / mu_earth) / 60.0
#         mean_motion_rev_per_day = 1440.0 / period_minutes  # revolutions per day
        
#         # Create TLE with realistic parameters
#         # Line 1: Catalog number, classification, launch year, launch number, etc.
#         line1 = f"1 99999U {epoch_year-2000:02d}001A   {epoch_year%100:02d}001.00000000  .00000000  00000+0  00000+0 0  9990"
        
#         # Line 2: Inclination, RAAN, eccentricity, arg of perigee, mean anomaly, mean motion
#         line2 = f"2 99999 {inclination_deg:8.4f}   0.0000 0000000   0.0000   0.0000 {mean_motion_rev_per_day:11.8f}     09"
        
#         return line1, line2
    
#     def analyze_gimbal_angles(self, duration_hours=24, time_step_minutes=2, 
#                             altitude_km=550, inclination_deg=98.0):
#         """
#         Analyze gimbal angles for Munich ground station
        
#         Args:
#             duration_hours: Analysis duration in hours
#             time_step_minutes: Time step for analysis in minutes
#             altitude_km: Satellite altitude in km
#             inclination_deg: Orbital inclination in degrees
            
#         Returns:
#             Dictionary with analysis results
#         """
#         print(f"\nStarting Professional Orbital Mechanics Analysis")
#         print(f"=" * 55)
#         print(f"Satellite altitude: {altitude_km} km")
#         print(f"Orbital inclination: {inclination_deg}°")
#         print(f"Duration: {duration_hours} hours")
#         print(f"Time step: {time_step_minutes} minutes")
#         print(f"Ground station: Munich ({MUNICH_LAT:.4f}°N, {MUNICH_LON:.4f}°E)")
        
#         # Create satellite from TLE
#         line1, line2 = self.create_realistic_tle(altitude_km, inclination_deg)
#         satellite = EarthSatellite(line1, line2, 'LEO Analysis Satellite', self.ts)
        
#         # Create Munich ground station
#         munich = Topos(MUNICH_LAT, MUNICH_LON, elevation_m=MUNICH_ALT*1000)
        
#         # Generate time array
#         start_time = self.ts.now()
#         num_points = int(duration_hours * 60 / time_step_minutes)
#         time_points = self.ts.tt_jd(start_time.tt + np.arange(num_points) * time_step_minutes / (24 * 60))
        
#         # Storage for results
#         results = {
#             'times': [],
#             'elevations_gimbal': [],
#             'azimuths_gimbal': [],
#             'elevations_ground': [],
#             'azimuths_ground': [],
#             'distances': [],
#             'sat_positions_gcrs': [],
#             'sat_velocities_gcrs': [],
#             'ground_positions_gcrs': [],
#             'visible_times': [],
#             'satellite_info': {
#                 'altitude_km': altitude_km,
#                 'inclination_deg': inclination_deg,
#                 'orbital_period_min': 2 * np.pi * np.sqrt(((EARTH_RADIUS + altitude_km)**3) / 398600.4418) / 60.0
#             }
#         }
        
#         print(f"Calculating {num_points} satellite positions...")
        
#         visible_count = 0
#         for i, t in enumerate(time_points):
#             try:
#                 # Get satellite position and velocity in GCRS
#                 satellite_gcrs = satellite.at(t)
#                 sat_pos_km = satellite_gcrs.position.km
#                 sat_vel_km_s = satellite_gcrs.velocity.km_per_s
                
#                 # Get ground station position in GCRS (accounting for Earth rotation)
#                 ground_gcrs = munich.at(t)
#                 ground_pos_km = ground_gcrs.position.km
                
#                 # Calculate topocentric coordinates (satellite as seen from ground station)
#                 difference = satellite.at(t) - munich.at(t)
#                 alt, az, distance = difference.altaz()
                
#                 # Only analyze when satellite is visible (above minimum elevation)
#                 min_elevation_deg = 10.0
#                 if alt.degrees >= min_elevation_deg:
#                     # Calculate gimbal angles using proper orbital mechanics
#                     elevation_gimbal, azimuth_gimbal = self.calculate_gimbal_angles_professional(
#                         sat_pos_km, sat_vel_km_s, ground_pos_km
#                     )
                    
#                     # Store results
#                     results['times'].append(t.tt)
#                     results['elevations_gimbal'].append(elevation_gimbal)
#                     results['azimuths_gimbal'].append(azimuth_gimbal)
#                     results['elevations_ground'].append(alt.degrees)
#                     results['azimuths_ground'].append(az.degrees)
#                     results['distances'].append(distance.km)
#                     results['sat_positions_gcrs'].append(sat_pos_km)
#                     results['sat_velocities_gcrs'].append(sat_vel_km_s)
#                     results['ground_positions_gcrs'].append(ground_pos_km)
#                     results['visible_times'].append(t)
                    
#                     visible_count += 1
                
#                 # Progress indicator
#                 if (i + 1) % 100 == 0 or i == 0:
#                     progress = (i + 1) / num_points * 100
#                     print(f"Progress: {progress:5.1f}% - Visible points: {visible_count}")
                    
#             except Exception as e:
#                 print(f"Error at time point {i}: {e}")
#                 continue
        
#         print(f"\nAnalysis complete!")
#         print(f"Total visible data points: {visible_count}")
#         print(f"Visibility percentage: {visible_count/num_points*100:.1f}%")
        
#         return results
    
#     def calculate_gimbal_angles_professional(self, sat_pos_km, sat_vel_km_s, ground_pos_km):
#         """
#         Calculate gimbal elevation and azimuth angles using rigorous orbital mechanics
        
#         This function calculates the angles that a satellite payload gimbal must rotate
#         from its nominal horizontal orientation to point toward a ground station.
        
#         Args:
#             sat_pos_km: Satellite position in GCRS [x, y, z] km
#             sat_vel_km_s: Satellite velocity in GCRS [vx, vy, vz] km/s
#             ground_pos_km: Ground station position in GCRS [x, y, z] km
        
#         Returns:
#             elevation: Gimbal elevation angle in degrees (+ = below horizon)
#             azimuth: Gimbal azimuth angle in degrees (from along-track direction)
#         """
#         # Convert to numpy arrays for vector operations
#         r_sat = np.array(sat_pos_km)
#         v_sat = np.array(sat_vel_km_s) 
#         r_ground = np.array(ground_pos_km)
        
#         # Line-of-sight vector from satellite to ground station
#         los_vector = r_ground - r_sat
#         los_unit = los_vector / np.linalg.norm(los_vector)
        
#         # Build satellite's local orbital coordinate system
#         # 1. Radial direction (from Earth center toward satellite)
#         radial_unit = r_sat / np.linalg.norm(r_sat)
        
#         # 2. Angular momentum vector (normal to orbital plane)
#         h_vector = np.cross(r_sat, v_sat)
#         cross_track_unit = h_vector / np.linalg.norm(h_vector)
        
#         # 3. Along-track direction (in orbital plane, perpendicular to radial)
#         along_track_unit = np.cross(cross_track_unit, radial_unit)
        
#         # 4. Nadir direction (from satellite toward Earth center)
#         nadir_unit = -radial_unit
        
#         # Calculate elevation angle from satellite's local horizontal plane
#         # The local horizontal plane is perpendicular to the nadir direction
#         # Positive elevation means pointing below the horizon (toward Earth)
#         elevation_rad = np.arcsin(-np.clip(np.dot(los_unit, nadir_unit), -1.0, 1.0))
#         elevation_deg = np.degrees(elevation_rad)
        
#         # Calculate azimuth angle in the local horizontal plane
#         # Project LOS vector onto the horizontal plane (perpendicular to nadir)
#         los_horizontal = los_unit - np.dot(los_unit, nadir_unit) * nadir_unit
        
#         # Handle the case where LOS is nearly vertical (satellite nearly overhead)
#         horizontal_magnitude = np.linalg.norm(los_horizontal)
#         if horizontal_magnitude < 1e-8:
#             azimuth_deg = 0.0  # Azimuth undefined when pointing straight down
#         else:
#             los_horizontal_unit = los_horizontal / horizontal_magnitude
            
#             # Calculate azimuth relative to along-track direction
#             cos_azimuth = np.dot(los_horizontal_unit, along_track_unit)
#             sin_azimuth = np.dot(los_horizontal_unit, cross_track_unit)
            
#             azimuth_rad = np.arctan2(sin_azimuth, cos_azimuth)
#             azimuth_deg = np.degrees(azimuth_rad)
        
#         return elevation_deg, azimuth_deg
    
#     def plot_comprehensive_analysis(self, results):
#         """Create comprehensive plots and statistical analysis"""
        
#         if len(results['elevations_gimbal']) == 0:
#             print("ERROR: No visible satellite passes found!")
#             print("Try increasing the analysis duration or lowering the minimum elevation angle.")
#             return None
        
#         # Convert to numpy arrays for analysis
#         elevations = np.array(results['elevations_gimbal'])
#         azimuths = np.array(results['azimuths_gimbal'])
#         distances = np.array(results['distances'])
#         ground_elevations = np.array(results['elevations_ground'])
#         ground_azimuths = np.array(results['azimuths_ground'])
        
#         # Calculate comprehensive statistics
#         stats = {
#             'count': len(elevations),
#             'mean': np.mean(elevations),
#             'median': np.median(elevations),
#             'std': np.std(elevations),
#             'min': np.min(elevations),
#             'max': np.max(elevations),
#             'q25': np.percentile(elevations, 25),
#             'q75': np.percentile(elevations, 75),
#             'mean_distance': np.mean(distances),
#             'min_distance': np.min(distances),
#             'max_distance': np.max(distances)
#         }
        
#         # Print comprehensive results
#         print("\n" + "="*70)
#         print("PROFESSIONAL GIMBAL ELEVATION ANGLE ANALYSIS")
#         print("="*70)
#         print(f"Satellite Parameters:")
#         print(f"  Altitude: {results['satellite_info']['altitude_km']} km")
#         print(f"  Inclination: {results['satellite_info']['inclination_deg']}°")
#         print(f"  Orbital Period: {results['satellite_info']['orbital_period_min']:.1f} minutes")
#         print(f"\nGround Station: Munich ({MUNICH_LAT:.4f}°N, {MUNICH_LON:.4f}°E)")
#         print(f"Analysis Points: {stats['count']} visible observations")
        
#         print(f"\nGIMBAL ELEVATION ANGLE STATISTICS:")
#         print(f"  Mean:           {stats['mean']:6.1f}°")
#         print(f"  Median:         {stats['median']:6.1f}°")
#         print(f"  Standard Dev:   {stats['std']:6.1f}°")
#         print(f"  Range:          {stats['min']:6.1f}° to {stats['max']:6.1f}°")
#         print(f"  25th percentile: {stats['q25']:6.1f}°")
#         print(f"  75th percentile: {stats['q75']:6.1f}°")
        
#         print(f"\nDISTANCE STATISTICS:")
#         print(f"  Mean distance:  {stats['mean_distance']:6.0f} km")
#         print(f"  Range:          {stats['min_distance']:6.0f} - {stats['max_distance']:6.0f} km")
        
#         # Elevation angle distribution analysis
#         ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90)]
#         print(f"\nELEVATION ANGLE DISTRIBUTION:")
#         for low, high in ranges:
#             count = np.sum((elevations >= low) & (elevations < high))
#             percentage = count / len(elevations) * 100
#             print(f"  {low:2d}-{high:2d}°: {count:4d} points ({percentage:5.1f}%)")
        
#         print("="*70)
        
#         # Create comprehensive plots
#         fig = plt.figure(figsize=(18, 14))
        
#         # Plot 1: Time series of gimbal elevation
#         plt.subplot(3, 3, 1)
#         time_hours = [(t - results['times'][0]) * 24 for t in results['times']]
#         plt.plot(time_hours, elevations, 'b-', alpha=0.8, linewidth=1.2)
#         plt.axhline(y=stats['mean'], color='red', linestyle='--', linewidth=2,
#                    label=f'Mean: {stats["mean"]:.1f}°')
#         plt.axhline(y=stats['median'], color='green', linestyle=':', linewidth=2,
#                    label=f'Median: {stats["median"]:.1f}°')
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Gimbal Elevation Angle vs Time\n(Professional Skyfield Analysis)')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 2: Histogram with statistics
#         plt.subplot(3, 3, 2)
#         n, bins, patches = plt.hist(elevations, bins=30, alpha=0.7, color='lightblue', 
#                                    edgecolor='black', density=True)
#         plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
#                    label=f'Mean: {stats["mean"]:.1f}°')
#         plt.axvline(stats['median'], color='green', linestyle=':', linewidth=2,
#                    label=f'Median: {stats["median"]:.1f}°')
#         plt.axvline(stats['q25'], color='orange', linestyle='-.', alpha=0.7,
#                    label=f'Q25: {stats["q25"]:.1f}°')
#         plt.axvline(stats['q75'], color='orange', linestyle='-.', alpha=0.7,
#                    label=f'Q75: {stats["q75"]:.1f}°')
#         plt.xlabel('Gimbal Elevation Angle (°)')
#         plt.ylabel('Probability Density')
#         plt.title('Elevation Angle Distribution')
#         plt.legend(fontsize=8)
#         plt.grid(True, alpha=0.3)
        
#         # Plot 3: Polar plot of gimbal pointing directions
#         plt.subplot(3, 3, 3, projection='polar')
#         scatter = plt.scatter(np.radians(azimuths), elevations, c=distances, 
#                             cmap='viridis', alpha=0.7, s=25)
#         plt.ylabel('Elevation (°)', labelpad=35)
#         plt.title('Gimbal Pointing Directions\n(Color = Distance)', pad=20)
#         cbar = plt.colorbar(scatter, shrink=0.8, pad=0.1)
#         cbar.set_label('Distance (km)', rotation=270, labelpad=15)
        
#         # Plot 4: Distance vs Elevation with trend
#         plt.subplot(3, 3, 4)
#         plt.scatter(distances, elevations, alpha=0.6, c='darkorange', s=20)
#         # Add polynomial trend line
#         z = np.polyfit(distances, elevations, 2)  # 2nd order polynomial
#         p = np.poly1d(z)
#         dist_trend = np.linspace(np.min(distances), np.max(distances), 100)
#         plt.plot(dist_trend, p(dist_trend), "r-", linewidth=2, alpha=0.8, label='Trend')
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Distance vs Gimbal Elevation')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 5: Ground elevation vs Gimbal elevation
#         plt.subplot(3, 3, 5)
#         plt.scatter(ground_elevations, elevations, alpha=0.6, c='purple', s=20)
#         # Add 1:1 reference line and trend
#         min_val = min(np.min(ground_elevations), np.min(elevations))
#         max_val = max(np.max(ground_elevations), np.max(elevations))
#         plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
#                 label='1:1 Reference')
        
#         # Linear fit
#         coeff = np.polyfit(ground_elevations, elevations, 1)
#         trend_line = np.poly1d(coeff)
#         plt.plot(ground_elevations, trend_line(ground_elevations), 'r-', 
#                 linewidth=2, alpha=0.8, label=f'Trend (slope: {coeff[0]:.2f})')
        
#         plt.xlabel('Ground Station Elevation (°)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Ground vs Gimbal Elevation')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 6: Cumulative distribution
#         plt.subplot(3, 3, 6)
#         sorted_elevations = np.sort(elevations)
#         cumulative = np.arange(1, len(sorted_elevations) + 1) / len(sorted_elevations)
#         plt.plot(sorted_elevations, cumulative * 100, 'b-', linewidth=2)
#         plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
#         plt.axhline(25, color='orange', linestyle=':', alpha=0.7, label='25th percentile')
#         plt.axhline(75, color='orange', linestyle=':', alpha=0.7, label='75th percentile')
#         plt.xlabel('Gimbal Elevation Angle (°)')
#         plt.ylabel('Cumulative Percentage')
#         plt.title('Cumulative Distribution')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 7: Box plot with outliers
#         plt.subplot(3, 3, 7)
#         box_plot = plt.boxplot(elevations, patch_artist=True, 
#                               boxprops=dict(facecolor='lightblue', alpha=0.7),
#                               medianprops=dict(color='red', linewidth=2))
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Statistical Summary\n(Box Plot)')
#         plt.grid(True, alpha=0.3)
        
#         # Add statistical annotations
#         plt.text(1.1, stats['q75'], f'Q3: {stats["q75"]:.1f}°', 
#                 verticalalignment='center')
#         plt.text(1.1, stats['median'], f'Median: {stats["median"]:.1f}°', 
#                 verticalalignment='center')
#         plt.text(1.1, stats['q25'], f'Q1: {stats["q25"]:.1f}°', 
#                 verticalalignment='center')
        
#         # Plot 8: Azimuth distribution
#         plt.subplot(3, 3, 8)
#         plt.hist(azimuths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
#         plt.xlabel('Gimbal Azimuth (°)')
#         plt.ylabel('Frequency')
#         plt.title('Azimuth Angle Distribution')
#         plt.grid(True, alpha=0.3)
        
#         # Plot 9: 3D scatter plot
#         ax9 = plt.subplot(3, 3, 9, projection='3d')
#         scatter = ax9.scatter(ground_elevations, distances/1000, elevations, 
#                             c=elevations, cmap='coolwarm', alpha=0.6, s=20)
#         ax9.set_xlabel('Ground Elevation (°)')
#         ax9.set_ylabel('Distance (1000 km)')
#         ax9.set_zlabel('Gimbal Elevation (°)')
#         ax9.set_title('3D Analysis View')
        
#         plt.tight_layout()
#         plt.show()
        
#         return stats

# # Main execution and demonstration
# if __name__ == "__main__" and SKYFIELD_AVAILABLE:
#     try:
#         print("Initializing Professional LEO Gimbal Analyzer...")
#         analyzer = LEOGimbalAnalyzer()
        
#         print("Running comprehensive analysis...")
#         results = analyzer.analyze_gimbal_angles(
#             duration_hours=24,
#             time_step_minutes=2,
#             altitude_km=550,
#             inclination_deg=98.0
#         )
        
#         print("Generating comprehensive plots and statistics...")
#         final_stats = analyzer.plot_comprehensive_analysis(results)
        
#         if final_stats:
#             print(f"\nKEY FINDINGS:")
#             print(f"The average gimbal elevation angle of {final_stats['mean']:.1f}° represents")
#             print(f"the typical angle that a LEO satellite payload must tilt DOWN from")
#             print(f"its nominal horizontal orientation to communicate with Munich.")
#             print(f"\nThis result accounts for:")
#             print(f"• Accurate orbital mechanics (Skyfield)")
#             print(f"• Earth rotation effects")
#             print(f"• Proper coordinate frame transformations")
#             print(f"• Realistic satellite orbital parameters")
        
#     except Exception as e:
#         print(f"Error during analysis: {e}")
#         print("Make sure Skyfield is properly installed: pip install skyfield")

# elif not SKYFIELD_AVAILABLE:
#     print("\n" + "="*60)
#     print("INSTALLATION REQUIRED")
#     print("="*60)
#     print("To run this professional orbital mechanics analysis, install:")
#     print("pip install skyfield numpy matplotlib")
#     print("\nSkyfield provides:")
#     print("• NASA/JPL quality orbital mechanics")
#     print("• Accurate coordinate transformations")
#     print("• Earth rotation and precession effects")
#     print("• Professional satellite tracking capabilities")
#     print("="*60)
# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Professional orbital mechanics libraries
# try:
#     from skyfield.api import load, Topos, EarthSatellite
#     from skyfield.sgp4lib import EarthSatellite as SGP4Satellite
#     SKYFIELD_AVAILABLE = True
#     print("Using Skyfield for professional orbital mechanics calculations")
# except ImportError:
#     SKYFIELD_AVAILABLE = False
#     print("Skyfield not available. Install with: pip install skyfield")
#     print("Run: pip install skyfield numpy matplotlib")

# # Physical constants
# EARTH_RADIUS = 6371.0  # km
# LEO_ALTITUDE = 550.0   # km

# # Munich coordinates
# MUNICH_LAT = 48.1351  # degrees N
# MUNICH_LON = 11.5820  # degrees E
# MUNICH_ALT = 0.520    # km above sea level

# class LEOGimbalAnalyzer:
#     """Professional LEO satellite gimbal angle analyzer using Skyfield"""
    
#     def __init__(self):
#         if SKYFIELD_AVAILABLE:
#             # Load timescale for accurate time calculations
#             self.ts = load.timescale()
#             print("Skyfield timescale loaded successfully")
#         else:
#             raise ImportError("Skyfield is required for accurate calculations")
    
#     def create_realistic_tle(self, altitude_km=550, inclination_deg=98.0, epoch_year=2024):
#         """
#         Create realistic TLE for LEO satellite
        
#         Args:
#             altitude_km: Satellite altitude in km
#             inclination_deg: Orbital inclination in degrees
#             epoch_year: Epoch year for TLE
        
#         Returns:
#             TLE lines as strings
#         """
#         # Calculate mean motion from altitude
#         semi_major_axis = EARTH_RADIUS + altitude_km  # km
#         mu_earth = 398600.4418  # km³/s²
#         period_minutes = 2 * np.pi * np.sqrt((semi_major_axis)**3 / mu_earth) / 60.0
#         mean_motion_rev_per_day = 1440.0 / period_minutes  # revolutions per day
        
#         # Create TLE with realistic parameters
#         # Line 1: Catalog number, classification, launch year, launch number, etc.
#         line1 = f"1 99999U {epoch_year-2000:02d}001A   {epoch_year%100:02d}001.00000000  .00000000  00000+0  00000+0 0  9990"
        
#         # Line 2: Inclination, RAAN, eccentricity, arg of perigee, mean anomaly, mean motion
#         line2 = f"2 99999 {inclination_deg:8.4f}   0.0000 0000000   0.0000   0.0000 {mean_motion_rev_per_day:11.8f}     09"
        
#         return line1, line2
    
#     def analyze_gimbal_angles(self, duration_hours=24, time_step_minutes=2, 
#                             altitude_km=550, inclination_deg=98.0):
#         """
#         Analyze gimbal angles for Munich ground station
        
#         Args:
#             duration_hours: Analysis duration in hours
#             time_step_minutes: Time step for analysis in minutes
#             altitude_km: Satellite altitude in km
#             inclination_deg: Orbital inclination in degrees
            
#         Returns:
#             Dictionary with analysis results
#         """
#         print(f"\nStarting Professional Orbital Mechanics Analysis")
#         print(f"=" * 55)
#         print(f"Satellite altitude: {altitude_km} km")
#         print(f"Orbital inclination: {inclination_deg}°")
#         print(f"Duration: {duration_hours} hours")
#         print(f"Time step: {time_step_minutes} minutes")
#         print(f"Ground station: Munich ({MUNICH_LAT:.4f}°N, {MUNICH_LON:.4f}°E)")
        
#         # Create satellite from TLE
#         line1, line2 = self.create_realistic_tle(altitude_km, inclination_deg)
#         satellite = EarthSatellite(line1, line2, 'LEO Analysis Satellite', self.ts)
        
#         # Create Munich ground station
#         munich = Topos(MUNICH_LAT, MUNICH_LON, elevation_m=MUNICH_ALT*1000)
        
#         # Generate time array
#         start_time = self.ts.now()
#         num_points = int(duration_hours * 60 / time_step_minutes)
#         time_points = self.ts.tt_jd(start_time.tt + np.arange(num_points) * time_step_minutes / (24 * 60))
        
#         # Storage for results
#         results = {
#             'times': [],
#             'elevations_gimbal': [],
#             'azimuths_gimbal': [],
#             'elevations_ground': [],
#             'azimuths_ground': [],
#             'distances': [],
#             'sat_positions_gcrs': [],
#             'sat_velocities_gcrs': [],
#             'ground_positions_gcrs': [],
#             'visible_times': [],
#             'satellite_info': {
#                 'altitude_km': altitude_km,
#                 'inclination_deg': inclination_deg,
#                 'orbital_period_min': 2 * np.pi * np.sqrt(((EARTH_RADIUS + altitude_km)**3) / 398600.4418) / 60.0
#             }
#         }
        
#         print(f"Calculating {num_points} satellite positions...")
        
#         visible_count = 0
#         for i, t in enumerate(time_points):
#             try:
#                 # Get satellite position and velocity in GCRS
#                 satellite_gcrs = satellite.at(t)
#                 sat_pos_km = satellite_gcrs.position.km
#                 sat_vel_km_s = satellite_gcrs.velocity.km_per_s
                
#                 # Get ground station position in GCRS (accounting for Earth rotation)
#                 ground_gcrs = munich.at(t)
#                 ground_pos_km = ground_gcrs.position.km
                
#                 # Calculate topocentric coordinates (satellite as seen from ground station)
#                 difference = satellite.at(t) - munich.at(t)
#                 alt, az, distance = difference.altaz()
                
#                 # Only analyze when satellite is visible (above minimum elevation)
#                 min_elevation_deg = 10.0
#                 if alt.degrees >= min_elevation_deg:
#                     # Calculate gimbal angles using proper orbital mechanics
#                     elevation_gimbal, azimuth_gimbal = self.calculate_gimbal_angles_professional(
#                         sat_pos_km, sat_vel_km_s, ground_pos_km
#                     )
                    
#                     # Store results
#                     results['times'].append(t.tt)
#                     results['elevations_gimbal'].append(elevation_gimbal)
#                     results['azimuths_gimbal'].append(azimuth_gimbal)
#                     results['elevations_ground'].append(alt.degrees)
#                     results['azimuths_ground'].append(az.degrees)
#                     results['distances'].append(distance.km)
#                     results['sat_positions_gcrs'].append(sat_pos_km)
#                     results['sat_velocities_gcrs'].append(sat_vel_km_s)
#                     results['ground_positions_gcrs'].append(ground_pos_km)
#                     results['visible_times'].append(t)
                    
#                     visible_count += 1
                
#                 # Progress indicator
#                 if (i + 1) % 100 == 0 or i == 0:
#                     progress = (i + 1) / num_points * 100
#                     print(f"Progress: {progress:5.1f}% - Visible points: {visible_count}")
                    
#             except Exception as e:
#                 print(f"Error at time point {i}: {e}")
#                 continue
        
#         print(f"\nAnalysis complete!")
#         print(f"Total visible data points: {visible_count}")
#         print(f"Visibility percentage: {visible_count/num_points*100:.1f}%")
        
#         return results
    
#     def calculate_gimbal_angles_professional(self, sat_pos_km, sat_vel_km_s, ground_pos_km):
#         """
#         Calculate gimbal elevation and azimuth angles using rigorous orbital mechanics
        
#         This function calculates the angles that a satellite payload gimbal must rotate
#         from its nominal horizontal orientation to point toward a ground station.
        
#         Args:
#             sat_pos_km: Satellite position in GCRS [x, y, z] km
#             sat_vel_km_s: Satellite velocity in GCRS [vx, vy, vz] km/s
#             ground_pos_km: Ground station position in GCRS [x, y, z] km
        
#         Returns:
#             elevation: Gimbal elevation angle in degrees (+ = below horizon)
#             azimuth: Gimbal azimuth angle in degrees (from along-track direction)
#         """
#         # Convert to numpy arrays for vector operations
#         r_sat = np.array(sat_pos_km)
#         v_sat = np.array(sat_vel_km_s) 
#         r_ground = np.array(ground_pos_km)
        
#         # Line-of-sight vector from satellite to ground station
#         los_vector = r_ground - r_sat
#         los_unit = los_vector / np.linalg.norm(los_vector)
        
#         # Build satellite's local orbital coordinate system
#         # 1. Radial direction (from Earth center toward satellite)
#         radial_unit = r_sat / np.linalg.norm(r_sat)
        
#         # 2. Angular momentum vector (normal to orbital plane)
#         h_vector = np.cross(r_sat, v_sat)
#         cross_track_unit = h_vector / np.linalg.norm(h_vector)
        
#         # 3. Along-track direction (in orbital plane, perpendicular to radial)
#         along_track_unit = np.cross(cross_track_unit, radial_unit)
        
#         # 4. Nadir direction (from satellite toward Earth center)
#         nadir_unit = -radial_unit
        
#         # Calculate elevation angle from satellite's local horizontal plane
#         # The local horizontal plane is perpendicular to the nadir direction
#         # Positive elevation means pointing below the horizon (toward Earth)
#         elevation_rad = np.arcsin(np.clip(np.dot(los_unit, nadir_unit), -1.0, 1.0))
#         elevation_deg = np.degrees(elevation_rad)
        
#         # Calculate azimuth angle in the local horizontal plane
#         # Project LOS vector onto the horizontal plane (perpendicular to nadir)
#         los_horizontal = los_unit - np.dot(los_unit, nadir_unit) * nadir_unit
        
#         # Handle the case where LOS is nearly vertical (satellite nearly overhead)
#         horizontal_magnitude = np.linalg.norm(los_horizontal)
#         if horizontal_magnitude < 1e-8:
#             azimuth_deg = 0.0  # Azimuth undefined when pointing straight down
#         else:
#             los_horizontal_unit = los_horizontal / horizontal_magnitude
            
#             # Calculate azimuth relative to along-track direction
#             cos_azimuth = np.dot(los_horizontal_unit, along_track_unit)
#             sin_azimuth = np.dot(los_horizontal_unit, cross_track_unit)
            
#             azimuth_rad = np.arctan2(sin_azimuth, cos_azimuth)
#             azimuth_deg = np.degrees(azimuth_rad)
        
#         return elevation_deg, azimuth_deg
    
#     def plot_comprehensive_analysis(self, results):
#         """Create comprehensive plots and statistical analysis"""
        
#         if len(results['elevations_gimbal']) == 0:
#             print("ERROR: No visible satellite passes found!")
#             print("Try increasing the analysis duration or lowering the minimum elevation angle.")
#             return None
        
#         # Convert to numpy arrays for analysis
#         elevations = np.array(results['elevations_gimbal'])
#         azimuths = np.array(results['azimuths_gimbal'])
#         distances = np.array(results['distances'])
#         ground_elevations = np.array(results['elevations_ground'])
#         ground_azimuths = np.array(results['azimuths_ground'])
        
#         # Calculate comprehensive statistics
#         stats = {
#             'count': len(elevations),
#             'mean': np.mean(elevations),
#             'median': np.median(elevations),
#             'std': np.std(elevations),
#             'min': np.min(elevations),
#             'max': np.max(elevations),
#             'q25': np.percentile(elevations, 25),
#             'q75': np.percentile(elevations, 75),
#             'mean_distance': np.mean(distances),
#             'min_distance': np.min(distances),
#             'max_distance': np.max(distances)
#         }
        
#         # Print comprehensive results
#         print("\n" + "="*70)
#         print("PROFESSIONAL GIMBAL ELEVATION ANGLE ANALYSIS")
#         print("="*70)
#         print(f"Satellite Parameters:")
#         print(f"  Altitude: {results['satellite_info']['altitude_km']} km")
#         print(f"  Inclination: {results['satellite_info']['inclination_deg']}°")
#         print(f"  Orbital Period: {results['satellite_info']['orbital_period_min']:.1f} minutes")
#         print(f"\nGround Station: Munich ({MUNICH_LAT:.4f}°N, {MUNICH_LON:.4f}°E)")
#         print(f"Analysis Points: {stats['count']} visible observations")
        
#         print(f"\nGIMBAL ELEVATION ANGLE STATISTICS:")
#         print(f"  Mean:           {stats['mean']:6.1f}°")
#         print(f"  Median:         {stats['median']:6.1f}°")
#         print(f"  Standard Dev:   {stats['std']:6.1f}°")
#         print(f"  Range:          {stats['min']:6.1f}° to {stats['max']:6.1f}°")
#         print(f"  25th percentile: {stats['q25']:6.1f}°")
#         print(f"  75th percentile: {stats['q75']:6.1f}°")
        
#         print(f"\nDISTANCE STATISTICS:")
#         print(f"  Mean distance:  {stats['mean_distance']:6.0f} km")
#         print(f"  Range:          {stats['min_distance']:6.0f} - {stats['max_distance']:6.0f} km")
        
#         # Elevation angle distribution analysis
#         ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90)]
#         print(f"\nELEVATION ANGLE DISTRIBUTION:")
#         for low, high in ranges:
#             count = np.sum((elevations >= low) & (elevations < high))
#             percentage = count / len(elevations) * 100
#             print(f"  {low:2d}-{high:2d}°: {count:4d} points ({percentage:5.1f}%)")
        
#         print("="*70)
        
#         # Create comprehensive plots
#         fig = plt.figure(figsize=(18, 14))
        
#         # Plot 1: Time series of gimbal elevation
#         plt.subplot(3, 3, 1)
#         time_hours = [(t - results['times'][0]) * 24 for t in results['times']]
#         plt.plot(time_hours, elevations, 'b-', alpha=0.8, linewidth=1.2)
#         plt.axhline(y=stats['mean'], color='red', linestyle='--', linewidth=2,
#                    label=f'Mean: {stats["mean"]:.1f}°')
#         plt.axhline(y=stats['median'], color='green', linestyle=':', linewidth=2,
#                    label=f'Median: {stats["median"]:.1f}°')
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Gimbal Elevation Angle vs Time\n(Professional Skyfield Analysis)')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 2: Histogram with statistics
#         plt.subplot(3, 3, 2)
#         n, bins, patches = plt.hist(elevations, bins=30, alpha=0.7, color='lightblue', 
#                                    edgecolor='black', density=True)
#         plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
#                    label=f'Mean: {stats["mean"]:.1f}°')
#         plt.axvline(stats['median'], color='green', linestyle=':', linewidth=2,
#                    label=f'Median: {stats["median"]:.1f}°')
#         plt.axvline(stats['q25'], color='orange', linestyle='-.', alpha=0.7,
#                    label=f'Q25: {stats["q25"]:.1f}°')
#         plt.axvline(stats['q75'], color='orange', linestyle='-.', alpha=0.7,
#                    label=f'Q75: {stats["q75"]:.1f}°')
#         plt.xlabel('Gimbal Elevation Angle (°)')
#         plt.ylabel('Probability Density')
#         plt.title('Elevation Angle Distribution')
#         plt.legend(fontsize=8)
#         plt.grid(True, alpha=0.3)
        
#         # Plot 3: Polar plot of gimbal pointing directions
#         plt.subplot(3, 3, 3, projection='polar')
#         scatter = plt.scatter(np.radians(azimuths), elevations, c=distances, 
#                             cmap='viridis', alpha=0.7, s=25)
#         plt.ylabel('Elevation (°)', labelpad=35)
#         plt.title('Gimbal Pointing Directions\n(Color = Distance)', pad=20)
#         cbar = plt.colorbar(scatter, shrink=0.8, pad=0.1)
#         cbar.set_label('Distance (km)', rotation=270, labelpad=15)
        
#         # Plot 4: Distance vs Elevation with trend
#         plt.subplot(3, 3, 4)
#         plt.scatter(distances, elevations, alpha=0.6, c='darkorange', s=20)
#         # Add polynomial trend line
#         z = np.polyfit(distances, elevations, 2)  # 2nd order polynomial
#         p = np.poly1d(z)
#         dist_trend = np.linspace(np.min(distances), np.max(distances), 100)
#         plt.plot(dist_trend, p(dist_trend), "r-", linewidth=2, alpha=0.8, label='Trend')
#         plt.xlabel('Distance (km)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Distance vs Gimbal Elevation')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 5: Ground elevation vs Gimbal elevation
#         plt.subplot(3, 3, 5)
#         plt.scatter(ground_elevations, elevations, alpha=0.6, c='purple', s=20)
#         # Add 1:1 reference line and trend
#         min_val = min(np.min(ground_elevations), np.min(elevations))
#         max_val = max(np.max(ground_elevations), np.max(elevations))
#         plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
#                 label='1:1 Reference')
        
#         # Linear fit
#         coeff = np.polyfit(ground_elevations, elevations, 1)
#         trend_line = np.poly1d(coeff)
#         plt.plot(ground_elevations, trend_line(ground_elevations), 'r-', 
#                 linewidth=2, alpha=0.8, label=f'Trend (slope: {coeff[0]:.2f})')
        
#         plt.xlabel('Ground Station Elevation (°)')
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Ground vs Gimbal Elevation')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 6: Cumulative distribution
#         plt.subplot(3, 3, 6)
#         sorted_elevations = np.sort(elevations)
#         cumulative = np.arange(1, len(sorted_elevations) + 1) / len(sorted_elevations)
#         plt.plot(sorted_elevations, cumulative * 100, 'b-', linewidth=2)
#         plt.axhline(50, color='red', linestyle='--', alpha=0.7, label='50th percentile')
#         plt.axhline(25, color='orange', linestyle=':', alpha=0.7, label='25th percentile')
#         plt.axhline(75, color='orange', linestyle=':', alpha=0.7, label='75th percentile')
#         plt.xlabel('Gimbal Elevation Angle (°)')
#         plt.ylabel('Cumulative Percentage')
#         plt.title('Cumulative Distribution')
#         plt.grid(True, alpha=0.3)
#         plt.legend()
        
#         # Plot 7: Box plot with outliers
#         plt.subplot(3, 3, 7)
#         box_plot = plt.boxplot(elevations, patch_artist=True, 
#                               boxprops=dict(facecolor='lightblue', alpha=0.7),
#                               medianprops=dict(color='red', linewidth=2))
#         plt.ylabel('Gimbal Elevation (°)')
#         plt.title('Statistical Summary\n(Box Plot)')
#         plt.grid(True, alpha=0.3)
        
#         # Add statistical annotations
#         plt.text(1.1, stats['q75'], f'Q3: {stats["q75"]:.1f}°', 
#                 verticalalignment='center')
#         plt.text(1.1, stats['median'], f'Median: {stats["median"]:.1f}°', 
#                 verticalalignment='center')
#         plt.text(1.1, stats['q25'], f'Q1: {stats["q25"]:.1f}°', 
#                 verticalalignment='center')
        
#         # Plot 8: Azimuth distribution
#         plt.subplot(3, 3, 8)
#         plt.hist(azimuths, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
#         plt.xlabel('Gimbal Azimuth (°)')
#         plt.ylabel('Frequency')
#         plt.title('Azimuth Angle Distribution')
#         plt.grid(True, alpha=0.3)
        
#         # Plot 9: 3D scatter plot
#         ax9 = plt.subplot(3, 3, 9, projection='3d')
#         scatter = ax9.scatter(ground_elevations, distances/1000, elevations, 
#                             c=elevations, cmap='coolwarm', alpha=0.6, s=20)
#         ax9.set_xlabel('Ground Elevation (°)')
#         ax9.set_ylabel('Distance (1000 km)')
#         ax9.set_zlabel('Gimbal Elevation (°)')
#         ax9.set_title('3D Analysis View')
        
#         plt.tight_layout()
#         plt.show()
        
#         return stats

# # Main execution and demonstration
# if __name__ == "__main__" and SKYFIELD_AVAILABLE:
#     try:
#         print("Initializing Professional LEO Gimbal Analyzer...")
#         analyzer = LEOGimbalAnalyzer()
        
#         print("Running comprehensive analysis...")
#         results = analyzer.analyze_gimbal_angles(
#             duration_hours=24,
#             time_step_minutes=2,
#             altitude_km=550,
#             inclination_deg=98.0
#         )
        
#         print("Generating comprehensive plots and statistics...")
#         final_stats = analyzer.plot_comprehensive_analysis(results)
        
#         if final_stats:
#             print(f"\nKEY FINDINGS (CORRECTED CONVENTION):")
#             print(f"The average gimbal elevation angle of {final_stats['mean']:.1f}° represents")
#             print(f"the angle BELOW the local horizon (negative = toward Earth).")
#             print(f"This means the payload must tilt {abs(final_stats['mean']):.1f}° DOWN from")
#             print(f"its nominal horizontal orientation to communicate with Munich.")
#             print(f"\nConvention used:")
#             print(f"• Positive elevation = above local horizon (toward space)")
#             print(f"• Negative elevation = below local horizon (toward Earth)")
#             print(f"• Local horizon = tangent to orbit (parallel to velocity)")
        
#     except Exception as e:
#         print(f"Error during analysis: {e}")
#         print("Make sure Skyfield is properly installed: pip install skyfield")

# elif not SKYFIELD_AVAILABLE:
#     print("\n" + "="*60)
#     print("INSTALLATION REQUIRED")
#     print("="*60)
#     print("To run this professional orbital mechanics analysis, install:")
#     print("pip install skyfield numpy matplotlib")
#     print("\nSkyfield provides:")
#     print("• NASA/JPL quality orbital mechanics")
#     print("• Accurate coordinate transformations")
#     print("• Earth rotation and precession effects")
#     print("• Professional satellite tracking capabilities")
#     print("="*60)
# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import warnings
# warnings.filterwarnings('ignore')

# # Try to use professional libraries
# try:
#     from skyfield.api import load, EarthSatellite
#     from skyfield.framelib import itrs
#     SKYFIELD_AVAILABLE = True
#     print("Using Skyfield with proper ECI (GCRS) frame")
# except ImportError:
#     SKYFIELD_AVAILABLE = False
#     print("Skyfield not available - using analytical ECI orbital mechanics")

# # Physical constants
# EARTH_RADIUS = 6371.0  # km
# SATELLITE_ALTITUDE = 500.0  # km
# ORBIT_RADIUS = EARTH_RADIUS + SATELLITE_ALTITUDE
# MU_EARTH = 398600.4418  # km³/s²

# class InterSatelliteLinkAnalyzer_ECI:
#     """
#     Inter-satellite link analysis using ECI (Earth-Centered Inertial) frame
    
#     ECI is the proper choice for orbital mechanics because:
#     - Inertial frame (non-rotating)
#     - Keplerian motion is natural
#     - No fictitious forces
#     - Orbital elements remain constant
#     """
    
#     def __init__(self):
#         # Calculate orbital parameters
#         self.orbital_period = 2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / MU_EARTH)  # seconds
#         self.orbital_period_min = self.orbital_period / 60.0
#         self.mean_motion = 2 * np.pi / self.orbital_period  # rad/s
#         self.orbital_velocity = np.sqrt(MU_EARTH / ORBIT_RADIUS)  # km/s
        
#         print(f"Inter-Satellite Link Analysis in ECI Frame")
#         print(f"=" * 50)
#         print(f"Orbital Parameters for 500km Polar Orbit:")
#         print(f"  Orbital radius: {ORBIT_RADIUS} km")
#         print(f"  Orbital period: {self.orbital_period_min:.1f} minutes")
#         print(f"  Orbital velocity: {self.orbital_velocity:.2f} km/s")
#         print(f"  Mean motion: {np.degrees(self.mean_motion)*60:.2f} °/min")
        
#         if SKYFIELD_AVAILABLE:
#             self.ts = load.timescale()
#             print(f"Using Skyfield GCRS (ECI-equivalent) frame")
#         else:
#             print(f"Using analytical ECI calculations")
    
#     def eci_satellite_state(self, time_sec, inclination_deg=90.0, raan_deg=0.0, 
#                            mean_anomaly_offset=0.0):
#         """
#         Calculate satellite state vectors in ECI frame
        
#         Args:
#             time_sec: Time since epoch (seconds)
#             inclination_deg: Orbital inclination (degrees)
#             raan_deg: Right Ascension of Ascending Node (degrees)
#             mean_anomaly_offset: Initial mean anomaly offset (radians)
        
#         Returns:
#             pos_eci: Position vector in ECI [x, y, z] km
#             vel_eci: Velocity vector in ECI [vx, vy, vz] km/s
#         """
#         # Mean anomaly (includes initial offset for phase separation)
#         mean_anomaly = self.mean_motion * time_sec + mean_anomaly_offset
        
#         # For circular orbit: true anomaly = mean anomaly
#         true_anomaly = mean_anomaly
        
#         # Position and velocity in orbital plane (perifocal coordinates)
#         cos_nu = np.cos(true_anomaly)
#         sin_nu = np.sin(true_anomaly)
        
#         # Position in perifocal frame
#         r_pqw = np.array([
#             ORBIT_RADIUS * cos_nu,
#             ORBIT_RADIUS * sin_nu,
#             0.0
#         ])
        
#         # Velocity in perifocal frame
#         v_pqw = np.array([
#             -self.orbital_velocity * sin_nu,
#             self.orbital_velocity * cos_nu,
#             0.0
#         ])
        
#         # Transformation matrix from perifocal to ECI
#         # Uses classical orbital elements: RAAN (Ω), inclination (i), argument of periapsis (ω=0)
#         omega = 0.0  # Argument of periapsis (0 for circular orbit)
#         inc_rad = np.radians(inclination_deg)
#         raan_rad = np.radians(raan_deg)
#         omega_rad = np.radians(omega)
        
#         # Rotation matrix elements
#         cos_raan = np.cos(raan_rad)
#         sin_raan = np.sin(raan_rad)
#         cos_inc = np.cos(inc_rad)
#         sin_inc = np.sin(inc_rad)
#         cos_omega = np.cos(omega_rad)
#         sin_omega = np.sin(omega_rad)
        
#         # Complete transformation matrix from PQW to ECI
#         R11 = cos_raan * cos_omega - sin_raan * sin_omega * cos_inc
#         R12 = -cos_raan * sin_omega - sin_raan * cos_omega * cos_inc
#         R13 = sin_raan * sin_inc
        
#         R21 = sin_raan * cos_omega + cos_raan * sin_omega * cos_inc
#         R22 = -sin_raan * sin_omega + cos_raan * cos_omega * cos_inc
#         R23 = -cos_raan * sin_inc
        
#         R31 = sin_omega * sin_inc
#         R32 = cos_omega * sin_inc
#         R33 = cos_inc
        
#         R_pqw_to_eci = np.array([
#             [R11, R12, R13],
#             [R21, R22, R23],
#             [R31, R32, R33]
#         ])
        
#         # Transform to ECI
#         pos_eci = R_pqw_to_eci @ r_pqw
#         vel_eci = R_pqw_to_eci @ v_pqw
        
#         return pos_eci, vel_eci
    
#     def satellite_local_frame_eci(self, pos_eci, vel_eci):
#         """
#         Calculate satellite's local coordinate frame in ECI
        
#         Args:
#             pos_eci: Satellite position in ECI [x, y, z] km
#             vel_eci: Satellite velocity in ECI [vx, vy, vz] km/s
        
#         Returns:
#             radial: Radial unit vector (Earth center to satellite)
#             along_track: Along-track unit vector (velocity direction)
#             cross_track: Cross-track unit vector (angular momentum direction)
#         """
#         # Radial direction (from Earth center to satellite)
#         radial = pos_eci / np.linalg.norm(pos_eci)
        
#         # Cross-track direction (angular momentum vector)
#         h_vec = np.cross(pos_eci, vel_eci)
#         cross_track = h_vec / np.linalg.norm(h_vec)
        
#         # Along-track direction (completes right-handed system)
#         along_track = np.cross(cross_track, radial)
        
#         return radial, along_track, cross_track
    
#     def calculate_intersatellite_eci_angles(self, sat1_pos_eci, sat1_vel_eci, sat2_pos_eci):
#         """
#         Calculate inter-satellite link angles in ECI frame
        
#         Args:
#             sat1_pos_eci: Position of satellite 1 in ECI [x, y, z] km
#             sat1_vel_eci: Velocity of satellite 1 in ECI [vx, vy, vz] km/s
#             sat2_pos_eci: Position of satellite 2 in ECI [x, y, z] km
        
#         Returns:
#             elevation: Elevation angle (degrees) from local horizon
#             azimuth: Azimuth angle (degrees) from along-track direction
#             range_km: Distance between satellites (km)
#         """
#         # Line-of-sight vector from sat1 to sat2
#         los_vector = sat2_pos_eci - sat1_pos_eci
#         range_km = np.linalg.norm(los_vector)
#         los_unit = los_vector / range_km
        
#         # Get satellite 1's local coordinate frame
#         radial, along_track, cross_track = self.satellite_local_frame_eci(sat1_pos_eci, sat1_vel_eci)
        
#         # Zenith direction (radial outward from Earth)
#         zenith = radial
        
#         # Calculate elevation angle (standard aerospace convention)
#         # Positive = above local horizon, Negative = below local horizon
#         elevation_rad = np.arcsin(np.clip(np.dot(los_unit, zenith), -1.0, 1.0))
#         elevation_deg = np.degrees(elevation_rad)
        
#         # Calculate azimuth in local horizontal plane
#         # Local horizontal plane is perpendicular to zenith (radial) direction
#         los_horizontal = los_unit - np.dot(los_unit, zenith) * zenith
        
#         if np.linalg.norm(los_horizontal) < 1e-8:
#             azimuth_deg = 0.0  # Undefined when pointing straight up/down
#         else:
#             los_horizontal_unit = los_horizontal / np.linalg.norm(los_horizontal)
            
#             # Azimuth from along-track direction
#             cos_az = np.dot(los_horizontal_unit, along_track)
#             sin_az = np.dot(los_horizontal_unit, cross_track)
#             azimuth_deg = np.degrees(np.arctan2(sin_az, cos_az))
        
#         return elevation_deg, azimuth_deg, range_km
    
#     def analyze_in_plane_links_eci(self, phase_separation_deg=90, duration_hours=2):
#         """
#         Analyze in-plane inter-satellite links using ECI frame
        
#         Both satellites in same orbital plane with phase separation
#         """
#         print(f"\nIN-PLANE Analysis (ECI Frame)")
#         print(f"Phase separation: {phase_separation_deg}°")
#         print(f"Duration: {duration_hours} hours")
        
#         # Time array (higher resolution for smoother plots)
#         time_steps = int(duration_hours * 60 * 4)  # 15-second steps
#         time_array = np.linspace(0, duration_hours * 3600, time_steps)
        
#         # Convert phase separation to radians
#         phase_offset = np.radians(phase_separation_deg)
        
#         results = {
#             'times_hours': time_array / 3600,
#             'elevations': [],
#             'azimuths': [],
#             'ranges': [],
#             'sat1_positions_eci': [],
#             'sat2_positions_eci': [],
#             'sat1_velocities_eci': [],
#             'sat2_velocities_eci': []
#         }
        
#         print("Computing in-plane satellite states...")
#         for i, t in enumerate(time_array):
#             # Satellite 1: Reference satellite (RAAN=0°, i=90°)
#             pos1_eci, vel1_eci = self.eci_satellite_state(
#                 t, inclination_deg=90.0, raan_deg=0.0, mean_anomaly_offset=0.0
#             )
            
#             # Satellite 2: Same orbit, phase-shifted by phase_offset
#             pos2_eci, vel2_eci = self.eci_satellite_state(
#                 t, inclination_deg=90.0, raan_deg=0.0, mean_anomaly_offset=phase_offset
#             )
            
#             # Calculate inter-satellite link geometry
#             elevation, azimuth, range_km = self.calculate_intersatellite_eci_angles(
#                 pos1_eci, vel1_eci, pos2_eci
#             )
            
#             # Store results
#             results['elevations'].append(elevation)
#             results['azimuths'].append(azimuth)
#             results['ranges'].append(range_km)
#             results['sat1_positions_eci'].append(pos1_eci.copy())
#             results['sat2_positions_eci'].append(pos2_eci.copy())
#             results['sat1_velocities_eci'].append(vel1_eci.copy())
#             results['sat2_velocities_eci'].append(vel2_eci.copy())
            
#             # Progress indicator
#             if (i + 1) % 100 == 0:
#                 progress = (i + 1) / len(time_array) * 100
#                 print(f"  Progress: {progress:5.1f}%")
        
#         print(f"In-plane analysis complete: {len(results['elevations'])} points")
#         return results
    
#     def analyze_cross_plane_links_eci(self, raan_separation_deg=90, duration_hours=2):
#         """
#         Analyze cross-plane inter-satellite links using ECI frame
        
#         Satellites in different orbital planes (different RAAN)
#         """
#         print(f"\nCROSS-PLANE Analysis (ECI Frame)")
#         print(f"RAAN separation: {raan_separation_deg}°")
#         print(f"Duration: {duration_hours} hours")
        
#         # Time array
#         time_steps = int(duration_hours * 60 * 4)  # 15-second steps
#         time_array = np.linspace(0, duration_hours * 3600, time_steps)
        
#         results = {
#             'times_hours': time_array / 3600,
#             'elevations': [],
#             'azimuths': [],
#             'ranges': [],
#             'sat1_positions_eci': [],
#             'sat2_positions_eci': [],
#             'sat1_velocities_eci': [],
#             'sat2_velocities_eci': []
#         }
        
#         print("Computing cross-plane satellite states...")
#         for i, t in enumerate(time_array):
#             # Satellite 1: Reference satellite (RAAN=0°, i=90°)
#             pos1_eci, vel1_eci = self.eci_satellite_state(
#                 t, inclination_deg=90.0, raan_deg=0.0, mean_anomaly_offset=0.0
#             )
            
#             # Satellite 2: Different orbital plane (RAAN=raan_separation_deg, i=90°)
#             pos2_eci, vel2_eci = self.eci_satellite_state(
#                 t, inclination_deg=90.0, raan_deg=raan_separation_deg, mean_anomaly_offset=0.0
#             )
            
#             # Calculate inter-satellite link geometry
#             elevation, azimuth, range_km = self.calculate_intersatellite_eci_angles(
#                 pos1_eci, vel1_eci, pos2_eci
#             )
            
#             # Store results
#             results['elevations'].append(elevation)
#             results['azimuths'].append(azimuth)
#             results['ranges'].append(range_km)
#             results['sat1_positions_eci'].append(pos1_eci.copy())
#             results['sat2_positions_eci'].append(pos2_eci.copy())
#             results['sat1_velocities_eci'].append(vel1_eci.copy())
#             results['sat2_velocities_eci'].append(vel2_eci.copy())
            
#             # Progress indicator
#             if (i + 1) % 100 == 0:
#                 progress = (i + 1) / len(time_array) * 100
#                 print(f"  Progress: {progress:5.1f}%")
        
#         print(f"Cross-plane analysis complete: {len(results['elevations'])} points")
#         return results
    
#     def plot_eci_analysis(self, in_plane_results, cross_plane_results):
#         """Create comprehensive plots for ECI-based inter-satellite analysis"""
        
#         print(f"\nGenerating ECI-based analysis plots...")
        
#         fig = plt.figure(figsize=(20, 16))
        
#         # Convert to numpy arrays for analysis
#         in_plane = {
#             'times': np.array(in_plane_results['times_hours']),
#             'elevations': np.array(in_plane_results['elevations']),
#             'azimuths': np.array(in_plane_results['azimuths']),
#             'ranges': np.array(in_plane_results['ranges']),
#             'sat1_pos': np.array(in_plane_results['sat1_positions_eci']),
#             'sat2_pos': np.array(in_plane_results['sat2_positions_eci'])
#         }
        
#         cross_plane = {
#             'times': np.array(cross_plane_results['times_hours']),
#             'elevations': np.array(cross_plane_results['elevations']),
#             'azimuths': np.array(cross_plane_results['azimuths']),
#             'ranges': np.array(cross_plane_results['ranges']),
#             'sat1_pos': np.array(cross_plane_results['sat1_positions_eci']),
#             'sat2_pos': np.array(cross_plane_results['sat2_positions_eci'])
#         }
        
#         # Plot 1: Elevation angles time series (ECI frame)
#         plt.subplot(3, 4, 1)
#         plt.plot(in_plane['times'], in_plane['elevations'], 'b-', linewidth=2, 
#                 label=f'In-Plane (mean: {np.mean(in_plane["elevations"]):.1f}°)', alpha=0.8)
#         plt.plot(cross_plane['times'], cross_plane['elevations'], 'r-', linewidth=2, 
#                 label=f'Cross-Plane (mean: {np.mean(cross_plane["elevations"]):.1f}°)', alpha=0.8)
#         plt.axhline(0, color='black', linestyle='--', alpha=0.6, label='Local Horizon')
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Elevation Angle (°)')
#         plt.title('Elevation Angles (ECI Frame)\nPositive = Above Horizon')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 2: Range comparison (ECI frame)
#         plt.subplot(3, 4, 2)
#         plt.plot(in_plane['times'], in_plane['ranges'], 'b-', linewidth=2, 
#                 label=f'In-Plane (avg: {np.mean(in_plane["ranges"]):.0f} km)')
#         plt.plot(cross_plane['times'], cross_plane['ranges'], 'r-', linewidth=2, 
#                 label=f'Cross-Plane (avg: {np.mean(cross_plane["ranges"]):.0f} km)')
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Inter-Satellite Range (km)')
#         plt.title('Range Variations (ECI Frame)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 3: Azimuth angles (ECI frame)
#         plt.subplot(3, 4, 3)
#         plt.plot(in_plane['times'], in_plane['azimuths'], 'b.', markersize=3, 
#                 label='In-Plane Links', alpha=0.7)
#         plt.plot(cross_plane['times'], cross_plane['azimuths'], 'r.', markersize=3, 
#                 label='Cross-Plane Links', alpha=0.7)
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Azimuth Angle (°)')
#         plt.title('Azimuth Variations (ECI Frame)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 4: 3D ECI visualization
#         ax4 = plt.subplot(3, 4, 4, projection='3d')
        
#         # Sample every 20th point for clarity
#         sample_step = 20
#         indices = range(0, len(in_plane['sat1_pos']), sample_step)
        
#         # Plot Earth
#         u = np.linspace(0, 2 * np.pi, 15)
#         v = np.linspace(0, np.pi, 15)
#         earth_x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
#         earth_y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
#         earth_z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
#         ax4.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue')
        
#         # Plot satellite trajectories
#         ax4.plot(in_plane['sat1_pos'][indices, 0], in_plane['sat1_pos'][indices, 1], 
#                 in_plane['sat1_pos'][indices, 2], 'g-', linewidth=2, label='Satellite 1')
#         ax4.plot(in_plane['sat2_pos'][indices, 0], in_plane['sat2_pos'][indices, 1], 
#                 in_plane['sat2_pos'][indices, 2], 'b-', linewidth=2, label='Sat 2 (In-Plane)')
#         ax4.plot(cross_plane['sat2_pos'][indices, 0], cross_plane['sat2_pos'][indices, 1], 
#                 cross_plane['sat2_pos'][indices, 2], 'r-', linewidth=2, label='Sat 2 (Cross-Plane)')
        
#         # Add some inter-satellite links
#         for i in indices[::5]:  # Every 5th sample
#             if i < len(in_plane['sat1_pos']):
#                 ax4.plot([in_plane['sat1_pos'][i, 0], in_plane['sat2_pos'][i, 0]],
#                         [in_plane['sat1_pos'][i, 1], in_plane['sat2_pos'][i, 1]],
#                         [in_plane['sat1_pos'][i, 2], in_plane['sat2_pos'][i, 2]], 
#                         'b--', alpha=0.3, linewidth=1)
#                 ax4.plot([in_plane['sat1_pos'][i, 0], cross_plane['sat2_pos'][i, 0]],
#                         [in_plane['sat1_pos'][i, 1], cross_plane['sat2_pos'][i, 1]],
#                         [in_plane['sat1_pos'][i, 2], cross_plane['sat2_pos'][i, 2]], 
#                         'r--', alpha=0.3, linewidth=1)
        
#         ax4.set_xlabel('ECI X (km)')
#         ax4.set_ylabel('ECI Y (km)')
#         ax4.set_zlabel('ECI Z (km)')
#         ax4.set_title('3D ECI Orbital Configuration\nwith Inter-Satellite Links')
#         ax4.legend(fontsize=8)
        
#         # Plot 5: Range vs Time with orbital periods marked
#         plt.subplot(3, 4, 5)
#         plt.plot(in_plane['times'], in_plane['ranges'], 'b-', linewidth=1.5, 
#                 label='In-Plane', alpha=0.8)
#         plt.plot(cross_plane['times'], cross_plane['ranges'], 'r-', linewidth=1.5, 
#                 label='Cross-Plane', alpha=0.8)
        
#         # Mark orbital periods
#         for period_num in range(1, int(np.max(in_plane['times']) / (self.orbital_period_min/60)) + 1):
#             period_time = period_num * self.orbital_period_min / 60
#             plt.axvline(period_time, color='gray', linestyle=':', alpha=0.5)
        
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Range (km)')
#         plt.title(f'Range Evolution\n(Orbital Period: {self.orbital_period_min:.1f} min)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 6: Elevation statistics comparison
#         plt.subplot(3, 4, 6)
#         plt.hist(in_plane['elevations'], bins=40, alpha=0.6, color='blue', 
#                 label=f'In-Plane\nμ={np.mean(in_plane["elevations"]):.1f}°\nσ={np.std(in_plane["elevations"]):.1f}°',
#                 density=True)
#         plt.hist(cross_plane['elevations'], bins=40, alpha=0.6, color='red', 
#                 label=f'Cross-Plane\nμ={np.mean(cross_plane["elevations"]):.1f}°\nσ={np.std(cross_plane["elevations"]):.1f}°',
#                 density=True)
#         plt.axvline(0, color='black', linestyle='--', alpha=0.8, label='Horizon')
#         plt.xlabel('Elevation Angle (°)')
#         plt.ylabel('Probability Density')
#         plt.title('Elevation Distribution (ECI)')
#         plt.legend(fontsize=8)
#         plt.grid(True, alpha=0.3)
        
#         # Plot 7: Polar plot - In-plane pointing directions
#         plt.subplot(3, 4, 7, projection='polar')
#         scatter1 = plt.scatter(np.radians(in_plane['azimuths']), 
#                               np.abs(in_plane['elevations']), 
#                               c=in_plane['ranges'], cmap='Blues', alpha=0.6, s=15)
#         plt.ylabel('|Elevation| (°)', labelpad=25)
#         plt.title('In-Plane Link Directions\n(ECI Frame)', pad=15)
#         cbar1 = plt.colorbar(scatter1, shrink=0.6, pad=0.1)
#         cbar1.set_label('Range (km)', fontsize=8)
        
#         # Plot 8: Polar plot - Cross-plane pointing directions
#         plt.subplot(3, 4, 8, projection='polar')
#         scatter2 = plt.scatter(np.radians(cross_plane['azimuths']), 
#                               np.abs(cross_plane['elevations']), 
#                               c=cross_plane['ranges'], cmap='Reds', alpha=0.6, s=15)
#         plt.ylabel('|Elevation| (°)', labelpad=25)
#         plt.title('Cross-Plane Link Directions\n(ECI Frame)', pad=15)
#         cbar2 = plt.colorbar(scatter2, shrink=0.6, pad=0.1)
#         cbar2.set_label('Range (km)', fontsize=8)
        
#         # Plot 9: Range statistics
#         plt.subplot(3, 4, 9)
#         plt.hist(in_plane['ranges']/1000, bins=30, alpha=0.7, color='blue', 
#                 label=f'In-Plane\nMean: {np.mean(in_plane["ranges"]):.0f} km', density=True)
#         plt.hist(cross_plane['ranges']/1000, bins=30, alpha=0.7, color='red', 
#                 label=f'Cross-Plane\nMean: {np.mean(cross_plane["ranges"]):.0f} km', density=True)
#         plt.xlabel('Range (1000 km)')
#         plt.ylabel('Probability Density')
#         plt.title('Range Distribution (ECI)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 10: Gimbal requirements (elevation magnitude)
#         plt.subplot(3, 4, 10)
#         in_plane_gimbal = np.abs(in_plane['elevations'])
#         cross_plane_gimbal = np.abs(cross_plane['elevations'])
        
#         plt.plot(in_plane['times'], in_plane_gimbal, 'b-', linewidth=1.5, 
#                 label=f'In-Plane (max: {np.max(in_plane_gimbal):.1f}°)', alpha=0.8)
#         plt.plot(cross_plane['times'], cross_plane_gimbal, 'r-', linewidth=1.5, 
#                 label=f'Cross-Plane (max: {np.max(cross_plane_gimbal):.1f}°)', alpha=0.8)
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Gimbal Deflection |Elevation| (°)')
#         plt.title('Gimbal Angle Requirements')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot 11: Link availability (range threshold)
#         plt.subplot(3, 4, 11)
#         max_range_km = 5000  # Example maximum communication range
        
#         in_plane_available = (in_plane['ranges'] <= max_range_km).astype(float)
#         cross_plane_available = (cross_plane['ranges'] <= max_range_km).astype(float)
        
#         plt.plot(in_plane['times'], in_plane_available, 'b-', linewidth=2, 
#                 label=f'In-Plane ({np.mean(in_plane_available)*100:.1f}% available)')
#         plt.plot(cross_plane['times'], cross_plane_available, 'r-', linewidth=2, 
#                 label=f'Cross-Plane ({np.mean(cross_plane_available)*100:.1f}% available)')
#         plt.xlabel('Time (hours)')
#         plt.ylabel('Link Available (1=Yes, 0=No)')
#         plt.title(f'Link Availability\n(Max Range: {max_range_km} km)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.ylim(-0.1, 1.1)
        
#         # Plot 12: ECI coordinate evolution
#         plt.subplot(3, 4, 12)
#         plt.plot(in_plane['times'], in_plane['sat1_pos'][:, 0], 'g-', linewidth=1, 
#                 label='Sat 1 (X)', alpha=0.7)
#         plt.plot(in_plane['times'], in_plane['sat2_pos'][:, 0], 'b-', linewidth=1, 
#                 label='Sat 2 In-Plane (X)', alpha=0.7)
        
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------
# Helpers
# --------------------------
def d2r(d): return np.deg2rad(d)

def normalized(v):
    return v/np.linalg.norm(v)

def perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg):
    """ECI = R3(Ω) * R1(i) * R3(ω) * PQW."""
    Ω = d2r(raan_deg)
    i = d2r(inc_deg)
    ω = d2r(argp_deg)

    R3Ω = np.array([[ np.cos(Ω),-np.sin(Ω),0],
                    [ np.sin(Ω), np.cos(Ω),0],
                    [         0,         0,1]])
    R1i = np.array([[1,        0,         0],
                    [0, np.cos(i),-np.sin(i)],
                    [0, np.sin(i), np.cos(i)]])
    R3ω = np.array([[ np.cos(ω),-np.sin(ω),0],
                    [ np.sin(ω), np.cos(ω),0],
                    [         0,         0,1]])
    return R3Ω @ R1i @ R3ω

def orbit_positions_eci(a_km, e, inc_deg, raan_deg, argp_deg, num=800):
    """Return ECI positions along the orbit (true anomaly sweep)."""
    ν = np.linspace(0, 2*np.pi, num)
    p = a_km * (1 - e**2)
    r_pf = np.vstack([
        p * np.cos(ν) / (1 + e*np.cos(ν)),
        p * np.sin(ν) / (1 + e*np.cos(ν)),
        np.zeros_like(ν)
    ])  # in perifocal PQW
    Q = perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg)
    r_eci = Q @ r_pf  # (3, N)
    return r_eci

def draw_earth(ax, Re=6371.0, alpha=0.15, color='#6699cc'):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = Re * np.outer(np.cos(u), np.sin(v))
    y = Re * np.outer(np.sin(u), np.sin(v))
    z = Re * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, alpha=alpha, color=color, shade=True)

def plane_patch_from_normal(n, size, zoff=0.0, alpha=0.08, color='C0'):
    """
    Build a square patch centered at origin, oriented with normal n (|n|=1).
    """
    n = n / np.linalg.norm(n)
    # pick an arbitrary vector not parallel to n, to build basis
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)

    half = size / 2.0
    corners = np.array([
        -half*u - half*v,
         half*u - half*v,
         half*u + half*v,
        -half*u + half*v
    ])
    return corners

def set_axes_equal_3d(ax):
    """Make 3D axes equal scale for a better sphere/orbit look."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    mid_x = np.mean(x_limits); mid_y = np.mean(y_limits); mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def orbit_positions_and_velocity(a, e, inc_deg, raan_deg, argp_deg, ν_deg):
    """Return r and v in ECI for given orbital elements at true anomaly ν."""
    μ = 398600.4418  # km^3/s^2 Earth
    ν = d2r(ν_deg)
    p = a * (1 - e**2)

    # Position in PQW frame
    r_pf = np.array([p*np.cos(ν)/(1+e*np.cos(ν)),
                     p*np.sin(ν)/(1+e*np.cos(ν)),
                     0.0])
    # Velocity in PQW
    v_pf = np.array([-np.sqrt(μ/p)*np.sin(ν),
                      np.sqrt(μ/p)*(e+np.cos(ν)),
                      0.0])
    # Rotate into ECI
    Q = perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg)
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci

def los_from_nus(a, e, i_deg, raan1, raan2, argp, nu1_deg, nu2_deg):
    """Return LOS range (km), az(deg), el(deg) from sat1->sat2
       using single true anomalies nu1 and nu2 (degrees)."""
    r1, v1 = orbit_positions_and_velocity(a, e, i_deg, raan1, argp, nu1_deg)
    r2, v2 = orbit_positions_and_velocity(a, e, i_deg, raan2, argp, nu2_deg)

    # LOS in ECI
    los_eci = r2 - r1
    rng = np.linalg.norm(los_eci)

    # RTN at sat1 (R outward)
    R_hat = -r1 / np.linalg.norm(r1)
    T_hat = v1 / np.linalg.norm(v1)
    N_hat = np.cross(R_hat, T_hat); N_hat /= np.linalg.norm(N_hat)

    Q_eci2rtn = np.column_stack((R_hat, T_hat, N_hat))
    los_rtn = Q_eci2rtn.T @ los_eci
    x_R, y_T, z_N = los_rtn.astype(float)

    # Elevation above local horizon (deg): asin(R_component / range)
    el_deg = np.degrees(np.arcsin(-x_R / rng))
    # Azimuth in local horizontal plane measured from T_hat toward N_hat
    az_deg = np.degrees(np.arctan2(z_N, y_T))

    return {
        'r1': r1, 'v1': v1, 'r2': r2, 'v2': v2,
        'range_km': float(rng), 'az_deg': float(az_deg), 'el_deg': float(el_deg),
        'los_eci': los_eci
    }
# --------------------------
# Define the two orbits
# --------------------------
Re = 6378.0              # Earth radius (km)
a  = Re + 700.0          # semi-major axis (700 km circular LEO)
e  = 0.0                 # circular
i_deg = 89.0             # inclination (pick any to visualize a tilted plane)
ω_deg = 0.0              # argument of perigee (irrelevant for circular)
ν_deg = 0.0
ν_deg2 = -30.0

RAAN_1 = 0.0             # orbit A
RAAN_2 = 0.0            # orbit B (rotated about Z by +45°)
RAANs = [RAAN_1, RAAN_2]
colors = ['C0','C1']
# Compute orbits
r1, v1 = orbit_positions_and_velocity(a, e, i_deg, RAAN_1, ω_deg, ν_deg)
r2, v2 = orbit_positions_and_velocity(a, e, i_deg, RAAN_2, ω_deg, ν_deg2)


res = los_from_nus(a, e, i_deg, RAAN_1, RAAN_2, ω_deg, ν_deg,  ν_deg2)
print("Range (km):", res['range_km'])
print("Az (deg)  :", res['az_deg'])
print("El (deg)  :", res['el_deg'])

# Ascending node directions (unit vectors on equatorial plane)
Ω1 = d2r(RAAN_1); Ω2 = d2r(RAAN_2)
asc1_dir = np.array([np.cos(Ω1), np.sin(Ω1), 0.0])
asc2_dir = np.array([np.cos(Ω2), np.sin(Ω2), 0.0])

# Plane normals (same magnitude, rotated by RAAN about Z)
# Normal of orbital plane in ECI for given i, Ω: n = R3(Ω) * [0, sin(i), cos(i)] (up to scale)
n1 = np.array([ np.sin(d2r(i_deg))*np.sin(Ω1),
               -np.sin(d2r(i_deg))*np.cos(Ω1),
                np.cos(d2r(i_deg)) ])
n2 = np.array([ np.sin(d2r(i_deg))*np.sin(Ω2),
               -np.sin(d2r(i_deg))*np.cos(Ω2),
                np.cos(d2r(i_deg)) ])
n1 = n1/np.linalg.norm(n1)
n2 = n2/np.linalg.norm(n2)

# RTN basis at Sat1
R_hat = r1 / np.linalg.norm(r1)             # Radial (towards nadir)
T_hat = v1 / np.linalg.norm(v1)             # Along-track
N_hat = np.cross(R_hat, T_hat)              # Normal
N_hat /= np.linalg.norm(N_hat)

# Build transformation matrix: ECI -> RTN
Q_eci2rtn = np.column_stack((R_hat, T_hat, N_hat))  # (3x3)

# LOS in ECI
los_eci = r2 - r1
range_km = np.linalg.norm(los_eci)

# LOS in RTN frame
los_rtn = Q_eci2rtn.T @ normalized(los_eci)
x_rtn, y_rtn, z_rtn = los_rtn

# Convert to Azimuth/Elevation
az = np.degrees(np.arctan2(y_rtn, x_rtn))  # azimuth in XY plane of RTN
el = np.degrees(np.arcsin(z_rtn / range_km))


# --------------------------
# Plot
# --------------------------
fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111, projection='3d')

# Earth
draw_earth(ax, Re=Re, alpha=0.20, color='#88aadd')

# Orbits
ax.plot(r1[0], r1[1], r1[2], label=f'Orbit A: RAAN {RAAN_1:.0f}°', lw=2)
ax.plot(r2[0], r2[1], r2[2], label=f'Orbit B: RAAN {RAAN_2:.0f}°', lw=2)

# Orbital planes (translucent square patches)
plane_size = 2.2*a
for n, col, name in [(n1, 'C0', 'Plane A'), (n2, 'C1', 'Plane B')]:
    corners = plane_patch_from_normal(n, size=plane_size)
    poly = Poly3DCollection([corners], alpha=0.08, facecolor=col, edgecolor=col)
    ax.add_collection3d(poly)

# Show ECI axes
axis_len = 1.2 * a
ax.quiver(0,0,0, axis_len,0,0, color='k', arrow_length_ratio=0.05)
ax.text(axis_len, 0, 0, 'X', color='k')
ax.quiver(0,0,0, 0,axis_len,0, color='k', arrow_length_ratio=0.05)
ax.text(0, axis_len, 0, 'Y', color='k')
ax.quiver(0,0,0, 0,0,axis_len, color='k', arrow_length_ratio=0.05)
ax.text(0, 0, axis_len, 'Z', color='k')

for RAAN, col in zip(RAANs, colors):
    # Satellite state
    if RAAN == RAAN_2:
        ν_deg = ν_deg2
    r, v = orbit_positions_and_velocity(a, e, i_deg, RAAN, ω_deg, ν_deg)

    # Local RTN frame
    R_hat = r/np.linalg.norm(r)
    T_hat = v/np.linalg.norm(v)
    N_hat = np.cross(R_hat, T_hat); N_hat /= np.linalg.norm(N_hat)

    # Orbit arc
    νs = np.linspace(0,360,400)
    rs = np.array([orbit_positions_and_velocity(a,e,i_deg,RAAN,ω_deg,ν)[0] for ν in νs])
    ax.plot(rs[:,0], rs[:,1], rs[:,2], col, label=f'RAAN {RAAN:.0f}° orbit')

    # Plot satellite
    ax.scatter(r[0],r[1],r[2], color=col, s=60, marker='o')
    ax.quiver(r1[0], r1[1], r1[2],
          los_eci[0], los_eci[1], los_eci[2],
          color='m', arrow_length_ratio=0.2, linewidth=2, label="LOS")


    # Draw frame arrows (scaled)
    L = 1500  # km arrow length
    ax.quiver(r[0],r[1],r[2], L*R_hat[0],L*R_hat[1],L*R_hat[2], color='r')
    ax.quiver(r[0],r[1],r[2], L*T_hat[0],L*T_hat[1],L*T_hat[2], color='g')
    ax.quiver(r[0],r[1],r[2], L*N_hat[0],L*N_hat[1],L*N_hat[2], color='b')
    ax.text(r[0]+L*R_hat[0], r[1]+L*R_hat[1], r[2]+L*R_hat[2], f'R ({col})', color='r')
    ax.text(r[0]+L*T_hat[0], r[1]+L*T_hat[1], r[2]+L*T_hat[2], f'T ({col})', color='g')
    ax.text(r[0]+L*N_hat[0], r[1]+L*N_hat[1], r[2]+L*N_hat[2], f'N ({col})', color='b')

# Draw ascending-node directions on equator
node_arrow = Re * 1.5
ax.quiver(0,0,0, node_arrow*asc1_dir[0], node_arrow*asc1_dir[1], 0, color='C0', lw=2)
ax.text(node_arrow*asc1_dir[0], node_arrow*asc1_dir[1], 0, 'Ω=0°', color='C0')
ax.quiver(0,0,0, node_arrow*asc2_dir[0], node_arrow*asc2_dir[1], 0, color='C1', lw=2)
ax.text(node_arrow*asc2_dir[0], node_arrow*asc2_dir[1], 0, 'Ω=45°', color='C1')

# Cosmetics
ax.set_title('Two identical orbits, different RAAN (Ω): 0° vs 45°')
ax.set_xlabel('ECI X [km]'); ax.set_ylabel('ECI Y [km]'); ax.set_zlabel('ECI Z [km]')
ax.legend(loc='upper left')
ax.view_init(elev=25, azim=35)
set_axes_equal_3d(ax)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --------------------------
# Helpers
# --------------------------
def d2r(d): return np.deg2rad(d)

def normalized(v):
    return v/np.linalg.norm(v)

def perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg):
    """ECI = R3(Ω) * R1(i) * R3(ω) * PQW."""
    Ω = d2r(raan_deg)
    i = d2r(inc_deg)
    ω = d2r(argp_deg)

    R3Ω = np.array([[ np.cos(Ω),-np.sin(Ω),0],
                    [ np.sin(Ω), np.cos(Ω),0],
                    [         0,         0,1]])
    R1i = np.array([[1,        0,         0],
                    [0, np.cos(i),-np.sin(i)],
                    [0, np.sin(i), np.cos(i)]])
    R3ω = np.array([[ np.cos(ω),-np.sin(ω),0],
                    [ np.sin(ω), np.cos(ω),0],
                    [         0,         0,1]])
    return R3Ω @ R1i @ R3ω

def orbit_positions_eci(a_km, e, inc_deg, raan_deg, argp_deg, num=800):
    """Return ECI positions along the orbit (true anomaly sweep)."""
    ν = np.linspace(0, 2*np.pi, num)
    p = a_km * (1 - e**2)
    r_pf = np.vstack([
        p * np.cos(ν) / (1 + e*np.cos(ν)),
        p * np.sin(ν) / (1 + e*np.cos(ν)),
        np.zeros_like(ν)
    ])  # in perifocal PQW
    Q = perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg)
    r_eci = Q @ r_pf  # (3, N)
    return r_eci

def draw_earth(ax, Re=6371.0, alpha=0.15, color='#6699cc'):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = Re * np.outer(np.cos(u), np.sin(v))
    y = Re * np.outer(np.sin(u), np.sin(v))
    z = Re * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0, alpha=alpha, color=color, shade=True)

def plane_patch_from_normal(n, size, zoff=0.0, alpha=0.08, color='C0'):
    """
    Build a square patch centered at origin, oriented with normal n (|n|=1).
    """
    n = n / np.linalg.norm(n)
    # pick an arbitrary vector not parallel to n, to build basis
    a = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)

    half = size / 2.0
    corners = np.array([
        -half*u - half*v,
         half*u - half*v,
         half*u + half*v,
        -half*u + half*v
    ])
    return corners

def set_axes_equal_3d(ax):
    """Make 3D axes equal scale for a better sphere/orbit look."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    mid_x = np.mean(x_limits); mid_y = np.mean(y_limits); mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def orbit_positions_and_velocity(a, e, inc_deg, raan_deg, argp_deg, ν_deg):
    """Return r and v in ECI for given orbital elements at true anomaly ν."""
    μ = 398600.4418  # km^3/s^2 Earth
    ν = d2r(ν_deg)
    p = a * (1 - e**2)

    # Position in PQW frame
    r_pf = np.array([p*np.cos(ν)/(1+e*np.cos(ν)),
                     p*np.sin(ν)/(1+e*np.cos(ν)),
                     0.0])
    # Velocity in PQW
    v_pf = np.array([-np.sqrt(μ/p)*np.sin(ν),
                      np.sqrt(μ/p)*(e+np.cos(ν)),
                      0.0])
    # Rotate into ECI
    Q = perifocal_to_eci_matrix(raan_deg, inc_deg, argp_deg)
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci

def los_analysis_full_orbit(a, e, i_deg, raan1, raan2, argp, nu_offset_deg=0, num_points=360):
    """
    Perform LOS analysis over full orbit for nadir-pointing satellites.
    Returns arrays of azimuth, elevation, range, and true anomalies.
    """
    nu_array = np.linspace(0, 360, num_points)  # True anomaly for sat1
    
    ranges = []
    azimuths = []
    elevations = []
    
    for nu1 in nu_array:
        # Satellite 2 has an offset in true anomaly
        nu2 = nu1 + nu_offset_deg
        
        # Get positions and velocities
        r1, v1 = orbit_positions_and_velocity(a, e, i_deg, raan1, argp, nu1)
        r2, v2 = orbit_positions_and_velocity(a, e, i_deg, raan2, argp, nu2)
        
        # LOS vector in ECI
        los_eci = r2 - r1
        rng = np.linalg.norm(los_eci)
        
        # Build RTN frame at sat1 (nadir-pointing satellite)
        R_hat = r1 / np.linalg.norm(r1)   # Radial (away from Earth center, toward zenith)
        T_hat = v1 / np.linalg.norm(v1)   # Along-track (velocity direction)
        N_hat = np.cross(R_hat, T_hat)    # Normal (completes right-handed system)
        N_hat = N_hat / np.linalg.norm(N_hat)
        
        # Transform matrix ECI -> RTN
        Q_eci2rtn = np.column_stack((R_hat, T_hat, N_hat))
        
        # LOS in RTN frame
        los_rtn = Q_eci2rtn.T @ los_eci
        x_R, y_T, z_N = los_rtn
        
        # Calculate azimuth and elevation for nadir-pointing satellite
        # Elevation: angle above local horizon (positive when target is above horizon)
        el_deg = np.degrees(np.arcsin(x_R / rng))
        
        # Azimuth: angle in horizontal plane from T toward N
        az_deg = np.degrees(np.arctan2(z_N, y_T))
        
        ranges.append(rng)
        azimuths.append(az_deg)
        elevations.append(el_deg)
    
    return np.array(nu_array), np.array(ranges), np.array(azimuths), np.array(elevations)

# --------------------------
# Define the two orbits
# --------------------------
Re = 6378.0              # Earth radius (km)
a  = Re + 700.0          # semi-major axis (700 km circular LEO)
e  = 0.0                 # circular
i_deg = 89.0             # inclination
ω_deg = 0.0              # argument of perigee
RAAN_1 = 0.0             # orbit A
RAAN_2 = 0.0            # orbit B
nu_offset = 30.0        # True anomaly offset between satellites

# Perform full orbit LOS analysis
nu_array, ranges, azimuths, elevations = los_analysis_full_orbit(
    a, e, i_deg, RAAN_1, RAAN_2, ω_deg, nu_offset_deg=nu_offset, num_points=360
)

print(f"LOS Analysis Results:")
print(f"Range: {ranges.min():.1f} to {ranges.max():.1f} km")
print(f"Azimuth: {azimuths.min():.1f} to {azimuths.max():.1f} deg")
print(f"Elevation: {elevations.min():.1f} to {elevations.max():.1f} deg")

# --------------------------
# Create comprehensive plots
# --------------------------
fig = plt.figure(figsize=(16, 12))

# 1. 3D Orbital geometry
ax1 = plt.subplot(2, 3, 1, projection='3d')
draw_earth(ax1, Re=Re, alpha=0.20, color='#88aadd')

# Plot complete orbits
orbit1_eci = orbit_positions_eci(a, e, i_deg, RAAN_1, ω_deg, num=400)
orbit2_eci = orbit_positions_eci(a, e, i_deg, RAAN_2, ω_deg, num=400)

ax1.plot(orbit1_eci[0], orbit1_eci[1], orbit1_eci[2], 'C0-', 
         label=f'Orbit A (RAAN={RAAN_1}°)', linewidth=2)
ax1.plot(orbit2_eci[0], orbit2_eci[1], orbit2_eci[2], 'C1-', 
         label=f'Orbit B (RAAN={RAAN_2}°)', linewidth=2)

# Show current satellite positions
r1_current, v1_current = orbit_positions_and_velocity(a, e, i_deg, RAAN_1, ω_deg, 0)
r2_current, v2_current = orbit_positions_and_velocity(a, e, i_deg, RAAN_2, ω_deg, nu_offset)

ax1.scatter(*r1_current, color='C0', s=100, label='Sat A')
ax1.scatter(*r2_current, color='C1', s=100, label='Sat B')

# Show LOS
los_current = r2_current - r1_current
ax1.plot([r1_current[0], r2_current[0]], 
         [r1_current[1], r2_current[1]], 
         [r1_current[2], r2_current[2]], 
         'm-', linewidth=3, alpha=0.7, label='LOS')

ax1.set_title('3D Orbital Geometry')
ax1.set_xlabel('ECI X [km]')
ax1.set_ylabel('ECI Y [km]')
ax1.set_zlabel('ECI Z [km]')
ax1.legend()
set_axes_equal_3d(ax1)

# 2. Range vs True Anomaly
ax2 = plt.subplot(2, 3, 2)
ax2.plot(nu_array, ranges/1000, 'b-', linewidth=2)
ax2.set_xlabel('True Anomaly [deg]')
ax2.set_ylabel('Range [1000 km]')
ax2.set_title('LOS Range vs True Anomaly')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 360])

# 3. Azimuth vs True Anomaly
ax3 = plt.subplot(2, 3, 3)
ax3.plot(nu_array, azimuths, 'g-', linewidth=2)
ax3.set_xlabel('True Anomaly [deg]')
ax3.set_ylabel('Azimuth [deg]')
ax3.set_title('LOS Azimuth vs True Anomaly')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 360])

# 4. Elevation vs True Anomaly
ax4 = plt.subplot(2, 3, 4)
ax4.plot(nu_array, elevations, 'r-', linewidth=2)
ax4.set_xlabel('True Anomaly [deg]')
ax4.set_ylabel('Elevation [deg]')
ax4.set_title('LOS Elevation vs True Anomaly')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 360])

# 5. Azimuth vs Elevation (antenna pointing pattern)
ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(azimuths, elevations, c=ranges/1000, cmap='viridis', 
                     s=20, alpha=0.7)
ax5.set_xlabel('Azimuth [deg]')
ax5.set_ylabel('Elevation [deg]')
ax5.set_title('Antenna Pointing Pattern')
ax5.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Range [1000 km]')

# 6. Combined Az/El/Range plot
ax6 = plt.subplot(2, 3, 6)
# Create dual y-axis
ax6_twin = ax6.twinx()

line1 = ax6.plot(nu_array, azimuths, 'g-', linewidth=2, label='Azimuth')
line2 = ax6.plot(nu_array, elevations, 'r-', linewidth=2, label='Elevation')
line3 = ax6_twin.plot(nu_array, ranges/1000, 'b--', linewidth=2, label='Range')

ax6.set_xlabel('True Anomaly [deg]')
ax6.set_ylabel('Az/El [deg]', color='black')
ax6_twin.set_ylabel('Range [1000 km]', color='blue')
ax6.set_title('Combined LOS Parameters')

# Combine legends
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax6.grid(True, alpha=0.3)
ax6.set_xlim([0, 360])

plt.tight_layout()
plt.show()

# Test the original single-point calculation with corrected RTN frame
r1, v1 = orbit_positions_and_velocity(a, e, i_deg, RAAN_1, ω_deg, 0)
r2, v2 = orbit_positions_and_velocity(a, e, i_deg, RAAN_2, ω_deg, nu_offset)

# RTN basis at Sat1 (nadir-pointing - corrected)
R_hat = r1 / np.linalg.norm(r1)              # Radial (away from Earth center, toward zenith)
T_hat = v1 / np.linalg.norm(v1)             # Along-track
N_hat = np.cross(R_hat, T_hat)              # Normal
N_hat /= np.linalg.norm(N_hat)

# Build transformation matrix: ECI -> RTN
Q_eci2rtn = np.column_stack((R_hat, T_hat, N_hat))  # (3x3)

# LOS in ECI
los_eci = r2 - r1
range_km = np.linalg.norm(los_eci)

# LOS in RTN frame
los_rtn = Q_eci2rtn.T @ los_eci
x_rtn, y_rtn, z_rtn = los_rtn

# Convert to Azimuth/Elevation for nadir-pointing satellite
az = np.degrees(np.arctan2(z_rtn, y_rtn))  # azimuth in horizontal plane
el = np.degrees(np.arcsin(x_rtn / range_km))  # elevation (positive above horizon)

print(f"\nSingle point verification (nadir-pointing corrected):")
print(f"Range: {range_km:.1f} km")
print(f"Azimuth: {az:.1f} deg")  
print(f"Elevation: {el:.1f} deg")

# --------------------------
# Additional analysis
# --------------------------
print("\n" + "="*50)
print("DETAILED LOS ANALYSIS")
print("="*50)

# Find minimum and maximum ranges
min_range_idx = np.argmin(ranges)
max_range_idx = np.argmax(ranges)

print(f"\nRange Analysis:")
print(f"Minimum range: {ranges[min_range_idx]:.1f} km at ν = {nu_array[min_range_idx]:.1f}°")
print(f"Maximum range: {ranges[max_range_idx]:.1f} km at ν = {nu_array[max_range_idx]:.1f}°")

# Find high elevation events (good for communication)
high_el_mask = elevations > 10  # Above 10 degrees
if np.any(high_el_mask):
    high_el_periods = nu_array[high_el_mask]
    print(f"\nHigh elevation periods (>10°):")
    print(f"True anomaly ranges: {high_el_periods.min():.1f}° to {high_el_periods.max():.1f}°")
    print(f"Maximum elevation: {elevations.max():.1f}° at ν = {nu_array[np.argmax(elevations)]:.1f}°")
else:
    print(f"\nNo high elevation periods found. Maximum elevation: {elevations.max():.1f}°")

# Communication window analysis (assuming minimum elevation threshold)
min_elevation_threshold = 0  # degrees
comm_windows = elevations >= min_elevation_threshold
comm_percentage = np.sum(comm_windows) / len(comm_windows) * 100

print(f"\nCommunication Analysis:")
print(f"Visibility percentage (El ≥ {min_elevation_threshold}°): {comm_percentage:.1f}%")

# Statistical summary
print(f"\nStatistical Summary:")
print(f"Range - Mean: {np.mean(ranges):.1f} km, Std: {np.std(ranges):.1f} km")
print(f"Azimuth - Mean: {np.mean(azimuths):.1f}°, Std: {np.std(azimuths):.1f}°")
print(f"Elevation - Mean: {np.mean(elevations):.1f}°, Std: {np.std(elevations):.1f}°")
# %%
