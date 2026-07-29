from skyfield.api import EarthSatellite, load, wgs84
import simplekml
from datetime import datetime, timedelta
import csv
from tkinter import Tk, simpledialog

# --- Setup GUI to get parameters ---
Tk().withdraw()

# Input datetime string (UTC) to center around
start_datetime_str = simpledialog.askstring("Start DateTime (UTC)",
    "Enter start date/time (UTC) as YYYY-MM-DD HH:MM:SS", initialvalue=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
if not start_datetime_str:
    exit("No start datetime provided.")
try:
    center_time = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
except ValueError:
    exit("Invalid datetime format.")

# Input total hours before and after the center time
hours_before = simpledialog.askfloat("Hours Before",
    "Enter number of hours before start time", initialvalue=3.0)
hours_after = simpledialog.askfloat("Hours After",
    "Enter number of hours after start time", initialvalue=3.0)
if hours_before is None or hours_after is None:
    exit("Hours before/after not provided.")

# Fixed step size of 5 seconds (no prompt, hardcoded)
step_seconds = 1

# --- TLE Data ---
name = "AETHER-2 (58299)"
line1 = "1 58299U 23174AV  25205.23068287  .00000000  00000-0 -92872-2 0    04"
line2 = "2 58299  97.4169 283.7198 0006774 230.4729  96.2984 15.28996586    07"
satellite = EarthSatellite(line1, line2, name)
ts = load.timescale()

# --- Calculate time window ---
start_time = center_time - timedelta(hours=hours_before)
end_time = center_time + timedelta(hours=hours_after)

# --- Ground Station Coordinates ---
ground_lat = 34.947951075484106
ground_lon = -106.51217526827344
ground_alt = 1684.8471257174388  # meters
ground_station = wgs84.latlon(ground_lat, ground_lon, elevation_m=ground_alt)

# --- Storage ---
visible_coords = []
visible_times = []
az_el_table = []

invisible_coords = []
invisible_times = []

# --- Generate Coordinates and Visibility Info ---
current_time = start_time
while current_time <= end_time:
    t = ts.utc(current_time.year, current_time.month, current_time.day,
               current_time.hour, current_time.minute, current_time.second)

    geocentric = satellite.at(t)
    subpoint = wgs84.subpoint(geocentric)
    lat = subpoint.latitude.degrees
    lon = subpoint.longitude.degrees
    alt = subpoint.elevation.m
    coord = (lon, lat, alt)

    timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Visibility check from ground station
    topocentric = (satellite - ground_station).at(t)
    alt_angle, az, distance = topocentric.altaz()

    if alt_angle.degrees > 0:
        visible_coords.append(coord)
        visible_times.append(timestamp)
        az_el_table.append((
            timestamp,
            round(az.degrees, 1),
            round(alt_angle.degrees, 1),
            round(distance.km, 1)
        ))
    else:
        invisible_coords.append(coord)
        invisible_times.append(timestamp)

    current_time += timedelta(seconds=step_seconds)

# --- Write Azimuth Table ---
with open("visibility_table.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Time (UTC)", "Azimuth (°)", "Elevation (°)", "Range (km)"])
    writer.writerows(az_el_table)

print("📄 Saved azimuth/elevation table to 'visibility_table.csv'")

# --- Create KML ---
kml = simplekml.Kml()

# --- Visible Path (Green) ---
if visible_coords:
    track_visible = kml.newgxtrack(name="Visible Path (Green)")
    track_visible.newwhen(visible_times)
    track_visible.newgxcoord(visible_coords)
    track_visible.altitudemode = simplekml.AltitudeMode.absolute
    track_visible.extrude = 1
    track_visible.linestyle.color = simplekml.Color.green
    track_visible.linestyle.width = 2
    track_visible.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/satellite.png'
    track_visible.iconstyle.scale = 1.2

# --- Invisible Path (Red) ---
if invisible_coords:
    track_invisible = kml.newgxtrack(name="Invisible Path (Red)")
    track_invisible.newwhen(invisible_times)
    track_invisible.newgxcoord(invisible_coords)
    track_invisible.altitudemode = simplekml.AltitudeMode.absolute
    track_invisible.extrude = 1
    track_invisible.linestyle.color = simplekml.Color.red
    track_invisible.linestyle.width = 2
    track_invisible.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/track.png'
    track_invisible.iconstyle.scale = 1.0

# --- Animated Line of Sight (Cyan) ---
if visible_coords:
    los_track = kml.newgxtrack(name="Line of Sight")
    los_track.altitudemode = simplekml.AltitudeMode.absolute
    los_track.extrude = 1
    los_track.linestyle.color = simplekml.Color.cyan
    los_track.linestyle.width = 1.5

    for time_str, sat_coord in zip(visible_times, visible_coords):
        # Line from ground -> satellite
        los_track.newwhen([time_str, time_str])
        los_track.newgxcoord([
            (ground_lon, ground_lat, ground_alt),
            sat_coord
        ])

# --- Start Point Marker ---
start_coord = visible_coords[0] if visible_coords else invisible_coords[0]
start_lon, start_lat, start_alt = start_coord
kml.newpoint(name="Start Position",
             coords=[(start_lon, start_lat, start_alt)],
             description="Start of path",
             altitudemode=simplekml.AltitudeMode.absolute)

# --- Ground Station Marker ---
station_marker = kml.newpoint(name="Ground Station",
                              coords=[(ground_lon, ground_lat, ground_alt)],
                              description="Observer Location")
station_marker.altitudemode = simplekml.AltitudeMode.clamptoground
from simplekml import Icon
station_marker.style.iconstyle.icon = Icon(href="http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png")
station_marker.style.iconstyle.color = simplekml.Color.yellow
station_marker.style.iconstyle.scale = 1.3

# --- Save KML ---
kml.save("AETHER-2_visibility.kml")
print("✅ KML saved as 'AETHER-2_orbit_visibility.kml'")
