# ==================================================
# Solar Eclipse Ground Track – 12 August 2026
# Sun/Moon generated on-the-fly (no CSV)
# ==================================================
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time, TimeDelta
from astropy.coordinates import (
    get_body_barycentric, solar_system_ephemeris,
    CartesianRepresentation, GCRS, ITRS
)
import astropy.units as u
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from astropy.coordinates import get_body
# We will use cartopy only to extract coastlines (optional but recommended)
import cartopy.io.shapereader as shpreader
from cartopy.feature import COASTLINE, BORDERS

# -------------------- CONFIG --------------------
R_EARTH_A = 6378137.0
R_EARTH_B = 6356752.314245
R_SUN     = 695700000.0
R_MOON    = 1737400.0

# 24-hour window centred on the eclipse day
START_UTC = "2026-08-12 00:00:00"
END_UTC   = "2026-08-13 00:00:00"
DT_SECONDS = 30.0                 # sampling step

TRACK_STEP_SECONDS = 30.0
GRID_RESOLUTION_DEG = 0.35
MUNICH_LON, MUNICH_LAT = 11.58, 48.14

# -------------------- EPHEMERIS --------------------
# solar_system_ephemeris.set("de432s")   # high-precision, downloads once
# Best practical choice for 2026
solar_system_ephemeris.set("de440s")

def generate_sun_moon(times):
    """Return geocentric GCRS positions [m] for Sun and Moon."""
    sun_b   = get_body_barycentric("sun",   times)
    moon_b  = get_body_barycentric("moon",  times)
    earth_b = get_body_barycentric("earth", times)

    rS = (sun_b  - earth_b).xyz.to(u.m).value.T   # (N,3)
    rM = (moon_b - earth_b).xyz.to(u.m).value.T
    return rS, rM

# -------------------- GEOMETRY HELPERS --------------------
def angle_between(a, b):
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return np.arccos(np.clip(np.sum(a*b, axis=-1) / np.maximum(na*nb, 1e-12), -1.0, 1.0))

def solar_obscuration(theta, alpha_s, alpha_m):
    theta   = np.asarray(theta, dtype=float)
    alpha_s = np.asarray(alpha_s, dtype=float)
    alpha_m = np.asarray(alpha_m, dtype=float)
    result  = np.zeros_like(theta)

    total = (alpha_m >= alpha_s) & (theta <= alpha_m - alpha_s)
    result[total] = 1.0

    inside = (alpha_s > alpha_m) & (theta <= alpha_s - alpha_m)
    result[inside] = (alpha_m[inside] / alpha_s[inside])**2

    no_overlap = theta >= alpha_s + alpha_m
    partial = ~(total | inside | no_overlap)

    if np.any(partial):
        d  = np.maximum(theta[partial], 1e-15)
        rs = alpha_s[partial]
        rm = alpha_m[partial]
        c1 = np.clip((d**2 + rs**2 - rm**2)/(2*d*rs), -1, 1)
        c2 = np.clip((d**2 + rm**2 - rs**2)/(2*d*rm), -1, 1)
        a1 = rs**2 * np.arccos(c1)
        a2 = rm**2 * np.arccos(c2)
        rad = np.maximum((-d+rs+rm)*(d+rs-rm)*(d-rs+rm)*(d+rs+rm), 0.0)
        result[partial] = (a1 + a2 - 0.5*np.sqrt(rad)) / (np.pi * rs**2)

    return np.clip(result, 0.0, 1.0)

def geodetic_to_ecef(lat_deg, lon_deg):
    a, b = R_EARTH_A, R_EARTH_B
    f = (a-b)/a
    e2 = f*(2-f)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    N = a / np.sqrt(1 - e2*sin_lat**2)
    x = N * cos_lat * np.cos(lon)
    y = N * cos_lat * np.sin(lon)
    z = N * (1-e2) * sin_lat
    return np.column_stack((x, y, z))

def ecef_to_geodetic(r):
    x, y, z = r
    a, b = R_EARTH_A, R_EARTH_B
    e2 = 1 - (b/a)**2
    ep2 = (a/b)**2 - 1
    p = np.sqrt(x*x + y*y)
    theta = np.arctan2(a*z, b*p)
    lat = np.arctan2(z + ep2*b*np.sin(theta)**3,
                     p - e2*a*np.cos(theta)**3)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon)

def shadow_center_ecef(rS, rM):
    axis = rM - rS
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None
    axis /= n
    x0, y0, z0 = rM
    dx, dy, dz = axis
    a, b = R_EARTH_A, R_EARTH_B
    aa = (dx*dx + dy*dy)/(a*a) + dz*dz/(b*b)
    bb = 2*((x0*dx + y0*dy)/(a*a) + z0*dz/(b*b))
    cc = (x0*x0 + y0*y0)/(a*a) + z0*z0/(b*b) - 1.0
    disc = bb*bb - 4*aa*cc
    if disc < 0:
        return None
    root = np.sqrt(disc)
    l1 = (-bb - root)/(2*aa)
    l2 = (-bb + root)/(2*aa)
    cands = [l for l in (l1, l2) if l > 0]
    if not cands:
        return None
    return rM + min(cands)*axis

def eci_to_itrs_batch(r_eci, times):
    rep = CartesianRepresentation(r_eci[:,0]*u.m, r_eci[:,1]*u.m, r_eci[:,2]*u.m)
    gcrs = GCRS(rep, obstime=times)
    itrs = gcrs.transform_to(ITRS(obstime=times))
    return np.column_stack((
        itrs.cartesian.x.to_value(u.m),
        itrs.cartesian.y.to_value(u.m),
        itrs.cartesian.z.to_value(u.m)
    ))

def solar_elevation(lat, lon, rS_ecef):
    """
    Approximate solar elevation angle (degrees) for a grid of points.
    rS_ecef : Sun position in ECEF at the chosen epoch [m]
    """
    # Unit vector toward the Sun
    sun_dir = rS_ecef / np.linalg.norm(rS_ecef)
    
    # ECEF positions of the grid
    xyz = geodetic_to_ecef(lat.ravel(), lon.ravel())
    # Local up (geocentric approximation is fine for this purpose)
    up = xyz / np.linalg.norm(xyz, axis=1)[:, None]
    
    cos_zenith = np.sum(up * sun_dir, axis=1)
    elev = np.degrees(np.arcsin(np.clip(cos_zenith, -1.0, 1.0)))
    return elev.reshape(lat.shape)

def add_day_night(ax, lat_grid, lon_grid, rS_ecef, alpha=0.35):
    """
    Shade the night side on an existing cartopy axis.
    """
    elev = solar_elevation(lat_grid, lon_grid, rS_ecef)
    
    # Night mask
    night = np.ma.masked_where(elev >= 0, np.ones_like(elev))
    
    ax.contourf(lon_grid, lat_grid, night,
                levels=[0.5, 1.5],
                colors=["#1a1a2e"],
                alpha=alpha,
                transform=ccrs.PlateCarree(),
                zorder=3)


def plot_eclipse_3d_plotly(
    track_lat, track_lon,
    maximum, lat_grid, lon_grid,
    rS_ecef_rep,                 # Sun ECEF vector at representative time (for day/night)
    munich_lat=48.14, munich_lon=11.58
):
    """
    3D globe that re-uses everything already computed in main().
    """

    def ll2xyz(lat, lon, r=1.0):
        lat = np.radians(np.asarray(lat, dtype=float))
        lon = np.radians(np.asarray(lon, dtype=float))
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return x, y, z

    fig = go.Figure()

    # =========================================================
    # 1. Base Earth sphere
    # =========================================================
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#1a5276'], [1, '#2980b9']],
        showscale=False, opacity=0.92,
        hoverinfo='skip', name='Ocean'
    ))

    # =========================================================
    # 2. Coastlines + Borders
    # =========================================================
    def add_lines(feature, color, width=1.8):
        for geom in feature.geometries():
            lines = [geom] if geom.geom_type == 'LineString' else geom.geoms
            for line in lines:
                lons, lats = line.xy
                xs, ys, zs = ll2xyz(lats, lons, r=1.011)
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='skip', showlegend=False
                ))

    add_lines(COASTLINE, 'white', 2.2)
    add_lines(BORDERS,   '#cccccc', 1.0)

    # =========================================================
    # 3. Day / Night shading (correct for Plotly)
    # =========================================================
    # Higher resolution night points
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    u, v = np.meshgrid(u, v)

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Normal = position vector on unit sphere
    normals = np.stack((x, y, z), axis=-1)
    sun_dir = rS_ecef_rep / np.linalg.norm(rS_ecef_rep)

    cos_zenith = np.sum(normals * sun_dir, axis=-1)

    # Create a scalar field: 1 = night, 0 = day
    night_field = np.where(cos_zenith < 0.02, 1.0, 0.0)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=night_field,
        colorscale=[
            [0.0, 'rgba(0,0,0,0)'],          # day → fully transparent
            [1.0, 'rgba(10,15,30,0.75)']     # night → dark semi-transparent
        ],
        showscale=False,
        name='Night side',
        hoverinfo='skip'
    ))
    # =========================================================
    # 4. Obscuration field + contour-like rings
    # =========================================================
    levels = [10, 25, 50, 75, 90, 95, 99]
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#ff9f1c', '#ff0000']

    for lev, col in zip(levels, colors):
        mask = (maximum >= lev) & (maximum < lev + 8)   # thin band for "contour"
        if np.any(mask):
            xf, yf, zf = ll2xyz(lat_grid[mask], lon_grid[mask], r=1.018)
            fig.add_trace(go.Scatter3d(
                x=xf, y=yf, z=zf,
                mode='markers',
                marker=dict(size=2, color=col, opacity=0.05),
                name=f'{lev}%',
                showlegend=True
            ))

    # Full high-resolution obscuration cloud (semi-transparent)
    mask_high = maximum > 15
    if np.any(mask_high):
        xf, yf, zf = ll2xyz(lat_grid[mask_high], lon_grid[mask_high], r=1.015)
        fig.add_trace(go.Scatter3d(
            x=xf, y=yf, z=zf,
            mode='markers',
            marker=dict(
                size=2.8,
                color=maximum[mask_high],
                colorscale='Inferno',
                cmin=0, cmax=100,
                opacity=0.05,
                colorbar=dict(
                    title=dict(text='Obscuration %', side='right'),
                    x=0.99, len=0.75
                )
            ),
            name='Obscuration field',
            showlegend=False
        ))

    # =========================================================
    # 5. Shadow axis + markers
    # =========================================================
    if len(track_lat) > 0:
        xt, yt, zt = ll2xyz(track_lat, track_lon, r=1.03)
        fig.add_trace(go.Scatter3d(
            x=xt, y=yt, z=zt,
            mode='lines',
            line=dict(color='red', width=8),
            name='Shadow axis'
        ))
        fig.add_trace(go.Scatter3d(
            x=[xt[0]], y=[yt[0]], z=[zt[0]],
            mode='markers', marker=dict(size=8, color='lime'),
            name='Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[xt[-1]], y=[yt[-1]], z=[zt[-1]],
            mode='markers', marker=dict(size=8, color='blue'),
            name='End'
        ))

    # Munich
    xm, ym, zm = ll2xyz(munich_lat, munich_lon, r=1.04)
    fig.add_trace(go.Scatter3d(
        x=[xm], y=[ym], z=[zm],
        mode='markers',
        marker=dict(size=11, color='cyan', symbol='diamond'),
        name='Munich'
    ))

    # =========================================================
    # Layout
    # =========================================================
    fig.update_layout(
        title=dict(text='Solar Eclipse 12 Aug 2026 – 3D Globe with Contours & Day/Night', x=0.5),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.55, y=1.25, z=0.85))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
    )

    fig.show()

# -------------------- MAIN PIPELINE --------------------
def main():
    print("="*60)
    print("SOLAR ECLIPSE GROUND TRACK – generated ephemeris")
    print("="*60)

    # ---- time grid ----
    t0 = Time(START_UTC, scale="utc")
    t1 = Time(END_UTC,   scale="utc")
    n  = int((t1 - t0).sec / DT_SECONDS) + 1
    times = t0 + TimeDelta(np.arange(n)*DT_SECONDS, format="sec")
    times_utc = times.utc
    t_gps = times.gps

    # print(f"Epochs          : {n}")
    print(f"Start UTC       : {times_utc[0].iso}")
    print(f"End   UTC       : {times_utc[-1].iso}")

    # ---- Sun / Moon in GCRS ----
    print("Generating Sun & Moon positions …")
    rS, rM = generate_sun_moon(times)   

    # Munich in ECEF (constant)
    rH_ecef = geodetic_to_ecef(np.array([MUNICH_LAT]), np.array([MUNICH_LON]))[0]
    # Transform Sun & Moon to ITRS for the whole day (or at least the interesting window)
    print("Transforming Sun/Moon to ITRS for observer calculation …")
    rS_itrs = eci_to_itrs_batch(rS, times)
    rM_itrs = eci_to_itrs_batch(rM, times)

    fractions = np.zeros(len(times))
    for i in range(len(times)):
        s = rS_itrs[i] - rH_ecef
        m = rM_itrs[i] - rH_ecef
        ds = np.linalg.norm(s)
        dm = np.linalg.norm(m)
        alpha_s = np.arcsin(np.clip(R_SUN / ds, -1.0, 1.0))
        alpha_m = np.arcsin(np.clip(R_MOON / dm, -1.0, 1.0))
        theta   = angle_between(s[None, :], m[None, :])[0]
        fractions[i] = solar_obscuration(theta, alpha_s, alpha_m)

    best_idx = np.argmax(fractions)
    print(f"\nMunich maximum obscuration : {100*fractions[best_idx]:.1f} %")
    print(f"Time of maximum (UTC)      : {times_utc[best_idx].iso}")
    print(f"Time of maximum (CEST)     : {times_utc[best_idx].to_datetime(timezone=ZoneInfo('Europe/Berlin'))}")




    # ---- Ground track (shadow axis) ----
    # Sample a bit more densely only around the interesting window
    mask = (times_utc >= Time("2026-08-12 15:00:00")) & (times_utc <= Time("2026-08-12 19:30:00"))
    idx  = np.where(mask)[0]
    step = max(1, len(idx)//800)
    idx  = idx[::step]

    print(f"\nGround-track epochs : {len(idx)}")
    rS_ecef = eci_to_itrs_batch(rS[idx], times[idx])
    rM_ecef = eci_to_itrs_batch(rM[idx], times[idx])

    track_lat, track_lon, track_time = [], [], []
    for k in range(len(idx)):
        c = shadow_center_ecef(rS_ecef[k], rM_ecef[k])
        if c is None:
            continue
        lat, lon = ecef_to_geodetic(c)
        if np.dot(c, rS_ecef[k]) > 0:          # day side
            track_lat.append(lat)
            track_lon.append(lon)
            track_time.append(t_gps[idx[k]])
    track_lat = np.asarray(track_lat)
    track_lon = np.asarray(track_lon)

    print(f"Valid track points : {len(track_lat)}")

    if len(track_lat) == 0:
        raise RuntimeError("No Earth intersection found")

    MAP_LON_MIN = np.floor(np.min(track_lon) - 2.0)
    MAP_LON_MAX = np.ceil(np.max(track_lon) + 2.0)
    MAP_LAT_MIN = np.floor(np.min(track_lat) - 2.0)
    MAP_LAT_MAX = np.ceil(np.max(track_lat) + 2.0)

    print("Map longitude:",MAP_LON_MIN,"to",MAP_LON_MAX)
    print("Map latitude:",MAP_LAT_MIN,"to",MAP_LAT_MAX)

    lat_grid = np.arange(MAP_LAT_MIN,MAP_LAT_MAX + GRID_RESOLUTION_DEG,GRID_RESOLUTION_DEG)
    lon_grid = np.arange(MAP_LON_MIN,MAP_LON_MAX + GRID_RESOLUTION_DEG,GRID_RESOLUTION_DEG)

    if len(lat_grid) < 2 or len(lon_grid) < 2:
        raise RuntimeError("Map grid is too small. Check MAP limits and GRID_RESOLUTION_DEG.")

    lon_m,lat_m = np.meshgrid(lon_grid,lat_grid)

    surface = geodetic_to_ecef(lat_m.ravel(),lon_m.ravel())

    maximum = np.zeros(surface.shape[0],dtype=np.float32)
    for k in range(len(idx)):
        if k % 20 == 0:
            print(f"  {k+1}/{len(idx)}", end="\r")
        sun_v  = rS_ecef[k] - surface
        moon_v = rM_ecef[k] - surface
        ds = np.linalg.norm(sun_v, axis=1)
        dm = np.linalg.norm(moon_v, axis=1)
        alpha_s = np.arcsin(np.clip(R_SUN/ds, -1, 1))
        alpha_m = np.arcsin(np.clip(R_MOON/dm, -1, 1))
        theta   = angle_between(sun_v, moon_v)
        frac    = solar_obscuration(theta, alpha_s, alpha_m)
        maximum = np.maximum(maximum, 100*frac)
    print()
    maximum = maximum.reshape(lat_m.shape)

    # ---- Plot ----
    fig=plt.figure(figsize=(14,10))
    ax0=fig.add_subplot(2,1,1,projection=ccrs.PlateCarree())
    ax0.set_extent([MAP_LON_MIN,MAP_LON_MAX,MAP_LAT_MIN,MAP_LAT_MAX],crs=ccrs.PlateCarree())
    ax0.add_feature(cfeature.LAND,facecolor="#dddddd")
    ax0.add_feature(cfeature.OCEAN,facecolor="#b9ddeb")
    ax0.add_feature(cfeature.COASTLINE,linewidth=0.6)
    ax0.add_feature(cfeature.BORDERS,linewidth=0.4)

    levels=[1,10,25,50,75,90,95,99,100]
    cf=ax0.contourf(lon_m,lat_m,maximum,levels=levels,cmap="jet",extend="max",alpha=0.85,transform=ccrs.PlateCarree())
    cs=ax0.contour(lon_m,lat_m,maximum,levels=[10,25,50,75,90,95,99],colors="white",linewidths=0.6,transform=ccrs.PlateCarree())
    ax0.clabel(cs,cs.levels,inline=True,inline_spacing=5,fmt=lambda x:f"{x:.0f}%",fontsize=8,colors="white")
    ax0.contour(lon_m,lat_m,np.where(maximum>=99.9,1,np.nan),levels=[0.5],colors="cyan",linewidths=2.0,transform=ccrs.PlateCarree())
    ax0.plot(track_lon,track_lat,color="red",linewidth=1.8,label="Shadow axis",transform=ccrs.PlateCarree())

    if len(track_lat):
        ax0.scatter(track_lon[0],track_lat[0],color="lime",s=50,zorder=5,label="Start",transform=ccrs.PlateCarree())
        ax0.scatter(track_lon[-1],track_lat[-1],color="blue",s=50,zorder=5,label="End",transform=ccrs.PlateCarree())

    ax0.scatter(MUNICH_LON,MUNICH_LAT,color="cyan",s=120,marker="*",zorder=6,label="Munich",transform=ccrs.PlateCarree())

    k_rep=len(rS_ecef)//2
    rS_rep = rS_ecef[k_rep]
    add_day_night(ax0,lat_m,lon_m,rS_ecef[k_rep],alpha=0.38)

    cbar=fig.colorbar(cf,ax=ax0,orientation="horizontal",pad=0.04,shrink=0.75)
    cbar.set_label("Solar disk obscuration (%)")
    ax0.set_title("Solar Eclipse Ground Track – 12 August 2026\n(generated with Astropy + DE432s)")
    ax0.legend(loc="lower left")

    ax1=fig.add_subplot(2,1,2)
    mask=fractions>0.01
    if np.any(mask):
        t_plot=times_utc[mask].to_datetime(timezone=ZoneInfo("Europe/Berlin"))
        ax1.plot(t_plot,100*fractions[mask],"k-",lw=1.3)
        ax1.set_xlabel("Germany local time (CEST)")
        ax1.set_ylabel("Solar disk obscured (%)")
        ax1.set_title("Observer Solar Eclipse – Munich")
        ax1.grid(True,alpha=0.3)

    fig.tight_layout()
    
    
    
    # ---------- 3D interactive globe ----------
    plot_eclipse_3d_plotly(
        track_lat  = track_lat,
        track_lon  = track_lon,
        maximum    = maximum,
        lat_grid   = lat_m,
        lon_grid   = lon_m,
        rS_ecef_rep = rS_rep,
        munich_lat = MUNICH_LAT,
        munich_lon = MUNICH_LON
    )
    # plt.show()

if __name__ == "__main__":
    main()