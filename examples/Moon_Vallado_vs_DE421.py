# ==================================================
# File: Moon_Vallado_vs_DE421.py
# Author: Bayajid Khan
# Created: 2026-08-20
# Description: 
""" Compare Vallado (2013), 4th ed., p. 288, Algorithm 31 (MOON) 
against JPL DE421. Output: 3-D position error, radial error, 
angular error, statistics, worst cases, CSV file, and plots. 
Coordinates: geocentric equatorial J2000/ICRF-like frame. """

# ==================================================
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jplephem.spk import SPK
from pathlib import Path


kernel_path = Path(__file__).resolve().parent.parent / "de421.bsp"
kernel = SPK.open(str(kernel_path))

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

def datetime_to_jd(dt):
    return dt.to_julian_date()

def vallado_algorithm_31(jd):
    """Vallado 2013, 4th ed., p. 288, Algorithm 31."""
    T = (jd - 2451545.0) / 36525.0

    lon = (218.32 + 481267.8813*T
           + 6.29*np.sin((134.9 + 477198.85*T)*DEG2RAD)
           - 1.27*np.sin((259.2 - 413335.38*T)*DEG2RAD)
           + 0.66*np.sin((235.7 + 890534.23*T)*DEG2RAD)
           + 0.21*np.sin((269.9 + 954397.70*T)*DEG2RAD)
           - 0.19*np.sin((357.5 + 35999.05*T)*DEG2RAD)
           - 0.11*np.sin((186.6 + 966404.05*T)*DEG2RAD))

    lat = (5.13*np.sin((93.3 + 483202.03*T)*DEG2RAD)
           + 0.28*np.sin((228.2 + 960400.87*T)*DEG2RAD)
           - 0.28*np.sin((318.3 + 6003.18*T)*DEG2RAD)
           - 0.17*np.sin((217.6 - 407332.21*T)*DEG2RAD))

    radius = (385000.26
              - 20905.355*np.cos((134.9 + 477198.85*T)*DEG2RAD)
              - 3699.111*np.cos((259.2 - 413335.38*T)*DEG2RAD)
              - 2955.968*np.cos((235.7 + 890534.23*T)*DEG2RAD)
              - 569.925*np.cos((269.9 + 954397.70*T)*DEG2RAD)
              + 48.888*np.cos((357.5 + 35999.05*T)*DEG2RAD)
              - 3.149*np.cos((186.6 + 966404.05*T)*DEG2RAD))

    lon = (lon % 360.0) * DEG2RAD
    lat = lat * DEG2RAD

    x_ecl = radius*np.cos(lat)*np.cos(lon)
    y_ecl = radius*np.cos(lat)*np.sin(lon)
    z_ecl = radius*np.sin(lat)

    eps = 23.439291 * DEG2RAD

    x = x_ecl
    y = np.cos(eps)*y_ecl - np.sin(eps)*z_ecl
    z = np.sin(eps)*y_ecl + np.cos(eps)*z_ecl

    return np.array([x, y, z])

def de421_moon_geocentric(kernel, jd):
    """Moon relative to Earth from DE421, in km."""
    moon_emb = kernel[3, 301].compute(jd)
    earth_emb = kernel[3, 399].compute(jd)
    return moon_emb - earth_emb

def norm(v):
    return np.linalg.norm(v)

def angular_separation(r1, r2):
    a = r1 / norm(r1)
    b = r2 / norm(r2)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    arcsec= np.arccos(c) * RAD2DEG * 3600.0
    return np.arccos(c)*1e6  # returns in µrads

def spherical_elements(r):
    x, y, z = r
    rr = norm(r)
    lon = np.arctan2(y, x) * RAD2DEG % 360.0
    lat = np.arcsin(z / rr) * RAD2DEG
    return lon, lat, rr

def generate_dates(start="2000-01-01", end="2050-01-01", step_hours=24):
    return pd.date_range(start=start, end=end,
                         freq=pd.Timedelta(hours=step_hours))

def run_comparison(start="2000-01-01", end="2050-01-01", step_hours=24):
    # try:
    #     import de421
    #     kernel_path = de421.__path__[0] + "/de421.bsp"
    # except Exception:
    #     kernel_path = "de421.bsp"

    kernel = SPK.open(kernel_path)
    dates = generate_dates(start, end, step_hours)
    results = []

    for dt in dates:
        jd = datetime_to_jd(dt)
        rv = vallado_algorithm_31(jd)
        rd = de421_moon_geocentric(kernel, jd)
        dr = rv - rd

        lv, bv, rv_range = spherical_elements(rv)
        ld, bd, rd_range = spherical_elements(rd)

        dlon = (lv - ld + 180.0) % 360.0 - 180.0
        dlat = bv - bd

        results.append({
            "date": dt,
            "jd": jd,
            "vallado_x_km": rv[0],
            "vallado_y_km": rv[1],
            "vallado_z_km": rv[2],
            "de421_x_km": rd[0],
            "de421_y_km": rd[1],
            "de421_z_km": rd[2],
            "dx_km": dr[0],
            "dy_km": dr[1],
            "dz_km": dr[2],
            "error_3d_km": norm(dr),
            "radial_error_km": rv_range - rd_range,
            "angular_error_µrad": angular_separation(rv, rd),
            "vallado_lon_deg": lv,
            "de421_lon_deg": ld,
            "longitude_error_deg": dlon,
            "vallado_lat_deg": bv,
            "de421_lat_deg": bd,
            "latitude_error_deg": dlat,
            "vallado_range_km": rv_range,
            "de421_range_km": rd_range
        })

    return pd.DataFrame(results)

def print_statistics(df):
    e = df["error_3d_km"].to_numpy()
    radial = np.abs(df["radial_error_km"].to_numpy())
    angular = df["angular_error_µrad"].to_numpy()
    lon = np.abs(df["longitude_error_deg"].to_numpy())
    lat = np.abs(df["latitude_error_deg"].to_numpy())

    print("\n" + "="*60)
    print("VALLADO ALGORITHM 31 vs DE421")
    print("="*60)
    print("\n3-D POSITION ERROR")
    print(f"Mean                 : {np.mean(e):12.3f} km")
    print(f"Median               : {np.median(e):12.3f} km")
    print(f"RMS                  : {np.sqrt(np.mean(e**2)):12.3f} km")
    print(f"Std                  : {np.std(e):12.3f} km")
    print(f"95th percentile      : {np.percentile(e,95):12.3f} km")
    print(f"99th percentile      : {np.percentile(e,99):12.3f} km")
    print(f"Maximum              : {np.max(e):12.3f} km")

    print("\nRADIAL ERROR")
    print(f"Mean absolute        : {np.mean(radial):12.3f} km")
    print(f"RMS                  : {np.sqrt(np.mean(radial**2)):12.3f} km")
    print(f"95th percentile      : {np.percentile(radial,95):12.3f} km")
    print(f"Maximum              : {np.max(radial):12.3f} km")

    print("\nANGULAR SEPARATION")
    print(f"Mean                 : {np.mean(angular):12.3f} µrad")
    print(f"Median               : {np.median(angular):12.3f} µrad")
    print(f"RMS                  : {np.sqrt(np.mean(angular**2)):12.3f} µrad")
    print(f"95th percentile      : {np.percentile(angular,95):12.3f} µrad")
    print(f"99th percentile      : {np.percentile(angular,99):12.3f} µrad")
    print(f"Maximum              : {np.max(angular):12.3f} µrad")

    print("\nLONGITUDE ERROR")
    print(f"Mean absolute        : {np.mean(lon):12.6f} deg")
    print(f"RMS                  : {np.sqrt(np.mean(lon**2)):12.6f} deg")
    print(f"95th percentile      : {np.percentile(lon,95):12.6f} deg")
    print(f"Maximum              : {np.max(lon):12.6f} deg")

    print("\nLATITUDE ERROR")
    print(f"Mean absolute        : {np.mean(lat):12.6f} deg")
    print(f"RMS                  : {np.sqrt(np.mean(lat**2)):12.6f} deg")
    print(f"95th percentile      : {np.percentile(lat,95):12.6f} deg")
    print(f"Maximum              : {np.max(lat):12.6f} deg")
    print("="*60)

def print_worst_cases(df, n=20):
    cols = ["date", "error_3d_km", "radial_error_km",
            "angular_error_µrad", "longitude_error_deg",
            "latitude_error_deg"]
    print(f"\nWORST {n} CASES")
    print(df.nlargest(n, "error_3d_km")[cols].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))


def plot_results(df):
    """Timeline of Vallado vs DE421 errors with global statistics."""

    # Data
    e = df["error_3d_km"].to_numpy()
    r = np.abs(df["radial_error_km"].to_numpy())
    a = df["angular_error_µrad"].to_numpy()

    # Statistics helper
    def stats(x):
        return (
            np.std(x),
            np.sqrt(np.mean(x**2)),
            np.percentile(x, 95),
            np.percentile(x, 99)
        )

    e_std, e_rms, e_p95, e_p99 = stats(e)
    r_std, r_rms, r_p95, _     = stats(r)
    a_std, a_rms, a_p95, _     = stats(a)

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    def timeline(axis, x, ylabel, color, log=False):
        std, rms, p95, p99 = stats(x)

        axis.plot(df["date"], x, color=color, lw=0.7,
                  label="Error")

        axis.axhline(std, color="#F2CF5B", ls=":", lw=1.4,
                     label=f"STD {std:.2f}")
        axis.axhline(rms, color="#F58518", ls="--", lw=1.4,
                     label=f"RMS {rms:.2f}")
        axis.axhline(p95, color="#E45756", ls="-.", lw=1.5,
                     label=f"P95 {p95:.2f}")

        if log:
            axis.set_yscale("log")

        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.2)
        axis.legend(loc="upper right", fontsize=8, ncol=3)

    # 3-D position error
    timeline(ax[0], e,"3-D error [km]","#2C7FB8"    )
    ax[0].set_title("Vallado Algorithm 31 vs JPL DE421",fontsize=14, fontweight="bold"    )

    # Radial error
    timeline(ax[1], r,"Radial error [km]","#2CA25F"    )

    # Angular error
    timeline(ax[2], a,"Angular error [µrad]","#756BB1",log=True    )

    ax[2].set_xlabel("Year")

    plt.tight_layout()
    #plt.show()


def plot_statistics(df):
    """
    Create presentation-quality statistical plots for
    Vallado Algorithm 31 vs JPL DE421.

    Produces:
        1. 3-D error summary bar chart
        2. Concentric-circle error visualization
        3. Error distribution / CDF
        4. Radial vs angular error summary
    """

    # --------------------------------------------------
    # Extract statistics
    # --------------------------------------------------

    e = df["error_3d_km"].to_numpy()
    radial = np.abs(df["radial_error_km"].to_numpy())
    angular = df["angular_error_µrad"].to_numpy()

    stats_3d = {
        "Mean": np.mean(e),
        "Median": np.median(e),
        "STD": np.std(e),
        "RMS": np.sqrt(np.mean(e**2)),
        "P95": np.percentile(e, 95),
        "P99": np.percentile(e, 99),
        "Maximum": np.max(e),
    }

    stats_radial = {
        "Mean": np.mean(radial),
        "RMS": np.sqrt(np.mean(radial**2)),
        "P95": np.percentile(radial, 95),
        "Maximum": np.max(radial),
    }

    stats_angular = {
        "Mean": np.mean(angular),
        "RMS": np.sqrt(np.mean(angular**2)),
        "P95": np.percentile(angular, 95),
        "Maximum": np.max(angular),
    }

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # ==================================================
    # 1. 3-D ERROR STATISTICS
    # ==================================================

    ax1 = fig.add_subplot(gs[0, 0])

    labels = ["Mean", "Median", "STD", "RMS", "P95", "P99", "Maximum"]
    values = [stats_3d[x] for x in labels]

    colors = [
        "#4C78A8",
        "#72B7B2",
        "#F2CF5B",
        "#F58518",
        "#E45756",
        "#B279A2",
        "#54A24B"
    ]

    bars = ax1.barh(labels, values, color=colors, edgecolor="black",
                    linewidth=0.5)

    ax1.set_xlabel("Position error [km]")
    ax1.set_title("3-D Position Error Statistics",
                  fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    # Put values at end of bars
    xmax = max(values)

    for bar, value in zip(bars, values):ax1.text(value + xmax * 0.015,bar.get_y() 
                + bar.get_height() / 2,f"{value:.2f}",va="center",fontsize=9)

    ax1.set_xlim(0, xmax * 1.18)

    # ==================================================
    # 2. CONCENTRIC ERROR CIRCLES
    # ==================================================

    ax2 = fig.add_subplot(gs[0, 1])

    # Main statistics we want to emphasize
    circle_stats = [
        ("STD", stats_3d["STD"], "#F2CF5B"),
        ("RMS", stats_3d["RMS"], "#F58518"),
        ("P95", stats_3d["P95"], "#E45756"),
        ("P99", stats_3d["P99"], "#B279A2"),
        ("Maximum", stats_3d["Maximum"], "#54A24B"),
    ]

    max_radius = stats_3d["Maximum"]

    # Draw largest circles first
    for name, radius, color in reversed(circle_stats):
        circle = plt.Circle((0, 0),radius,fill=False,linewidth=2.5,color=color,alpha=0.85)
        ax2.add_patch(circle)

        # Label near upper-right part of circle
        angle = 35 * DEG2RAD
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        ax2.text(x,y,f"{name}\n{radius:.2f} km",color=color,
                 fontsize=9,fontweight="bold",ha="left",va="bottom")

    # Center point
    ax2.scatter(0, 0, s=35, color="black", zorder=5)
    ax2.set_xlim(-max_radius * 1.15, max_radius * 1.15)
    ax2.set_ylim(-max_radius * 1.15, max_radius * 1.15)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Error scale [km]")
    ax2.set_ylabel("Error scale [km]")

    ax2.set_title("Concentric Error Scales",fontsize=13,fontweight="bold")

    ax2.grid(True, alpha=0.2)

    # ==================================================
    # 3. CUMULATIVE DISTRIBUTION
    # ==================================================

    ax3 = fig.add_subplot(gs[1, 0])

    sorted_error = np.sort(e)
    cumulative = np.arange(1, len(e) + 1) / len(e) * 100

    ax3.plot(sorted_error,cumulative,color="#4C78A8",linewidth=2    )

    # P95 marker
    p95 = stats_3d["P95"]

    ax3.axvline(p95,color="#E45756",linestyle="--",linewidth=1.8,label=f"P95 = {p95:.2f} km"    )

    ax3.axhline(
        95,
        color="#E45756",
        linestyle=":",
        linewidth=1.2
    )

    # RMS marker
    rms = stats_3d["RMS"]

    ax3.axvline(
        rms,
        color="#F58518",
        linestyle="--",
        linewidth=1.8,
        label=f"RMS = {rms:.2f} km"
    )

    ax3.set_xlabel("3-D position error [km]")
    ax3.set_ylabel("Cumulative probability [%]")

    ax3.set_title(
        "Cumulative Distribution of 3-D Error",
        fontsize=13,
        fontweight="bold"
    )

    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.25)

    ax3.legend(frameon=True)

    # ==================================================
    # 4. RADIAL / ANGULAR SUMMARY
    # ==================================================

    ax4 = fig.add_subplot(gs[1, 1])

    # Convert angular µrad to equivalent transverse distance
    # using mean Earth-Moon distance. This gives a physically
    # intuitive comparison with radial error.
    mean_range = np.mean(df["de421_range_km"])

    angular_km = angular * 1e-6 * mean_range

    comparison_labels = [
        "3-D\nRMS",
        "Radial\nRMS",
        "Angular\nRMS",
        "3-D\nP95",
        "Radial\nP95",
        "Angular\nP95"
    ]

    comparison_values = [
        stats_3d["RMS"],
        stats_radial["RMS"],
        stats_angular["RMS"] * 1e-6 * mean_range,
        stats_3d["P95"],
        stats_radial["P95"],
        stats_angular["P95"] * 1e-6 * mean_range
    ]

    comparison_colors = [
        "#4C78A8",
        "#72B7B2",
        "#F2CF5B",
        "#E45756",
        "#54A24B",
        "#B279A2"
    ]

    bars = ax4.bar(
        comparison_labels,
        comparison_values,
        color=comparison_colors,
        edgecolor="black",
        linewidth=0.5
    )

    for bar, value in zip(bars, comparison_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax4.set_ylabel("Equivalent error [km]")
    ax4.set_title(
        "Radial / Angular Error Contribution",
        fontsize=13,
        fontweight="bold"
    )

    ax4.grid(axis="y", alpha=0.25)

    # --------------------------------------------------
    # Overall title
    # --------------------------------------------------

    fig.suptitle(
        "Vallado Algorithm 31 vs JPL DE421 — Statistical Error Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    

if __name__ == "__main__":
    df = run_comparison(start="2000-01-01",end="2050-01-01",step_hours=24)

    print_statistics(df)
    # print_worst_cases(df, n=20)

    df.to_csv("vallado_algorithm31_vs_de421.csv",index=False)
    print("\nResults saved to vallado_algorithm31_vs_de421.csv")

    # Time history
    plot_results(df)

    # Statistical plots
    plot_statistics(df)
    plt.show()



# %%
