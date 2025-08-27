#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal Summary Analysis for NC Fault Creep

Author: Danielle Lindsay
Date: 2024-11-18

This script:
- Loads alignment array displacement data and precipitation (PRISM) + seismicity (NCEDC).
- Computes long-term velocities, step sizes, detrended time series.
- Fits decay models following last step.
- Performs seasonal harmonic velocity analysis.
- Plots displacement, detrended, velocity, precipitation, and seismicity in a 3x2 panel figure.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import linregress, pearsonr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Point, Polygon
import geopandas as gpd

import numpy as np
from scipy.optimize import curve_fit, brentq

from NC_creep_filepaths import common_paths
import creep_utils as utils


# ============================================================
# CONFIGURATION / PARAMETERS
# ============================================================

STA_ID = "MWIL"                                # Station ID
REF_LON, REF_LAT = -123.35612, 39.41242        # Reference location for distance calc

# Step-change times (decimal years) and fitting window (yrs)
STEPS = [1994.118, 1996.965, 2002.3315]
STEP_WIN = 1.5

# Seasonal velocity analysis bounds
SEASON_START, SEASON_END = 1992.0, 2008.0

# Shaded highlight region
START_SHADE = pd.Timestamp("2002-03-30", tz="UTC")
END_SHADE   = pd.Timestamp("2002-06-02", tz="UTC")


# Plotting
FIGSIZE = (15, 9)
SEG_COLORS = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
DECAY_COLORS = {"Exponential": "green", "Power-law": "purple",
                "Logarithmic": "brown", "Stretched-exp": "teal"}
CMAP_EQ = "magma"
CUM_MOMENT_COLOR = "#023E8A"
CUM_COUNT_COLOR  = "#e34a33"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_decimal_year(df, date_col="date"):
    """Add decimal year column from datetime column."""
    df[date_col] = pd.to_datetime(df[date_col])

    def to_decimal_year(ts):
        year_start = pd.Timestamp(year=ts.year, month=1, day=1)
        year_end   = pd.Timestamp(year=ts.year + 1, month=1, day=1)
        year_len   = (year_end - year_start).days
        return ts.year + (ts - year_start).days / year_len

    df["decimal_year"] = df[date_col].apply(to_decimal_year)
    return df


def add_cumulative_precip(df, date_col="Date", precip_col="Precipitation (mm)"):
    """Add cumulative precipitation that resets each water year (Oct–Sep)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["water_year"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "water_year"] += 1
    df["cumulative_precip_mm"] = df.groupby("water_year")[precip_col].cumsum()
    return df


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between two points given lon/lat in degrees."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # Earth radius in km


def fit_segment(x, y):
    """Fit linear regression and return slope, intercept, stderr, n."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    s, c, _, _, se = linregress(x, y)
    return s, c, se, len(x)

# --- Shaded highlight region (decimal years) ---
def to_decimal_year(ts):
    year_start = pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")
    year_end   = pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")
    return ts.year + (ts - year_start).days / (year_end - year_start).days


# ============================================================
# DATA LOADING
# ============================================================

# Alignment array metadata
AA_table_path = common_paths["AA"]["table"]
AA_dir        = common_paths["AA"]["dir"]

# Precipitation
precip_monthly = pd.read_csv(common_paths["PRISM"]["monthly"])
precip_monthly = add_decimal_year(precip_monthly, date_col="Date")
precip_monthly = add_cumulative_precip(precip_monthly)

precip_daily = pd.read_csv(common_paths["PRISM"]["daily"])
precip_daily = add_decimal_year(precip_daily, date_col="date")
precip_daily_sorted = precip_daily.sort_values("decimal_year").copy()
precip_daily['cum_precip'] = precip_daily['rain_mm'].cumsum()

shade_start_dec = to_decimal_year(START_SHADE)
shade_end_dec   = to_decimal_year(END_SHADE)

# -------------------------
# Earthquake data
# -------------------------
eq_data = pd.read_csv(common_paths["NCEDC"]["MWIL_15km"])
eq_data["time"] = pd.to_datetime(eq_data["time"], utc=True)

# Decimal year conversion
eq_data["decimal_year"] = eq_data["time"].apply(
    lambda ts: ts.year + (ts - pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")).days /
               (pd.Timestamp(year=ts.year+1, month=1, day=1, tz="UTC") -
                pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")).days
)

# Window around step
window_mask = (eq_data['decimal_year'] >= STEPS[2] - 1.5) & \
              (eq_data['decimal_year'] <= STEPS[2] + 1.5)
eq_window = eq_data.loc[window_mask].copy()

# Seismic moment (Hanks & Kanamori, 1979)
eq_window["seismic_moment"] = 10**(1.5*eq_window["mag"] + 16.1)

# Distance from reference location
eq_window["distance"] = haversine(
    REF_LON, REF_LAT,
    eq_window["longitude"], eq_window["latitude"]
)

# ---- Sort by time for cumulative calcs ----
eq_window_sorted = eq_window.sort_values("time").copy()
eq_window_sorted['cum_seismic_moment'] = eq_window_sorted['seismic_moment'].cumsum()
eq_window_sorted['cum_count'] = np.arange(1, len(eq_window_sorted) + 1)

# Normalize (0–1)
norm_moment = eq_window_sorted['cum_seismic_moment'] / eq_window_sorted['cum_seismic_moment'].max()
norm_count  = eq_window_sorted['cum_count'] / eq_window_sorted['cum_count'].max()

# ---- Sort by distance for scatter plotting ----
eq_window_dist = eq_window_sorted.sort_values("distance", ascending=False).copy()

# ============================================================
# ALIGNMENT ARRAY DATA (example for MWIL)
# ============================================================

# Read the Table 1b.csv file into a DataFrame, skipping the first line and using the third as the header
AA_table_df = pd.read_csv(AA_table_path, encoding='ISO-8859-1', skiprows=1)

# Remove leading/trailing whitespace from column names
AA_table_df.columns = AA_table_df.columns.str.strip()

# Extract the station codes and relevant information
AA_station_info = AA_table_df[['Site Code', 'Fault', 'Site Name', 'Longitude (WGS84)', 'Latitude (WGS84)']].dropna()
AA_station_info = AA_station_info.rename(columns={
    'Site Code': 'Sta_ID',
    'Longitude (WGS84)': 'Lon',
    'Latitude (WGS84)': 'Lat'
})
AA_station_info = AA_station_info.sort_values(by="Lat", ascending=False)

# Initialize new columns for velocity and standard error
AA_station_info['Velocity(mm/yr)'] = pd.NA
AA_station_info['Std_Err'] = pd.NA

# Initialize an empty dictionary to hold station data
AA_stations_dict = {}

# Function to load station data and update AA_station_info
def load_station_data(AA_station_info):
    global AA_stations_dict  # Use global dictionary to store station data

    # Process each station file based on the extracted station info
    for index, row in AA_station_info.iterrows():
        station_code = row['Sta_ID']
        station_file = f"{station_code}.csv"
        
        # Define the full file path for the station file
        file_path = f'{AA_dir}/{station_file}'

        # Check if the file exists before attempting to read it
        if os.path.exists(file_path):
            # Read the corresponding station file into a DataFrame, using the 3rd line (index 2) as the header
            station_data = pd.read_csv(file_path, header=2, encoding='ISO-8859-1')
            
            # Remove leading/trailing whitespace from column names
            station_data.columns = station_data.columns.str.strip()
            
            # Check if required columns are present
            if 'Year' in station_data.columns and 'Cumulative movement (mm)' in station_data.columns:
                # Filter out non-numeric and NaN entries for 'Year'
                valid_years = station_data['Year'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna().tolist()
                
                # Filter out non-numeric and NaN entries for 'Cumulative movement (mm)'
                valid_cumulative_movements = station_data['Cumulative movement (mm)'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna().tolist()

                start_year = valid_years[0] if valid_years else None  # First valid year if it exists
                end_year = valid_years[-1] if valid_years else None   # Last valid year if it exists
            else:
                print(f"Error: Required columns not found in {station_file}")
                continue  # Skip processing for this file
            
            if station_code == "BPCE":
                valid_cumulative_movements.insert(0, 0.0)
                
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(valid_years, valid_cumulative_movements)
            
            # Print to confirm correct slope and standard error
            #print(f"Station: {station_code}, Slope: {slope}, Std_Err: {std_err}")

            # Add velocity and standard error to the DataFrame
            AA_station_info.loc[index, 'Velocity(mm/yr)'] = slope
            AA_station_info.loc[index, 'Std_Err'] = std_err
            
            # Populate the dictionary with station information and raw measurements
            AA_stations_dict[station_code] = {
                'Sta_ID': station_code,
                'Lat': row['Lat'],  
                'Lon': row['Lon'],  
                'Start': start_year, 
                'End': end_year,
                'Year': valid_years,
                'Cumulative movement (mm)': valid_cumulative_movements,
            }
        else:
            print(f"File not found: {station_file}")

    return AA_station_info  # Explicitly return the updated DataFrame

# Load station data and reassign the updated DataFrame
AA_station_info = load_station_data(AA_station_info)


# Function to check for mismatched lengths of 'Year' and 'Cumulative movement (mm)'
def check_length_mismatch(AA_stations_dict):
    for station_code, data in AA_stations_dict.items():
        years = data['Year']
        cumulative_movements = data['Cumulative movement (mm)']
        
        if len(years) != len(cumulative_movements):
            print(f"Error: Length mismatch in {station_code}. Year count: {len(years)}, Cumulative movement count: {len(cumulative_movements)}")

# Perform the length check
check_length_mismatch(AA_stations_dict)

# Extract time and displacement from the data
data = AA_stations_dict[STA_ID]
years_all = np.asarray(data["Year"], dtype=float)                     # decimal years
disp_all  = np.asarray(data["Cumulative movement (mm)"], dtype=float) # mm

order = np.argsort(years_all)
years_all = years_all[order]
disp_all  = disp_all[order]

# ============================================================
# Long-term trend & velocity
# ============================================================

# Linear regression on full displacement record
slope, intercept, r_value, p_value, std_err = linregress(years_all, disp_all)
trend_all = slope * years_all + intercept
det_all   = disp_all - trend_all
vel_all   = np.gradient(disp_all, years_all)

print(f"[TREND] Overall linear velocity = {slope:.3f} ± {std_err:.3f} mm/yr (R²={r_value**2:.3f})")

# Sampling intervals (pre/post 2007)
dt = np.diff(years_all)
pre_mask_dt  = years_all[:-1] < 2007.0
post_mask_dt = ~pre_mask_dt
mean_sampling_interval_pre_2007  = float(np.mean(dt[pre_mask_dt]))  if np.any(pre_mask_dt)  else np.nan
mean_sampling_interval_post_2007 = float(np.mean(dt[post_mask_dt])) if np.any(post_mask_dt) else np.nan
print(f"[SAMPLE] Mean Δt pre-2007 = {mean_sampling_interval_pre_2007:.3f} yr, post-2007 = {mean_sampling_interval_post_2007:.3f} yr")

# ============================================================
# Segments & step sizes
# ============================================================

t_edges = [years_all.min(), STEPS[0], STEPS[1], STEPS[2], 2012.0, 2023.0]  # force segment end at 2007


# Inter-event segments (on raw displacement)
segments = []
for i in range(len(t_edges) - 1):
    m = (years_all >= t_edges[i]) & (years_all <= t_edges[i+1])
    s_i, c_i, se_i, n_i = fit_segment(years_all[m], disp_all[m])
    segments.append(dict(mask=m, t0=t_edges[i], t1=t_edges[i+1],
                         slope=s_i, intercept=c_i, stderr=se_i, n=n_i))

print("\n[INTERVAL VELOCITIES] (raw displacement)")
for k, seg in enumerate(segments, 1):
    print(f"  S{k}: {seg['t0']:.3f}–{seg['t1']:.3f}  "
          f"{seg['slope']:.2f} ± {seg['stderr']:.2f} mm/yr  (n={seg['n']})")
    
print("\nS4 velocity decrese from linear rate:",
      f"\n{np.round(((slope-segments[3]['slope'])/slope)*100,1)}% reduction")

# Step size estimation (raw displacement, local fits)
def step_size_raw_displacement(t_step, win=STEP_WIN):
    """Displacement step (mm) at time t_step using local fits before/after."""
    eps = 1e-6
    pre_mask  = (years_all >= (t_step - win)) & (years_all <  (t_step - eps))
    post_mask = (years_all >  (t_step + eps)) & (years_all <= (t_step + win))
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return np.nan, (np.nan, np.nan, 0), (np.nan, np.nan, 0)
    sp, cp, sep, npre   = fit_segment(years_all[pre_mask], disp_all[pre_mask])
    so, co, seo, npost  = fit_segment(years_all[post_mask], disp_all[post_mask])
    step_mm = (so * t_step + co) - (sp * t_step + cp)
    return step_mm, (sp, sep, npre), (so, seo, npost)

print(f"\n[STEP SIZES] (+/- {STEP_WIN})")
step_results = {}
for t in STEPS:
    step_mm, pre_stats, post_stats = step_size_raw_displacement(t)
    sp, sep, npre   = pre_stats
    so, seo, npost  = post_stats
    print(f"  @ {t:.4f}: {step_mm:.2f} mm | pre {sp:.2f}±{sep:.2f} (n={npre}) | "
          f"post {so:.2f}±{seo:.2f} (n={npost})")
    step_results[t] = dict(step_mm=step_mm, pre_slope=sp, post_slope=so, win=STEP_WIN)

# # ============================================================
# # Detrended modeling & decay after last step
# # ============================================================

import numpy as np
from scipy.optimize import curve_fit, brentq

# --- Last step time & size (from your step_results) ---
t_last = float(STEPS[-1])
step_size_last = float(step_results[t_last]["step_mm"])
last_Step_size = step_size_last  # if you need this alias

# --- Post-step series (detrended displacement minus the step) ---
post_mask = years_all > t_last
x_post = years_all[post_mask]
y_post = det_all[post_mask] - step_size_last

if len(x_post) < 3:
    print("[DECAY] Not enough post-step samples.")
else:
    # --------- helpers & models ----------
    def rmse(y, yhat):
        y = np.asarray(y, float); yhat = np.asarray(yhat, float)
        return float(np.sqrt(np.mean((y - yhat)**2)))

    def exp_model(t, a, tau, c, t0):   return a*np.exp(-(t - t0)/tau) + c
    def log_model(t, a, tau, c, t0):   return a*np.log1p((t - t0)/tau) + c
    def pwr_model(t, a, b, c, t0):     return a*np.maximum(t - t0, 1e-9)**(-b) + c
    def biexp_model(t, A1, tau1, A2, tau2, c, t0):
        return A1*np.exp(-(t - t0)/tau1) + A2*np.exp(-(t - t0)/tau2) + c

    models = {}
    n = len(x_post)
    late_med = float(np.median(y_post[-max(3, n//5):]))
    span = max(x_post[-1] - x_post[0], 1.0)

    # Exponential
    try:
        p0 = [y_post[0] - late_med, max(span/3, 0.5), late_med]
        popt, _ = curve_fit(lambda t,a,tau,c: exp_model(t,a,tau,c,t_last),
                            x_post, y_post, p0=p0,
                            bounds=([-np.inf,1e-4,-np.inf],[np.inf,np.inf,np.inf]),
                            maxfev=50000)
        yhat = exp_model(x_post, *popt, t_last)
        models["Exponential"] = dict(params=popt, yhat=yhat, rmse=rmse(y_post,yhat))
    except Exception as e:
        print("[WARN] Exponential fit failed:", e)

    # Logarithmic
    try:
        p0 = [y_post[0] - late_med, max(span/3, 0.5), late_med]
        popt, _ = curve_fit(lambda t,a,tau,c: log_model(t,a,tau,c,t_last),
                            x_post, y_post, p0=p0,
                            bounds=([-np.inf,1e-4,-np.inf],[np.inf,np.inf,np.inf]),
                            maxfev=50000)
        yhat = log_model(x_post, *popt, t_last)
        models["Logarithmic"] = dict(params=popt, yhat=yhat, rmse=rmse(y_post,yhat))
    except Exception as e:
        print("[WARN] Logarithmic fit failed:", e)

    # Power-law
    try:
        p0 = [y_post[0] - late_med, 1.0, late_med]
        popt, _ = curve_fit(lambda t,a,b,c: pwr_model(t,a,b,c,t_last),
                            x_post, y_post, p0=p0,
                            bounds=([-np.inf,1e-3,-np.inf],[np.inf,5.0,np.inf]),
                            maxfev=50000)
        yhat = pwr_model(x_post, *popt, t_last)
        models["Power-law"] = dict(params=popt, yhat=yhat, rmse=rmse(y_post,yhat))
    except Exception as e:
        print("[WARN] Power-law fit failed:", e)

    # Bi-exponential
    try:
        A_tot = float(y_post[0] - late_med)
        p0 = [0.6*A_tot, max(span/10, 0.3), 0.4*A_tot, max(span/2, 1.0), late_med]
        popt, _ = curve_fit(lambda t,a1,tau1,a2,tau2,c: biexp_model(t,a1,tau1,a2,tau2,c,t_last),
                            x_post, y_post, p0=p0,
                            bounds=([-np.inf,1e-4,-np.inf,1e-4,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf]),
                            maxfev=60000)
        yhat = biexp_model(x_post, *popt, t_last)
        models["Bi-exponential"] = dict(params=popt, yhat=yhat, rmse=rmse(y_post,yhat))
    except Exception as e:
        print("[WARN] Bi-exponential fit failed:", e)

    # --- RMSEs and best model ---
    print("\n[DECAY] Post-step decay fits (sorted by RMSE):")
    if not models:
        print("  No valid fits.")
        best_name = None
    else:
        for name, m in sorted(models.items(), key=lambda kv: kv[1]["rmse"]):
            print(f"  {name:<14} RMSE={m['rmse']:.4f}")
        best_name = min(models, key=lambda n: models[n]["rmse"])

    # --- Bi-exponential basics (optional) ---
    if "Bi-exponential" in models:
        A1, tau1, A2, tau2, c_bi = models["Bi-exponential"]["params"]

        def mag(T):  # |y(T)-c|
            return abs(A1*np.exp(-(T - t_last)/tau1) + A2*np.exp(-(T - t_last)/tau2))
        init_mag = mag(t_last + 1e-9)

        def _solve(frac, max_years=200.0):
            target = frac * init_mag
            f = lambda T: mag(T) - target
            lo = t_last
            hi = t_last + max(5.0*max(tau1, tau2), 5.0)
            tries = 0
            while np.sign(f(lo)) == np.sign(f(hi)) and hi - t_last < max_years:
                hi += 2.0*max(tau1, tau2); tries += 1
                if tries > 50: break
            return float(brentq(f, lo, hi)) if np.sign(f(lo)) != np.sign(f(hi)) else np.nan

        t_half = _solve(0.5); t_95 = _solve(0.05)
        d_half = (t_half - t_last) if np.isfinite(t_half) else np.nan
        d_95   = (t_95   - t_last) if np.isfinite(t_95)   else np.nan

        print("\n[Bi-exp basics]")
        print(f"  tau1={tau1:.2f} yr, tau2={tau2:.2f} yr")
        if np.isfinite(t_half): print(f"  t_half = {t_half:.3f}  (Δ = {d_half:.3f} yr)")
        if np.isfinite(t_95):   print(f"  t_95   = {t_95:.3f}  (Δ = {d_95:.3f} yr)")

    
# ============================================================
# Seasonal harmonic VELOCITY analysis (panel c style, no legend)
# ============================================================

# --- helpers (local to this block) ---
def harmonic_model(t, v0, a1, b1):
    return v0 + a1*np.sin(2*np.pi*t) + b1*np.cos(2*np.pi*t)

def fit_harmonic_with_uncertainty(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(t), np.sin(2*np.pi*t), np.cos(2*np.pi*t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    v0, a1, b1 = beta
    A   = np.hypot(a1, b1)
    phi = np.arctan2(b1, a1)                            # <-- NOTE ordering
    peak_frac = (0.25 - phi/(2*np.pi)) % 1.0            # fraction-of-year of seasonal peak

    # covariance & uncertainty
    resid = y - X @ beta
    dof = len(y) - 3
    sigma2 = np.sum(resid**2) / max(dof, 1)
    covB = sigma2 * np.linalg.inv(X.T @ X)
    var_a1, var_b1, cov_ab = covB[1,1], covB[2,2], covB[1,2]
    dphi_da = -b1/(a1**2 + b1**2 + 1e-12)
    dphi_db =  a1/(a1**2 + b1**2 + 1e-12)
    var_phi = (dphi_da**2)*var_a1 + (dphi_db**2)*var_b1 + 2*dphi_da*dphi_db*cov_ab
    phi_std = float(np.sqrt(max(var_phi, 0.0)))
    peak_std = phi_std/(2*np.pi)                         # in fraction of a year
    return dict(v0=float(v0), a1=float(a1), b1=float(b1),
                A=float(A), phi=float(phi),
                peak_frac=float(peak_frac), peak_std=float(peak_std))

import calendar
def _frac_to_month_day(frac):
    doy = frac * 365.2425
    remain = doy
    for m in range(1, 13):
        dim = calendar.monthrange(2001, m)[1]
        if remain <= dim:
            return calendar.month_name[m], int(round(remain))
        remain -= dim
    return "December", 31

# --- data slice and derivative (no smoothing) ---
m_season    = (years_all >= SEASON_START) & (years_all <= SEASON_END)
m_season    = (years_all >= t_edges[0]) & (years_all <= SEASON_END)
years_season = years_all[m_season]
vel_season   = np.gradient(disp_all[m_season], years_all[m_season])


# --- fit & diagnostics ---
vf = fit_harmonic_with_uncertainty(years_season, vel_season)
t_fit_v = np.linspace(years_season.min(), years_season.max(), 1200)
y_fit_v = harmonic_model(t_fit_v, vf["v0"], vf["a1"], vf["b1"])

# mean observed–model peak timing offset (circular, by year)
def _frac(x): return x - np.floor(x)
obs_fracs = []
for Y in range(int(np.floor(years_season.min())), int(np.ceil(years_season.max()))+1):
    mY = (years_season >= Y) & (years_season < Y+1)
    if np.any(mY):
        i = np.argmax(vel_season[mY])
        obs_fracs.append(_frac(years_season[mY][i]))
if len(obs_fracs):
    diffs = [((_frac - vf["peak_frac"] + 0.5) % 1.0) - 0.5 for _frac in obs_fracs]
    mean_offset_years = float(np.mean(diffs))
else:
    mean_offset_years = np.nan
mean_offset_days = mean_offset_years * 365.2425

# --- print stats ---
phi_deg = np.degrees(vf["phi"])
phi_std_deg = np.degrees(vf["peak_std"]*2*np.pi)
peak_days = int(round(vf["peak_frac"] * 365.2425))
peak_days_std = int(round(vf["peak_std"] * 365.2425))

v_mo, v_day = _frac_to_month_day(vf["peak_frac"])
v_day_std = int(round(vf["peak_std"] * 365.2425))


print(f"[VELOCITY] \nv0={vf['v0']:.3f} mm/yr, A={vf['A']:.3f} mm/yr, φ={phi_deg:.1f}°")
print(f"Peak fraction = {vf['peak_frac']:.3f} ± {vf['peak_std']:.3f} "
      f"(~day {peak_days} ± {peak_days_std} d)\n",
      f"Peak ~ {v_mo} {v_day} (± {v_day_std} days)\n")
print(f"Mean observed–model peak offset = {mean_offset_years:+.3f} yr "
      f"(~{mean_offset_days:+.1f} d)\n")


# --- Calculate mean velocity for post 2007 
vel_all   = np.gradient(disp_all, years_all)

time_periods = list(zip(t_edges[:-1], t_edges[1:]))  # create start/end pairs
v0_periods = {}

for i, (start, end) in enumerate(time_periods, start=1):
    mask = (years_all > start) & (years_all < end)
    if mask.any():
        v0 = np.mean(vel_all[mask])   # mean velocity for that period
        v0_periods[f"T{i}"] = v0
        print(f"T{i}: v0 = {v0:.2f} mm/yr, start={start}, end={end}")
    else:
        v0_periods[f"T{i}"] = np.nan
        print(f"T{i}: no data in this period ({start}–{end})")

# ------------------------------------------------------------
# STEP-YEAR PEAK OFFSETS (Velocity vs. Harmonic model)
# ------------------------------------------------------------
print("\n[STEP-YEAR PEAK OFFSETS]")

# 1) Build per-year observed peak times from the raw velocity
obs_peaks_by_year = {}
year_lo = int(np.floor(years_season.min()))
year_hi = int(np.ceil(years_season.max()))
for Y in range(year_lo, year_hi + 1):
    mY = (years_season >= Y) & (years_season < Y + 1)
    if np.any(mY):
        idx = np.argmax(vel_season[mY])              # observed annual max
        obs_t = float(years_season[mY][idx])         # absolute time (decimal year)
        obs_peaks_by_year[Y] = obs_t

obs_peak_times = np.array(list(obs_peaks_by_year.values())) if obs_peaks_by_year else np.array([])

# 2) For each step time, compare observed vs. nearest modeled peak
for t_step in STEPS:
    year_label = int(np.floor(t_step))

    # Observed: prefer the same calendar year; else nearest observed peak in time
    if year_label in obs_peaks_by_year:
        obs_t = obs_peaks_by_year[year_label]
    elif obs_peak_times.size:
        obs_t = float(obs_peak_times[np.argmin(np.abs(obs_peak_times - t_step))])
    else:
        obs_t = np.nan

    # Modeled: nearest harmonic peak to the step time
    # For a 1-year period model, peaks occur at k + vf["peak_frac"], k ∈ Z
    k = int(np.round(t_step - vf["peak_frac"]))
    mod_t = float(vf["peak_frac"] + k)

    if np.isfinite(obs_t):
        diff_yr = obs_t - mod_t
        diff_mon = diff_yr * 12.0
        print(f"{year_label}: Observed {obs_t:.2f}, Model {mod_t:.2f}, "
              f"Offset = {diff_yr:+.3f} yr (~{diff_mon:+.1f} months)")
    else:
        print(f"{year_label}: No observed peak found nearby\n")

# ============================================================
# Precipitation (panel f style) + daily harmonic peak & window
# ============================================================

# ensure decimal_year is present
if "decimal_year" not in precip_daily.columns:
    precip_daily = add_decimal_year(precip_daily, date_col="date")

# fit to DAILY totals (not monthly) with same 1-yr harmonic
t_p = precip_daily["decimal_year"].to_numpy(dtype=float)
y_p = precip_daily["rain_mm"].to_numpy(dtype=float)

pf = fit_harmonic_with_uncertainty(t_p, y_p)
t_fit_p = np.linspace(t_p.min(), t_p.max(), 1500)
y_fit_p = harmonic_model(t_fit_p, pf["v0"], pf["a1"], pf["b1"])

# velocity–precip peak offset (signed, shortest wrap)
signed_delta = ((pf["peak_frac"] - vf["peak_frac"] + 0.5) % 1.0) - 0.5
delta_days = signed_delta * 365.2425

# print precip stats
phi_p_deg = np.degrees(pf["phi"])
phi_p_std_deg = np.degrees(pf["peak_std"]*2*np.pi)
peak_p_days = int(round(pf["peak_frac"] * 365.2425))
peak_p_days_std = int(round(pf["peak_std"] * 365.2425))

p_mo, p_day = _frac_to_month_day(pf["peak_frac"])
p_day_std = int(round(pf["peak_std"] * 365.2425))

print(f"[PRECIP]   \nv0={pf['v0']:.3f}, A={pf['A']:.3f}, φ={phi_p_deg:.1f}°")
print(f"[Peak fraction = {pf['peak_frac']:.3f} ± {pf['peak_std']:.3f} "
      f"(~day {peak_p_days} ± {peak_p_days_std} d)",
      f"Peak ~ {p_mo} {p_day} (± {p_day_std} days)\n")
print(f"\n[OFFSET]   Precip – Velocity = {signed_delta:+.3f} yr (~{delta_days:+.1f} d)")

# ============================================================
# CRE analysis
# ============================================================

# Load Takaaki's Cat
taka_column_names = ['YYYY', 'Lat', 'Lon', 'Depth', 'cumD(cm)', 'EVID', 'CSID', 'Mag', 'D(cm)']

# Read the file into a DataFrame for 'taka'
taka_df = pd.read_csv(common_paths["CRE_Taka"]["MA"], sep=',', header=None, comment='#', names=taka_column_names, skiprows=1)
taka_df = taka_df.sort_values(by='CSID')

# Load Seno 
# Define column names for the new dataset
seno_column_names = ['eventID', 'NC_lon', 'NC_lat', 'NC_dep', 'NC_mag', 'MM/DD/YY','HH:MM:SS', 
                     'DD_lon', 'DD_lat', 'DD_depth', 'DD_mag', 'seqID']

# Read the file into a DataFrame
df1 = pd.read_csv(common_paths["CRE_Seno"]["S1"], names=seno_column_names, header=0) # confirmed repeater families
df2 = pd.read_csv(common_paths["CRE_Seno"]["S2"], names=seno_column_names, header=0) # possible repeaters
df2 = pd.read_csv(common_paths["CRE_Seno"]["S3"], names=seno_column_names, header=0) # families with only 2 events
seno_df = pd.concat([df1, df2], ignore_index=True)

# Apply the function to convert 'MM/DD/YY' and 'HH:MM:SS' to decimal years
seno_df['YYYY'] = seno_df.apply(lambda row: utils.date_to_decimal_year_seno(row['MM/DD/YY'], row['HH:MM:SS']), axis=1)

# Load Waldhauser Cat
# Define the column names for the final DataFrame
columns = ['YR', 'MO', 'DY', 'HR', 'MN', 'SC', 'DAYS', 
           'Lat', 'Lon', 'Depth', 'EX', 'EY', 'EZ', 
           'MAG', 'DMAG', 'DMAGE', 'CCm', 'evID', 'seqID']

# Initialize an empty list to store the event data
event_data = []

# Open the file and read it line by line
with open(common_paths["CRE_Wald"], 'r') as file:
    current_seq_id = None  # Variable to keep track of the current seqID
    
    # Skip the first 39 lines as per your description
    for _ in range(39):
        next(file)
    
    for line in file:
        line = line.strip()
        if line.startswith('#'):  # Header line with seqID
            parts = line[1:].split()  # Split the line without the '#'
            current_seq_id = parts[-1]  # Get the seqID (last part of the line)
        elif current_seq_id:  # Event line
            parts = line.split()  # Split the event line into components
            # Append the seqID to the event data
            parts.append(current_seq_id)
            # Add the data to the list
            event_data.append(parts)

# Create a DataFrame from the collected event data
wald_df = pd.DataFrame(event_data, columns=columns)

# Convert appropriate columns to numeric types, ignoring errors for non-numeric data
wald_df = wald_df.apply(pd.to_numeric, errors='ignore')

# Calculate the decimal year for each row and add it to the DataFrame
wald_df['YYYY'] = wald_df.apply(utils.to_decimal_year_wald, axis=1)

##############################
# Calculate event slip, filter by region, and calculate cummulative fault slip
##############################



CRE_taka_df = taka_df[["YYYY", 'Lon',    'Lat',    'Depth',   'Mag',    'CSID']]
CRE_seno_df = seno_df[["YYYY", 'NC_lon', 'NC_lat', 'NC_dep',  'NC_mag', 'seqID']]
CRE_wald_df = wald_df[["YYYY", 'Lon',    'Lat',    'Depth',  'MAG', 'seqID']]

# Rename columns to be the same 
CRE_taka_df = CRE_taka_df.rename(columns={'CSID': "seqID"})
CRE_seno_df = CRE_seno_df.rename(columns={'NC_lon': "Lon", "NC_lat":"Lat", "NC_dep":"Depth", "NC_mag": "Mag"})
CRE_wald_df = CRE_wald_df.rename(columns={"MAG":"Mag"})

# Calculate Slip perevent
CRE_taka_df["slip"] = CRE_taka_df["Mag"].apply(utils.get_CRE_slip)
CRE_seno_df["slip"] = CRE_seno_df["Mag"].apply(utils.get_CRE_slip)
CRE_wald_df["slip"] = CRE_wald_df["Mag"].apply(utils.get_CRE_slip)

# Load polygon geometries
MA_CRE_polygon = utils.load_polygon_gmt(common_paths["MA_CRE_poly"])
if isinstance(MA_CRE_polygon, gpd.GeoDataFrame):
    MA_CRE_polygon = MA_CRE_polygon.unary_union  # ensure it's a shapely geometry

# Convert to GeoDataFrames
taka_gdf = gpd.GeoDataFrame(CRE_taka_df, geometry=gpd.points_from_xy(CRE_taka_df['Lon'], CRE_taka_df['Lat']), crs="EPSG:4326")
seno_gdf = gpd.GeoDataFrame(CRE_seno_df, geometry=gpd.points_from_xy(CRE_seno_df['Lon'], CRE_seno_df['Lat']), crs="EPSG:4326")
wald_gdf = gpd.GeoDataFrame(CRE_wald_df, geometry=gpd.points_from_xy(CRE_wald_df['Lon'], CRE_wald_df['Lat']), crs="EPSG:4326")

# Initialize filtered dicts
filtered_taka, filtered_seno, filtered_wald = {}, {}, {}

# MA via polygon, then split north/south at 39.1° lat
ma_taka = taka_gdf[taka_gdf.geometry.within(MA_CRE_polygon)].copy()
ma_seno = seno_gdf[seno_gdf.geometry.within(MA_CRE_polygon)].copy()
ma_wald = wald_gdf[wald_gdf.geometry.within(MA_CRE_polygon)].copy()

lat_split = 39.1
filtered_taka["NM"] = ma_taka[ma_taka["Lat"] >= lat_split].copy()
filtered_seno["NM"] = ma_seno[ma_seno["Lat"] >= lat_split].copy()
filtered_wald["NM"] = ma_wald[ma_wald["Lat"] >= lat_split].copy()

# Apply cumulative slip calculation (guarding empties if needed)
processed_dfs_taka = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_taka.items() if not df.empty}
processed_dfs_seno = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_seno.items() if not df.empty}
processed_dfs_wald = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_wald.items() if not df.empty}

# Subtract mean for plotting
processed_dfs_taka["NM"]["CumSlip/numSeqID"] = processed_dfs_taka["NM"]["CumSlip/numSeqID"]- np.nanmean(processed_dfs_taka["NM"]["CumSlip/numSeqID"])
processed_dfs_seno["NM"]["CumSlip/numSeqID"] = processed_dfs_seno["NM"]["CumSlip/numSeqID"]- np.nanmean(processed_dfs_seno["NM"]["CumSlip/numSeqID"])
processed_dfs_wald["NM"]["CumSlip/numSeqID"] = processed_dfs_wald["NM"]["CumSlip/numSeqID"]- np.nanmean(processed_dfs_wald["NM"]["CumSlip/numSeqID"])

# Calculate linear model and change rates
results = {}

for cat_name, datasets in [("Taka", processed_dfs_taka), 
                           ("Seno", processed_dfs_seno), 
                           ("Wald", processed_dfs_wald)]:
    for seg, df in datasets.items():
        pre_mask  = (df["YYYY"] >= STEPS[2] - 10) & (df["YYYY"] < STEPS[2])
        post_mask = (df["YYYY"] >= STEPS[2]) & (df["YYYY"] < STEPS[2] + 10)
        
        if pre_mask.sum() > 2 and post_mask.sum() > 2:
            pre_slope, pre_intercept   = np.polyfit(df.loc[pre_mask,"YYYY"], df.loc[pre_mask,"CumSlip/numSeqID"], 1)
            post_slope, post_intercept = np.polyfit(df.loc[post_mask,"YYYY"], df.loc[post_mask,"CumSlip/numSeqID"], 1)
            
            pct_change = (post_slope - pre_slope) / pre_slope * 100
            
            results[(seg, cat_name)] = {
                "pre_slope": pre_slope,
                "pre_intercept": pre_intercept,
                "post_slope": post_slope,
                "post_intercept": post_intercept,
                "pct_change": pct_change
            }
            
# Print results
print("\n[CRE CHANGE IN RATE]")
for (seg, cat), vals in results.items():
    pre = vals['pre_slope']
    post = vals['post_slope']
    pct = vals['pct_change']
    
    if pct > 0:
        trend = "increase"
    else:
        trend = "decrease"
    
    print(f"[{seg} - {cat}]")
    print(f"  Pre-event velocity : {pre:.2f} cm/yr")
    print(f"  Post-event velocity: {post:.2f} cm/yr")
    print(f"  Change            : {pct:+.1f}% ({trend})")
    print("")


# ============================================================
# Plotting
# ============================================================

fig, axs = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)


# ---------------------------------------
# Dispalcement - top left
# ---------------------------------------
ax = axs[0,0]

# Plot displacement and long-term linear trend
ax.plot(years_all, trend_all, color="orange", ls="--", lw=1.5, label=f"1992–2022: {slope:.1f} ± {std_err:.1f} mm/yr")
ax.plot(years_all, disp_all, color="#1f77b4", marker="o", ms=2.5, lw=1.2, label="Displacement")

# Draw step markers (short vertical lines around y_here)
for i, t in enumerate(STEPS):
    y_here = disp_all[np.argmin(np.abs(years_all - t))]
    ax.plot([t, t], [y_here-20, y_here+20], color="k", ls=":", alpha=0.7)
    ax.text(t, y_here+25, f"S{i+1}", color="black", fontsize=9, ha="center", va="bottom",)

# Add text block with slopes in bottom-right corner
slope_text = "\n".join([f"T{i+1} {seg['t0']:.0f}–{seg['t1']:.0f}: {seg['slope']:.1f} mm/yr"
                        for i, seg in enumerate(segments)])
ax.text(0.98, 0.03, slope_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# Labels and formatting
ax.set_title(f"{STA_ID} Displacement", fontsize=13)
ax.set_ylabel("Displacement (mm)")
ax.set_xlim(1991, 2024)
ax.set_ylim(-10, disp_all.max() + 15)
ax.legend(fontsize=9, loc="upper left", frameon=False)


# ---------------------------------------
# Detrended Displacement — middle left
# ---------------------------------------

axs[1,0].plot(
    years_all, det_all, color="#1f77b4",
    marker="o", ms=3, lw=1.2, label="Detrended displacement"
)
axs[1,0].axhline(0.0, color="k", ls=":", lw=0.7, zorder=0)

# Step markers + labels
for i, t in enumerate(STEPS):
    step_mm = step_results[t]["step_mm"]
    y_here = det_all[int(np.argmin(np.abs(years_all - t)))]
    axs[1,0].plot([t, t], [y_here-5, y_here+5], color="k", ls=":", alpha=0.6)
    axs[1,0].text(t + 0.2, y_here-4, f"S{i+1}: {step_mm:.1f} mm", color="black", fontsize=9)

# Best-decay overlay (offset back by step_size_last), always orange, no RMSE in label
if ('best_name' in locals()) and (best_name is not None) and (best_name in models):
    yhat_plot = models[best_name]["yhat"] + step_size_last
    axs[1,0].plot(
        x_post, yhat_plot, ls="--", lw=2.0, color="#FF7F0E",
        label=f"{best_name} decay", zorder=3
    )

axs[1,0].set_title("Detrended displacement", fontsize=13)
axs[1,0].set_ylabel("Detrended (mm)")
axs[1,0].set_xlim(1991, 2024)
axs[1,0].legend(fontsize=9, frameon=False)

# ---------------------------------------
# Seasonal Analysis - Bottom left
# ---------------------------------------

axs[2,0].clear()
# step markers
for i, t in enumerate(STEPS):
    axs[2,0].plot([t,t], [-5, 25], color="k", ls=":", alpha=0.6)
    axs[2,0].text(t, 26, f"S{i+1}", color="black", fontsize=9, ha="center", va="bottom")

# bands: ±1σ around modeled peak each year
for Y in range(int(np.floor(years_season.min())), int(np.ceil(years_season.max()))+1):
    axs[2,0].axvspan(Y + vf["peak_frac"] - vf["peak_std"],
                     Y + vf["peak_frac"] + vf["peak_std"], alpha=0.15)

# Plot mean for each Time segment (no label)
for i, (start, end) in enumerate(zip(t_edges[:-1], t_edges[1:]), start=1):
    v0 = v0_periods.get(f"T{i}", np.nan)
    if not np.isnan(v0):
        axs[2,0].plot([start, end], [v0, v0], lw=1.0, color="tab:green", alpha=0.8)
        
# data + model
axs[2,0].plot(t_fit_v, y_fit_v, color="orange", lw=1.2, label="Harmonic Model", alpha=0.80)
axs[2,0].plot(years_all, vel_all, color="#1f77b4", lw=1.2, label="Velocity")




axs[2,0].set_title("Instantaneous Velocity", fontsize=13)
axs[2,0].set_xlabel("Year"); axs[2,0].set_ylabel("Velocity (mm/yr)")
#axs[2,0].set_xlim(SEASON_START-0.5, SEASON_END+0.5)
axs[2,0].set_ylim(-15, 35)

axs[2,0].legend(fontsize=9, loc="upper right", frameon=False)

axs[2,0].text(
    0.02, 0.95,
    f"Peak ~ {v_mo} {v_day} (± {v_day_std} days)\n",
    transform=axs[2,0].transAxes, ha="left", va="top", fontsize=9
)


axs[2,0].set_xlim(1991, 2024)

# Create slope text block using v0_periods
slope_text = "\n".join([
    f"T{i}: {v0_periods[f'T{i}']:.1f} mm/yr"
    for i in range(1, len(t_edges))
    if not np.isnan(v0_periods[f"T{i}"])
])

# Add text to bottom-left of the axis
axs[2,0].text(0.98, 0.03, slope_text,
              transform=axs[2,0].transAxes,
              ha="right", va="bottom", fontsize=9,
              bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# ---------------------------------------
# CRE cumulative slip – top right
# ---------------------------------------

axs[0,1].set_title("CRE Cumulative Slip - NMF Segment", fontsize=13)

for i, t in enumerate(STEPS):
    disp = processed_dfs_wald["NM"]["CumSlip/numSeqID"] - np.nanmean(processed_dfs_wald["NM"]["CumSlip/numSeqID"])
    t_vals = processed_dfs_wald["NM"]["YYYY"].values
    idx = np.argmin(np.abs(t_vals - t))
    y_here = disp.iloc[idx]  

    axs[0,1].plot([t, t], [y_here - 6.5, y_here + 3], color="k", ls=":", alpha=0.7)
    axs[0,1].text(t, y_here + 3, f"S{i+1}", color="black", fontsize=9, ha="center", va="bottom")

axs[0,1].plot(processed_dfs_taka["NM"]["YYYY"],
              processed_dfs_taka["NM"]["CumSlip/numSeqID"] - np.nanmean(processed_dfs_taka["NM"]["CumSlip/numSeqID"]),
              color="tab:blue", lw=1.2, label="Taira (this study)", marker="o", ms=2.5)

axs[0,1].plot(processed_dfs_seno["NM"]["YYYY"],
              processed_dfs_seno["NM"]["CumSlip/numSeqID"] - np.nanmean(processed_dfs_seno["NM"]["CumSlip/numSeqID"]),
              color="tab:orange", lw=1.2, label="Shakibay (2019)", marker="o", ms=2.5)

axs[0,1].plot(processed_dfs_wald["NM"]["YYYY"],
              processed_dfs_wald["NM"]["CumSlip/numSeqID"] - np.nanmean(processed_dfs_wald["NM"]["CumSlip/numSeqID"]),
              color="tab:green", lw=1.2, label="Waldhauser (2021)", marker="o", ms=2.5)

catalog_colors = {
    "Taka": "tab:blue",
    "Seno": "tab:orange",
    "Wald": "tab:green"
}

for cat_name, color in catalog_colors.items():
    df = {
        "Taka": processed_dfs_taka["NM"],
        "Seno": processed_dfs_seno["NM"],
        "Wald": processed_dfs_wald["NM"]
    }[cat_name]

    slip = df["CumSlip/numSeqID"].values - np.nanmean(df["CumSlip/numSeqID"])

    pre_slope  = results[("NM", cat_name)]["pre_slope"]
    pre_int    = results[("NM", cat_name)]["pre_intercept"]
    post_slope = results[("NM", cat_name)]["post_slope"]
    post_int   = results[("NM", cat_name)]["post_intercept"]

    pre_years = np.linspace(STEPS[2] - 12, STEPS[2], 100)
    post_years = np.linspace(STEPS[2], STEPS[2] + 12, 100)

    # Adjust intercepts to match zero-centered slip
    pre_trend  = pre_slope * pre_years + pre_int - np.nanmean(slip)
    post_trend = post_slope * post_years + post_int - np.nanmean(slip)

    axs[0,1].plot(pre_years, pre_trend, color=color, linestyle="--", lw=1)
    axs[0,1].plot(post_years, post_trend, color=color, linestyle="-.", lw=1)

# Annotate % change after the step
for cat_name, color in catalog_colors.items():
    vals = results[("NM", cat_name)]
    pct_change = vals["pct_change"]

    # Format label
    direction = "↓" if pct_change < 0 else "↑"
    pct_text = f"{abs(pct_change):.0f}%{direction}"

    # Place text near end of post-event trend line
    post_years = np.linspace(STEPS[2], STEPS[2] + 12, 100)
    post_trend = vals["post_slope"] * post_years + vals["post_intercept"]
    post_trend -= np.nanmean({
        "Taka": processed_dfs_taka["NM"]["CumSlip/numSeqID"],
        "Seno": processed_dfs_seno["NM"]["CumSlip/numSeqID"],
        "Wald": processed_dfs_wald["NM"]["CumSlip/numSeqID"]
    }[cat_name])

    # Pick a clean location for label
    x_text = post_years[-1] + 2
    y_text = post_trend[-1]

    axs[0,1].text(x_text, y_text, pct_text,
                  color=color, fontsize=10,
                  ha="left", va="center")
    
# Labels / limits
axs[0,1].set_ylabel("Cumulative Slip (cm)")
#axs[0,1].set_xlim(STEPS[2]-20, STEPS[2]+20)
axs[0,1].set_xlim(1991, 2024)
axs[0,1].legend(fontsize=9, frameon=False)

# ---------------------------------------
# Precipitation panel (middle right)
# ---------------------------------------

axs[1,1].clear()
for i, t in enumerate(STEPS):
    axs[1,1].plot([t, t], [0, 140], color="k", ls=":", alpha=0.7)
    axs[1,1].text(t, 140, f"S{i+1}", color="black", fontsize=9, ha="center", va="bottom")

# daily series (thin) + harmonic (dashed)
line_daily, = axs[1,1].plot(
    t_p, y_p, color="orange", lw=0.8, label="Daily Precipitation"
)
line_model, = axs[1,1].plot(
    t_fit_p, y_fit_p+3, lw=1.2, color="green", label="Harmonic Model"
)

# shade precip ±1σ window per year
for Y in range(int(np.floor(t_p.min())), int(np.ceil(t_p.max()))+1):
    axs[1,1].axvspan(Y + pf["peak_frac"] - pf["peak_std"],
                     Y + pf["peak_frac"] + pf["peak_std"], alpha=0.15, color="green")

axs[1,1].set_title("Precipitation at MWIL", fontsize=13)
axs[1,1].set_ylabel("Daily Precipitation (mm)")
axs[1,1].set_ylim(0, 200)

# twin axis: cumulative precip (keep your monthly series)
ax2 = axs[1,1].twinx()
line_cum, = ax2.plot(
    precip_monthly['decimal_year'],
    precip_monthly['cumulative_precip_mm'],
    color="#1f77b4", lw=1.2, label="Cumulative Precipitation"
)

# line_cum, = ax2.plot(
#     precip_daily['decimal_year'],
#     precip_daily['cum_precip'],
#     color="#1f77b4", lw=1.2, label="Cumulative Precipitation"
# )

ax2.set_ylabel("Cumulative Precipitation (mm)")
ax2.set_ylim(0, 3000)

axs[1,1].set_xlim(1991, 2024)
lines = [line_daily, line_model, line_cum]
labels = ["Daily Precipitation", "Harmonic Model", "Cumulative Precipitation"]
axs[1,1].legend(lines, labels, fontsize=9, loc="upper right", frameon=False)

axs[1,1].text(
    0.02, 0.95,
    f"Peak ~ {p_mo} {p_day} (± {p_day_std} days)\n",
    transform=axs[1,1].transAxes, ha="left", va="top", fontsize=9
)

# ---------------------------------------
# Seismicity panel (bottom right)
# ---------------------------------------

axs[2,1].set_title("Seismicity within 15km of MWIL", fontsize=13)

scatter_short = axs[2,1].scatter(
    eq_window_dist['decimal_year'], eq_window_dist['mag'],
    c=eq_window_dist['distance'], cmap='viridis',
    alpha=1, s=15, zorder=1,
    vmin=0, vmax=15
)

norm_moment = eq_window_sorted['cum_seismic_moment'] / eq_window_sorted['cum_seismic_moment'].max()
norm_count  = eq_window_sorted['cum_count'] / eq_window_sorted['cum_count'].max()

ax_norm = axs[2,1].twinx()
line_moment, = ax_norm.plot(
    eq_window_sorted['decimal_year'], norm_moment,
    color="#1f77b4", linewidth=1.2, label='Norm. Cumulative Moment'
)
line_count, = ax_norm.plot(
    eq_window_sorted['decimal_year'], norm_count,
    color="orange", linewidth=1.2, label='Norm. Cumulative Count'
)

axs[2,1].legend([line_moment, line_count],
                ['Norm. Cumulative Moment', 'Norm. Cumulative Count'],
                fontsize=9, loc="upper left", frameon=False)

axs[2,1].set_ylabel("Magnitude")
axs[2,1].set_ylim(0.8, 4.5)
ax_norm.set_ylabel("Normalized (0–1)")
ax_norm.set_ylim(0, 1.05)
axs[2,1].set_xlim(STEPS[2] - 1.5, STEPS[2] + 1.5)

cbar = fig.colorbar(
    scatter_short, ax=axs[2,1], orientation='vertical',
    shrink=0.7, pad=0.03
)
cbar.set_label("Distance from MWIL (km)")

axs[2,1].axvspan(
    shade_start_dec, shade_end_dec,
    color="gray", alpha=0.3, zorder=0
)


# ---------------------------------------
# Define subplot labels
# ---------------------------------------

subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']

# Add labels (a)–(f), top-left of each subplot
for i in range(3):  # 3 rows
    # Left column
    axs[i, 0].text(
        0.0, 1.02, f'{subplot_labels[i]})',
        transform=axs[i, 0].transAxes,
        fontsize=12, va='bottom', ha='left'
    )
    
    # Right column
    axs[i, 1].text(
        0.0, 1.02, f'{subplot_labels[i + 3]})',
        transform=axs[i, 1].transAxes,
        fontsize=12, va='bottom', ha='left'
    )

# Save & show
out_path = f'{common_paths["fig_dir"]}/Fig_6_{STA_ID}_Disp_Vel_Seasonal_2_anualcumPrecip.png'
plt.savefig(out_path, dpi=300)
plt.show()
print(f"\nSaved: {out_path}")




# # ---------------------------------------
# # Seismicity panel (middle right)
# # ---------------------------------------

# axs[1,1].set_title("Seismicity within 15km of MWIL", fontsize=13)

# # Scatter plot of magnitudes, colored by distance (forced 0–15 km)
# scatter_short = axs[1,1].scatter(
#     eq_window_dist['decimal_year'], eq_window_dist['mag'],
#     c=eq_window_dist['distance'], cmap='viridis',
#     alpha=1, s=15, zorder=1,
#     vmin=0, vmax=15
# )

# # Normalized cumulative curves
# norm_moment = eq_window_sorted['cum_seismic_moment'] / eq_window_sorted['cum_seismic_moment'].max()
# norm_count  = eq_window_sorted['cum_count'] / eq_window_sorted['cum_count'].max()

# ax_norm = axs[1,1].twinx()
# line_moment, = ax_norm.plot(
#     eq_window_sorted['decimal_year'], norm_moment,
#     color="#1f77b4", linewidth=1.2, label='Norm. Cumulative Moment'
# )
# line_count, = ax_norm.plot(
#     eq_window_sorted['decimal_year'], norm_count,
#     color="orange", linewidth=1.2, label='Norm. Cumulative Count'
# )

# # Legend (only for the two lines)
# axs[1,1].legend([line_moment, line_count],
#                 ['Norm. Cumulative Moment', 'Norm. Cumulative Count'],
#                 fontsize=9, loc="upper left", frameon=False)

# axs[1,1].set_ylabel("Magnitude")
# axs[1,1].set_ylim(0.8, 4.5)
# ax_norm.set_ylabel("Normalized (0–1)")
# ax_norm.set_ylim(0, 1.05)
# axs[1,1].set_xlim(STEPS[2] - 1.5, STEPS[2] + 1.5)

# # Colorbar
# cbar = fig.colorbar(
#     scatter_short, ax=axs[1,1], orientation='vertical',
#     shrink=0.7, pad=0.03
# )
# cbar.set_label("Distance from MWIL (km)")

# axs[1,1].axvspan(
#     shade_start_dec, shade_end_dec,
#     color="gray", alpha=0.3, zorder=0
# )

# # ---------------------------------------
# # Precipitation panel (bottom right)
# # ---------------------------------------

# # plot into axs[2,1]
# axs[2,1].clear()
# #axs[2,1].axvspan(shade_start_dec, shade_end_dec, color="gray", alpha=0.3, zorder=0)
# for i, t in enumerate(STEPS):
#     axs[2,1].plot([t, t], [0, 120], color="k", ls=":", alpha=0.7)
#     axs[2,1].text(t, 120, f"S{i+1}", color="black", fontsize=9, ha="center", va="bottom")

# # daily series (thin) + harmonic (dashed)
# line_daily, = axs[2,1].plot(
#     t_p, y_p, color="orange", lw=0.8, label="Daily Precipitation"
# )
# line_model, = axs[2,1].plot(
#     t_fit_p, y_fit_p, ls="--", lw=1.8, color="green", label="Harmonic Model"
# )

# axs[2,1].plot(t_fit_p, y_fit_p, ls="--", lw=1.8, color="green", label="Harmonic Model")

# # shade precip ±1σ window per year
# for Y in range(int(np.floor(t_p.min())), int(np.ceil(t_p.max()))+1):
#     axs[2,1].axvspan(Y + pf["peak_frac"] - pf["peak_std"],
#                      Y + pf["peak_frac"] + pf["peak_std"], alpha=0.15, color="green")

# axs[2,1].set_title("Precipitation at MWIL", fontsize=13)
# axs[2,1].set_ylabel("Daily Precipitation (mm)")

# # twin axis: cumulative precip (keep your monthly series)
# ax2 = axs[2,1].twinx()
# line_cum, = ax2.plot(
#     precip_monthly['decimal_year'],
#     precip_monthly['cumulative_precip_mm'],
#     color="#1f77b4", lw=1.2, label="Cumulative Precipitation"
# )
# ax2.set_ylabel("Cumulative Precipitation (mm)")
# ax2.set_ylim(-100, 3000)

# # x-lims and legend (only for daily & cumulative lines to keep it clean)
# #axs[2,1].set_xlim(STEPS[2] - 10, STEPS[2] + 10)
# axs[2,1].set_xlim(1991, 2024)
# lines = [line_daily, line_model, line_cum]
# labels = ["Daily Precipitation", "Harmonic Model", "Cumulative Precipitation"]
# axs[2,1].legend(lines, labels, fontsize=9, loc="upper right", frameon=False)

# axs[2,1].text(
#     0.02, 0.95,
#     f"Peak ~ {p_mo} {p_day} (± {p_day_std} days)\n",
#     transform=axs[2,1].transAxes, ha="left", va="top", fontsize=9
# )
