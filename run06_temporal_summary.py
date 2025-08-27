#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:19:58 2024

@author: daniellelindsay
"""

import pandas as pd
import numpy as np
import os
#from NC_manuscript_filepaths_Aug24 import *

from scipy.stats import linregress

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from NC_creep_filepaths import common_paths

def add_decimal_year(df, date_col='date'):
    """
    Add a decimal year column to a DataFrame based on a date column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the date column.
    date_col : str, optional
        Name of the date column (default 'date').

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a new column 'decimal_year'.
    """
    # Ensure date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Function to calculate decimal year for a single timestamp
    def to_decimal_year(ts):
        year_start = pd.Timestamp(year=ts.year, month=1, day=1)
        year_end = pd.Timestamp(year=ts.year + 1, month=1, day=1)
        year_length = (year_end - year_start).days
        fraction = (ts - year_start).days / year_length
        return ts.year + fraction

    df['decimal_year'] = df[date_col].apply(to_decimal_year)
    return df

StaID = "MWIL"

def add_cumulative_precip(df, date_col="Date", precip_col="Precipitation (mm)"):
    """
    Add a cumulative precipitation column that resets each Oct 1 (water year).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Define water year (Oct–Sep)
    df["water_year"] = df[date_col].dt.year
    df.loc[df[date_col].dt.month >= 10, "water_year"] += 1

    # Compute cumulative sum within each water year
    df["cumulative_precip_mm"] = df.groupby("water_year")[precip_col].cumsum()
    return df


# Define a function to compute the Haversine distance between two points (in km)
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the Earth specified in decimal degrees.
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Compute differences
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    # Haversine formula 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth's radius in kilometers
    return km

def to_decimal_year(ts):
    # Now ts is a pandas.Timestamp with tz=UTC
    year_start = pd.Timestamp(year=ts.year, month=1, day=1, tz="UTC")
    year_end   = pd.Timestamp(year=ts.year + 1, month=1, day=1, tz="UTC")
    return ts.year + (ts - year_start).days / (year_end - year_start).days


##################3
# Set variables and hardcoded values
##################3

# Define shaded region (make these timezone-aware as well)
start_shade = pd.Timestamp('2002-03-30', tz='UTC')
end_shade   = pd.Timestamp('2002-06-02', tz='UTC')

# Compute distance (in km) from the reference point (-123.35612, 39.41242)
MWIL_lon = -123.35612
MWIL_lat = 39.41242

##################3
# Load easy data... 
##################3

# Define the file path for Table 1b.csv
AA_table_path = common_paths["AA"]["table"]
AA_dir = common_paths["AA"]["dir"]

MWIL_precip_file = common_paths["PRISM"]["MWIL_mm"]

precip_monthly =  pd.read_csv(common_paths["PRISM"]["monthly"], header=0)
precip_monthly = add_decimal_year(precip_monthly, date_col='Date')
# Example usage
precip_monthly = add_cumulative_precip(precip_monthly.copy())

precip_daily = pd.read_csv(common_paths["PRISM"]["daily"], header=0)
precip_daily = add_decimal_year(precip_daily, date_col='date')

eq_data = pd.read_csv(common_paths["NCEDC"]["MWIL_15km"])
eq_data['time'] = pd.to_datetime(eq_data['time'], utc=True)
eq_data_sorted = eq_data.sort_values('time')

eq_data_sorted['cum_magnitude'] = eq_data_sorted['mag'].cumsum()
eq_data_sorted['cum_count']     = range(1, len(eq_data_sorted) + 1)

# Compute seismic moment using Hanks & Kanamori (1979): log10(M0) = 1.5*M + 16.1
eq_data_sorted['seismic_moment']    = 10**(1.5 * eq_data_sorted['mag'] + 16.1)
eq_data_sorted['cum_seismic_moment'] = eq_data_sorted['seismic_moment'].cumsum()
eq_data_sorted['decimal_year'] = eq_data_sorted['time'].apply(to_decimal_year)

norm_moment = eq_data_sorted['cum_seismic_moment'] / eq_data_sorted['cum_seismic_moment'].max()
norm_count  = eq_data_sorted['cum_count'] / eq_data_sorted['cum_count'].max()



# Assumes the CSVs include 'longitude' and 'latitude'
eq_data_sorted['distance'] = haversine(MWIL_lon, MWIL_lat, 
                                          eq_data_sorted['longitude'], 
                                          eq_data_sorted['latitude'])

eq_data_sorted_dist = eq_data_sorted.sort_values('distance', ascending=False)


##################3
# Load AA timeseries
##################3

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



# Function to load station data
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
            print(f"Station: {station_code}, Slope: {slope}, Std_Err: {std_err}")

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
data = AA_stations_dict[StaID]
years_all = np.asarray(data["Year"], dtype=float)                     # decimal years
disp_all  = np.asarray(data["Cumulative movement (mm)"], dtype=float) # mm

order = np.argsort(years_all)
years_all = years_all[order]
disp_all  = disp_all[order]

# ============================================================
# Long-term trend & velocity
# ============================================================
slope, intercept, r_value, p_value, std_err = linregress(years_all, disp_all)
trend_all = slope * years_all + intercept
det_all   = disp_all - trend_all
vel_all   = np.gradient(disp_all, years_all)

print(f"[TREND] Overall linear velocity = {slope:.3f} ± {std_err:.3f} mm/yr (R²={r_value**2:.3f})")

# Sampling intervals (pre/post 2008)
dt = np.diff(years_all)
pre_mask_dt  = years_all[:-1] < 2008.0
post_mask_dt = ~pre_mask_dt
mean_sampling_interval_pre_2008  = float(np.mean(dt[pre_mask_dt]))  if np.any(pre_mask_dt)  else np.nan
mean_sampling_interval_post_2008 = float(np.mean(dt[post_mask_dt])) if np.any(post_mask_dt) else np.nan
print(f"[SAMPLE] Mean Δt pre-2008 = {mean_sampling_interval_pre_2008:.3f} yr, post-2008 = {mean_sampling_interval_post_2008:.3f} yr")

# ============================================================
# Segments & step sizes
#   - Segment 4 ends at 2008.0
#   - Step sizes = displacement jumps from local ±STEP_WIN yr fits (raw displacement)
# ============================================================
steps   = [1994.118, 1996.965, 2002.3315]   # step times T1, T2, T3
STEP_WIN = 1.5                               # years for local step fits
t_edges = [years_all.min(), steps[0], steps[1], steps[2], 2008.0]  # force S4 end at 2008

def fit_segment_with_stats(x, y):
    """Return slope, intercept, stderr, n (on raw displacement)."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    s, c, r, p, se = linregress(x, y)
    return s, c, se, len(x)

# Inter-event segments (on RAW displacement)
segments = []
for i in range(len(t_edges) - 1):
    m = (years_all >= t_edges[i]) & (years_all <= t_edges[i+1])
    s_i, c_i, se_i, n_i = fit_segment_with_stats(years_all[m], disp_all[m])
    segments.append(dict(mask=m, t0=t_edges[i], t1=t_edges[i+1], slope=s_i, intercept=c_i, stderr=se_i, n=n_i))

print("\n[INTERVAL VELOCITIES] (raw displacement)")
for k, seg in enumerate(segments, 1):
    print(f"  S{k}: {seg['t0']:.3f}–{seg['t1']:.3f}  {seg['slope']:.2f} ± {seg['stderr']:.2f} mm/yr  (n={seg['n']})")

def step_size_raw_displacement(t_step, win=STEP_WIN):
    """
    Displacement step at t_step (mm) from raw displacement:
      Fit y = m t + c on [t_step-win, t_step) and (t_step, t_step+win],
      then Δ = y_post(t_step) - y_pre(t_step).
    """
    eps = 1e-6
    pre_mask  = (years_all >= (t_step - win)) & (years_all <  (t_step - eps))
    post_mask = (years_all >  (t_step + eps)) & (years_all <= (t_step + win))
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return np.nan, (np.nan, np.nan, 0), (np.nan, np.nan, 0)
    sp, cp, sep, npre   = fit_segment_with_stats(years_all[pre_mask],  disp_all[pre_mask])
    so, co, seo, npost  = fit_segment_with_stats(years_all[post_mask], disp_all[post_mask])
    step_mm = (so * t_step + co) - (sp * t_step + cp)
    return step_mm, (sp, sep, npre), (so, seo, npost)

print("\n[STEP SIZES] (displacement jump from local fits)")
step_results = {}
for t in steps:
    step_mm, pre_stats, post_stats = step_size_raw_displacement(t)
    sp, sep, npre   = pre_stats
    so, seo, npost  = post_stats
    print(f"  @ {t:.4f}: {step_mm:.2f} mm | pre-slope {sp:.2f}±{sep:.2f} (n={npre}) | post-slope {so:.2f}±{seo:.2f} (n={npost})")
    step_results[t] = dict(step_mm=step_mm, pre_slope=sp, post_slope=so, win=STEP_WIN)

# ============================================================
# Detrended modeling — baseline = inter-event fit (step2 -> step3), test decays after last step
# ============================================================
t1, t2, t3 = steps[0], steps[1], steps[2]
t_last = t3

# Baseline (detrended) over [step2, step3]
baseline_mask = (years_all >= t2) & (years_all <= t3)
m_lin_last, c_lin_last, se_lin_last, n_lin_last = fit_segment_with_stats(years_all[baseline_mask], det_all[baseline_mask])

# Step size at last step from detrended series using that baseline
idx_last      = int(np.argmin(np.abs(years_all - t_last)))
det_at_last   = det_all[idx_last]
step_size_last = det_at_last - (m_lin_last * t_last + c_lin_last)

# Work with post-step residuals (detrended, minus step)
post_mask = years_all > t_last
x_post = years_all[post_mask]
y_post = det_all[post_mask] - step_size_last

# Inter-event detrended fits to plot (S1: start–t1, S2: t1–t2, S3: t2–t3)
inter_event_ranges = [(years_all.min(), t1), (t1, t2), (t2, t3)]
det_lines = []  # list of (mask, y_fit, is_baseline)
for (ta, tb) in inter_event_ranges:
    msk = (years_all >= ta) & (years_all <= tb)
    m_d, c_d, _, n_d = fit_segment_with_stats(years_all[msk], det_all[msk])
    if n_d >= 2:
        y_fit = m_d * years_all[msk] + c_d
        is_base = (abs(ta - t2) < 1e-6 and abs(tb - t3) < 1e-6)
        det_lines.append((msk, y_fit, is_base))

# ---------- Decay models after last step (detrended & step-removed) ----------
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def aic(n, rss, k): return float(n*np.log(rss/n) + 2*k)

models = {}

# Exponential: a * exp(-(t-t0)/tau) + c
def exp_model(t, a, tau, c, t0): return a * np.exp(-(t - t0)/tau) + c
try:
    p0 = [y_post[0] if y_post.size else 1.0, 1.0, 0.0]
    popt, _ = curve_fit(lambda t,a,tau,c: exp_model(t,a,tau,c,t_last),
                        x_post, y_post, p0=p0,
                        bounds=([-np.inf, 1e-4, -np.inf],[np.inf, np.inf, np.inf]), maxfev=20000)
    yhat = exp_model(x_post, *popt, t_last)
    RSS  = float(np.sum((y_post - yhat)**2)); n=len(y_post); kpar=3
    models["Exponential"] = dict(params=dict(a=popt[0], tau=popt[1], c=popt[2]),
                                 rmse=rmse(y_post,yhat), aic=aic(n,RSS,kpar), yhat=yhat)
except Exception:
    pass

# Logarithmic: a * log(1 + (t-t0)/tau) + c
def log_model(t, a, tau, c, t0): return a * np.log1p((t - t0)/tau) + c
try:
    p0 = [y_post[0] if y_post.size else 1.0, 1.0, 0.0]
    popt, _ = curve_fit(lambda t,a,tau,c: log_model(t,a,tau,c,t_last),
                        x_post, y_post, p0=p0,
                        bounds=([-np.inf, 1e-4, -np.inf],[np.inf, np.inf, np.inf]), maxfev=20000)
    yhat = log_model(x_post, *popt, t_last)
    RSS  = float(np.sum((y_post - yhat)**2)); n=len(y_post); kpar=3
    models["Logarithmic"] = dict(params=dict(a=popt[0], tau=popt[1], c=popt[2]),
                                 rmse=rmse(y_post,yhat), aic=aic(n,RSS,kpar), yhat=yhat)
except Exception:
    pass

# Power-law: a * (t-t0)^(-b) + c
def pwr_model(t, a, b, c, t0): return a * np.power(np.maximum(t - t0, 1e-6), -b) + c
try:
    p0 = [y_post[0] if y_post.size else 1.0, 1.0, 0.0]
    popt, _ = curve_fit(lambda t,a,b,c: pwr_model(t,a,b,c,t_last),
                        x_post, y_post, p0=p0,
                        bounds=([-np.inf, 1e-3, -np.inf],[np.inf, 5.0, np.inf]), maxfev=20000)
    yhat = pwr_model(x_post, *popt, t_last)
    RSS  = float(np.sum((y_post - yhat)**2)); n=len(y_post); kpar=3
    models["Power-law"] = dict(params=dict(a=popt[0], b=popt[1], c=popt[2]),
                               rmse=rmse(y_post,yhat), aic=aic(n,RSS,kpar), yhat=yhat)
except Exception:
    pass

# Stretched-exponential: a * exp(-((t-t0)/tau)^beta) + c
def strexp_model(t, a, tau, beta, c, t0): return a * np.exp(-np.power((t - t0)/tau, beta)) + c
try:
    p0 = [y_post[0] if y_post.size else 1.0, 1.0, 0.8, 0.0]
    bounds = ([-np.inf, 1e-4, 0.2, -np.inf], [np.inf, np.inf, 2.0,  np.inf])
    popt, _ = curve_fit(lambda t,a,tau,beta,c: strexp_model(t,a,tau,beta,c,t_last),
                        x_post, y_post, p0=p0, bounds=bounds, maxfev=30000)
    yhat = strexp_model(x_post, *popt, t_last)
    RSS  = float(np.sum((y_post - yhat)**2)); n=len(y_post); kpar=4
    models["Stretched-exp"] = dict(params=dict(a=popt[0], tau=popt[1], beta=popt[2], c=popt[3]),
                                   rmse=rmse(y_post,yhat), aic=aic(n,RSS,kpar), yhat=yhat)
except Exception:
    pass

print("\n[DECAY] Post-last-step decay models (detrended, step removed):")
if models:
    for name, m in sorted(models.items(), key=lambda kv: (kv[1]["aic"], kv[1]["rmse"])):
        p = m["params"]; ptxt = ", ".join([f"{k}={v:.4f}" for k, v in p.items()])
        print(f"  {name:<14} RMSE={m['rmse']:.4f}  AIC={m['aic']:.2f}  |  {ptxt}")
    best_name = min(models.keys(), key=lambda n: (models[n]["aic"], models[n]["rmse"]))
else:
    best_name = None
    print("  No valid decay fits.")

# ============================================================
# Seasonal analysis — automatic peaks & fit quality
# ============================================================
season_start, season_end = 1991.0, 2008.0
m_season = (years_all >= season_start) & (years_all <= season_end)
years_season = years_all[m_season]
vel_season   = np.gradient(disp_all[m_season], years_all[m_season])

# Smooth (no phase shift) for robust peak finding
#vel_smooth = gaussian_filter1d(vel_season, sigma=1, mode="nearest")

def harmonic_model(t, v0, a1, b1):
    return v0 + a1 * np.sin(2*np.pi*t) + b1 * np.cos(2*np.pi*t)

if len(years_season) >= 3:
    p0 = [np.nanmean(vel_season), 1.0, 1.0]
    (v0, a1, b1), _ = curve_fit(harmonic_model, years_season, vel_season, p0=p0, maxfev=20000)
    model_vel = harmonic_model(years_season, v0, a1, b1)

    # Fit quality
    resid = vel_season - model_vel
    rmse_seas = float(np.sqrt(np.mean(resid**2)))
    r2_seas   = 1.0 - (np.var(resid) / np.var(vel_season))
    r_seas, _ = pearsonr(vel_season, model_vel)
    amplitude = float(np.sqrt(a1**2 + b1**2))
    phase     = float(np.arctan2(a1, b1))

    # Model & observed peak times (automatic)
    dense_t = np.linspace(years_season.min(), years_season.max(),
                          int((years_season.max() - years_season.min()) * 24))
    dense_model = harmonic_model(dense_t, v0, a1, b1)
    i_model, _ = find_peaks(dense_model, prominence=np.std(dense_model)*0.2)
    model_peak_times = dense_t[i_model]

    i_obs, _ = find_peaks(vel_season, prominence=np.std(vel_season)*0.2)
    obs_peak_times = years_season[i_obs]

    # Circular mean absolute timing error (mod 1 year)
    def circular_mean_abs_err_years(obs, mod):
        if obs.size == 0 or mod.size == 0: return np.nan
        phi_obs = np.modf(obs)[0]; phi_mod = np.modf(mod)[0]
        errs = []
        for po in phi_obs:
            d = np.abs(phi_mod - po)
            d = np.minimum(d, 1 - d)
            errs.append(np.min(d))
        return float(np.mean(errs))
    mean_abs_phase_err_years = circular_mean_abs_err_years(obs_peak_times, model_peak_times)

    print("\n[SEASONAL] Harmonic fit (velocity, 1991–2008, automatic peaks)")
    print(f"  v0 = {v0:.3f} mm/yr, amplitude = {amplitude:.3f} mm/yr, phase = {phase:.3f} rad")
    print(f"  Fit quality: RMSE = {rmse_seas:.3f} mm/yr, R² = {r2_seas:.3f}, r = {r_seas:.3f}")
    print(f"  Observed peak times (auto): {np.array2string(obs_peak_times, precision=3)}")
    print(f"  Model peak times (auto):    {np.array2string(model_peak_times, precision=3)}")
    print(f"  Mean |timing offset| (circular, years) = {mean_abs_phase_err_years:.3f}")
else:
    print("\n[SEASONAL] Not enough samples for seasonal fit.")
    v0 = amplitude = phase = rmse_seas = r2_seas = r_seas = np.nan
    obs_peak_times = model_peak_times = np.array([])
    dense_t = np.array([]); dense_model = np.array([])

# ============================================================
# Plotting
# ============================================================
fig, axs = plt.subplots(3, 2, figsize=(15, 9), constrained_layout=True)

# ---------- Panel 1: Raw displacement + long-term trend + interval velocities ----------
axs[0,0].plot(years_all, disp_all, color="#1f77b4", marker="o", ms=3, lw=0.8, label="Displacement")
axs[0,0].plot(years_all, trend_all, color="gray", ls="--", lw=1.5, label=f"{slope:.1f} ± {std_err:.1f} mm/yr")
for t in steps: axs[0,0].axvline(t, color="k", ls=":", alpha=0.6)


seg_colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"] # Removed "tab:" prefix
for i, seg in enumerate(segments):
    m = seg["mask"]; s = seg["slope"]; c = seg["intercept"]
    axs[0,0].plot(years_all[m], s*years_all[m] + c, lw=2, color=seg_colors[i],
                label=f"T{i} {seg['t0']:.2f}–{seg['t1']:.2f}: {s:.1f} mm/yr")

axs[0,0].set_title(f"{StaID} Displacement with Long-term Linear Trend", fontsize=13)
axs[0,0].set_ylabel("Displacement (mm)")
axs[0,0].legend(fontsize=9, ncol=2, loc='lower right')
# axs[0,0].text(0.98, 0.02,
#             f"Sampling Interval Pre-2008: {mean_sampling_interval_pre_2008:.2f} yr, "
#             f"Post-2008: {mean_sampling_interval_post_2008:.2f} yr",
#             transform=axs[0,0].transAxes, ha="right", fontsize=9)
axs[0,0].set_xlim(1991, 2024)

# ---------- Panel 2: Detrended + inter-event lines + step sizes + best decay ----------
axs[1,0].plot(years_all, det_all, color="#1f77b4", marker="o", ms=3, lw=0.8, label="Detrended Displacement")
axs[1,0].axhline(0.0, color="k", ls=":", lw=0.7)
for t in steps: axs[1,0].axvline(t, color="k", ls=":", alpha=0.6)

# inter-event detrended lines (S1–S3); label the baseline (S2->S3) only
for msk, y_fit, is_baseline in det_lines:
    axs[1,0].plot(years_all[msk], y_fit, ls="--", lw=2,
                color="orange" if is_baseline else "orange",
                label=f"Linear Model" if is_baseline else None)

# step-size annotations at each step (mm)
for t in steps:
    step_mm = step_results[t]["step_mm"]
    y_here = det_all[int(np.argmin(np.abs(years_all - t)))]
    axs[1,0].text(t + 0.2, y_here + 1.5, f"{step_mm:.1f} mm", color="black", fontsize=10)

# best decay after last step (offset back to detrended level)
if models and best_name is not None:
    m = models[best_name]
    color = {"Exponential":"green", "Power-law":"purple", "Logarithmic":"brown", "Stretched-exp":"teal"}.get(best_name, "green")
    axs[1,0].plot(x_post, m["yhat"] + step_size_last, ls="--", lw=2, color=color,
                label=f"{best_name} decay")

axs[1,0].set_title("Detrended Displacement", fontsize=13)
axs[1,0].set_ylabel("Detrended (mm)")
axs[1,0].legend(fontsize=9)
axs[1,0].set_xlim(1991, 2024)

# ---------- Panel 3: Velocity & seasonal harmonic (1991–2008) ----------
years_season = years_all[m_season]
axs[2,0].plot(years_season, vel_season, label="Velocity", color="#1f77b4", lw=0.9)
for t in steps: axs[2,0].axvline(t, color="k", ls=":", alpha=0.6)
axs[2,0].axvline(t_last, color="k", ls=":", alpha=0.6)
if dense_t.size:
    axs[2,0].plot(dense_t, dense_model, color="orange", label="Harmonic Model")
    # observed/model peak markers
    if 'obs_peak_times' in locals() and obs_peak_times.size:
        axs[2,0].scatter(obs_peak_times, np.interp(obs_peak_times, years_season, vel_season),
                       s=14, color="#1f77b4", zorder=3,)
    if 'model_peak_times' in locals() and model_peak_times.size:
        axs[2,0].scatter(model_peak_times, np.interp(model_peak_times, dense_t, dense_model),
                       s=14, color="orange", zorder=3, )
axs[2,0].set_title("Velocity Analysis (1991–2008)", fontsize=13)
axs[2,0].set_xlabel("Year"); axs[2,0].set_ylabel("Velocity (mm/yr)")
axs[2,0].legend(fontsize=9)
axs[2,0].set_xlim(1991, 2008)



axs[0,1].set_title("CRE time series", fontsize=13)
axs[0,1].axvline(steps[2], color="k", ls=":", alpha=0.6)
axs[0,1].set_xlim(steps[2] - 6, steps[2] + 6 )

axs[1,1].set_title("Precipitation", fontsize=13)
axs[1,1].axvline(steps[2], color="k", ls=":", alpha=0.6)

line1, = axs[1,1].plot(
    precip_monthly['decimal_year'],
    precip_monthly['cumulative_precip_mm'],
    label="Cumulative Precipitation",
    color="#1f77b4", lw=1.2
)
axs[1,1].set_ylabel("Cumulative Precipitation (mm)")

# Right y-axis: daily precipitation
ax2 = axs[1,1].twinx()  # create secondary y-axis sharing the same x
line2, = ax2.plot(
    precip_daily['decimal_year'],
    precip_daily['rain_mm'],
    label="Daily Precipitation",
    color="orange", lw=0.8, alpha=0.7
)
ax2.set_ylabel("Daily Precipitation (mm)")

# Align x-axis
axs[1,1].set_xlim(steps[2] - 6, steps[2] + 6)

# Combined legend (grab handles from both axes)
lines = [line1, line2]
labels = [l.get_label() for l in lines]
axs[1,1].legend(lines, labels, fontsize=9, loc="upper left")

# Filter seismicity data for time window around step
mask = (eq_data_sorted['decimal_year'] >= steps[2] - 1.5) & \
       (eq_data_sorted['decimal_year'] <= steps[2] + 1.5)
eq_window = eq_data_sorted.loc[mask].copy()

# Now plot only filtered data
scatter_short = axs[2,1].scatter(
    eq_window['decimal_year'], eq_window['mag'],
    c=eq_window['distance'], cmap='magma',
    alpha=1, label='Magnitude', s=15, zorder=1,
    vmin=0, vmax=15
)

# Normalized cumulative curves restricted to same window
norm_moment = eq_window['cum_seismic_moment'] / eq_window['cum_seismic_moment'].max()
norm_count  = eq_window['cum_count'] / eq_window['cum_count'].max()

ax_norm = axs[2,1].twinx()
line_moment, = ax_norm.plot(
    eq_window['decimal_year'], norm_moment,
    color="#023E8A", linewidth=1.5, label='Norm. Cumulative Moment'
)
line_count, = ax_norm.plot(
    eq_window['decimal_year'], norm_count,
    color="#e34a33", linewidth=1.5, ls="--", label='Norm. Cumulative Count'
)

ax_norm.set_ylabel("Normalized (0–1)")
ax_norm.set_ylim(0, 1.05)

axs[2,1].set_xlim(steps[2] - 1.5, steps[2] + 1.5)


import string # To easily get 'a', 'b', 'c', etc.
subplot_labels = string.ascii_lowercase # Gives 'a', 'b', 'c', ...

# for i, ax in enumerate(axs): # Loop through your axes objects
#     # Example plotting for demonstration
#     # ax.plot([1, 2, 3], [i, i+1, i+2])
#     # ax.set_title(f"Subplot {i+1}")

#     # Add the label (a), b), etc.
#     # Adjust x and y (e.g., -0.1, 1.05) and transform=ax.transAxes
#     # to place it just outside the top-left corner of the subplot in axes coordinates.
#     # Add a buffer of (0.05, 0) to shift it slightly right.
#     ax.text(0.0, 1.02, f'({subplot_labels[i]})', transform=ax.transAxes,
#             fontsize=12, va='bottom', ha='left') #

# Save & show
out_path = f'{common_paths["fig_dir"]}/Fig_6_{StaID}_Disp_Vel_Seasonal.png'
plt.savefig(out_path, dpi=300)
plt.show()
print(f"\nSaved: {out_path}")

# # Print results
# print(f"Mean Velocity (v0): {v0:.3f} mm/year")
# print(f"Seasonal Amplitude: {amplitude:.3f} mm/year")
# print(f"Seasonal Phase: {phase:.3f} radians")
# print("Manual Peaks:", manual_peaks)
# print("Harmonic Model Peaks:", harmonic_peak_times)
# print("Manual Peak Timing Offsets:", manual_peak_differences)
# print(f"Mean Absolute Timing Offset (Manual): {manual_mean_peak_offset:.3f} years")
# print(f"Mean Sampling Interval Pre-2008: {mean_sampling_interval_pre_2008:.3f} years")
# print(f"Mean Sampling Interval Post-2008: {mean_sampling_interval_post_2008:.3f} years")
# print(f"Within Sampling Error Pre-2008: {within_error_pre_2008}")

# fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)

# # Panel 1: Raw displacement with trend
# axs[0,0].plot(years, trend_displacement, label="Long-term Trend", color="red", linestyle="--")
# axs[0,0].plot(years, displacement, label="Raw Displacement", color="blue", marker="o")
# axs[0,0].set_title(f"{StaID} Displacement with Long-term Linear Trend", fontsize=14)
# axs[0,0].set_ylabel("Displacement (mm)")
# axs[0,0].legend(fontsize=10)
# axs[0,0].text(0.5, 0.90, f"Velocity = {slope:.1f} ± {std_err:.1f} mm/yr",
#             transform=axs[0,0].transAxes, ha="center", fontsize=12)
# axs[0,0].set_xlim(start_year, end_year)
# axs[0,0].plot([step, step], [45, 80], color="black", alpha=0.5, linestyle=":")

# # Panel 2: Detrended displacement
# axs[1,0].plot(years, detrended_displacement, label="Detrended Displacement", color="blue", marker="o")
# axs[1,0].axhline(0, color="black", linestyle=":", linewidth=0.5)  # Dotted zero line
# axs[1,0].set_title("Detrended Displacement", fontsize=14)
# axs[1,0].set_ylabel("Detrended Displacement (mm)")
# axs[1,0].set_xlim(start_year, end_year)
# axs[1,0].plot([step, step], [0, 10], color="black", alpha=0.5, linestyle=":")

# # Panel 3: Velocity analysis
# axs[2,0].plot(years, velocity_raw, label="Raw Velocity", color="blue")
# axs[2,0].plot(smooth_years, smooth_seasonal_velocity, label="Modeled Seasonal Velocity", color="orange")
# axs[2,0].axhline(0, color="black", linestyle=":", linewidth=0.5)  # Dotted zero line
# axs[2,0].scatter(manual_peaks, manual_peak_values, color="blue")
# axs[2,0].scatter(harmonic_peak_times, harmonic_peak_values,
#                color="orange",)
# axs[2,0].set_title("Velocity Analysis", fontsize=14)
# axs[2,0].set_xlabel("Year")
# axs[2,0].set_ylabel("Velocity (mm/year)")
# axs[2,0].legend(fontsize=10)
# axs[2,0].text(0.5, 0.02, f"Mean Peak Offset = {manual_mean_peak_offset:.3f} years",
#                 transform=axs[2,0].transAxes, ha="center", fontsize=10)
# axs[2,0].set_xlim(start_year, end_year)
# axs[2,0].plot([step, step], [-4, 25], color="black", alpha=0.5, linestyle=":")

# plt.savefig(f'{common_paths["fig_dir"]}/Fig_6_{StaID}_{start_year}_{end_year}_Disp_Vel_Seasonal.png', dpi=300)

# plt.show()


# ##################################3
# # Normalized velocites and precipation
# ##################################3

# # Load the CSV data into a DataFrame
# precip_df = pd.read_csv(MWIL_precip_file)

# # Normalize precipitation values between 0 and 1
# precip_df["Normalized Precipitation"] = (
#     precip_df["Mean Monthly Precipitation (mm)"] - precip_df["Mean Monthly Precipitation (mm)"].min()
# ) / (
#     precip_df["Mean Monthly Precipitation (mm)"].max() - precip_df["Mean Monthly Precipitation (mm)"].min()
# )

# # Normalized time for precipitation, where 0.0 is January 1
# precip_df["Normalized Time"] = precip_df["Decimal Year"] - np.floor(precip_df["Decimal Year"])

# # Ensure the precipitation data is sorted by normalized time
# precip_df = precip_df.sort_values(by="Normalized Time")

# # Normalize the unique velocity years to a range [0, 1] for the colormap
# norm_velocities_by_year = {}
# unique_years = np.unique(np.floor(years).astype(int))
# year_norm = mcolors.Normalize(vmin=unique_years.min(), vmax=unique_years.max())
# colormap = cm.get_cmap("viridis")

# # Group velocities by calendar year
# for year in unique_years:
#     year_start = year
#     year_end = year + 1
#     indices = np.where((years >= year_start) & (years < year_end))[0]
#     if len(indices) > 0:
#         velocities = velocity_raw[indices]
#         year_times = years[indices]
#         min_velocity = velocities.min()
#         max_velocity = velocities.max()
#         norm_velocities = (velocities - min_velocity) / (max_velocity - min_velocity)
#         norm_times = (year_times - year_start) / (year_end - year_start)
#         norm_velocities_by_year[year] = (norm_times, norm_velocities)

# # Plot normalized velocities and precipitation
# fig, ax = plt.subplots(figsize=(12, 6))
# for year, (norm_times, norm_velocities) in norm_velocities_by_year.items():
#     if year in [1991, 2007]:
#         continue  # Skip years 1991 and 2007
#     color = colormap(year_norm(year))
#     ax.plot(norm_times, norm_velocities, color=color, label=f"{year}", alpha=0.5)

# # Explicitly replot the year 2002 in green to ensure it stands out
# if 2002 in norm_velocities_by_year:
#     norm_times, norm_velocities = norm_velocities_by_year[2002]
#     ax.plot(norm_times, norm_velocities, color="red", label="2002 (Step)", linewidth=2)

# # Plot smoothed seasonal cycle
# seasonal_year_start = 1995.0
# seasonal_year_end = 1996.0
# indices = np.where((smooth_years >= seasonal_year_start) & (smooth_years < seasonal_year_end))[0]
# if len(indices) > 0:
#     smoothed_years_clipped = smooth_years[indices]
#     smoothed_velocity_clipped = smooth_seasonal_velocity[indices]
#     min_velocity = smoothed_velocity_clipped.min()
#     max_velocity = smoothed_velocity_clipped.max()
#     norm_velocity = (smoothed_velocity_clipped - min_velocity) / (max_velocity - min_velocity)
#     norm_times = (smoothed_years_clipped - seasonal_year_start) / (seasonal_year_end - seasonal_year_start)
#     ax.plot(norm_times, norm_velocity, label="Smoothed Seasonal Cycle", color="orange", marker="o", linewidth=3)

# # Plot precipitation
# ax.plot(precip_df["Normalized Time"], precip_df["Normalized Precipitation"], marker="o", label="Mean Monthly Precipitation", color="blue", linewidth=3)

# # Customize plot
# plt.title(f"{StaID} Normalized Velocity and Precipitation (Calendar Year)", fontsize=14)
# plt.xlabel("Normalized Time (Jan 1 = 0.0)", fontsize=12)
# plt.ylabel("Normalized Velocity / Precipitation", fontsize=12)
# plt.grid(alpha=0.3)
# plt.xlim(0, 1)  # Set x-axis limits to [0, 1]

# # Move legend to the right
# plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
# plt.savefig(f'{common_paths["fig_dir"]}/Fig_6_{StaID}_Stacked_Vel_Precip_Seasonal_Calendar.png', dpi=300)

# plt.show()
