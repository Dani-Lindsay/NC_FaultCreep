#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 07:30:41 2025

@author: daniellelindsay
"""


import pandas as pd
import numpy as np
import pygmt
import os
import creep_utils as utils
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

import geopandas as gpd
from shapely.geometry import Point, Polygon

from NC_creep_filepaths import common_paths

##############################
# Load Data
##############################

# Load dd catalogue
columns = ['Date', 'Time', 'Lat', 'Lon', 'Depth', 'Mag', 'Magt', 'Nst', 'Gap', 'Clo', 'RMS', 'SRC', 'EventID']
dd_df = pd.read_csv(common_paths["NCEDC"]["M1"], delim_whitespace=True, skiprows=2, names=columns)

#geysers_dd_file = f'{data_dir}/Proj_1_MTJ_creeprates/data/Geysers_ddcat_2005-2025_M2.txt'
columns = ['Date', 'Time', 'Lat', 'Lon', 'Depth', 'Mag', 'Magt', 'Nst', 'Gap', 'Clo', 'RMS', 'SRC', 'EventID']
dd_geysers_df = pd.read_csv(common_paths["NCEDC"]["Geysers"], delim_whitespace=True, skiprows=2, names=columns)

# Load creep rate files 
params = common_paths["params"]

NMF_creep_info_df = pd.read_csv(common_paths["NMF"]["170_creeprate"], sep=',', header=0)
SMF_creep_info_df = pd.read_csv(common_paths["SMF"]["170_creeprate"], sep=',', header=0)
RCF_creep_info_df = pd.read_csv(common_paths["RCF"]["170_creeprate"], sep=',', header=0)

places = {
    "Santa Rosa": {"lat": 38.4404, "lon": -122.7141},
    "Cloverdale": {"lat": 38.8055, "lon": -123.0172},
    "Ukiah": {"lat": 39.1502, "lon": -123.2078},
    "Willits": {"lat": 39.4096, "lon": -123.3556},  
    "Laytonville": {"lat": 39.6882, "lon": -123.4828},
    "Hopland": {"lat": 38.9702, "lon": -123.1166},
    "Jimtown": {"lat": 38.6666, "lon": -122.8197}
}

# Convert dictionary to DataFrame
places_df = pd.DataFrame([
    {"Place": place, "Lat": coords["lat"], "Lon": coords["lon"]}
    for place, coords in places.items()
])

# Example usage:
for place, coords in places.items():
    print(f"{place}: Latitude {coords['lat']}, Longitude {coords['lon']}")

##############################
# Load Takaaki's Cat
##############################

# Define column names for CRE_taka_info_df
taka_column_names = ['YYYY', 'Lat', 'Lon', 'Depth', 'cumD(cm)', 'EVID', 'CSID', 'Mag', 'D(cm)']

# Read the file into a DataFrame for 'taka'
taka_df_MA = pd.read_csv(common_paths["CRE_Taka"]["MA"], sep=',', header=None, comment='#', names=taka_column_names, skiprows=1)
taka_df_MA = taka_df_MA.sort_values(by='CSID')

taka_df_RC = pd.read_csv(common_paths["CRE_Taka"]["RC"], sep='\s+', header=None, comment='#', names=taka_column_names, skiprows=1)
taka_df_RC = taka_df_RC.sort_values(by='CSID')

taka_df = pd.concat([taka_df_MA, taka_df_RC], ignore_index=True)

# Initialize storage for DataFrame and dictionary for 'taka'
CRE_taka_info_data = []
CRE_taka_info_dict = {}

# Function to calculate recurrence statistics
def calculate_recurrence_statistics(events_df):
    events_df = events_df.sort_values(by='YYYY')
    intervals = events_df['YYYY'].diff().dropna()
    
    if len(intervals) < 1:
        return {
            'RCm': np.nan,
            'RCs': np.nan,
            'RCcv': np.nan,
            'RCm1': np.nan,
            'RCs1': np.nan,
            'RCcv1': np.nan
        }
    
    rc_median = intervals.median()
    rc_std = intervals.std()
    rc_cv = rc_std / rc_median if rc_median != 0 else np.nan
    is_quasi_periodic = len(intervals) > 2
    
    return {
        'RCm': rc_median,
        'RCs': rc_std,
        'RCcv': rc_cv,
        'RCm1': rc_median if is_quasi_periodic else np.nan,
        'RCs1': rc_std if is_quasi_periodic else np.nan,
        'RCcv1': rc_cv if is_quasi_periodic else np.nan
    }

# Process 'taka' dataset
for csid, group_df in taka_df.groupby('CSID'):
    CRE_taka_info_dict[csid] = group_df
    
    nevents = len(group_df)
    latm = group_df['Lat'].median()
    lonm = group_df['Lon'].median()
    depm = group_df['Depth'].median()
    mag_median = group_df['Mag'].median()
    mag_std = group_df['Mag'].std()
    cumD_median = group_df['cumD(cm)'].median()
    cumD_std = group_df['cumD(cm)'].std()
    
    recurrence_stats = calculate_recurrence_statistics(group_df)
    
    cc_median = np.nan
    
    CRE_taka_info_data.append({
        'NEV': nevents,
        'LATm': latm,
        'LONm': lonm,
        'DEPm': depm,
        'DMAGm': mag_median,
        'DMAGs': mag_std,
        'RCm': recurrence_stats['RCm'],
        'RCs': recurrence_stats['RCs'],
        'RCcv': recurrence_stats['RCcv'],
        'RCm1': recurrence_stats['RCm1'],
        'RCs1': recurrence_stats['RCs1'],
        'RCcv1': recurrence_stats['RCcv1'],
        'CCm': cc_median,
        'seqID': csid
    })

CRE_taka_info_df = pd.DataFrame(CRE_taka_info_data)


##################3
# Load AA timeseries
##################3

# Read the Table 1b.csv file into a DataFrame, skipping the first line and using the third as the header
AA_table_df = pd.read_csv(common_paths["AA"]["table"], encoding='ISO-8859-1', skiprows=1)

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

AA_station_info = AA_station_info.dropna()

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
        file_path = f'{common_paths["AA"]["dir"]}/{station_file}'

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

MA_AA_station_info = AA_station_info [AA_station_info["Fault"] == "Maacama"]
RC_AA_station_info = AA_station_info [(AA_station_info["Fault"] == "Rodgers Creek") & (AA_station_info["Velocity(mm/yr)"] > 0)].copy()



###################
# Load my creep data
###################

# Load polygon geometries
RC_CRE_polygon = utils.load_polygon_gmt(common_paths["RC_CRE_poly"])
MA_CRE_polygon = utils.load_polygon_gmt(common_paths["MA_CRE_poly"])

# Convert CRE_taka_info_df to a GeoDataFrame
cre_gdf = gpd.GeoDataFrame(
    CRE_taka_info_df,
    geometry=gpd.points_from_xy(CRE_taka_info_df['LONm'], CRE_taka_info_df['LATm']),
    crs="EPSG:4326"
)

dd_gdf = gpd.GeoDataFrame(
    dd_df,
    geometry=gpd.points_from_xy(dd_df['Lon'], dd_df['Lat']),
    crs="EPSG:4326"
)

# Filter points inside each polygon
RC_CRE_taka_info_df = cre_gdf[cre_gdf.geometry.within(RC_CRE_polygon)].copy()
MA_CRE_taka_info_df = cre_gdf[cre_gdf.geometry.within(MA_CRE_polygon)].copy()

RC_dd_info_df = dd_gdf[dd_gdf.geometry.within(RC_CRE_polygon)].copy()
MA_dd_info_df = dd_gdf[dd_gdf.geometry.within(MA_CRE_polygon)].copy()

# Define profile variables 
MA_cen_la, MA_cen_lo, MA_az, MA_dist, MA_width = 39.0611, -123.0857, 150, 93, 10
RC_cen_la, RC_cen_lo, RC_az, RC_dist, RC_width = 38.4315, -122.6723, 148, 43, 10

# Get start and end points using provided utility function
MA_start_lon, MA_start_lat, MA_end_lon, MA_end_lat = utils.get_start_end_points(MA_cen_lo, MA_cen_la, MA_az, MA_dist)
RC_start_lon, RC_start_lat, RC_end_lon, RC_end_lat = utils.get_start_end_points(RC_cen_lo, RC_cen_la, RC_az, RC_dist)

MA_AA_df = MA_AA_station_info[['Lon', 'Lat', 'Velocity(mm/yr)']].copy()
MA_AA_df['Velocity(mm/yr)'] = pd.to_numeric(MA_AA_df['Velocity(mm/yr)'], errors='coerce')
#MA_AA_df = MA_AA_df.dropna()  # Drop any rows with NaN values

# Repeat for RC
RC_AA_df = RC_AA_station_info[['Lon', 'Lat', 'Velocity(mm/yr)']].copy()
RC_AA_df['Velocity(mm/yr)'] = pd.to_numeric(RC_AA_df['Velocity(mm/yr)'], errors='coerce')
#RC_AA_df = RC_AA_df.dropna()

# Extracting necessary data from CRE DataFrames


NMF_creep_df = NMF_creep_info_df[['ft_lon','ft_lat','170_fault_parallel_creep_rate']]
SMF_creep_df = SMF_creep_info_df[['ft_lon','ft_lat','170_fault_parallel_creep_rate']]
RC_creep_df = RCF_creep_info_df[['ft_lon','ft_lat','170_fault_parallel_creep_rate']]
RC_dd_df = RC_dd_info_df[['Lon','Lat','Depth']]
MA_dd_df = MA_dd_info_df[['Lon','Lat','Depth']]
RC_cre_df = RC_CRE_taka_info_df[['LONm', 'LATm', 'DEPm']]
MA_cre_df = MA_CRE_taka_info_df[['LONm', 'LATm', 'DEPm']]

# Project data to extract the profile for both wald and seno
MA_cross_AA =   pygmt.project(data=MA_AA_df, center=[MA_cen_lo, MA_cen_la], length=[-MA_dist, MA_dist], width=[-MA_width, MA_width], azimuth=MA_az, unit=True)
MA_cross_dd =   pygmt.project(data=MA_dd_df, center=[MA_cen_lo, MA_cen_la], length=[-MA_dist, MA_dist], width=[-MA_width, MA_width], azimuth=MA_az, unit=True)
NMF_cross_creep =   pygmt.project(data=NMF_creep_df, center=[MA_cen_lo, MA_cen_la], length=[-MA_dist, MA_dist], width=[-MA_width, MA_width], azimuth=MA_az, unit=True)
SMF_cross_creep =   pygmt.project(data=SMF_creep_df, center=[MA_cen_lo, MA_cen_la], length=[-MA_dist, MA_dist], width=[-MA_width, MA_width], azimuth=MA_az, unit=True)
MA_cross_cre = pygmt.project(data=MA_cre_df, center=[MA_cen_lo, MA_cen_la], length=[-MA_dist, MA_dist], width=[-MA_width, MA_width], azimuth=MA_az, unit=True)

RC_cross_AA =   pygmt.project(data=RC_AA_df, center=[RC_cen_lo, RC_cen_la], length=[-RC_dist, RC_dist], width=[-RC_width, RC_width], azimuth=RC_az, unit=True)
RC_cross_dd =   pygmt.project(data=RC_dd_df, center=[RC_cen_lo, RC_cen_la], length=[-RC_dist, RC_dist], width=[-RC_width, RC_width], azimuth=RC_az, unit=True)
RC_cross_creep =   pygmt.project(data=RC_creep_df, center=[RC_cen_lo, RC_cen_la], length=[-RC_dist, RC_dist], width=[-RC_width, RC_width], azimuth=RC_az, unit=True)
RC_cross_cre = pygmt.project(data=RC_cre_df, center=[RC_cen_lo, RC_cen_la], length=[-RC_dist, RC_dist], width=[-RC_width, RC_width], azimuth=RC_az, unit=True)


# Assign appropriate column names
MA_cross_AA.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
MA_cross_dd.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
NMF_cross_creep.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
SMF_cross_creep.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
MA_cross_cre.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']

RC_cross_AA.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
RC_cross_dd.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
RC_cross_creep.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']
RC_cross_cre.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']


MA_d90_df = utils.calculate_seismo_depth(MA_cross_dd)
RC_d90_df = utils.calculate_seismo_depth(RC_cross_dd)
RC_d90_df = RC_d90_df[RC_d90_df['profile_km'] <= 20].reset_index(drop=True)

# def moving_average_within_range(df, value_col, p_col, window_range):
#     """For each row in df, compute the mean of value_col for all rows 
#     with p values within ±window_range of that row's p value."""
#     return df.apply(lambda row: df.loc[(df[p_col] >= row[p_col] - window_range) & 
#                                        (df[p_col] <= row[p_col] + window_range), value_col].mean(), axis=1)



# MA_d90_df['D90_depth_smoothed'] = moving_average_within_range(MA_d90_df,'D90_depth_km','profile_km', 10)
# RC_d90_df['D90_depth_smoothed'] = moving_average_within_range(RC_d90_df,'D90_depth_km','profile_km', 10)

# MA_d90_df['D95_depth_smoothed'] = moving_average_within_range(MA_d90_df,'D95_depth_km','profile_km', 10)
# RC_d90_df['D95_depth_smoothed'] = moving_average_within_range(RC_d90_df,'D95_depth_km','profile_km', 10)


#MA_d90_df['seismo_depth_smoothed'] = gaussian_filter1d(MA_d90_df['seismo_depth'], sigma=2)
#RC_d90_df['seismo_depth_smoothed'] = gaussian_filter1d(RC_d90_df['seismo_depth'], sigma=2)
#MA_d90_df['D90_depth_smoothed'] = gaussian_filter1d(MA_d90_df['D90_depth_km'], sigma=2)
#RC_d90_df['D90_depth_smoothed'] = gaussian_filter1d(RC_d90_df['D90_depth_km'], sigma=2)


def gaussian_smooth_within_range(df, value_col, p_col, sigma_range, cutoff=3.0):
    """
    Centered Gaussian smoothing in p-units (e.g., km), no phase shift.
    Returns a Series aligned to df.index.
      - sigma_range: Gaussian sigma in same units as p_col
      - cutoff: kernel half-width in sigmas (default 3σ)
    """
    # Work on a copy with just the needed columns; preserve original index
    sub = df[[p_col, value_col]].copy()

    # Drop rows with NaN in either column; remember which to put back
    valid = sub[p_col].notna() & sub[value_col].notna()
    sub = sub.loc[valid]

    if sub.empty:
        return pd.Series(np.nan, index=df.index, name=f"{value_col}_gauss")

    # Sort by p for a proper 1-D pass
    sub = sub.sort_values(p_col)
    x = sub[p_col].to_numpy()
    y = sub[value_col].to_numpy(dtype=float)

    # Decide if spacing is "near-uniform" (enable fast path)
    dx = np.diff(x)
    uniform = False
    if dx.size > 0:
        dx_med = np.median(dx)
        if dx_med > 0 and np.all(dx > 0):
            uniform = (np.max(np.abs(dx - dx_med)) / dx_med) < 0.2  # <=20% variation

    if uniform:
        # Convert sigma from p-units to samples
        sigma_samples = max(sigma_range / dx_med, 1e-9)
        y_s = gaussian_filter1d(y, sigma=sigma_samples, mode="nearest", truncate=cutoff)
    else:
        # Irregular spacing: explicit Gaussian in p-units
        n = len(x)
        y_s = np.empty(n, dtype=float)
        halfwidth = cutoff * sigma_range
        for i in range(n):
            xi = x[i]
            d = x - xi
            m = np.abs(d) <= halfwidth
            w = np.exp(-0.5 * (d[m] / sigma_range) ** 2)
            ym = y[m]
            y_s[i] = np.sum(w * ym) / np.sum(w)

    # Put smoothed values back on the sorted sub's index, then reindex to original df
    smooth_series = pd.Series(y_s, index=sub.index, name=f"{value_col}_gauss")
    # Fill NaNs for rows we dropped earlier; align to original df.index
    return smooth_series.reindex(df.index)

sigma_km = 10 / np.sqrt(3)   # ≈ 5.77 km for a ±10 km boxcar-equivalent


MA_d90_df['D90_depth_smoothed'] = gaussian_smooth_within_range(MA_d90_df, 'D90_depth_km', 'profile_km', sigma_km)
RC_d90_df['D90_depth_smoothed'] = gaussian_smooth_within_range(RC_d90_df, 'D90_depth_km', 'profile_km', sigma_km)

MA_d90_df['D95_depth_smoothed'] = gaussian_smooth_within_range(MA_d90_df, 'D95_depth_km', 'profile_km', sigma_km)
RC_d90_df['D95_depth_smoothed'] = gaussian_smooth_within_range(RC_d90_df, 'D95_depth_km', 'profile_km', sigma_km)

MA_d90_df_env = pd.DataFrame(
 data={
     "x": MA_d90_df["profile_km"],
     "y": MA_d90_df["seismo_depth"],
     "y_deviation_low":  MA_d90_df["std_seismo_depth"],
     "y_deviation_upp":  MA_d90_df["std_seismo_depth"],
     }
)

RC_d90_df_env = pd.DataFrame(
    data={
        "x": RC_d90_df["profile_km"],
        "y": RC_d90_df["seismo_depth"],
        "y_deviation_low":  RC_d90_df["std_seismo_depth"],
        "y_deviation_upp":  RC_d90_df["std_seismo_depth"],
    }
)

NMF_merged_df = pd.merge(NMF_cross_creep, NMF_creep_info_df, left_on='z', right_on='170_fault_parallel_creep_rate', how='inner')
SMF_merged_df = pd.merge(SMF_cross_creep, SMF_creep_info_df, left_on='z', right_on='170_fault_parallel_creep_rate', how='inner')

# Combine the two merged DataFrames
combined_df = pd.concat([NMF_merged_df, SMF_merged_df], ignore_index=True)

# Sort by ft_lat descending (assuming higher ft_lat means more northern)
combined_df = combined_df.sort_values(by='ft_lat', ascending=False)

# Drop duplicate rows based on ft_lat, keeping the first (northern) record
MA_cross_creep = combined_df.drop_duplicates(subset=['ft_lat'], keep='first')


# # Apply the smoothing using a ±5 km window based on the 'p' column
# #NMF_cross_creep['z_smooth'] = moving_average_within_range(NMF_cross_creep, 'z', 'p', 10)
# MA_cross_creep['z_smooth'] = moving_average_within_range(MA_cross_creep, 'z', 'p', 10)
# RC_cross_creep['z_smooth'] = moving_average_within_range(RC_cross_creep, 'z', 'p', 10)

# Gaussian smoothing in p-units (km), no phase shift
sigma_km = 10 / np.sqrt(3)   # ≈ 5.77 km  (boxcar ±10 km → Gaussian σ match)

MA_cross_creep['z_smooth'] = gaussian_smooth_within_range(MA_cross_creep, 'z', 'p', sigma_km)
RC_cross_creep['z_smooth'] = gaussian_smooth_within_range(RC_cross_creep, 'z', 'p', sigma_km)


MA_creep_env = pd.DataFrame(
    data={
        "x": MA_cross_creep['p'], 
        "y": MA_cross_creep['z'],
        "y_deviation_low":  MA_cross_creep["170_fault_parallel_std_err"]*3,
        "y_deviation_upp":  MA_cross_creep["170_fault_parallel_std_err"]*3,
    }
)

RC_creep_env = pd.DataFrame(
    data={
        "x": RC_cross_creep['p'], 
        "y": RC_cross_creep['z'],
        "y_deviation_low":  RCF_creep_info_df["170_fault_parallel_std_err"]*3,
        "y_deviation_upp":  RCF_creep_info_df["170_fault_parallel_std_err"]*3,
    }
)


MA_cross_AA = pd.concat([MA_cross_AA.reset_index(drop=True),
                                MA_AA_station_info.reset_index(drop=True)], axis=1)

RC_cross_AA = pd.concat([RC_cross_AA.reset_index(drop=True),
                                RC_AA_station_info.reset_index(drop=True)], axis=1)

#######################
# Begin Plot
########################

proj_MF = "Oa-120/25/-30/11.8c"
region_MF = "-123.8/39.7/-122.4/38.4+r"
proj_MF_x = "X11.8/2"
region_MF_x = f"-{MA_dist}, {MA_dist}, -10, 10"
proj_MF_xd = "X11.8/-3"
region_MF_xd = f"-{MA_dist}, {MA_dist}, 0, 15"

proj_RC = "Oa-120/25/-30/6.4c"
region_RC = "-123.05/38.7/-122.3/38.15+r"
proj_RC_x = "X6.4/2"
region_RC_x = f"-{RC_dist}, {RC_dist}, -10, 10"
proj_RC_xd = "X6.4/-3"
region_RC_xd = f"-{RC_dist}, {RC_dist}, 0, 15"

taka_style = "c.15c"  # Circle style for Taka dataset

RC_ref_lat = 38.44183
RC_ref_lon = -122.74641

SM_ref_lat = 38.809587
SM_ref_lon = -123.012755

NM_ref_lat = 39.389548
NM_ref_lon = -123.347155

# Sort by NEV in descending order (larger families last) and Depth in ascending order (shallowest last)
CRE_taka_info_df = CRE_taka_info_df.sort_values(by=['DEPm'], ascending=[False])

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=9)

with fig.subplot(nrows=2, ncols=2, figsize=("25c", "4c"), frame="lrtb", autolabel=True):
    # Create color palette for depth
    pygmt.makecpt(cmap="inferno", series=[0, 12], reverse=True)
    
    ######## MA CREs
    fig.coast(panel=True, projection=proj_MF, region=region_MF, frame=["WbrN", "xa0.3", "ya0.2"], land="gray91", water="steelblue",)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, projection=proj_MF, region=region_MF, pen="0.5p,black", transparency=50)
    fig.plot(x=dd_df['Lon'], y=dd_df['Lat'], style="c.05c", fill='black', 
                 transparency=70, projection=proj_MF, region=region_MF,)
    fig.plot(x=dd_geysers_df['Lon'], y=dd_geysers_df['Lat'], style="c.05c", fill='black', 
                 transparency=70, projection=proj_MF, region=region_MF,)
    fig.plot(projection=proj_MF, region=region_MF, x=CRE_taka_info_df['LONm'], y=CRE_taka_info_df['LATm'], 
             style=taka_style, fill=CRE_taka_info_df['DEPm'], cmap=True, pen="0.2p,black")
    fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_MF, region=region_MF,
             style="t.2c", fill="dodgerblue", pen="0.5p")
    
    fig.basemap(frame="lrtb", map_scale="jBL+w10k+o0.2/0.5c", projection=proj_RC, region=region_RC)
    
    ######## RC CREs
    fig.coast(panel=True, projection=proj_RC, region=region_RC, frame=["lbrN", "xa0.3", "ya0.2"], land="gray91", water="white",  shorelines=True,)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, projection=proj_RC, region=region_RC, pen="0.5p,black", transparency=70)
    
    fig.plot(x=dd_df['Lon'], y=dd_df['Lat'], style="c.05c", fill='black', 
                 transparency=70, projection=proj_RC, region=region_RC,)
    fig.plot(projection=proj_RC, region=region_RC, x=CRE_taka_info_df['LONm'], y=CRE_taka_info_df['LATm'], 
             style=taka_style, fill=CRE_taka_info_df['DEPm'], cmap=True, pen="0.2p,black")
    fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_RC, region=region_RC,
                 style="t.2c", fill="dodgerblue", pen="0.5p")
    
    fig.basemap(frame="lrtb", map_scale="jBL+w10k+o0.2/0.5c", projection=proj_RC, region=region_RC)
    
    ######## MA InSAR - LOS
    fig.coast(panel=True, projection=proj_MF, region=region_MF, frame=["Wbrt", "xa0.3", "ya0.2"], land="gray91", water="steelblue",)
    
    pygmt.makecpt(cmap="vik", series=[-0.006, 0.006])
    fig.grdimage(grid=common_paths["NMF"]["vel_grd"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    fig.grdimage(grid=common_paths["SMF"]["vel_grd"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    #fig.grdimage(grid=common_paths["NMF"]["up_068_170"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    #fig.grdimage(grid=common_paths["SMF"]["up_068_170"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    fig.plot(x=SM_ref_lon, y=SM_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    fig.plot(x=NM_ref_lon, y=NM_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, projection=proj_MF, region=region_MF, pen="0.5p,black", transparency=50)
        
    fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_MF, region=region_MF,
             style="t.2c", fill="dodgerblue", pen="0.5p")
    fig.coast(projection=proj_MF, region=region_MF, frame="Wrtb", land=None, water="white", shorelines=True,)
    fig.text(text="Line-of-sight", position="BL", offset="0.2/0.2c", projection=proj_MF, region=region_MF,)

    
    ######## RC InSAR - LOS
    fig.coast(panel=True, projection=proj_RC, region=region_RC, frame="lrtb", land="gray91", water="white", shorelines=True,)
    pygmt.makecpt(cmap="vik", series=[-0.006, 0.006])
    fig.grdimage(grid=common_paths["RCF"]["vel_grd"], cmap=True, projection=proj_RC, region=region_RC,)
    #fig.grdimage(grid=common_paths["RCF"]["up_068_170"], cmap=True, projection=proj_RC, region=region_RC,)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, projection=proj_RC, region=region_RC, pen="0.5p,black", transparency=50)
    fig.plot(x=RC_ref_lon, y=RC_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_RC, region=region_RC,
                 style="t.2c", fill="dodgerblue", pen="0.5p")
    
    fig.coast(projection=proj_RC, region=region_RC, frame="lrtb", land=None, water="white", shorelines=True,)
    


    # ######## MA InSAR - FP
    # fig.coast(panel=True, projection=proj_MF, region=region_MF, frame=["Wbrt", "xa0.3", "ya0.2"], land="gray91", water="steelblue",)
    
    # pygmt.makecpt(cmap="vik", series=[-0.006, 0.006])
    # #fig.grdimage(grid=common_paths["NMF"]["vel_grd"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    # #fig.grdimage(grid=common_paths["SMF"]["vel_grd"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    # fig.grdimage(grid=common_paths["NMF"]["FP_068_170"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    # fig.grdimage(grid=common_paths["SMF"]["FP_068_170"], cmap=True, projection=proj_MF, region=region_MF, nan_transparent=True)
    # fig.plot(x=SM_ref_lon, y=SM_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    # fig.plot(x=NM_ref_lon, y=NM_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    
    # for fault_file in common_paths["fault_files"]:
    #     fig.plot(data=fault_file, projection=proj_MF, region=region_MF, pen="0.5p,black", transparency=50)
        
    # fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_MF, region=region_MF,
    #          style="t.2c", fill="dodgerblue", pen="0.5p")
    # fig.coast(projection=proj_MF, region=region_MF, frame="Wrtb", land=None, water="white", shorelines=True,)
    
    # fig.text(text="Fault Parallel", position="BL", offset="0.2/0.2c", projection=proj_MF, region=region_MF,)

    
    # ######## RC InSAR - FP
    # fig.coast(panel=True, projection=proj_RC, region=region_RC, frame="lrtb", land="gray91", water="white", shorelines=True,)
    # pygmt.makecpt(cmap="vik", series=[-0.006, 0.006])
    # #fig.grdimage(grid=common_paths["RCF"]["vel_grd"], cmap=True, projection=proj_RC, region=region_RC,)
    # fig.grdimage(grid=common_paths["RCF"]["FP_068_170"], cmap=True, projection=proj_RC, region=region_RC,)
    # for fault_file in common_paths["fault_files"]:
    #     fig.plot(data=fault_file, projection=proj_RC, region=region_RC, pen="0.5p,black", transparency=50)
    # fig.plot(x=RC_ref_lon, y=RC_ref_lat, style="s.2c", fill="black", pen="0.2p,black",  projection=proj_RC, region=region_RC,)
    # fig.plot(x=AA_station_info["Lon"], y=AA_station_info["Lat"], projection=proj_RC, region=region_RC,
    #              style="t.2c", fill="dodgerblue", pen="0.5p")
    
    # fig.coast(projection=proj_RC, region=region_RC, frame="lrtb", land=None, water="white", shorelines=True,)
    


fig.shift_origin(yshift="-2.5c")

c_min = -3
c_max = 13 

with fig.subplot(nrows=1, ncols=2, figsize=("25", "2.0c"), frame="lrtb", autolabel="e)"):
    fig.basemap(panel=True, projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max], 
                frame=['xaf+l"Distance"', 'ya+lCreep Rate (mm/yr)', "Ws"])
    
    
    fig.plot(x=[-MA_dist, MA_dist], y=[0, 0], pen="0.5p,black", label="ALOS-2",
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    # Plot a symmetrical envelope based on the deviations ("+d")
    fig.plot(data=MA_creep_env, close="+d", fill="gray@50", projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    fig.plot(x=[-MA_dist, MA_dist], y=[0, 0], pen="0.5p,gray,--", transparency=50,
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    fig.plot(x=MA_cross_creep['p'], y=MA_cross_creep['z'], pen="0.5p", 
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max],)
    
    fig.plot(x=MA_cross_creep['p'], y=MA_cross_creep['z_smooth'], pen="1p,red",
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max],)


    fig.plot(x=MA_cross_AA['p'][0], y=3.1, style="-.2c", fill="purple", pen="1p", 
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max], label="Sentinel-1")
    
    
    # fig.plot(x=MA_cross_AA['p'][0], y=3.7, style="t.2c", fill="red", pen="0.5p", label="AA 2002-2010", 
    #          projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    # fig.plot(x=MA_cross_AA['p'][1], y=3.4, style="t.2c", fill="red", pen="0.5p", 
    #          projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    

    fig.plot(x=MA_cross_AA['p'], y=MA_cross_AA['z'], style="t.2c", fill="dodgerblue", pen="0.5p", label="Alignment",
             projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    fig.text(x=MA_cross_AA['p'], y=MA_cross_AA['z'], text=MA_cross_AA['Sta_ID'], 
             font="8,Helvetica", offset="0.0/0.4c", projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
    
    # For multi-column legends users have to provide the width via +w
    fig.legend(position="jTR+o0.1c/-0.1c", projection=proj_MF_x, region=[-MA_dist, MA_dist, c_min, c_max])
   
    fig.basemap(panel=True, projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max], 
                frame=['xaf+lDistance', 'ya', "Ws"])
    
    fig.plot(x=[-RC_dist, RC_dist], y=[0, 0], pen="0.5p,gray,--", transparency=50,
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])
    
    # Plot a symmetrical envelope based on the deviations ("+d")
    fig.plot(data=RC_creep_env, close="+d", fill="gray@50",
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])
    
    fig.plot(x=RC_cross_creep['p'], y=RC_cross_creep['z'], pen="0.5p", 
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max],)

    fig.plot(x=RC_cross_AA['p'][1], y=3.3, style="-.2c", fill="purple", pen="1p", 
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])
    
    fig.plot(x=RC_cross_AA['p'], y=RC_cross_AA['z'], style="t.2c", fill="dodgerblue", pen="0.5p", 
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])
    
    fig.text(x=RC_cross_AA['p'], y=RC_cross_AA['z'], text=RC_cross_AA['Sta_ID'], 
             font="8,Helvetica", offset="0.0/0.4c", projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])
    
    fig.plot(x=RC_cross_creep['p'], y=RC_cross_creep['z_smooth'], pen="1p,red", 
             projection=proj_RC_x, region=[-RC_dist, RC_dist, c_min, c_max])

fig.shift_origin(yshift="-3.5c")

z_min = 0
z_max = 13

with fig.subplot(nrows=1, ncols=2, figsize=("25", "3.0c"), frame="lrtb", autolabel="g)"):
    
    fig.basemap(panel=True, projection=proj_MF_xd, region=[-MA_dist, MA_dist, z_min, z_max], 
                frame=['xaf+lDistance (km)', 'yaf+lDepth (km)', "WSrt"])

    
    # Plot a symmetrical envelope based on the deviations ("+d")
    #fig.plot(data=MA_d90_df_env, close="+d", fill="gray@50", projection=proj_MF_xd, region=[-MA_dist, MA_dist, 0, 15])
    
    fig.plot(x=MA_cross_dd['p'], y=MA_cross_dd['z'], style="c.05c", fill='black', 
             transparency=70, projection=proj_MF_xd, region=[-MA_dist, MA_dist, z_min, z_max])
    
    #fig.plot(x=MA_d90_df['profile_km'], y=MA_d90_df['seismo_depth'], projection=proj_MF_xd, region=[-MA_dist, MA_dist, 0, 15], 
    #         pen="1p,dodgerblue")
    fig.plot(x=MA_d90_df['profile_km'], y=MA_d90_df['D90_depth_smoothed'], projection=proj_MF_xd, region=[-MA_dist, MA_dist, z_min, z_max], 
             pen="1p,dodgerblue")
    
    fig.plot(x=MA_d90_df['profile_km'], y=MA_d90_df['D95_depth_smoothed'], projection=proj_MF_xd, region=[-MA_dist, MA_dist, z_min, z_max], 
             pen="1p,dodgerblue4")
    
    pygmt.makecpt(cmap="inferno", series=[0, 12], reverse=True) 
    fig.plot(x=MA_cross_cre['p'], y=MA_cross_cre['z'], style=taka_style, fill=MA_cross_cre['z'],
             cmap=True, pen=0.5, projection=proj_MF_xd, region=[-MA_dist, MA_dist, z_min, z_max])
    
    
    
    fig.basemap(panel=True, projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max],
                frame=['xaf+lDistance (km)', 'yaf', "WSrt"])
    
    # Plot a symmetrical envelope based on the deviations ("+d")
    #fig.plot(data=RC_d90_df_env, close="+d", fill="gray@50", projection=proj_RC_xd, region=[-RC_dist, RC_dist, 0, 15])
    #
    #fig.plot(x=RC_d90_df['profile_km'], y=RC_d90_df['seismo_depth'], projection=proj_RC_xd, region=[-RC_dist, RC_dist, 0, 15], 
    #         pen="1p,dodgerblue")
    fig.plot(x=RC_d90_df['profile_km'], y=RC_d90_df['D90_depth_smoothed'], projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max], 
             pen="1p,dodgerblue,", label="Depth 90%")
    
    fig.plot(x=RC_d90_df['profile_km'], y=RC_d90_df['D95_depth_smoothed'], projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max], 
             pen="1p,dodgerblue4", label="Depth 95%")

    fig.plot(x=RC_cross_dd['p'], y=RC_cross_dd['z'], style="c.05c", fill='black', 
             transparency=70, projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max])
    
    pygmt.makecpt(cmap="inferno", series=[0, 12], reverse=True) 
    fig.plot(x=RC_cross_cre['p'], y=RC_cross_cre['z'], style=taka_style, fill=RC_cross_cre['z'],
             cmap=True, pen=0.5, projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max])
    
    # For multi-column legends users have to provide the width via +w
    fig.legend(position="jTR+o0.1c/0.1c", projection=proj_RC_xd, region=[-RC_dist, RC_dist, z_min, z_max])
    
# Save and show the figure
fig.savefig(f'{common_paths["fig_dir"]}/Fig_5_LOSvelgrd_summary_{params}.jpg', transparent=False, crop=True, anti_alias=True, show=False, dpi=400)
fig.savefig(f'{common_paths["fig_dir"]}/Fig_5_LOSvelgrd_summary_{params}.png', transparent=False, crop=True, anti_alias=True, show=False, dpi=400)
fig.savefig(f'{common_paths["fig_dir"]}/Fig_5_LOSvelgrd_summary_{params}.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()


        

