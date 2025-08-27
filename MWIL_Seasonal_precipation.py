#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:47:01 2024

Extract annual precipation from PRISM

@author: daniellelindsay

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pygmt
import os
from datetime import datetime, timedelta
import pandas as pd

from NC_creep_filepaths import common_paths



# Base directory where files are stored
#base_dir = '/Users/daniellelindsay/Google Drive/My Drive/Research_Archieve/Proj_9_Ferndale/data/PRISM_ppt_stable_4kmM3_198101_202304_bil'

base_dir = common_paths["PRISM"]["bil_dir"]
# File naming convention
file_prefix = 'PRISM_ppt_stable_4kmM3'

# Define the water year
start_WY = 9  # April
end_WY = 10    # March

# Data
start_data_year, start_data_month = 1993, 10 
end_data_year, end_data_month = 2023, 4

# Define the start and end years for the long term average
start_analysis_year = start_data_year

# Observation time period
start_obs_year, start_obs_month = 1991, 1  
end_obs_year, end_obs_month = 2024, 1

start_obs_date = datetime(start_obs_year, start_obs_month, 1)
end_obs_date = datetime(end_obs_year, end_obs_month, 1)

# Define the target latitude and longitude
target_lat =  39.41242
target_lon = -123.35612


lat_min = 39.3
lat_max = 39.6
lon_min = -123.8
lon_max = -123.0

# Define the latitude and longitude range
lat_range = [lat_min, lat_max]
lon_range = [lon_min, lon_max]

def read_bil_file(bil_path, hdr_path, lat_range, lon_range):
    # Read the header file to get metadata
    with open(hdr_path, 'r') as hdr_file:
        hdr_info = hdr_file.readlines()

    # Correctly parse the header information
    nrows = int(hdr_info[2].split()[1])
    ncols = int(hdr_info[3].split()[1])
    xllcorner = float(hdr_info[9].split()[1])
    yllcorner = float(hdr_info[10].split()[1])
    cellsize = float(hdr_info[11].split()[1])
    NODATA_value = float(hdr_info[13].split()[1])

    # Read the .bil file
    data = np.fromfile(bil_path, dtype=np.float32).reshape((nrows, ncols))
    #data = data[::-1, :]  # This will reverse the data array along the latitude axis
    data[data == NODATA_value] = np.nan
    data = data 

    # Generate full longitude and latitude arrays
    lons = np.linspace(xllcorner, xllcorner + cellsize * (ncols - 1), ncols)
    lats = np.linspace(yllcorner - cellsize * (nrows - 1), yllcorner, nrows)
    lats = np.flip(lats)  # This will reverse the latitude array

    # Assuming lats is a 1D array of latitude values
    row_min = np.argmin(np.abs(lats - lat_min))
    row_max = np.argmin(np.abs(lats - lat_max))

    # Ensure lat_min_idx is less than lat_max_idx
    if row_min > row_max:
        row_min, row_max = row_max, row_min

    col_min = int((lon_range[0] - xllcorner) / cellsize)
    col_max = int((lon_range[1] - xllcorner) / cellsize) + 1

    # Crop the data
    data_cropped = data[row_min:row_max, col_min:col_max]
    # Crop the latitude and longitude arrays to match the data
    lats_cropped = lats[row_min:row_max]
    lons_cropped = lons[col_min:col_max]

    return data_cropped, lats_cropped, lons_cropped, 


# Function to generate file paths and corresponding year-month pairs
def generate_monthly_filenames_and_dates(start_data_year, start_data_month, end_data_year, end_data_month, base_dir, file_prefix):
    start_date = datetime(start_data_year, start_data_month, 1)
    end_date = datetime(end_data_year, end_data_month, 1)
    current_date = start_date
    filenames = []
    dates = []
    while current_date <= end_date:
        year_month = current_date.strftime('%Y%m')
        bil_file = os.path.join(base_dir, f"{file_prefix}_{year_month}_bil.bil")
        hdr_file = os.path.join(base_dir, f"{file_prefix}_{year_month}_bil.hdr")
        filenames.append((bil_file, hdr_file))
        dates.append(current_date)
        current_date += timedelta(days=32)
        current_date = datetime(current_date.year, current_date.month, 1)
    return filenames, dates

# Function to generate a list of months
def generate_month_list(start_data_year, start_data_month, end_data_year, end_data_month):
    start_date = datetime(start_data_year, start_data_month, 1)
    end_date = datetime(end_data_year, end_data_month, 1)
    months = []
    while start_date <= end_date:
        months.append(start_date)
        start_date += timedelta(days=32)
        start_date = datetime(start_date.year, start_date.month, 1)
    return months

########################################
# Load data
########################################

# Generate file paths and dates
file_paths, dates = generate_monthly_filenames_and_dates(start_data_year, start_data_month, end_data_year, end_data_month, base_dir, file_prefix)

# Loop through files and stack data
stacked_data = []
for bil_path, hdr_path in file_paths:
    data, lats, lons = read_bil_file(bil_path, hdr_path, lat_range, lon_range)
    stacked_data.append(data)

# Convert to 3D numpy array
stacked_data_array = np.array(stacked_data)

########################################
# Adjusted: Filter data for the observation period
########################################

# Filter the file paths and dates for the observation window
filtered_file_paths = []
filtered_dates = []

for i, date in enumerate(dates):
    if start_obs_date <= date <= end_obs_date:
        filtered_file_paths.append(file_paths[i])
        filtered_dates.append(date)

# Loop through filtered files and stack data
filtered_stacked_data = []
for bil_path, hdr_path in filtered_file_paths:
    data, lats, lons = read_bil_file(bil_path, hdr_path, lat_range, lon_range)
    filtered_stacked_data.append(data)

# Convert to 3D numpy array for the observation window
filtered_stacked_data_array = np.array(filtered_stacked_data)

########################################
# Calculate mean monthly precipitation for observation window
########################################

# Create an array to store monthly precipitation values
monthly_precip = {month: [] for month in range(1, 13)}

# Loop through each date and stack data by month
for i, date in enumerate(filtered_dates):
    month = date.month
    monthly_precip[month].append(filtered_stacked_data_array[i])

# Calculate the mean precipitation for each month across the observation period
mean_monthly_precip = {}
for month in range(1, 13):
    # Stack the data for the current month
    monthly_data = np.stack(monthly_precip[month], axis=0)
    
    # Calculate the mean across the observation period
    mean_monthly_precip[month] = np.nanmean(monthly_data, axis=0)

# Find the indices of the target latitude and longitude
lat_idx = np.argmin(np.abs(lats - target_lat))
lon_idx = np.argmin(np.abs(lons - target_lon))

# Extract mean monthly precipitation for the target location
mean_monthly_precip_target = {}
for month in range(1, 13):
    mean_monthly_precip_target[month] = mean_monthly_precip[month][lat_idx, lon_idx]

# Convert the extracted data into a list for easier analysis or visualization
mean_monthly_precip_values = [mean_monthly_precip_target[month] for month in range(1, 13)]

# Print the mean monthly precipitation values
print("Mean Monthly Precipitation at Target Location (Observation Period):")
for month, value in enumerate(mean_monthly_precip_values, start=1):
    print(f"Month {month:02d}: {value:.2f} mm")

# Calculate decimal year for the mid-point of each month
decimal_years = [start_obs_year + (month - 0.5) / 12 for month in range(1, 13)]

# Create a DataFrame for the mean monthly precipitation values
df = pd.DataFrame({
    "Decimal Year": decimal_years,
    "Mean Monthly Precipitation (mm)": mean_monthly_precip_values
})

# Save the DataFrame to a CSV file
output_path = "/Users/daniellelindsay/Google Drive/My Drive/Proj_1_MTJ_creeprates/data/MWIL_mean_monthly_PRISM_obs_period.csv"
df.to_csv(output_path, index=False)

# ------------------------------
# 2) Full precipitation time series (new output)
# ------------------------------
# Extract precipitation for target lat/lon for every date
precip_timeseries = [
    filtered_stacked_data_array[i][lat_idx, lon_idx]
    for i in range(len(filtered_dates))
]

df_timeseries = pd.DataFrame({
    "Date": filtered_dates,
    "Precipitation (mm)": precip_timeseries
})

output_path_timeseries = "/Users/daniellelindsay/Google Drive/My Drive/Proj_1_MTJ_creeprates/data/MWIL_precipitation_timeseries_PRISM_obs_period.csv"
df_timeseries.to_csv(common_paths["PRISM"]["monthly"], index=False)
print(f"Saved full precipitation timeseries CSV: {output_path_timeseries}")


# Plot the mean monthly precipitation values
months = range(1, 13)  # Months from January to December

plt.figure(figsize=(10, 6))
plt.bar(months, mean_monthly_precip_values, color="skyblue", edgecolor="black")
plt.xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=12)
plt.ylabel("Mean Monthly Precipitation (mm)", fontsize=14)
plt.xlabel("Month", fontsize=14)
plt.title("Mean Monthly Precipitation at Target Location", fontsize=16)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ########################################
# # Calculate annual precipation for each water year and the average
# ########################################

# # Initialize a list to store the average precipitation data for each water year
# water_year_total = []
# water_year_periods = []

# # Loop through each date
# for i, date in enumerate(dates):
#     # Check if the month is September and if we're not at the end of the dates list
#     if date.month == start_WY and i + 12 <= len(dates):
#         start_idx = i
#         end_idx = i + 12

#         # Extract the relevant precipitation data for the water year
#         water_year_data = stacked_data_array[start_idx:end_idx, :, :]

#         # Calculate the total precipitation for the water year
#         total_precip = np.nansum(water_year_data, axis=0)
        
#         # Append the total precipitation data to the list
#         water_year_total.append(total_precip)
        
#         # Append the corresponding years to the water_year_periods list
#         water_year_periods.append((date.year, date.year + 1))
#         print(date.year, date.year + 1)

# # Convert the list of 2D arrays into a 3D numpy array
# water_year_total_array = np.array(water_year_total)

# # Calculate the long-term average annual precipitation
# long_term_total_precip = np.nanmean(water_year_total_array, axis=0)

# long_term_total_precip[long_term_total_precip == 0] = np.nan