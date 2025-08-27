#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 10:17:52 2025

@author: daniellelindsay
"""
import numpy as np
import pandas as pd
from NC_creep_filepaths import common_paths

import pandas as pd
import numpy as np
import pygmt
import os
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import distance
from datetime import datetime

def get_start_end_points(lon_ori, lat_ori, az, dist):
    start_lat, start_lon, start_z = distance(kilometers=-dist).destination((lat_ori, lon_ori), bearing=az)
    end_lat, end_lon, end_z = distance(kilometers=dist).destination((lat_ori, lon_ori), bearing=az)
    return start_lon, start_lat, end_lon, end_lat

# Load polygons from GMT files
def load_polygon_gmt(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    coords = [list(map(float, line.strip().split())) for line in lines if not line.startswith('>')]
    return Polygon(coords)

def calculate_seismo_depth(df, step=1, window=20, min_points=300):
    """
    Calculates the D90 depth profile and average depth of the deepest 10% using sliding windows.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'p' (profile distance in km) and 'z' (depth in km).
        step (float): Step size in km for the sliding window (default is 1 km).
        window (float): Window size in km over which to calculate D90 and deepest 10% (default is 20 km).
        min_points (int): Minimum number of points required in the window.

    Returns:
        pd.DataFrame: DataFrame with columns 'profile_km', 'D90_depth_km', and 'mean_deepest_10pct_km'.
    """
    df_sorted = df.sort_values(by='p').reset_index(drop=True)
    p_min, p_max = df_sorted['p'].min(), df_sorted['p'].max()
    p_centers = np.arange(p_min + window / 2, p_max - window / 2 + 1, step)

    result = []
    for center in p_centers:
        p_start = center - window / 2
        p_end = center + window / 2
        subset = df_sorted[(df_sorted['p'] >= p_start) & (df_sorted['p'] < p_end)]

        if len(subset) >= min_points:
            depths = subset['z'].values
            d90 = np.percentile(depths, 90)
            d95 = np.percentile(depths, 95)

            # Deepest 10%
            n_deep = max(1, int(0.10 * len(depths)))
            deepest_values = np.sort(depths)[-n_deep:]
            deepest_mean = np.mean(deepest_values)
            deepest_std = np.std(deepest_values, ddof=1)  # sample std deviation

            result.append({
                'profile_km': center,
                'D90_depth_km': d90,
                'D95_depth_km': d95,
                'seismo_depth': deepest_mean,
                'std_seismo_depth': deepest_std
            })

    return pd.DataFrame(result)

def date_to_decimal_year(date_input, format_str=None):
    """
    Convert a date to a decimal year.

    :param date_input: Date input which can be a string, byte string, or datetime object.
    :param format_str: Format string for parsing the date. If None, the function will attempt to infer the format.
                       Acceptable formats include: "%Y-%m-%d" (e.g., "2021-12-31"), 
                       "%m/%d/%Y" (e.g., "12/31/2021"), and "%Y%m%d" (e.g., "20211231").
    :return: Decimal year corresponding to the input date.
    """

    # If the input is a byte string, decode it to a regular string
    if isinstance(date_input, bytes):
        date_input = date_input.decode('utf-8')

    # If the input is already a datetime object, use it directly
    if isinstance(date_input, datetime):
        date = date_input
    else:
        # Attempt to infer the format if not provided
        if format_str is None:
            if "-" in date_input:
                format_str = "%Y-%m-%d"
            elif "/" in date_input:
                format_str = "%m/%d/%Y"
            elif len(date_input) == 8:
                format_str = "%Y%m%d"
            else:
                raise ValueError("Unknown date format. Please provide a format string.")

        # Parse the date string using the provided format
        date = datetime.strptime(date_input, format_str)

    start_of_year = datetime(year=date.year, month=1, day=1)
    start_of_next_year = datetime(year=date.year+1, month=1, day=1)
    year_length = (start_of_next_year - start_of_year).total_seconds()
    year_progress = (date - start_of_year).total_seconds()
    decimal_year = date.year + year_progress / year_length
    return decimal_year

# Function to convert date and time to decimal years
def date_to_decimal_year_seno(date_str, time_str):
    try:
        date_parts = pd.to_datetime(date_str).to_pydatetime().timetuple()
        year = date_parts.tm_year
        day_of_year = date_parts.tm_yday
        hour, minute, second = map(float, time_str.split(':')) if pd.notna(time_str) else (0.0, 0.0, 0.0)
        total_seconds = hour * 3600 + minute * 60 + second
        fraction_of_day = total_seconds / 86400
        return year + (day_of_year - 1 + fraction_of_day) / 365.25
    except Exception as e:
        print(f"Error processing date: {date_str} and time: {time_str} -> {e}")
        return np.nan

# Function to calculate the decimal year from date and time components
def to_decimal_year_wald(row):
    date_str = f"{int(row['YR'])}-{int(row['MO']):02d}-{int(row['DY']):02d} {int(row['HR']):02d}:{int(row['MN']):02d}:{int(row['SC']):02d}"
    date = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
    start_of_year = pd.to_datetime(f"{int(row['YR'])}-01-01")
    end_of_year = pd.to_datetime(f"{int(row['YR'])+1}-01-01")
    days_passed = (date - start_of_year).total_seconds() / (24 * 3600)
    total_days_in_year = (end_of_year - start_of_year).days
    decimal_year = row['YR'] + days_passed / total_days_in_year
    return round(decimal_year, 4)

def get_CRE_slip(mag):
    # Calculate slip using the given formula
    #slip = 10 ** (0.255 * (mag - 0.15) + 0.377)
    slip = 10 ** (0.255*mag + 0.377)
    return slip

# Function to sort by year and calculate cumulative slip
def calculate_cummulative_slip(df):
    # Ensure 'YYYY' is in numeric format for sorting
    df['YYYY'] = pd.to_numeric(df['YYYY'], errors='coerce')
    
    # Sort by year
    df_sorted = df.sort_values(by='YYYY')
    
    # Calculate cumulative slip
    df_sorted['Cumulative_Slip'] = df_sorted['slip'].cumsum()
    
    # Calculate number of unquie seqID's in the df
    unique_seq_ids = df_sorted["seqID"].nunique()
    
    # Divide cummulative slip by number of families
    df_sorted['CumSlip/numSeqID'] = df_sorted['Cumulative_Slip']/unique_seq_ids
    
    return df_sorted

def date_to_decimal_year_eq(datetime_str):
    try:
        # Parse the datetime string
        dt = pd.to_datetime(datetime_str, utc=True)
        year = dt.year
        day_of_year = dt.day_of_year
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        total_seconds = hour * 3600 + minute * 60 + second
        fraction_of_day = total_seconds / 86400
        return year + (day_of_year - 1 + fraction_of_day) / 365.25
    except Exception as e:
        print(f"Error processing datetime: {datetime_str} -> {e}")
        return np.nan
