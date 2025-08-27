#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:46:51 2024

Bob’s empirical equation
 - log10(d) = -2.36 + 0.17(M0)
d is fault slip inferred from repeating EQs
 - M0 is seismic moment in dyne-cm of a repeating EQ

Then we need to convert M (magnitude) to M0.
 - Two equations:
  1. Hanks and Kanamori (1979, JGR)
       - log10(M0) = 1.5xM + 16.1 (original is 16.05 though)
   2. Wyss et al. (2004, BSSA)
        - log(M0) = 1.6xM + 15.8
          - This is computed for Parkfield area with HRSN data

I used #2 Wyss’s equation sometime but now I use #1 Hanks and Kanamori equation for all target areas. FYI, Turner et al., 2013 use #2 Wyss's equation. 

Use log(M0) from M into Bob's equation.
 - log10(d) = -2.36 + 0.17(1.5xM +16.1)
 - If we need d then:
 - d = 10^(-2.36 + 0.17(1.5xM +16.1)
      =  10^(0.255xM + 0.377)

This equation is similar to what Dennis's paper shows but I think for some reason (I cannot find it from her paper) that she adjusts the catalog magnitude (M) by subtracting 0.15, which is:
   d = 10^(0.255x(M-0.15) + 0.377)

There are still questions about what types of magnitude you use, Ml, Mw, or Md? I think that the original Hanks and Kanamori paper is using Mw so if there is a systematic difference between Ml and Mw, then one may need to add a correction factor. There are actually such "offset" between Mw and Ml or Md in our NCSS catalog but it is always hard to quantify this. This is a part of the reason why Wyss's equation was developed for the Parkfield region. Given this, I would feel that "0.15" in Dennis's paper may be for this (to get Mw) but I did not find it when I quickly read this paper this morning. 

My take is that let's apply the Hanks and Kanamori equation, and yes there may be some inference or bias to get fault slip (d) but at least consistently applying the same equation would have some benefit to see temporal changes in fault slip. if I use different equations for each repeating EQs, then they would introduce an artifact. I know this is not a perfect argument (there would be many drawbacks) but this is what I have been doing.

@author: daniellelindsay
"""
import pandas as pd
import numpy as np
import pygmt
import math

from NC_creep_filepaths import common_paths
import creep_utils as utils
from shapely.geometry import Point, Polygon
import geopandas as gpd

step = 2002.33


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

##############################
# Load Seno 
##############################

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

##############################
# Load Waldhauser Cat
##############################

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
RC_CRE_polygon = utils.load_polygon_gmt(common_paths["RC_CRE_poly"])
MA_CRE_polygon = utils.load_polygon_gmt(common_paths["MA_CRE_poly"])

# Convert CRE_taka_info_df to a GeoDataFrame
taka_gdf = gpd.GeoDataFrame(CRE_taka_df, geometry=gpd.points_from_xy(CRE_taka_df['Lon'], CRE_taka_df['Lat']), crs="EPSG:4326")
seno_gdf = gpd.GeoDataFrame(CRE_seno_df, geometry=gpd.points_from_xy(CRE_seno_df['Lon'], CRE_seno_df['Lat']), crs="EPSG:4326")
wald_gdf = gpd.GeoDataFrame(CRE_wald_df, geometry=gpd.points_from_xy(CRE_wald_df['Lon'], CRE_wald_df['Lat']), crs="EPSG:4326")

# --- RC via polygon only ---
filtered_taka = {"RC": taka_gdf[taka_gdf.geometry.within(RC_CRE_polygon)].copy()}
filtered_seno = {"RC": seno_gdf[seno_gdf.geometry.within(RC_CRE_polygon)].copy()}
filtered_wald = {"RC": wald_gdf[wald_gdf.geometry.within(RC_CRE_polygon)].copy()}

# --- MA via polygon, then split north/south at 39.1° lat ---
ma_taka = taka_gdf[taka_gdf.geometry.within(MA_CRE_polygon)].copy()
ma_seno = seno_gdf[seno_gdf.geometry.within(MA_CRE_polygon)].copy()
ma_wald = wald_gdf[wald_gdf.geometry.within(MA_CRE_polygon)].copy()

lat_split = 39.1
filtered_taka["NM"] = ma_taka[ma_taka["Lat"] >= lat_split].copy()
filtered_taka["SM"] = ma_taka[ma_taka["Lat"] <  lat_split].copy()

filtered_seno["NM"] = ma_seno[ma_seno["Lat"] >= lat_split].copy()
filtered_seno["SM"] = ma_seno[ma_seno["Lat"] <  lat_split].copy()

filtered_wald["NM"] = ma_wald[ma_wald["Lat"] >= lat_split].copy()
filtered_wald["SM"] = ma_wald[ma_wald["Lat"] <  lat_split].copy()

# --- Apply cumulative slip calculation ---
processed_dfs_taka = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_taka.items()}
processed_dfs_seno = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_seno.items()}
processed_dfs_wald = {r: utils.calculate_cummulative_slip(df) for r, df in filtered_wald.items()}


results = {}

for cat_name, datasets in [("Taka", processed_dfs_taka), 
                           ("Seno", processed_dfs_seno), 
                           ("Wald", processed_dfs_wald)]:
    for seg, df in datasets.items():
        pre_mask  = (df["YYYY"] >= step - 10) & (df["YYYY"] < step)
        post_mask = (df["YYYY"] >= step) & (df["YYYY"] < step + 10)
        
        if pre_mask.sum() > 2 and post_mask.sum() > 2:
            pre_slope, pre_intercept   = np.polyfit(df.loc[pre_mask,"YYYY"], df.loc[pre_mask,"CumSlip/numSeqID"], 1)
            post_slope, post_intercept = np.polyfit(df.loc[post_mask,"YYYY"], df.loc[post_mask,"CumSlip/numSeqID"], 1)
            
            pct_change = (post_slope - pre_slope) / pre_slope * 100
            
            results[(seg, cat_name)] = {
                "pre_slope": pre_slope,
                "post_slope": post_slope,
                "pct_change": pct_change
            }
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
            
##############################
# Plot results
##############################


# Create a new PyGMT figure
fig = pygmt.Figure()

# Set the basemap with your specific region and frame settings
fig.basemap(projection="X10/10c", region=[1980, 2024, -2, 50], frame=['xaf+l"Time (year)"', 'yaf+l"Cumulative Fault Slip (cm)"', "WSrt"])

fig.plot(x=processed_dfs_taka["RC"]['YYYY'], y=processed_dfs_taka["RC"]['CumSlip/numSeqID']+5, style="c.15c", fill="67.375/60.25/132", )#label="RC - Taka")
fig.plot(x=processed_dfs_seno["RC"]['YYYY'], y=processed_dfs_seno["RC"]['CumSlip/numSeqID'], style="t.15c", fill="67.375/60.25/132", )#label="RC - Seno")
fig.plot(x=processed_dfs_wald["RC"]['YYYY'], y=processed_dfs_wald["RC"]['CumSlip/numSeqID'], style="s.15c", fill="67.375/60.25/132", )#label="RC - Wald")

fig.plot(x=processed_dfs_taka["SM"]['YYYY'], y=processed_dfs_taka["SM"]['CumSlip/numSeqID']+14, style="c.15", fill="56.25/185.12/118.88", )#label="SM - Taka")
fig.plot(x=processed_dfs_seno["SM"]['YYYY'], y=processed_dfs_seno["SM"]['CumSlip/numSeqID']+14, style="t.15c", fill="56.25/185.12/118.88", )#label="SM - Seno")
fig.plot(x=processed_dfs_wald["SM"]['YYYY'], y=processed_dfs_wald["SM"]['CumSlip/numSeqID']+12, style="s.15c", fill="56.25/185.12/118.88", )#label="SM - Wald")

fig.plot(x=processed_dfs_taka["NM"]['YYYY'], y=processed_dfs_taka["NM"]['CumSlip/numSeqID']+30, style="c.1c", fill="170.38/220/49.75", )#label="NM - Taka")
fig.plot(x=processed_dfs_seno["NM"]['YYYY'], y=processed_dfs_seno["NM"]['CumSlip/numSeqID']+24, style="t.15c", fill="170.38/220/49.75", )#label="NM - Seno")
fig.plot(x=processed_dfs_wald["NM"]['YYYY'], y=processed_dfs_wald["NM"]['CumSlip/numSeqID']+24, style="s.15c", fill="170.38/220/49.75", )#label="NM - Wald")

fig.plot(x=processed_dfs_taka["RC"]['YYYY'], y=processed_dfs_taka["RC"]['CumSlip/numSeqID']+5, pen="1p,67.375/60.25/132", transparency=50)
fig.plot(x=processed_dfs_seno["RC"]['YYYY'], y=processed_dfs_seno["RC"]['CumSlip/numSeqID'], pen="1p,67.375/60.25/132", transparency=50)
fig.plot(x=processed_dfs_wald["RC"]['YYYY'], y=processed_dfs_wald["RC"]['CumSlip/numSeqID'], pen="1p,67.375/60.25/132", transparency=50)

fig.plot(x=processed_dfs_taka["SM"]['YYYY'], y=processed_dfs_taka["SM"]['CumSlip/numSeqID']+14, pen="1p,56.25/185.12/118.88", transparency=50)
fig.plot(x=processed_dfs_seno["SM"]['YYYY'], y=processed_dfs_seno["SM"]['CumSlip/numSeqID']+14, pen="1p,56.25/185.12/118.88",  transparency=50)
fig.plot(x=processed_dfs_wald["SM"]['YYYY'], y=processed_dfs_wald["SM"]['CumSlip/numSeqID']+12, pen="1p,56.25/185.12/118.88",  transparency=50)

fig.plot(x=processed_dfs_taka["NM"]['YYYY'], y=processed_dfs_taka["NM"]['CumSlip/numSeqID']+30, pen="1p,170.38/220/49.75", transparency=50)
fig.plot(x=processed_dfs_seno["NM"]['YYYY'], y=processed_dfs_seno["NM"]['CumSlip/numSeqID']+24, pen="1p,170.38/220/49.75", transparency=50)
fig.plot(x=processed_dfs_wald["NM"]['YYYY'], y=processed_dfs_wald["NM"]['CumSlip/numSeqID']+24, pen="1p,170.38/220/49.75",  transparency=50)

fig.plot(x=[step, step], y=[0, 45], pen="1p,black,--", transparency=50)

fig.text(x=1980, y=0, text="RC", justify="BL", offset="0.2/0.1c", font="10,Helvetica")
fig.text(x=1980, y=10, text="SM", justify="BL", offset="0.2/0.1c", font="10,Helvetica")
fig.text(x=1980, y=20, text="NM", justify="BL", offset="0.2/0.1c", font="10,Helvetica")

fig.plot(x=-124.0, y=38.0, style="t.15c", fill="white", pen="0.2p,black", label = "Shakibay Senobari & Funning (2019)")
fig.plot(x=-124.0, y=38.0, style="s.15c", fill="white", pen="0.2p,black", label = "Waldhauser & Schaff (2021)")
fig.plot(x=-124.0, y=38.0, style="c.15c", fill="white", pen="0.2p,black", label = "Taira (this study)")

# Add a legend to the plot
fig.legend(box="+gwhite", position="jTL+o0.2c/0.2c") 

# Save and show the figure
fig.savefig(f'{common_paths["fig_dir"]}/ALL_cumulative_slip_plot_offset_10**(0.255*mag+0.377).png', transparent=True, crop=True, anti_alias=True, show=False, dpi=400)
fig.show()


'''
NH Center 13.66km from PP, length 13.5km, 37.901908°, -122.275188°.
SH center, 51 km from pp,  37.621041, -122.028837
angle 145. 
NC, 85 from pp, 10km length,  37.367742, -121.803585
SC, 117 from pp, 22.5km length,  37.146002, -121.578431
'''