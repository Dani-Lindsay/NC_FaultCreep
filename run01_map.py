#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 05:11:44 2025

@author: daniellelindsay
"""

import pygmt
import pandas as pd

from NC_creep_filepaths import common_paths


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



#creep_file = '/Users/daniellelindsay/Google Drive/My Drive/Proj_1_MTJ_creeprates/data/nshm2023_wus_creep.csv'
#creep = pd.read_csv(creep_file, sep=",", header=0)

creep_CM = pd.read_csv(common_paths["CM_locations"], sep=",", header=0)
creep_AA = pd.read_csv(common_paths["creep_rates"]["AA"], sep=",", header=0)

# Inset map boundaries
f1_center_lat =  39.41235
f1_center_lon = -123.35699
f1_min_lon = f1_center_lon - 0.26
f1_max_lon = f1_center_lon + 0.26
f1_min_lat = f1_center_lat - 0.33
f1_max_lat = f1_center_lat + 0.33
f1_sub_region = f"{f1_min_lon}/{f1_max_lon}/{f1_min_lat}/{f1_max_lat}"
f1_sub_size = "M6c"
f1_ref_lat = 39.389548
f1_ref_lon = -123.347155

# Inset map boundaries
f2_center_lat =  38.819349
f2_center_lon = -122.92
f2_min_lon = f2_center_lon - 0.28
f2_max_lon = f2_center_lon + 0.28
f2_min_lat = f2_center_lat - 0.33
f2_max_lat = f2_center_lat + 0.33
f2_sub_region = f"{f2_min_lon}/{f2_max_lon}/{f2_min_lat}/{f2_max_lat}"
f2_sub_size = "M6c"
f2_ref_lat = 38.809587
f2_ref_lon = -123.012755



f3_center_lat = 38.411953
f3_center_lon = -122.674709
f3_min_lon = f3_center_lon - 0.28
f3_max_lon = f3_center_lon + 0.28
f3_min_lat = f3_center_lat - 0.31
f3_max_lat = f3_center_lat + 0.31
f3_sub_region = f"{f3_min_lon}/{f3_max_lon}/{f3_min_lat}/{f3_max_lat}"
f3_sub_size = "M6c"
f3_ref_lat = 38.44183
f3_ref_lon = -122.74641

# Define region of interest 
min_lon=-123.9
max_lon=-121.8
min_lat=37.8
max_lat=40.0

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M9c"

ref_la = 38.39868
ref_lo = -122.72418

grid = pygmt.datasets.load_earth_relief(region=region, resolution="03s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)

### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=9, FONT_TITLE=9, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",)

# Plot DEM

fig.shift_origin(xshift="6.2c", yshift="5.9c")

fig.basemap(projection=fig_size, frame=["lbrt", "xa", "ya"], region=[region])
fig.coast(water="white", land="white", shorelines=True,lakes=False, borders="2/thin")
fig.grdimage(grid=grid, projection=fig_size, frame=["WSrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=region, transparency=50)
fig.coast(shorelines=True,lakes=False, borders="2/thin")

# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.8p,black", transparency=50)
    
fig.plot(data=common_paths['pb_file'] , pen="1.2p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")
fig.plot(data=common_paths['pb2_file'] , pen="1.2p,red3", style="f0.5c/0.15c+r+t", fill="red3")
fig.plot(data=common_paths['pb3_file'] , pen="1.2p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")

gey_lat, gey_lon = 38.84, -122.83 #Geysers
fig.plot(x=gey_lon, y=gey_lat, style="kvolcano/0.4c", pen="1p,black", fill="darkred")
fig.text(text="The Geysers", x=gey_lon, y=gey_lat, justify="ML", offset="0.2c/0c", font="9p,gray15", fill="white", transparency=60)
fig.text(text="The Geysers", x=gey_lon, y=gey_lat, justify="ML", offset="0.2c/0c", font="9p,gray15")

# sar_lat, sar_lon = 38.4404, -122.7141 # Santa Rosa
# wil_lat, wil_lon = 39.4096, -123.3556 # Wilits
# uki_lat, uki_lon = 39.1502, -123.2078 # Ukiah


# fig.plot(x=sar_lon, y=sar_lat, style="kcircle/0.15c", pen="1p,black", fill="dimgray")
# fig.plot(x=wil_lon, y=wil_lat, style="kcircle/0.15c", pen="1p,black", fill="dimgray")
# fig.plot(x=uki_lon, y=uki_lat, style="kcircle/0.15c", pen="1p,black", fill="dimgray")

# Draw bounding box for inset region on main map
fig.plot(x=[f1_min_lon, f1_min_lon, f1_max_lon, f1_max_lon, f1_min_lon], 
          y=[f1_min_lat, f1_max_lat, f1_max_lat, f1_min_lat, f1_min_lat], 
          pen="1p,black,--", transparency=40)

# Draw bounding box for inset region on main map
fig.plot(x=[f2_min_lon, f2_min_lon, f2_max_lon, f2_max_lon, f2_min_lon], 
          y=[f2_min_lat, f2_max_lat, f2_max_lat, f2_min_lat, f2_min_lat], 
          pen="1p,black,--", transparency=40)

# Draw bounding box for inset region on main map
fig.plot(x=[f3_min_lon, f3_min_lon, f3_max_lon, f3_max_lon, f3_min_lon], 
          y=[f3_min_lat, f3_max_lat, f3_max_lat, f3_min_lat, f3_min_lat], 
          pen="1p,black,--", transparency=40)



fig.text(x=(f1_min_lon+f1_max_lon)/2, y=f1_max_lat, text="North Maacama", font="10,Helvetica", justify="BC", offset="0.0/0.1c", fill="white", transparency=60)
fig.text(x=(f1_min_lon+f1_max_lon)/2, y=f1_max_lat, text="North Maacama", font="10,Helvetica", justify="BC", offset="0.0/0.1c", )

fig.text(x=f2_min_lon, y=f2_max_lat, text="South Maacama", font="10,Helvetica", justify="BL", offset="0.5/0.1c", fill="white", transparency=60)
fig.text(x=f2_min_lon, y=f2_max_lat, text="South Maacama", font="10,Helvetica", justify="BL", offset="0.5/0.1c")

fig.text(x=(f3_min_lon+f3_max_lon)/2, y=f3_min_lat, text="Rodgers Creek", font="10,Helvetica", justify="TC", offset="0.0/-0.1c", fill="white", transparency=60)
fig.text(x=(f3_min_lon+f3_max_lon)/2, y=f3_min_lat, text="Rodgers Creek", font="10,Helvetica", justify="TC", offset="0.0/-0.1c")

# fig.text(text="Santa Rosa",     x=sar_lon, y=sar_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15", fill="white", transparency=60 )
# fig.text(text="Wilits",         x=wil_lon, y=wil_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15", fill="white", transparency=60  )
# fig.text(text="Ukiah",     x=uki_lon, y=uki_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15",  fill="white", transparency=60  )

# fig.text(text="Santa Rosa",     x=sar_lon, y=sar_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15" )
# fig.text(text="Wilits",         x=wil_lon, y=wil_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15" )
# fig.text(text="Ukiah",     x=uki_lon, y=uki_lat, justify="MR", offset="-0.2c/0c", font="9p,gray15" )


for name, coords in places.items():
    lat = coords["lat"]
    lon = coords["lon"]
    fig.plot(x=lon, y=lat, style="s0.15c", fill="black")
    #fig.text(text=name, x=lon, y=lat, justify="MR", offset="-0.2c/0c", font="9p,gray15", fill="white", transparency=60)
    fig.text(text=name, x=lon, y=lat, justify="MR", offset="-0.2c/0c", font="9p,gray15")

fig.plot(x=creep_AA.lon, y=creep_AA.lat, style="t.2c", fill="dodgerblue", pen="0.2p", label='Alignment Array')
fig.plot(x=creep_CM.lon, y=creep_CM.lat, style="c.18c", fill="purple", pen="0.2p", label='Creep Meter')


fig.plot(x=-123.4, y=38.5, style="e308/1.25/0.5", fill="white", transparency=60)
fig.text(text="SAF", x=-123.4, y=38.5, justify="CM", font="10p,red3" , angle=308)


fig.plot(x=-123.27, y=39.3, style="e0/0.8.0/0.4", fill="white", transparency=30, pen="1p,black")
fig.text(text="MA", x=-123.27, y=39.3, justify="CM", font="9p,black" , angle=0)

fig.plot(x=-122.57, y=38.3, style="e0/0.8.0/0.4", fill="white", transparency=30,pen="1p,black")
fig.text(text="RC", x=-122.57, y=38.3, justify="CM", font="9p,black" , angle=0)

fig.plot(x=-122.85, y=39.3, style="e0/0.8.0/0.4", fill="white", transparency=30,pen="1p,black")
fig.text(text="BS", x=-122.85, y=39.3, justify="CM", font="9p,black" , angle=0)

fig.plot(x=-122.2, y=38.4, style="e0/0.8.0/0.4", fill="white", transparency=30,pen="1p,black")
fig.text(text="GV", x=-122.2, y=38.4, justify="CM", font="9p,black" , angle=0)

fig.plot(x=-122.25, y=37.9, style="e0/0.8.0/0.4", fill="white", transparency=30,pen="1p,black")
fig.text(text="HA", x=-122.25, y=37.9, justify="CM", font="9p,black" , angle=0)


# Example usage:
for place, coords in places.items():
    print(f"{place}: Latitude {coords['lat']}, Longitude {coords['lon']}")
    fig.text(text=place,     x=coords['lat'], y=coords['lon'], justify="MR", offset="-0.2c/0c", font="9p,gray15" )

fig.legend(box="+p1p+gwhite+c5p", position="jBL+o0.2c/0.2c", projection=fig_size)  
fig.basemap( frame=["WSrt", "xa", "ya"], map_scale="jBL+w30k+o0.2c/1.75c", projection=fig_size)



fig.shift_origin(xshift="7.2c", yshift="6.5c")



min_lon=-126.7
max_lon=-119.5
min_lat=35.5
max_lat=42.5



region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M5.0c"

grid = pygmt.datasets.load_earth_relief(region=region, resolution="15s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)

# # Inset map of frames

fig.basemap(projection=fig_size, frame=["lbrt"], region=[region])
fig.coast(water="white", land="white", shorelines=True,lakes=False, borders="2/thin",  frame=["lbrt"],)
# Plot DEM
fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt"], cmap='wiki-france.cpt', shading=dgrid, region=region, transparency=50)
fig.coast(shorelines=True,lakes=False, borders="2/thin",  frame=["lbrt"],)

# Plot Faults
for fault_file in common_paths["fault_files"]:
    fig.plot(data=fault_file, pen="0.5p,black", transparency=50)
    
fig.plot(data=common_paths['pb_file'] , pen="1p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")
fig.plot(data=common_paths['pb2_file'] , pen="1p,red3", style="f0.5c/0.15c+r+t", fill="red3")
fig.plot(data=common_paths['pb3_file'] , pen="1p,red3", style="f-1c/0.5c+r+s+p1.5p,red3,solid")

# Label Faults
fig.plot(x=-124.0, y=41.7, style="e280/1.25/0.5", fill="white", transparency=50)
fig.text(text="CSZ", x=-124.0, y=41.7, justify="CM", font="10p,red3" , angle=280)

fig.plot(x=-125.0, y=39.7, style="e0/1.25/0.5", fill="white", transparency=50)
fig.text(text="MFZ", x=-125.0, y=39.7, justify="CM", font="10p,red3" )

fig.plot(x=-123.25, y=37.2, style="e310/1.25/0.5", fill="white", transparency=50)
fig.text(text="SAF", x=-123.25, y=37.2, justify="CM", font="10p,red3" , angle=310)

df = pd.DataFrame(
    data={
        "x": [-125.8, -125.0],
        "y": [41.0, 38.5],
        "east_velocity": [17.20, -28.59],
        "north_velocity": [23.98, 43.59],
        "east_sigma": [0, 0],
        "north_sigma": [0, 0],
        "correlation_EN": [0, 0],
        "SITE": ["", ""],
        })

fig.velo(data=df, pen="1p,black", uncertaintyfill="lightblue1", line=True, spec="e0.030/0.39/18", vector="0.3c+p1p+e+gblack",)
fig.text(text=["JF(NA)","PA(NA)"], x=df.x, y=df.y, justify="TC",offset ="0/-0.2c",  font="8p,black")

fig.text(text=["SNGV"], x=-120.8, y=38.5, justify="TC",offset ="0/-0.2c",  font="8p,black")


# Subplot region subsers
sub_min_lon=-123.8
sub_max_lon=-121.8
sub_min_lat=37.8
sub_max_lat=40.0
fig.plot(x = [sub_min_lon, sub_min_lon, sub_max_lon, sub_max_lon, sub_min_lon], 
         y = [sub_min_lat, sub_max_lat, sub_max_lat, sub_min_lat, sub_min_lat], 
         pen="0.8p,black,--", transparency=0)


fig.plot(x=creep_CM.lon, y=creep_CM.lat, style="c.12c", fill="purple", pen="0.2p", label='Creep Meter')
fig.plot(x=creep_AA.lon, y=creep_AA.lat, style="t.15c", fill="dodgerblue", pen="0.2p", label='Alignment Array')




fig.basemap(  frame=["lbrt"], map_scale="jBL+w100k+o0.5/0.5c", projection=fig_size)

#fname = fig_dir+"Igarss_intro_map.png"
fig.savefig(f'{common_paths["fig_dir"]}/Fig_1_intro_map.png', transparent=True, crop=True, anti_alias=True, show=False)

fig.show()

