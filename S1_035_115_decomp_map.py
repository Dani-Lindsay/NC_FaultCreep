#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 23:10:46 2025

@author: daniellelindsay
"""

import numpy as np                
import pygmt
import pandas as pd
from NC_creep_filepaths import common_paths


fig_dir="/Volumes/WD2TB_Phd/NC_Fault_Creep/Figures/"
MWIL_up = "/Volumes/WD2TB_Phd/NC_Fault_Creep/MWIL_035_115/up_035_115.grd"
MWIL_hz = "/Volumes/WD2TB_Phd/NC_Fault_Creep/MWIL_035_115/N145_035_115.grd"

##################################
#### Plot results
##################################

# Define region of interest 
min_lon=-123.41504
max_lon=-123.29243
min_lat=39.35453
max_lat=39.45461
region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)

fig_size = "M7c"

MWIL_lon, MWIL_lat = -123.35612, 39.41242

fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=10, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",)


with fig.subplot(nrows=1, ncols=2, figsize=("18.0c", "7.35c"), sharey="l", frame=["WSrt"], margins=["0.3c", "0.3c"],):
    
    # Plot Faults

    
    grid = pygmt.datasets.load_earth_relief(region=region, resolution="01s")
    dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)
    
    fig.basemap(frame=["WSrt", "xa", "ya"], region=region, projection=fig_size, panel=True)
    #fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=region)
    pygmt.makecpt(cmap="magma", series=[-0.020, 0.002])
    fig.grdimage(grid=MWIL_up, cmap=True, region=region, projection=fig_size, nan_transparent=True)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.8p,black", transparency=50, region=region, projection=fig_size,)
    
    fig.plot(x=MWIL_lon, y=MWIL_lat, style="t.35c", fill="dodgerblue", pen="0.2p", region=region, projection=fig_size,)
    fig.text(text="MWIL", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="MWIL", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", region=region, projection=fig_size,)
    
    fig.text(text="Vertical", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="Vertical", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", region=region, projection=fig_size,)

    fig.text(text="e)", position="TL", justify="TL", offset="0.1c/-0.1c", font="18p,black", region=region, projection=fig_size,)
    
    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="magma", series=[-0.020*1000, 0.002*1000])
        fig.colorbar(position="JMR+o0.5c/0c+w5c/0.5c", frame=["xa", "y+lmm/yr"], projection=fig_size)
        
    fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jBL+w2k+o0.3/0.5c", projection=fig_size)
       
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region, projection=fig_size, panel=True)
    pygmt.makecpt(cmap="vik", series=[-0.012, 0.015])
    fig.grdimage(grid=MWIL_hz, cmap="vik", region=region, projection=fig_size)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.8p,black", transparency=50, region=region, projection=fig_size,)
    
    fig.plot(x=MWIL_lon, y=MWIL_lat, style="t.35c", fill="dodgerblue", pen="0.2p", region=region, projection=fig_size,)
    fig.text(text="MWIL", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="MWIL", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", region=region, projection=fig_size,)
    
    fig.text(text="Fault Parallel", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="Fault Parallel", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", region=region, projection=fig_size,)

    fig.text(text="f)", position="TL", justify="TL", offset="0.1c/-0.1c", font="18p,black", region=region, projection=fig_size,)
    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="vik", series=[-0.012*1000, 0.015*1000])
        fig.colorbar(position="JMR+o0.5c/0c+w5c/0.5c", frame=["xa", "y+lmm/yr"], projection=fig_size)
        
    #fig.plot(y=eel_lat, x=eel_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)

    fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jBL+w2k+o0.3/0.5c", projection=fig_size)
    
 
    
fig.savefig(fig_dir+"MWIL_035_115_decomp_map.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(fig_dir+"MWIL_035_115_decomp_map.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(fig_dir+"MWIL_035_115_decomp_map.jpg", transparent=False, crop=True, anti_alias=True, show=False)

fig.show()


ROSA_up = "/Volumes/WD2TB_Phd/NC_Fault_Creep/RCF_HzUp/up_035_115.grd"
ROSA_hz = "/Volumes/WD2TB_Phd/NC_Fault_Creep/RCF_HzUp/N145_035_115.grd"

##################################
#### Plot results
##################################

# Define region of interest 
min_lon=-122.73807-0.17
max_lon=-122.73807+0.17
min_lat=38.50169-0.141
max_lat=38.50169+0.141
region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)

fig_size = "M7c"

MWIL_lon, MWIL_lat = -122.73807,	38.50169

fig = pygmt.Figure()
pygmt.config(FONT=10, FONT_TITLE=10, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain",)


with fig.subplot(nrows=1, ncols=2, figsize=("18.0c", "7.35c"), sharey="l", frame=["WSrt"], margins=["0.3c", "0.3c"],):
    
    # Plot Faults

    
    grid = pygmt.datasets.load_earth_relief(region=region, resolution="01s")
    dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=region)
    
    fig.basemap(frame=["WSrt", "xa", "ya"], region=region, projection=fig_size, panel=True)
    #fig.grdimage(grid=grid, projection=fig_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=region)
    pygmt.makecpt(cmap="magma", series=[-0.022, 0.002])
    fig.grdimage(grid=ROSA_up, cmap=True, region=region, projection=fig_size, nan_transparent=True)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.8p,black", transparency=50, region=region, projection=fig_size,)
    
    fig.plot(x=MWIL_lon, y=MWIL_lat, style="t.35c", fill="dodgerblue", pen="0.2p", region=region, projection=fig_size,)
    fig.text(text="RCMW", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="RCMW", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", region=region, projection=fig_size,)
    
    fig.text(text="Vertical", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="Vertical", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", region=region, projection=fig_size,)
    
    fig.text(text="g)", position="TL", justify="TL", offset="0.1c/-0.1c", font="18p,black", region=region, projection=fig_size,)

    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="magma", series=[-0.022*1000, 0.002*1000])
        fig.colorbar(position="JMR+o0.5c/0c+w5c/0.5c", frame=["xa", "y+lmm/yr"], projection=fig_size)
        
    fig.basemap(frame=["WSrt", "xa", "ya"], map_scale="jBL+w2k+o0.3/0.5c", projection=fig_size)
       
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region, projection=fig_size, panel=True)
    pygmt.makecpt(cmap="vik", series=[-0.01, 0.008])
    fig.grdimage(grid=ROSA_hz, cmap=True, region=region, projection=fig_size)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.8p,black", transparency=50, region=region, projection=fig_size,)
    
    fig.plot(x=MWIL_lon, y=MWIL_lat, style="t.35c", fill="dodgerblue", pen="0.2p", region=region, projection=fig_size,)
    fig.text(text="RCMW", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="RCMW", x=MWIL_lon, y=MWIL_lat, justify="MR", offset="-0.2c/0.2c", font="10p,black,bold", region=region, projection=fig_size,)
    
    fig.text(text="Fault Parallel", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", fill="white", transparency=40, region=region, projection=fig_size,)
    fig.text(text="Fault Parallel", position="TC", justify="TC", offset="-0.0c/-0.2c", font="13p,black,bold", region=region, projection=fig_size,)

    fig.text(text="h)", position="TL", justify="TL", offset="0.1c/-0.1c", font="18p,black", region=region, projection=fig_size,)
    
    with pygmt.config(
             FONT_ANNOT_PRIMARY="18p,black", 
             FONT_ANNOT_SECONDARY="18p,black",
             FONT_LABEL="18p,black",
             ):
        pygmt.makecpt(cmap="vik", series=[-0.01*1000, 0.008*1000])
        fig.colorbar(position="JMR+o0.5c/0c+w5c/0.5c", frame=["xa", "y+lmm/yr"], projection=fig_size)
        
    #fig.plot(y=eel_lat, x=eel_lon, style="s.15c", fill="black", pen="0.8p,black", region=region, projection=fig_size)

    fig.basemap(frame=["wSrt", "xa", "ya"], map_scale="jBL+w2k+o0.3/0.5c", projection=fig_size)
    
 
    
fig.savefig(fig_dir+"RCMW_035_115_decomp_map.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(fig_dir+"RCMW_035_115_decomp_map.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(fig_dir+"RCMW_035_115_decomp_map.jpg", transparent=False, crop=True, anti_alias=True, show=False)

fig.show()

