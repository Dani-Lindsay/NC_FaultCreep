#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 05:00:37 2025

@author: daniellelindsay
"""

import glob
import os

proj_dir = "/Volumes/WD2TB_Phd/NC_Fault_Creep"
inputs_dir = "/Volumes/WD2TB_Phd/NC_Fault_Creep/Input_data"

params = "x_width0.5_dist_min0.5_dist_max1.5_minpoints20"

# ------------------------
# Common Paths
# ------------------------
common_paths = {
    "params": "x_width0.5_dist_min0.5_dist_max1.5_minpoints20",
    "fig_dir": "/Volumes/WD2TB_Phd/NC_Fault_Creep/Figures",
    "fault_files": glob.glob(os.path.join(inputs_dir, "qfaults", "*.txt")),
    "pb_file": os.path.join(inputs_dir, "transform.gmt"),
    "pb2_file": os.path.join(inputs_dir, "trench.gmt"),
    "pb3_file": os.path.join(inputs_dir, "ridge.gmt"),
    "MA_trace": os.path.join(inputs_dir, "MA_fault_trace.gmt"),
    "RC_trace": os.path.join(inputs_dir, "RC_fault_trace.gmt"),
    "MA_CRE_poly": os.path.join(inputs_dir, "MA_repeater_polygon.gmt"),
    "RC_CRE_poly": os.path.join(inputs_dir, "RC_repeater_polygon.gmt"),
    "gps": {
        "cGPS": os.path.join(inputs_dir, "GPS_Murray14", "Murrary2014_cGPS.csv"),
        "sGPS": os.path.join(inputs_dir, "GPS_Murray14", "Murrary2014_sGPS.csv"),
    },
    "CM_locations": os.path.join(inputs_dir, "creepmater_locations.csv"),
    "creep_rates": {
        "all": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep.csv"),
        "AA": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_AA.csv"),
        "CM": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_CM.csv"),
        "GPS": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_GPS.csv"),
        "Geod": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_Geod.csv"),
        "LiDAR": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_LiDAR.csv"),
        "SAR": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_creep_SAR.csv"),
        "gps_all": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_gps.csv"),
        "gps_ITRF14": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_gps_ITRF14.csv"),
        "gps_los": os.path.join(inputs_dir, "CreepRates_Johnson22", "nshm2023_wus_gps_los.csv"),
    },
    "AA": {
        "table": os.path.join(inputs_dir, "AA_2022_SF_Bay_Region_N", "Table 1b.csv"),
        "dir":  os.path.join(inputs_dir, "AA_2022_SF_Bay_Region_N"),
    },
    "CRE_Taka": {
        "MA": os.path.join(inputs_dir, "CRE_Taira24", "MTJMC.freq8-24Hz_maxdist3_coh9500_linkage_cluster.txt"),
        "RC": os.path.join(inputs_dir, "CRE_Taira24", "HFRC.freq8-24Hz_maxdist3_coh9500_linkage_cluster.txt"),
    },
    "CRE_Wald": os.path.join(inputs_dir, "CRE_Waldhauser21", "waldhauser_shaff_repeaters.txt"),
    "CRE_Seno": {
        "S1": os.path.join(inputs_dir, "CRE_Senobari19", "S1_Catalog_confirmed_RE.csv"),
        "S2": os.path.join(inputs_dir, "CRE_Senobari19", "S2_Catalog_of_possible_repeating_earthquakes.csv"),
        "S3": os.path.join(inputs_dir, "CRE_Senobari19", "S3_Catalog_of_repeating_earthquake_pairs.csv"),
    },
    "NCEDC": {
        "M0": os.path.join(inputs_dir, "NCEDC_seismicity", "RC_MA_ddcat_2005-2025_M0.txt"),
        "M1": os.path.join(inputs_dir, "NCEDC_seismicity", "RC_MA_ddcat_2005-2025_M1.txt"),
        "Geysers": os.path.join(inputs_dir, "NCEDC_seismicity", "Geysers_ddcat_2005-2025_M2.txt"),
        "MWIL_15km": os.path.join(inputs_dir, "NCEDC_seismicity", "MWIL_radius15km_1992_2012.csv"),
    },
    "PRISM": {
        "MWIL_mm": os.path.join(inputs_dir, "PRISM_precipatation", "MWIL_mean_monthly_PRISM_obs_period.csv"),
        "bil_dir": os.path.join(inputs_dir, "PRISM_precipatation", "PRISM_ppt_stable_4kmM3_198101_202304_bil"),
        "monthly": os.path.join(inputs_dir, "PRISM_precipatation", "MWIL_monthly_PRISM_obs_period.csv"),
        "daily": os.path.join(inputs_dir, "PRISM_precipatation", "MWIL_dailyrainfall_39.4096--123.3556.csv"),
    },
    "NMF": {
        "vel_grd": os.path.join(proj_dir, "Descending", "170_NMF", "geo", "geo_velocity_msk.grd"),
        "FP_035_115": os.path.join(proj_dir, "NMF_HzUp", "N145_035_115_Aug24.grd"),
        "FP_068_170": os.path.join(proj_dir, "NMF_HzUp", "N145_068_170_Aug24.grd"),
        "up_035_115": os.path.join(proj_dir, "NMF_HzUp", "up_N145_035_115_Aug24.grd"),
        "up_068_170": os.path.join(proj_dir, "NMF_HzUp", "up_N145_068_170_Aug24.grd"),
        "170_creeprate": os.path.join(proj_dir, "Creep_anaylsis", "NMF_170_west_east_fault_parallel_CreepRate_x_width0.5_dist_min0.5_dist_max1.5_minpoints20.csv"),
    },
    "SMF": {
        "vel_grd": os.path.join(proj_dir, "Descending", "170_SMF", "geo", "geo_velocity_msk.grd"),
        "FP_035_115": os.path.join(proj_dir, "SMF_HzUp", "N145_035_115_Aug24.grd"),
        "FP_068_170": os.path.join(proj_dir, "SMF_HzUp", "N145_068_170_Aug24.grd"),
        "up_035_115": os.path.join(proj_dir, "SMF_HzUp", "up_N145_035_115_Aug24.grd"),
        "up_068_170": os.path.join(proj_dir, "SMF_HzUp", "up_N145_068_170_Aug24.grd"),
        "170_creeprate": os.path.join(proj_dir, "Creep_anaylsis", "SMF_170_west_east_fault_parallel_CreepRate_x_width0.5_dist_min0.5_dist_max1.5_minpoints20.csv"),
    },
    "RCF": {
        "vel_grd": os.path.join(proj_dir, "Descending", "170_RCF", "geo", "geo_velocity_msk.grd"),
        "FP_035_115": os.path.join(proj_dir, "RCF_HzUp", "N145_035_115_Aug24.grd"),
        "FP_068_170": os.path.join(proj_dir, "RCF_HzUp", "N145_068_170_Aug24.grd"),
        "up_035_115": os.path.join(proj_dir, "RCF_HzUp", "up_N145_035_115_Aug24.grd"),
        "up_068_170": os.path.join(proj_dir, "RCF_HzUp", "up_N145_068_170_Aug24.grd"),
        "170_creeprate": os.path.join(proj_dir, "Creep_anaylsis", "RCF_170_west_east_fault_parallel_CreepRate_x_width0.5_dist_min0.5_dist_max1.5_minpoints20.csv"),
    },

}
