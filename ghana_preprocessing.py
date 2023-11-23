# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:00:30 2023

@author: tranq
"""

import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
import earthpy.plot as ep
import earthpy.spatial as es
import xarray as xr
import numpy as np

def norm_diff(b1, b2):
    return (b1 - b2) / (b1 + b2)

year_dict = {
    '2019': 0,
    '2020': 1,
    '2021': 2,
    '2022': 3,
}

def process_year(year):
    '''
    Param year: The year that will be processed.
    
    Returns: Geotiff in the outputs folder - ghana_prepped_{year}.tif
    Contains all sentinel bands 1-8, 8a, 9, 11, 12ds.c
    Contains NDVI, MNDWI, BSI
    Contains VV, VH, and 1 - (VV/(VV+VH))
    '''
    
    # Get spectral data
    ds = rxr.open_rasterio(f'raw_data/ghana_{year}_studyarea.tif', masked = True)

    # Get Radar data
    with rxr.open_rasterio('raw_data/ghana_VV_collection.tif', masked = True) as vv:
        vv_ds = vv[year_dict[year]]
        vv.close()

    with rxr.open_rasterio('raw_data/ghana_VH_collection.tif', masked = True) as vh:
        vh_ds = vh[year_dict[year]]
        vh.close()
        
    # norm_diff NIR/Red
    ndvi = norm_diff(ds[4], ds[3])

    # norm_diff Green/SWIR
    mndwi = norm_diff(ds[2], ds[10])

    # BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
    # from https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/barren_soil/
    bsi = 2.5 * norm_diff((ds[10] + ds[3]), (ds[6] + ds[1]))
    
    # Create m
    dop = vv_ds / (vv_ds + vh_ds)
    m = 1 - dop
    
    # Clip m to remove extreme outliers - may wish to remove this step in some cases
    modified_m = m.where(m > np.percentile(m, 0.1), np.percentile(m, 0.1))
    modified_m = modified_m.where(m < np.percentile(m, 99.9), np.percentile(m, 99.9))
    
    # Create final_output xarray
    # Note that we add band coords to the spectral indices and remove SCL band from sentinel data
    final_output = xr.concat([ds[:-1] / 10000, ndvi.assign_coords({'band': 1}), 
                              mndwi.assign_coords({'band': 1}), 
                              bsi.assign_coords({'band': 1}), 
                              vv_ds, vh_ds, modified_m], dim = 'band')
    
    final_output.assign_attrs(long_name = "B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, NDVI, MNDWI, BSI, \
                              VV, VH, M")
    
    final_output.rio.set_nodata(np.nan)
    final_output.rio.to_raster(f'outputs/ghana_prepped_{year}.tif')
    
# process_year('2020')
    
