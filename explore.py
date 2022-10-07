import pandas as pd
import numpy as np
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from dask import delayed
import functools
from skimage.io.collection import alphanumeric_key
import dask
import os
from pathlib import Path
import glob
import json
import xarray as xr
import matplotlib.pyplot as plt
import napari
from utils import utils
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
# explicitly connect to the cluster we just created
client = Client(cluster)
client

IS_CALCULATING_STATISTICS = False

if __name__ == "__main__":
    folder = "./data/ocean/SST_modified"
    filenames = sorted(glob.glob(os.path.join(folder, "*")), key=alphanumeric_key)
    print(f"number of files: {len(filenames)}")

    # Checking chunk size of original data
    for filename in filenames:
        with nc.Dataset(filename) as ds:
            print(f"{filename}: {ds['SST'].chunking()}")

    ds = xr.open_mfdataset(
        filenames,
        combine="by_coords",
        chunks={"time": 1, "nlat": 1200, "nlon": 1200},
    )

    # Get data and fill mask and nan values
    mask_region = ds["REGION_MASK"][0].load()
    land_value = -20.0
    fillna_value = land_value + 1
    mask_region = xr.where(mask_region != -1, 1, -land_value)
    data_name = "SST"
    da = ds[data_name]
    da = da * mask_region  # da = xr.where(da == -1, land_value, da)
    da = da.fillna(fillna_value)

    mask_value = max(land_value, fillna_value)
    if IS_CALCULATING_STATISTICS:
        df = utils.get_ocean_statistics(
            da,
            mask_value,
            verbose=True,
            saved_path="./outputs/ocean/SST/statistics.csv",
            is_save=True,
        )

    viewer = napari.view_image(
        da.isel(nlat=slice(None, None, -1)), multiscale=False, name="data"
    )
    viewer.layers["data"].contrast_limits = (land_value, 35)

    @viewer.bind_key("p")
    def print_names(viewer):
        print([layer.name for layer in viewer.layers])

    @viewer.bind_key("m")
    def print_message(viewer):
        print("hello")
        yield
        print("goodbye")

    napari.run()  # start the event loop and show viewer
