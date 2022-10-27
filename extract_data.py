import sys
import os
from pathlib import Path
import glob
from utils import utils
import numpy as np
import netCDF4 as nc
from skimage.io.collection import alphanumeric_key


def ncdump(nc_fid, verb=False):
    """
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    """

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    tmp_dict = {}
    for nc_attr in nc_attrs:
        tmp_dict.update({nc_attr: nc_fid.getncattr(nc_attr)})
    nc_attrs = tmp_dict

    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    tmp_dict = {}
    for dim in nc_dims:
        tmp_dict.update({dim: len(nc_fid.dimensions[dim])})
    nc_dims = tmp_dict

    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    tmp_dict = {}
    for var in nc_vars:
        var_info = {
            "dtype": nc_fid.variables[var].dtype,
            "dimensions": nc_fid.variables[var].dimensions,
            "size": nc_fid.variables[var].size,
        }
        for ncattr in nc_fid.variables[var].ncattrs():
            var_info.update({ncattr: nc_fid.variables[var].getncattr(ncattr)})

        tmp_dict.update({var: var_info})
    nc_vars = tmp_dict
    return nc_attrs, nc_dims, nc_vars


if __name__ == "__main__":
    folder = Path(os.getcwd()).parent.absolute()
    folder = os.path.join(folder, "data", "tccs", "ocean", "SST")
    filenames = sorted(glob.glob(os.path.join(folder, "*")), key=alphanumeric_key)
    # filenames = [f for f in filenames if f.find("nday1") != -1]
    print(f"number of files: {len(filenames)}")

    for filename in filenames:
        var_names = [
            "TLONG",
            "TLAT",
            "ULONG",
            "ULAT",
            "REGION_MASK",
            "SST",
            "time",
            # "z_t",
            # "z_t_150m",
            # "z_w",
            # "z_w_bot",
            # "z_w_top",
        ]

        # Get metadata
        with nc.Dataset(filename) as ds:
            nc_attrs, nc_dims, nc_vars = ncdump(ds, verb=True)
            del nc_dims["d2"]  # no available data

        # Write necessary data
        data_folder = "/".join(filenames[0].split("/")[:-1])
        data_folder = data_folder + "_modified"
        utils.mkdir_if_not_exist(data_folder)
        name = ".".join(filename.split(".")[-3:])
        output_path = os.path.join(data_folder, name)

        with nc.Dataset(output_path, mode="w", format="NETCDF4") as ncfile:
            ncfile.set_auto_maskandscale(True)
            # Create Dimension
            for k, v in nc_dims.items():
                if k.lower().find("time") != -1:
                    ncfile.createDimension(
                        k, None
                    )  # unlimited axis (can be appended to).
                else:
                    ncfile.createDimension(k, v)

            # Create Attributes
            for k, v in nc_attrs.items():
                if k.lower().find("history") == -1:
                    ncfile.setncattr(k, v)

            # Create Variables
            for var_name, var_attrs in nc_vars.items():
                if var_name in var_names:
                    if var_name in ["SST"]:
                        chunksizes = (1, 200, 200)
                    elif var_name in ["REGION_MASK", "TLONG", "TLAT", "ULONG", "ULAT"]:
                        chunksizes = (200, 200)
                    else:
                        chunksizes = None

                    if var_name == "SST":
                        var_attrs["dimensions"] = ("time", "nlat", "nlon")

                    variable = ncfile.createVariable(
                        var_name,
                        datatype=var_attrs["dtype"],
                        dimensions=var_attrs["dimensions"],
                        zlib=True,
                        chunksizes=chunksizes,
                    )
                    for attr_name, attr in var_attrs.items():
                        if attr_name not in [
                            "dimensions",
                            "dtype",
                            "size",
                            "_FillValue",
                            "missing_value",
                        ]:
                            variable.setncattr(attr_name, attr)
            for i, var_name in enumerate(var_names):
                # Assign data
                with nc.Dataset(filename) as ds:
                    data = ds.variables[var_name][:]
                    if var_name == "SST":
                        if len(data.shape) != len(nc_vars[var_name]["dimensions"]):
                            data = ds.variables[var_name][:, 0, :, :]
                    data = data.filled(np.nan)
                    ncfile.variables[var_name][:] = data

        print(f"complete {filename}")

    for filename in filenames:
        data_folder = "/".join(filenames[0].split("/")[:-1])
        data_folder = data_folder + "_modified"
        name = ".".join(filename.split(".")[-3:])
        output_path = os.path.join(data_folder, name)

        original_size = os.path.getsize(filename) / 1e9
        rebuilt_size = os.path.getsize(output_path) / 1e9
        print()
        print(
            f"original size: {original_size:0.2f} GB; "
            f"rebuilt size: {rebuilt_size:0.2f} GB"
        )
