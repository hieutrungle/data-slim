from pathlib import Path
import glob
from utils import utils
import numpy as np
import netCDF4 as nc
import os
from skimage.io.collection import alphanumeric_key


def ncdump(nc_fid):
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
    nc_attrs : dict of the NetCDF file global attributes
    nc_dims : dict of the NetCDF file dimensions
    nc_vars : dict of the NetCDF file variables
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


def create_dataset_with_only_metadata(input_file, output_file, ds_name, verbose=False):
    with nc.Dataset(input_file) as ds:
        nc_attrs, nc_dims, nc_vars = ncdump(ds)

    # Write necessary data
    filename = Path(output_file)
    parent_folder = filename.parent.absolute()
    utils.mkdir_if_not_exist(parent_folder)

    with nc.Dataset(output_file, mode="w", format="NETCDF4") as ncfile:
        ncfile.set_auto_maskandscale(True)
        # Create Dimension
        for k, v in nc_dims.items():
            if k.lower().find("time") != -1:
                ncfile.createDimension(k, None)  # unlimited axis (can be appended to).
            else:
                ncfile.createDimension(k, v)

        # Create Attributes
        for k, v in nc_attrs.items():
            ncfile.setncattr(k, v)

        with nc.Dataset(input_file) as ds:
            for i, (var_name, var_attrs) in enumerate(nc_vars.items()):

                chunksizes = set_chunksizes(var_attrs["dimensions"])
                if var_name.lower().find("time_bound") != -1:
                    chunksizes = (1, 2)
                if verbose:
                    print(
                        f"{i}: {var_name} - {var_attrs['dimensions']} "
                        f"- len_shape {len(var_attrs['dimensions'])} - {var_attrs['dtype']} "
                        f"- {var_attrs['dimensions']} - chunksizes {chunksizes}"
                    )
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

                if not (var_name.find(ds_name) != -1):
                    data = ds.variables[var_name][:]
                    data = data.filled(np.nan)
                    # ncfile.variables[var_name][:] = data
                    variable[:] = data


def set_chunksizes(dimensions):
    if len(dimensions) == 4:
        chunksizes = (1, 1, 400, 400)
    elif len(dimensions) == 3:
        chunksizes = (1, 400, 400)
    elif len(dimensions) == 2:
        chunksizes = (400, 400)
    elif len(dimensions) == 1 or len(dimensions) == 0:
        chunksizes = (1,)
    else:
        raise ValueError(f"shape of data {dimensions} is out of range")
    return chunksizes


def write_data_to_netcdf(output_file, da, ds_name, time_idx, verbose=False):

    # with nc.Dataset(output_file) as ds:
    #     nc_attrs, nc_dims, nc_vars = ncdump(ds)
    with nc.Dataset(output_file, mode="a", format="NETCDF4") as ncfile:
        nc_attrs, nc_dims, nc_vars = ncdump(ncfile)
        for i, (var_name, var_attrs) in enumerate(nc_vars.items()):
            if var_name.find(ds_name) != -1:
                # print(
                #     f"var_name: {var_name}; var_attrs: {var_attrs}; da.shape: {da.shape}"
                # )
                if len(var_attrs["dimensions"]) == 3:
                    ncfile.variables[var_name][time_idx, :, :] = da
                elif len(var_attrs["dimensions"]) == 4:
                    ncfile.variables[var_name][time_idx, 0, :, :] = da
                # ncfile.variables[var_name][time_idx, 0, ...] = da


def main():
    folder = Path(os.path.join("../data/tccs", "ocean", "SST"))
    filenames = sorted(glob.glob(os.path.join(folder, "*.nc")), key=alphanumeric_key)
    filename = filenames[0]

    data_folder = "/".join(filenames[0].split("/")[:-1])
    data_folder = data_folder + "_testing"
    name = ".".join(filename.split(".")[-3:])
    output_path = os.path.join(data_folder, name)
    create_dataset_with_only_metadata(filename, output_path)

    # Get all attributes of netcdf data


if __name__ == "__main__":
    main()
