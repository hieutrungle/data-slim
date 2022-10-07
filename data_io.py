import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import xarray as xr
import numpy as np
import logging
import random
import torch.utils.data as data
from utils import padder
from utils import sliding_window as sw
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io.collection import alphanumeric_key
import glob
import pandas as pd
import random
import dask

dask.config.set(scheduler="synchronous")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Dataio:
    def __init__(
        self,
        batch_size,
        data_patch_size,
        model_patch_size,
        original_shape,
    ):
        self.batch_size = batch_size
        self.data_patch_size = data_patch_size
        self.model_patch_size = model_patch_size
        self.original_shape = original_shape
        self.params = {}

    def get_num_batch_per_time_slice(self):
        # number of disjoint tiles per time slice for compression purpose
        sizes = []
        for size in self.original_shape[1:-1]:
            num_patch = size // self.data_patch_size
            if size % self.data_patch_size == 0:
                sizes.append(num_patch)
            else:
                sizes.append(num_patch + 1)
        return np.prod(sizes)

    def create_disjoint_generator(
        self, filenames, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = DisjointDataGen(
            filenames,
            batch_size=self.batch_size,
            data_patch_size=self.data_patch_size,
            original_shape=self.original_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def create_overlapping_generator(
        self, filenames, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = OverlappingDataGen(
            filenames,
            kernel_size=self.data_patch_size,
            stride=self.data_patch_size - 8,
            batch_size=self.batch_size,
            data_patch_size=self.data_patch_size,
            original_shape=self.original_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def create_compression_generator(
        self, filenames, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = CompressionDataGen(
            filenames,
            batch_size=self.batch_size,
            data_patch_size=self.data_patch_size,
            original_shape=self.original_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def get_data_loader(
        self,
        data_gen,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False,
    ):
        self._check_generator_type(data_gen)
        return data.DataLoader(
            data_gen,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
        )

    def _update_parameters(self, data_gen, name=None):
        self.params.update(
            {
                f"{name}.num_time_slices": data_gen.num_time_slices,
                f"{name}.num_patch_per_time_slice": data_gen.num_patch_per_time_slice,
                f"{name}.num_patches": data_gen.num_patches,
                f"{name}.num_batches": data_gen.num_batches,
                f"{name}.num_batch_per_time_slice": data_gen.num_batch_per_time_slice,
            }
        )
        # Quick hack to update params from custom data_gen objects (not good for other purposes)
        try:
            for k, v in data_gen.padder.get_attributes().items():
                self.params[f"{name}.{k}"] = v
        except:
            pass
        try:
            for k, v in data_gen.sliding_window.get_attributes().items():
                self.params[f"{name}.{k}"] = v
        except:
            pass

    def get_mask(self, data_gen):
        """Get mask"""
        self._check_generator_type(data_gen)
        return data_gen.mask

    def partition_data(self, ds, data_gen):
        """Partition ds to multiple tiles"""
        self._check_generator_type(data_gen)
        ds = data_gen.padder.pad_data(ds)
        ds = data_gen.padder.split_data(ds)
        return ds

    def revert_partition(self, ds, data_gen):
        """Revert partitioned data to original shape"""
        self._check_generator_type(data_gen)
        ds = data_gen.padder.unsplit_data(ds)
        ds = data_gen.padder.remove_pad_data(ds)
        return ds

    def _check_generator_type(self, data_gen):
        if not (isinstance(data_gen, BaseDataGen)):
            raise TypeError(
                "data_gen must be a TFDataGen object or CompressionDataGen object"
            )

    def get_training_parameters(self):
        return self.params

    def log_training_parameters(self):
        message = "\n"
        for k, v in self.get_training_parameters().items():
            message += k + " = " + str(v) + "\n"
        logger.info(f"Training Parameters:{message}")

    def get_filenames(self, data_path, prefix=".nc"):
        if data_path.endswith(".nc"):
            filenames = [data_path]
        else:
            filenames = sorted(
                glob.glob(os.path.join(data_path, "*" + prefix)), key=alphanumeric_key
            )
        return filenames

    def get_data_statistics(self, data_path):
        if data_path.endswith(".nc"):
            filename = Path(data_path)
            parent_folder = filename.parent.absolute()
            filename = glob.glob(os.path.join(parent_folder, "*.csv"))[0]
        else:
            filename = glob.glob(os.path.join(data_path, "*.csv"))[0]

        df = pd.read_csv(filename, index_col=0)
        stats = {}
        for col in ["mean", "median", "std"]:
            stats[col] = df[col].mean()
        return stats

    def get_filenames_and_fillna_value(self, data_path, prefix=".nc"):
        filenames = self.get_filenames(data_path, prefix=prefix)
        try:
            stats = self.get_data_statistics(data_path)
            fillna_value = stats["mean"]
        except:
            logger.info("No statistics file found. Using default nan_values = 0.")
            fillna_value = 0
        return filenames, fillna_value


class BaseDataGen(data.Dataset):
    """
    Data I/O, which prepares training, validation, and testing data sets.
    There are two types of data this can handle: 'image' and 'scientific' (binary) data.
    """

    def __init__(
        self,
        filenames,
        batch_size,
        original_shape,
        data_patch_size,
        fillna_value,
        shuffle,
        name,
    ):
        super().__init__()
        self.num_outputs = None
        self.filenames = filenames
        self.fillna_value = fillna_value
        self.name = name
        self.shuffle = shuffle
        self.data_patch_size = data_patch_size
        data_shape = [original_shape[0]]
        for i in range(len(original_shape) - 2):
            data_shape.append(data_patch_size)
        data_shape.append(original_shape[-1])
        self.data_shape = data_shape
        self.ds = None

    def __repr__(self):
        message = (
            f"{self.__class__.__name__}(name: {self.name}; " f"shuffle: {self.shuffle}"
        )
        if self.num_outputs is not None:
            message += f"; data_shape: {self.num_outputs}x{self.data_shape}"

        message += f")"
        return message

    def __str__(self):
        return self.__repr__()

    def print_instance_attributes(self):
        message = "\n"
        for attribute, value in self.__dict__.items():
            message += attribute + "=" + str(value) + "\n"
        logger.info(message)

    def get_xarray_data(self, filenames, fillna_value=0):

        ds = xr.open_mfdataset(
            filenames,
            combine="by_coords",
            chunks={"time": 1, "nlat": 1200, "nlon": 1200},
        )

        # masks value = nan, otherwise 1
        if len(ds["REGION_MASK"].shape) == 2:
            masks = ds["REGION_MASK"].load()
        elif len(ds["REGION_MASK"].shape) == 3:
            masks = ds["REGION_MASK"][0].load()
        else:
            raise ValueError(
                "REGION_MASK must be of shape (nlat, nlon) or (time, nlat, nlon)."
            )
        masks = xr.where(masks != -1, 1, np.nan)

        data_name = "SST"
        das = ds[data_name]  # (N, 2400, 3600)
        das = das * masks

        # masks = 0 where das is nan, otherwise 1
        masks = xr.where(das[0, ::-1, :].isnull(), 0, 1)
        masks = masks.to_numpy()

        das = das.fillna(fillna_value)

        return (das, masks)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index):
        raise NotImplementedError


class DisjointDataGen(BaseDataGen):
    """Data generator for disjoint patchers."""

    def __init__(
        self,
        filenames,
        batch_size,
        data_patch_size,
        original_shape,
        fillna_value=0,
        shuffle=True,
        name=None,
    ):
        super().__init__(
            filenames=filenames,
            batch_size=batch_size,
            original_shape=original_shape,
            data_patch_size=data_patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
        )

        self.num_outputs = 2
        if len(original_shape) == 4:
            self.padder = padder.Padder2D(data_patch_size, original_shape)
        elif len(original_shape) == 5:
            self.padder = padder.Padder3D(data_patch_size, original_shape)
        else:
            raise ValueError("data shape must be of length 4 or 5.")

        self.on_epoch_end()

        self.num_time_slices = self.ds.shape[0]
        self.num_patch_per_time_slice = (
            self.padder.padded_shape[1] // data_patch_size
        ) * (self.padder.padded_shape[2] // data_patch_size)
        self.num_patches = self.ds.shape[0] * self.num_patch_per_time_slice

        num_batches = self.num_patches / batch_size
        self.num_batches = int(
            num_batches if self.num_patches % batch_size == 0 else num_batches + 1
        )

        num_batch_per_time_slice = self.num_patch_per_time_slice / batch_size
        self.num_batch_per_time_slice = int(
            num_batch_per_time_slice
            if self.num_patch_per_time_slice % batch_size == 0
            else num_batch_per_time_slice + 1
        )

    def on_epoch_end(self):
        self.time_idx = -1
        self.da = None
        if self.ds == None or self.shuffle:
            random.shuffle(self.filenames)
            (self.ds, self.mask) = self.get_xarray_data(
                self.filenames, self.fillna_value
            )
            self.mask = self.mask[np.newaxis, ..., np.newaxis]
            self.mask = self.padder.pad_data(self.mask)
            self.mask = self.padder.split_data(self.mask)

    def __getitem__(self, index):
        # return data and its mask
        time_idx = index // self.num_patch_per_time_slice
        if time_idx > self.time_idx:
            self.time_idx = time_idx
            da = self.ds[time_idx]
            da = da.to_numpy()[::-1, :]
            da = da[np.newaxis, ..., np.newaxis]
            da = self.padder.pad_data(da)
            self.da = self.padder.split_data(da)
        da_idx = index % self.num_patch_per_time_slice
        return self.da[da_idx], self.mask[da_idx]


class OverlappingDataGen(BaseDataGen):
    """Data generator for overlapping patches."""

    def __init__(
        self,
        filenames,
        kernel_size,
        stride,
        batch_size,
        data_patch_size,
        original_shape,
        padding=True,
        fillna_value=0,
        shuffle=True,
        name=None,
    ):
        super().__init__(
            filenames=filenames,
            batch_size=batch_size,
            original_shape=original_shape,
            data_patch_size=data_patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
        )

        self.num_outputs = 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.ds = None
        self.sliding_window = sw.SlidingWindow(
            self.kernel_size, self.stride, padding=padding
        )
        self.coors = self.sliding_window.get_window_coors(original_shape)
        num_windows = self.sliding_window.get_total_num_windows(original_shape[1:-1])
        self.on_epoch_end()

        # Elliminate the windows that only contains masked values
        tmp = []
        for i in range(num_windows):
            coor = self.sliding_window.get_coor_given_index(self.coors, i)
            window_mask = self.sliding_window.get_window_with_coordinate(
                self.mask, coor
            )
            if np.sum(window_mask) != 0.0:
                tmp.append(coor)
        self.coors = np.array(tmp)
        num_windows = len(self.coors)

        self.num_time_slices = self.ds.shape[0]
        self.num_patch_per_time_slice = num_windows
        self.num_patches = self.num_time_slices * self.num_patch_per_time_slice

        num_batches = self.num_patches / batch_size
        self.num_batches = int(
            num_batches if self.num_patches % batch_size == 0 else num_batches + 1
        )

        num_batch_per_time_slice = self.num_patch_per_time_slice / batch_size
        self.num_batch_per_time_slice = int(
            num_batch_per_time_slice
            if self.num_patch_per_time_slice % batch_size == 0
            else num_batch_per_time_slice + 1
        )

    # implement sliding window
    def on_epoch_end(self):
        self.time_idx = -1
        self.da = None
        if self.ds == None or self.shuffle:
            random.shuffle(self.filenames)
            (self.ds, mask) = self.get_xarray_data(self.filenames, self.fillna_value)
            mask = mask[np.newaxis, ..., np.newaxis]
            self.mask = self.sliding_window.pad_data(mask)

    def __getitem__(self, index):
        # return data and its mask
        time_idx = index // self.num_patch_per_time_slice
        if time_idx > self.time_idx:

            self.time_idx = time_idx
            da = self.ds[time_idx]
            da = da.to_numpy()[::-1, :]
            da = da[np.newaxis, ..., np.newaxis]
            self.da = self.sliding_window.pad_data(da)

        window_idx = index % self.num_patch_per_time_slice
        window = self.sliding_window.get_window_with_coordinate(
            self.da, self.coors[window_idx]
        )
        window_mask = self.sliding_window.get_window_with_coordinate(
            self.mask, self.coors[window_idx]
        )

        return window, window_mask


class CompressionDataGen(BaseDataGen):
    """Data generator for compression, which only get the data (mask is not included)."""

    def __init__(
        self,
        filenames,
        batch_size,
        data_patch_size,
        original_shape,
        fillna_value=0,
        shuffle=True,
        name=None,
    ):
        super().__init__(
            filenames=filenames,
            batch_size=batch_size,
            original_shape=original_shape,
            data_patch_size=data_patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
        )

        self.num_outputs = 1
        if len(original_shape) == 4:
            self.padder = padder.Padder2D(data_patch_size, original_shape)
        elif len(original_shape) == 5:
            self.padder = padder.Padder3D(data_patch_size, original_shape)
        else:
            raise ValueError("data shape must be of length 4 or 5.")

        self.on_epoch_end()

        self.num_time_slices = self.ds.shape[0]
        self.num_patch_per_time_slice = (
            self.padder.padded_shape[1] // data_patch_size
        ) * (self.padder.padded_shape[2] // data_patch_size)
        self.num_patches = self.ds.shape[0] * self.num_patch_per_time_slice

        num_batches = self.num_patches / batch_size
        self.num_batches = int(
            num_batches if self.num_patches % batch_size == 0 else num_batches + 1
        )

        num_batch_per_time_slice = self.num_patch_per_time_slice / batch_size
        self.num_batch_per_time_slice = int(
            num_batch_per_time_slice
            if self.num_patch_per_time_slice % batch_size == 0
            else num_batch_per_time_slice + 1
        )

    def on_epoch_end(self):
        self.time_idx = -1
        self.da = None
        if self.ds == None or self.shuffle:
            random.shuffle(self.filenames)
            (self.ds, self.mask) = self.get_xarray_data(
                self.filenames, self.fillna_value
            )
            self.mask = self.mask[np.newaxis, ..., np.newaxis]
            self.mask = self.padder.pad_data(self.mask)
            self.mask = self.padder.split_data(self.mask)

    def __getitem__(self, index):
        # return data
        time_idx = index // self.num_patch_per_time_slice
        if time_idx > self.time_idx:
            self.time_idx = time_idx
            da = self.ds[time_idx]
            da = da.to_numpy()[::-1, :]
            da = da[np.newaxis, ..., np.newaxis]
            da = self.padder.pad_data(da)
            self.da = self.padder.split_data(da)
        da_idx = index % self.num_patch_per_time_slice
        return self.da[da_idx]
