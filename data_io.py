import sys
import os
import xarray as xr
import numpy as np
import random
import torch.utils.data as data
from utils import padder
from utils import sliding_window as sw
import matplotlib.pyplot as plt
import glob
import pandas as pd
import random
from torchvision import transforms
import torch
import dask
import copy

dask.config.set(scheduler="synchronous")

from utils import logger, utils

DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


class Dataio:
    def __init__(self, batch_size, patch_size, data_shape):
        self.padder = None
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.data_shape = data_shape
        self.params = {}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._init_padder()

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, value):
        self._patch_size = value
        if self.padder is not None:
            self._init_padder()

    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        self._data_shape = value
        if self.padder is not None:
            self._init_padder()

    def _init_padder(self):
        if len(self.data_shape) == 4:
            self.padder = padder.Padder2D(self.patch_size, self.data_shape)
        elif len(self.data_shape) == 5:
            self.padder = padder.Padder3D(self.patch_size, self.data_shape)
        else:
            raise ValueError("data shape must be of length 4 or 5.")

    def get_train_test_data_loader(
        self, data_path, data_type, ds_name, da_name, local_test=False
    ):
        if data_type.lower().find("netcdf") != -1:
            filenames, fillna_value = utils.get_filenames_and_fillna_value(data_path)
            if local_test:
                filenames = filenames[:2]
            split = int(len(filenames) * 0.99)
            train_files = filenames[:split]

            # Train data
            logger.log(f"number of train_files: {len(train_files)}")
            train_dataset = self.create_overlapping_generator(
                train_files,
                ds_name,
                da_name,
                fillna_value=fillna_value,
                name="train",
                shuffle=True,
            )
            # Test data
            test_files = filenames[split:]
            logger.log(f"number of test_files: {len(test_files)}")
            test_dataset = self.create_disjoint_generator(
                test_files,
                ds_name,
                da_name,
                fillna_value=fillna_value,
                name="test",
                shuffle=False,
            )
        elif data_type.lower().find("binary") != -1:
            # TODO: make create_disjoin_binary_data function

            train_dataset = OverlappingBinaryData(
                data_path,
                ds_name,
                da_name,
                kernel_size=self.patch_size,
                stride=self.patch_size - 8,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                data_shape=self.data_shape,
                fillna_value=0,
                shuffle=True,
                name="train",
                transform=self.transform,
            )
            self._update_parameters(train_dataset, name="train")
            test_dataset = DisjointBinaryData(
                data_path,
                ds_name,
                da_name,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                data_shape=self.data_shape,
                fillna_value=0,
                shuffle=True,
                name="test",
                transform=self.transform,
            )
            self._update_parameters(test_dataset, name="test")
            self.data_shape = test_dataset.data_shape
        else:
            raise ValueError(f"{data_type} is not supported")

        # data loader
        train_ds = self.get_data_loader(
            train_dataset,
            drop_last=True,
            shuffle=True,
            num_workers=4 * NUM_GPUS,
            pin_memory=True,
        )
        test_ds = self.get_data_loader(test_dataset, num_workers=4 * NUM_GPUS)
        return train_ds, test_ds

    def get_compression_data_loader(self, input_path, ds_name, da_name):
        filenames, fillna_value = utils.get_filenames_and_fillna_value(input_path)
        ds = self.create_disjoint_generator(
            filenames, ds_name, da_name, fillna_value, name="compression", shuffle=False
        )
        ds = self.get_data_loader(ds, num_workers=4 * NUM_GPUS)
        return ds

    def get_benchmark_compression_data_loader(self, input_path, ds_name, da_name):
        ds = DisjointBinaryData(
            input_path,
            ds_name,
            da_name,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            data_shape=self.data_shape,
            fillna_value=0,
            shuffle=True,
            name="compression",
            transform=self.transform,
        )
        self.data_shape = ds.data_shape
        ds = self.get_data_loader(ds, num_workers=4 * NUM_GPUS)
        return ds

    def get_num_batch_per_time_slice(self):
        # number of disjoint tiles per time slice for compression purpose
        sizes = []
        for size in self.data_shape[1:-1]:
            num_patch = size // self.patch_size
            if size % self.patch_size == 0:
                sizes.append(num_patch)
            else:
                sizes.append(num_patch + 1)
        return np.prod(sizes)

    def create_disjoint_generator(
        self, filenames, ds_name, da_name, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = DisjointDataGen(
            filenames,
            ds_name,
            da_name,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            data_shape=self.data_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
            transform=self.transform,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def create_overlapping_generator(
        self, filenames, ds_name, da_name, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = OverlappingDataGen(
            filenames,
            ds_name,
            da_name,
            kernel_size=self.patch_size,
            stride=self.patch_size - 8,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            data_shape=self.data_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
            transform=self.transform,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def create_train_generator(
        self, filenames, ds_name, da_name, fillna_value=0, name=None, shuffle=False
    ):
        data_gen = TrainOverlappingDataGen(
            filenames,
            ds_name,
            da_name,
            kernel_size=self.patch_size,
            stride=self.patch_size - 8,
            batch_size=self.batch_size,
            patch_size=self.patch_size,
            data_shape=self.data_shape,
            fillna_value=fillna_value,
            shuffle=shuffle,
            name=name,
            transform=self.transform,
        )
        self._update_parameters(data_gen, name=name)
        return data_gen

    def get_data_loader(
        self,
        data_gen,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        prefetch_factor=8,
        pin_memory=False,
    ):
        self._check_generator_type(data_gen)
        if num_workers == 0:
            num_workers = 1
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
                f"{name}.data_gen": data_gen.__repr__(),
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

    def change_data_shape(self, ds_name):
        if ds_name.lower().find("cesm") != -1:
            self.data_shape = (1, self.data_shape[-3], self.data_shape[-2], 1)
        elif (
            ds_name.lower().find("nyx") != -1
            or ds_name.lower().find("hurrican_isabel") != -1
        ):
            self.data_shape = (1, self.data_shape[-3], self.data_shape[-2], 2)

    def get_mask(self, data_gen):
        """Get mask"""
        self._check_generator_type(data_gen)
        return data_gen.mask

    def partition_data(self, ds):
        """Partition ds to multiple tiles"""
        ds = self.padder.pad_data(ds)
        ds = self.padder.split_data(ds)
        return ds

    def revert_partition(self, ds):
        """Revert partitioned data to original shape"""
        ds = self.padder.unsplit_data(ds)
        ds = self.padder.remove_pad_data(ds)
        return ds

    def _check_generator_type(self, data_gen):
        if not (isinstance(data_gen, BaseDataGen)):
            raise TypeError(
                "data_gen must be a TFDataGen object or CompressionDataGen object"
            )

    def get_training_parameters(self):
        return self.params

    def get_num_tiles(self):
        return self.padder.num_tiles

    def get_padded_dims(self):
        return self.padder.pad_dim

    def log_training_parameters(self):
        message = "\n"
        for k, v in self.get_training_parameters().items():
            message += k + " = " + str(v) + "\n"
        logger.log(f"DataIO Parameters:{message}")

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, "=", value)

    def get_attributes(self):
        return self.__dict__


class BaseDataGen(data.Dataset):
    """
    Data I/O, which prepares training, validation, and testing data sets.
    There are two types of data this can handle: 'image' and 'scientific' (binary) data.
    """

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        batch_size,
        data_shape,
        patch_size,
        fillna_value,
        shuffle,
        name,
        transform=None,
    ):
        super().__init__()
        self.num_outputs = None
        self.filenames = filenames
        self.ds_name = ds_name
        self.da_name = da_name
        self.fillna_value = fillna_value
        self.name = name
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.transform = transform
        self.data_shape = data_shape

        patch_shape = [data_shape[0]]
        patch_shape.append(data_shape[-1])
        for i in range(len(data_shape) - 2):
            patch_shape.append(patch_size)
        self.patch_shape = patch_shape
        self.ds = None

    def __repr__(self):
        message = (
            f"{self.__class__.__name__}(name: {self.name}; " f"shuffle: {self.shuffle}"
        )
        if self.num_outputs is not None:
            message += f"; patch_shape: {self.num_outputs}x{self.patch_shape}"
            message += f"; ds_name: {self.ds_name}"
            message += f"; da_name: {self.da_name}"

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
        raise NotImplementedError

    def __len__(self):
        return self.num_patches

    def __getitem__(self, index):
        raise NotImplementedError


class DisjointDataGen(BaseDataGen):
    """Data generator for disjoint patchers."""

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        batch_size,
        patch_size,
        data_shape,
        fillna_value=0,
        shuffle=False,
        name=None,
        transform=None,
    ):
        super().__init__(
            filenames=filenames,
            ds_name=ds_name,
            da_name=da_name,
            batch_size=batch_size,
            data_shape=data_shape,
            patch_size=patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
            transform=transform,
        )

        self.num_outputs = 2
        if len(data_shape) == 4:
            self.padder = padder.Padder2D(patch_size, data_shape)
        elif len(data_shape) == 5:
            self.padder = padder.Padder3D(patch_size, data_shape)
        else:
            raise ValueError("data shape must be of length 4 or 5.")

        self.on_epoch_end()

        self.num_time_slices = self.ds.shape[0]
        self.num_patch_per_time_slice = (self.padder.padded_shape[1] // patch_size) * (
            self.padder.padded_shape[2] // patch_size
        )
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
        if self.shuffle:
            random.shuffle(self.filenames)
        if self.ds == None:
            (self.ds, self.mask) = self.get_xarray_data(
                self.filenames, self.fillna_value
            )
            self.mask = self.mask[np.newaxis, ..., np.newaxis]
            self.mask = self.padder.pad_data(self.mask)
            self.mask = self.padder.split_data(self.mask)

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
        # Based on data type, fillna_value is different
        # TODO: change the value here, add mask_threshold to init args
        if self.ds_name == "SST":
            masks = xr.where(masks != -1, 1, np.nan)  # for SST dataset
        if self.ds_name == "TEMP":
            masks = xr.where(masks > 0, 1, np.nan)  # for volumetric TEMP

        if len(ds[self.ds_name].shape) == 4:
            das = ds[self.ds_name][:, 0, ...]  # (N, z_t, 2400, 3600)
        elif len(ds[self.ds_name].shape) == 3:
            das = ds[self.ds_name]
        das = das * masks

        # masks = 0 where das is nan, otherwise 1
        masks = xr.where(das[0, ::-1, :].isnull(), 0, 1)
        masks = masks.to_numpy()

        das = das.fillna(fillna_value)

        return (das, masks)

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

        sample = self.da[da_idx]
        mask = self.mask[da_idx]
        if self.transform:
            sample = self.transform(sample)
            mask = self.transform(mask)
        return sample, mask


class OverlappingDataGen(BaseDataGen):
    """Data generator for overlapping patches."""

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        kernel_size,
        stride,
        batch_size,
        patch_size,
        data_shape,
        padding=True,
        fillna_value=0,
        shuffle=False,
        name=None,
        transform=None,
    ):
        super().__init__(
            filenames=filenames,
            ds_name=ds_name,
            da_name=da_name,
            batch_size=batch_size,
            data_shape=data_shape,
            patch_size=patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
            transform=transform,
        )

        self.num_outputs = 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.ds = None
        self.sliding_window = sw.SlidingWindow(
            self.kernel_size, self.stride, padding=padding
        )
        self.coors = self.sliding_window.get_window_coors(data_shape)
        num_windows = self.sliding_window.get_total_num_windows(data_shape[1:-1])
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
        if self.shuffle:
            random.shuffle(self.filenames)
        if self.ds == None:
            (self.ds, mask) = self.get_xarray_data(self.filenames, self.fillna_value)
            mask = mask[np.newaxis, ..., np.newaxis]
            self.mask = self.sliding_window.pad_data(mask)

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

        # Based on data type, fillna_value is different
        # TODO: change the value here, add mask_threshold to init args
        if self.ds_name == "SST":
            masks = xr.where(masks != -1, 1, np.nan)  # for SST dataset
        if self.ds_name == "TEMP":
            masks = xr.where(masks > 0, 1, np.nan)  # for volumetric TEMP

        if len(ds[self.ds_name].shape) == 4:
            das = ds[self.ds_name][:, 0, ...]  # (N, z_t, 2400, 3600)
        elif len(ds[self.ds_name].shape) == 3:
            das = ds[self.ds_name]
        das = das * masks

        # masks = 0 where das is nan, otherwise 1
        masks = xr.where(das[0, ::-1, :].isnull(), 0, 1)
        masks = masks.to_numpy()

        das = das.fillna(fillna_value)

        return (das, masks)

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
        if self.transform:
            window = self.transform(window)
            window_mask = self.transform(window_mask)
        return window, window_mask


class TrainOverlappingDataGen(BaseDataGen):
    """Data generator for overlapping patches using weighted mask."""

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        kernel_size,
        stride,
        batch_size,
        patch_size,
        data_shape,
        padding=True,
        fillna_value=0,
        shuffle=False,
        name=None,
        transform=None,
    ):
        super().__init__(
            filenames=filenames,
            ds_name=ds_name,
            da_name=da_name,
            batch_size=batch_size,
            data_shape=data_shape,
            patch_size=patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
            transform=transform,
        )

        self.num_outputs = 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.ds = None
        self.sliding_window = sw.SlidingWindow(
            self.kernel_size, self.stride, padding=padding
        )
        self.coors = self.sliding_window.get_window_coors(data_shape)
        num_windows = self.sliding_window.get_total_num_windows(data_shape[1:-1])
        self.on_epoch_end()

        # Elliminate the windows that only contains masked values
        tmp = []
        for i in range(num_windows):
            coor = self.sliding_window.get_coor_given_index(self.coors, i)
            window_mask = self.sliding_window.get_window_with_coordinate(
                self.weighted_mask, coor
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
        if self.ds == None:
            (self.ds, weighted_mask) = self.get_xarray_data(
                self.filenames, self.fillna_value
            )
            weighted_mask = weighted_mask[np.newaxis, ..., np.newaxis]
            self.weighted_mask = self.sliding_window.pad_data(weighted_mask)

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
        weighted_mask = copy.deepcopy(masks)
        weighted_mask = xr.where(weighted_mask == -14, 0.7, weighted_mask)
        weighted_mask = xr.where(weighted_mask == -13, 0.8, weighted_mask)
        weighted_mask = xr.where(weighted_mask == -1, 0.0, weighted_mask)
        weighted_mask = xr.where(weighted_mask >= 8, 1.5, weighted_mask)
        weighted_mask = xr.where(weighted_mask >= 3, 2, weighted_mask)
        # weighted_mask = xr.where(weighted_mask == 2, 3, weighted_mask)
        # weighted_mask = xr.where(weighted_mask == 1.5, 2, weighted_mask)
        weighted_mask = weighted_mask[::-1, :]
        weighted_mask = weighted_mask.to_numpy()

        # Based on data type, fillna_value is different
        if self.ds_name == "SST":
            masks = xr.where(masks != -1, 1, np.nan)  # for SST dataset
        if self.ds_name == "TEMP":
            masks = xr.where(masks > 0, 1, np.nan)  # for volumetric TEMP

        if len(ds[self.ds_name].shape) == 4:
            das = ds[self.ds_name][:, 0, ...]  # (N, z_t, 2400, 3600)
        elif len(ds[self.ds_name].shape) == 3:
            das = ds[self.ds_name]
        das = das * masks
        das = das.fillna(fillna_value)

        return (das, weighted_mask)

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
        window_weighted_mask = self.sliding_window.get_window_with_coordinate(
            self.weighted_mask, self.coors[window_idx]
        )
        if self.transform:
            window = self.transform(window)
            window_weighted_mask = self.transform(window_weighted_mask)
        return window, window_weighted_mask


class DisjointBinaryData(BaseDataGen):
    """Data generator for disjoint patchers."""

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        batch_size,
        patch_size,
        data_shape,
        fillna_value=0,
        shuffle=False,
        name=None,
        transform=None,
    ):
        super().__init__(
            filenames=filenames,
            ds_name=ds_name,
            da_name=da_name,
            batch_size=batch_size,
            data_shape=data_shape,
            patch_size=patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
            transform=transform,
        )

        # get data
        ds = self.preprocess_data(filenames, data_shape)
        if name == "test":
            time_subset = int(np.ceil(ds.shape[0] * 0.01))
            ds = ds[-time_subset:, ...]
        self.data_shape = ds.shape
        self.num_outputs = 2
        if len(self.data_shape) == 4:
            self.padder = padder.Padder2D(patch_size, self.data_shape)
        elif len(self.data_shape) == 5:
            self.padder = padder.Padder3D(patch_size, self.data_shape)
        else:
            raise ValueError("data shape must be of length 4 or 5.")
        da = self.padder.pad_data(ds)
        self.da = self.padder.split_data(da)
        utils.get_data_info(self.da)

        self.num_time_slices = ds.shape[0]
        self.num_patch_per_time_slice = (self.padder.padded_shape[1] // patch_size) * (
            self.padder.padded_shape[2] // patch_size
        )
        self.num_patches = ds.shape[0] * self.num_patch_per_time_slice

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

    def preprocess_data(self, filenames, data_shape):
        ds = utils.get_raw_data(filenames)
        ds = np.reshape(ds, (-1, *data_shape[1:]))
        if self.ds_name.lower().find("cesm") != -1:
            ds = np.reshape(ds, (-1, *ds.shape[1:]))
        elif self.ds_name.lower().find("hurrican_isabel") != -1:
            ds = np.reshape(ds, (-1, 2, *ds.shape[2:-1]))
            ds = np.transpose(ds, (0, 2, 3, 1))
        elif self.ds_name.lower().find("nyx") != -1:
            ds = np.reshape(ds, (-1, 2, *ds.shape[2:-1]))
            ds = np.transpose(ds, (0, 2, 3, 1))
        else:
            raise ValueError("Unknown dataset name.")
        return ds

    def __getitem__(self, index):
        sample = self.da[index]
        if self.ds_name == "CESM":
            mask = copy.deepcopy(sample)
            mask[mask > 0] = 1
        else:
            mask = np.ones_like(sample)
        if self.transform:
            sample = self.transform(sample)
            mask = self.transform(mask)
        return sample, mask


class OverlappingBinaryData(BaseDataGen):
    """Data generator for overlapping patches."""

    def __init__(
        self,
        filenames,
        ds_name,
        da_name,
        kernel_size,
        stride,
        batch_size,
        patch_size,
        data_shape,
        padding=True,
        fillna_value=0,
        shuffle=False,
        name=None,
        transform=None,
    ):
        super().__init__(
            filenames=filenames,
            ds_name=ds_name,
            da_name=da_name,
            batch_size=batch_size,
            data_shape=data_shape,
            patch_size=patch_size,
            shuffle=shuffle,
            fillna_value=fillna_value,
            name=name,
            transform=transform,
        )
        self.zero_counter = 0
        self.num_outputs = 2
        self.kernel_size = kernel_size
        self.stride = stride
        ds = self._preprocess_data(filenames, data_shape)
        self.data_shape = ds.shape
        self.sliding_window = sw.SlidingWindow(
            self.kernel_size, self.stride, padding=padding
        )

        self.coors = self.sliding_window.get_window_coors(self.data_shape)
        num_windows = self.sliding_window.get_total_num_windows(self.data_shape[1:-1])
        coors = []
        for i in range(num_windows):
            coor = self.sliding_window.get_coor_given_index(self.coors, i)
            coors.append(coor)
        self.coors = np.array(coors)
        ds = self.sliding_window.pad_data(ds)
        self.da = []
        for time_idx in range(ds.shape[0]):
            for coor in self.coors:
                window = self.sliding_window.get_window_with_coordinate(
                    ds[time_idx][np.newaxis, ...], coor
                )
                if np.sum(window) != 0:
                    self.da.append(window)
        self.da = np.array(self.da)
        utils.get_data_info(self.da)

        self.num_time_slices = ds.shape[0]
        self.num_patch_per_time_slice = None
        self.num_patches = self.da.shape[0]
        num_batches = self.num_patches / batch_size
        self.num_batches = int(
            num_batches if self.num_patches % batch_size == 0 else num_batches + 1
        )
        self.num_batch_per_time_slice = None

    def _preprocess_data(self, filenames, data_shape):
        ds = utils.get_raw_data(filenames)
        ds = np.reshape(ds, (-1, *data_shape[1:]))
        if self.ds_name.lower().find("cesm") != -1:
            ds = np.reshape(ds, (-1, *ds.shape[1:]))
        elif self.ds_name.lower().find("hurrican_isabel") != -1:
            ds = np.reshape(ds, (-1, 2, *ds.shape[2:-1]))
            ds = np.transpose(ds, (0, 2, 3, 1))
        elif self.ds_name.lower().find("nyx") != -1:
            ds = np.reshape(ds, (-1, 2, *ds.shape[2:-1]))
            ds = np.transpose(ds, (0, 2, 3, 1))
        else:
            raise ValueError("Unknown dataset name.")
        return ds

    def __getitem__(self, index):
        # return data and its mask
        window = self.da[index]
        if self.ds_name == "CESM":
            window_mask = copy.deepcopy(window)
            window_mask[window_mask > 0] = 1
        else:
            window_mask = np.ones_like(window)
        if self.transform:
            window = self.transform(window)
            window_mask = self.transform(window_mask)
        return window, window_mask
