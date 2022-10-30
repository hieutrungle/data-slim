import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasePadder:
    def __init__(self, patch_size, original_shape):
        self.padded_shape = ()
        self.patch_size = patch_size
        self.original_shape = original_shape
        self.padded_shape = self.calculate_padded_data_shape(self.original_shape)
        self.pad_dim = self.calculate_padding_dim(
            self.padded_shape, self.original_shape
        )

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, value):
        self._patch_size = value
        if len(self.padded_shape) != 0:
            logger.warning(f"Recalculating padded shape")
            self.padded_shape = self.calculate_padded_data_shape(self.original_shape)
            self.pad_dim = self.calculate_padding_dim(
                self.padded_shape, self.original_shape
            )

    @property
    def original_shape(self):
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value):
        self._original_shape = value
        if len(self.padded_shape) != 0:
            logger.warning(f"Recalculating padded shape")
            self.padded_shape = self.calculate_padded_data_shape(self.original_shape)
            self.pad_dim = self.calculate_padding_dim(
                self.padded_shape, self.original_shape
            )

    def calculate_num_tiles(self, total_length, patch_size):
        num_tiles = total_length // patch_size
        if total_length % patch_size:
            num_tiles += 1
        return num_tiles

    def calculate_padded_data_shape(self, data_shape):
        """Calculate data shape after padding"""
        padded_shape = [data_shape[0]]
        for l in data_shape[1:-1]:
            num_tiles = self.calculate_num_tiles(l, self.patch_size)
            padded_shape.append(num_tiles * self.patch_size)
        padded_shape.append(data_shape[-1])
        return tuple(padded_shape)

    def calculate_padding_dim(self, padded_shape, original_shape):
        pad_dim = [new - old for new, old in zip(padded_shape, original_shape)]
        pad_dim = [
            (np.ceil(i / 2).astype(np.int16), np.floor(i / 2).astype(np.int16))
            for i in pad_dim
        ]
        return pad_dim

    def pad_data(self, ds):
        """
        Add paddings to the whole dataset (3d data)
        (B, len_x, len_y) -> (B, pad_len_x, pad_len_y)
        """
        ds = np.pad(ds, (self.pad_dim), mode="wrap")
        return np.array(ds)

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, "=", value)

    def get_attributes(self):
        return self.__dict__


class Padder2D(BasePadder):
    def __init__(self, patch_size, original_shape):
        super().__init__(patch_size=patch_size, original_shape=original_shape)
        self.padded_shape = ()
        self.patch_size = patch_size
        self.original_shape = original_shape
        self.padded_shape = self.calculate_padded_data_shape(self.original_shape)
        self.pad_dim = self.calculate_padding_dim(
            self.padded_shape, self.original_shape
        )

    def remove_pad_data(self, ds):
        """
        Remove added pads around the data
        (B, pad_len_x, pad_len_y) -> (B, len_x, len_y)
        """
        pad_x = self.pad_dim[1]
        pad_y = self.pad_dim[2]
        len_x = self.original_shape[1]
        len_y = self.original_shape[2]

        ds = ds[:, pad_x[0] : pad_x[0] + len_x, pad_y[0] : pad_y[0] + len_y, ...]
        return ds

    def split_data(self, data2d):
        """
        Split 2d data to multiple 3d tiles,
        (len_x, len_y)
        --> (num_tile_x, patch_size, num_tile_y, patch_size)
        --> (num_tile_x * num_tile_y, patch_size, patch_size)
        """
        len_x, len_y, num_channels = self.padded_shape[1:]
        # (num_tile_x, patch_size, num_tile_y, patch_size)
        tiled_array = np.reshape(
            data2d,
            (
                -1,
                len_x // self.patch_size,
                self.patch_size,
                len_y // self.patch_size,
                self.patch_size,
                num_channels,
            ),
        )

        # (batch, num_tile_x, num_tile_y, patch_size, patch_size, channels)
        tiled_array = np.transpose(tiled_array, (0, 1, 3, 2, 4, 5))
        # (batch*num_tile_x*num_tile_y, patch_size, patch_size, patch_size, channels)
        tiled_array = np.reshape(
            tiled_array,
            (-1, self.patch_size, self.patch_size, num_channels),
        )
        return tiled_array

    def unsplit_data(self, tiles3):
        """
        (num_tile_x * num_tile_y, patch_size, patch_size)
        --> (num_tile_x, patch_size, num_tile_y, patch_size)
        --> (len_x, len_y)
        """
        len_x, len_y, num_channels = self.padded_shape[1:]
        # (batch, num_tile_x, num_tile_y, num_tile_z, patch_size, patch_size, patch_size, channels)
        data = np.reshape(
            tiles3,
            (
                -1,
                len_x // self.patch_size,
                len_y // self.patch_size,
                self.patch_size,
                self.patch_size,
                num_channels,
            ),
        )
        # (batch, num_tile_x, patch_size, num_tile_y, patch_size, num_tile_z, patch_size, channels)
        data = np.transpose(data, (0, 1, 3, 2, 4, 5))
        data = np.reshape(data, (-1, len_x, len_y, num_channels))
        return data


class Padder3D(BasePadder):
    def __init__(self, patch_size, original_shape):
        self.padded_shape = ()
        self.patch_size = patch_size
        self.original_shape = original_shape
        self.padded_shape = self.calculate_padded_data_shape(self.original_shape)
        self.pad_dim = self.calculate_padding_dim(
            self.padded_shape, self.original_shape
        )

    def remove_pad_data(self, ds):
        """
        Remove added pads around the data
        (B, pad_len_x, pad_len_y, pad_len_z) -> (B, len_x, len_y, len_z)
        """
        pad_x = self.pad_dim[1]
        pad_y = self.pad_dim[2]
        pad_z = self.pad_dim[3]
        len_x = self.original_shape[1]
        len_y = self.original_shape[2]
        len_z = self.original_shape[3]

        ds = ds[
            :,
            pad_x[0] : pad_x[0] + len_x,
            pad_y[0] : pad_y[0] + len_y,
            pad_z[0] : pad_z[0] + len_z,
            ...,
        ]
        return ds

    def split_data(self, data3d):
        """
        Split 3d data to multiple 3d tiles,
        (batch, len_x, len_y, len_z, channels)
        --> (batch, num_tile_x, patch_size, num_tile_y, patch_size, num_tile_z, patch_size, channels)
        --> (batch*num_tile_x * num_tile_y * num_tile_z, patch_size, patch_size, patch_size, channels)
        """
        len_x, len_y, len_z, num_channels = self.padded_shape[1:]
        # (batch, num_tile_x, patch_size, num_tile_y, patch_size, num_tile_z, patch_size, channels)
        tiled_array = np.reshape(
            data3d,
            (
                -1,
                len_x // self.patch_size,
                self.patch_size,
                len_y // self.patch_size,
                self.patch_size,
                len_z // self.patch_size,
                self.patch_size,
                num_channels,
            ),
        )
        tiled_array = np.transpose(tiled_array, (0, 1, 3, 5, 2, 4, 6, 7))
        tiled_array = np.reshape(
            tiled_array,
            (-1, self.patch_size, self.patch_size, self.patch_size, num_channels),
        )
        return tiled_array

    def unsplit_data(self, tiles4):
        """
        (batch*num_tile_x * num_tile_y * num_tile_z, patch_size, patch_size, patch_size, channels)
        --> (batch, num_tile_x, patch_size, num_tile_y, patch_size, num_tile_z, patch_size, channels)
        --> (batch, len_x, len_y, len_z, channels)
        """
        len_x, len_y, len_z, num_channels = self.padded_shape[1:]
        # (batch, num_tile_x, num_tile_y, num_tile_z, patch_size, patch_size, patch_size, channels)
        data = np.reshape(
            tiles4,
            (
                -1,
                len_x // self.patch_size,
                len_y // self.patch_size,
                len_z // self.patch_size,
                self.patch_size,
                self.patch_size,
                self.patch_size,
                num_channels,
            ),
        )

        data = np.transpose(data, (0, 1, 4, 2, 5, 3, 6, 7))
        data = np.reshape(data, (-1, len_x, len_y, len_z, num_channels))
        return data
