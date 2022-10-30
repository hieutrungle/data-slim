import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import math


def _convert_if_not_numpy(x, dtype=np.float32):
    """Convert the input `x` to a numpy array of type `dtype`.
    Args:
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    Returns:
        A numpy array.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=dtype)
    return x


class SlidingWindow:
    """
    The procedure of sliding window is as follows:
    1. Obtain necessary parameters
    -- Get a list containing coordinates of the top-left corner of each window: coors = get_window_coors()
    -- Get the total number of windows: num_windows = get_total_num_windows()

    2. Process data
    -- padded = pad_data(data) # if padding is True

    3. Get data window
    main_idx in range(num_window_per_data * num_data) # Loop through all indices of all data
        window_idx = loop_idx % num_window_per_data # Get index of current data based on the main_idx of all data
        (y, x) = get_coor_given_index(coors, window_idx) # Get the top-left coordinate of the current window
        window = padded[B, y:y+kernel.h, x:x+kernel.w, :] # extract data window
    """

    def __init__(self, kernel_size, stride, padding=False):
        kernel_size = _convert_if_not_numpy(kernel_size, dtype=np.int16)
        stride = _convert_if_not_numpy(stride, dtype=np.int16)
        if len(kernel_size.shape) == 0:
            self.kernel_size = np.array([kernel_size, kernel_size])
        elif len(kernel_size.shape) == 1:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __repr__(self):
        return f"SlidingWindow(kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})"

    def __str__(self):
        return self.__repr__()

    def get_num_windows_each_dim_without_padding(self, image_size):
        # return number of windows without padding in all coordinate directions
        num_windows_each_dim = [
            (size - k_size) // self.stride + 1
            for size, k_size in zip(image_size, self.kernel_size)
        ]
        tmp = []
        for num_window, size, k_size in zip(
            num_windows_each_dim, image_size, self.kernel_size
        ):
            num_window = 0 if size - k_size < 0 else num_window
            tmp.append(num_window)
        num_windows_each_dim = tmp
        return np.array(num_windows_each_dim)

    def get_num_windows_each_dim_with_padding(self, data_size):
        # return number of windows with padding in all coordinate directions
        num_windows_each_dim = [math.ceil(size / self.stride) for size in data_size]
        return np.array(num_windows_each_dim)

    def get_num_windows_each_dim(self, image_size):
        if self.padding:
            return self.get_num_windows_each_dim_with_padding(image_size)
        else:
            return self.get_num_windows_each_dim_without_padding(image_size)

    def get_total_num_windows(self, image_size):
        return np.prod(self.get_num_windows_each_dim(image_size))

    def get_padding(self, image_size):
        # return padding in the coordinate directions
        num_windows_each_dim = self.get_num_windows_each_dim(image_size)
        padded_shape = []
        for num_windows, k_size, i_size in zip(
            num_windows_each_dim, self.kernel_size, image_size
        ):
            pad = (num_windows - 1) * self.stride + k_size - i_size
            pad = np.maximum(0, pad)
            padded_shape.append(pad)
        return padded_shape

    def calc_padding_dim(self, padded_shape, original_shape):
        pad_dim = [new - old for new, old in zip(padded_shape, original_shape)]
        pad_dim = [(0, i) for i in pad_dim]
        return pad_dim

    def pad_data(self, data):
        # data: a tensor of shape (batch_size, height, width, channels)
        # return: a tensor of shape (batch_size, padded_height, padded_width, channels)
        if self.padding:
            pad_dim = np.array([0, *self.get_padding(data.shape[1:-1]), 0])
            padded_shape = np.array(data.shape) + pad_dim
            pad_dim = self.calc_padding_dim(padded_shape, data.shape)
            ds = np.pad(data, (pad_dim), mode="wrap")
        else:
            ds = data
        return ds

    def get_window_coors_with_padding(self, image_shape):
        # data: a tensor of shape (batch_size, height, width, channels)
        # return: coors of windows, a tensor of shape (batch_size, num_windows, num_coors)
        num_windows_each_dim = self.get_num_windows_each_dim_with_padding(
            image_shape[1:-1]
        )
        coors = [
            np.arange(0, size, self.stride)
            for size in num_windows_each_dim * self.stride
        ]
        return coors

    def get_window_coors_without_padding(self, image_shape):
        # data: a tensor of shape (batch_size, height, width, channels)
        # return: coors of windows, a tensor of shape (batch_size, num_windows, num_coors)
        num_windows_each_dim = self.get_num_windows_each_dim_without_padding(
            image_shape[1:-1]
        )
        coors = [
            np.arange(0, size, self.stride)
            for size in num_windows_each_dim * self.stride
        ]
        return coors

    def get_window_coors(self, image_shape):
        if self.padding:
            return self.get_window_coors_with_padding(image_shape)
        else:
            return self.get_window_coors_without_padding(image_shape)

    def get_coor_given_index(self, coors, flatten_index):
        # get actual coordinate given a flatten index
        shape = [len(coor) for coor in coors]
        coor = self.reverse_flatten_indices(shape, flatten_index)
        coor = [coor_axis[i] for coor_axis, i in zip(coors, coor)]
        return np.array(coor)

    def reverse_flatten_indices(self, shape, flatten_index):
        # get the actual index from flatten_index
        full_index = []
        for len in reversed(shape):
            i = flatten_index % len
            flatten_index = flatten_index // len
            full_index.append(i)
        full_index.reverse()
        return tuple(full_index)

    def get_window_with_coordinate(self, data, coor):
        """_summary_
        get window with coordinate
        data: a tensor of shape (batch_size, height, width, channels)
        coor: a tensor containing the coordinate of the window, (y, x)
        """
        coor = np.array(coor)
        window = data[
            0,
            coor[0] : coor[0] + self.kernel_size[0],
            coor[1] : coor[1] + self.kernel_size[1],
            :,
        ]
        return window

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, "=", value)

    def get_attributes(self):
        return self.__dict__


if __name__ == "__main__":
    pass
