import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F

"""Warning: All Conv Transpose layers only work for odd kernel size"""
# Based on many comments in https://github.com/pytorch/pytorch/issues/3867


def _ntuple(n):
    """Copied from PyTorch since it's not importable as an internal function
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/utils.py#L6
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triplet = _ntuple(3)


class Conv1dSame(nn.Module):
    """Manual 1d convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer
        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _single(kernel_size)
        dilation_ = _single(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    @property
    def reversed_padding_repeated_twice(self):
        return self._reversed_padding_repeated_twice

    @reversed_padding_repeated_twice.setter
    def reversed_padding_repeated_twice(self, value):
        self._reversed_padding_repeated_twice = value

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are (N, C, ...) convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        padded = F.pad(x, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv1dTransposeSame(nn.Module):
    """Manual 1d transpose convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer
        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        output_padding = 1 if not return_odd else 0
        output_padding = 0 if stride <= 1 else output_padding

        # Setup internal representations
        kernel_size_ = _single(kernel_size)
        dilation_ = _single(dilation)
        self._padding = [0] * len(kernel_size_)
        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = (d * (k - 1)) // 2
            self._padding[i] = total_padding
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=self._padding,
            output_padding=output_padding,
            bias=bias,
            **kwargs,
        )

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are ``(N, C, ...)`` convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        return self.conv_transpose(x)


class Conv2dSame(nn.Module):
    """Manual 2d convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer
        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    @property
    def reversed_padding_repeated_twice(self):
        return self._reversed_padding_repeated_twice

    @reversed_padding_repeated_twice.setter
    def reversed_padding_repeated_twice(self, value):
        self._reversed_padding_repeated_twice = value

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are (N, C, ...) convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        padded = F.pad(x, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv2dTransposeSame(nn.Module):
    """Manual 2d transpose convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer
        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        output_padding = 1 if not return_odd else 0
        output_padding = 0 if stride <= 1 else output_padding

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._padding = [0] * len(kernel_size_)
        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = (d * (k - 1)) // 2
            self._padding[i] = total_padding
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=self._padding,
            output_padding=output_padding,
            bias=bias,
            **kwargs,
        )

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are ``(N, C, ...)`` convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        return self.conv_transpose(x)


class Conv3dSame(nn.Module):
    """Manual 3d convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer
        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            **kwargs,
        )

        # Setup internal representations
        kernel_size_ = _triplet(kernel_size)
        dilation_ = _triplet(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    @property
    def reversed_padding_repeated_twice(self):
        return self._reversed_padding_repeated_twice

    @reversed_padding_repeated_twice.setter
    def reversed_padding_repeated_twice(self, value):
        self._reversed_padding_repeated_twice = value

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are ``(N, C, ...)`` convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        padded = F.pad(x, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv3dTransposeSame(nn.Module):
    """Manual 3d tranpose convolution with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
        bias=True,
        **kwargs,
    ):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        output_padding = 1 if not return_odd else 0
        output_padding = 0 if stride <= 1 else output_padding

        # Setup internal representations
        kernel_size_ = _triplet(kernel_size)
        dilation_ = _triplet(dilation)
        self._padding = [0] * len(kernel_size_)
        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(
            dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)
        ):
            total_padding = (d * (k - 1)) // 2
            self._padding[i] = total_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=self._padding,
            output_padding=output_padding,
            bias=bias,
            **kwargs,
        )

    def forward(self, x):
        """Setup padding so same spatial dimensions are returned
        All shapes (input/output) are ``(N, C, ...)`` convention
        :param torch.Tensor data:
        :return torch.Tensor:
        """
        return self.conv_transpose(x)


if __name__ == "__main__":
    pass
