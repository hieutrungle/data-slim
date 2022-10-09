import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F


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
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
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

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv1dTransposeSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
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
            **kwargs,
        )

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        return self.conv_transpose(imgs)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
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

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv2dTransposeSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
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
            **kwargs,
        )

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        return self.conv_transpose(imgs)


class Conv3dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs
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

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


class Conv3dTransposeSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        return_odd=False,
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
            **kwargs,
        )

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        return self.conv_transpose(imgs)


if __name__ == "__main__":
    pass
