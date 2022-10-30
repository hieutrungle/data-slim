import collections
from itertools import repeat
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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


def scaled_dot_product(q, k, v, mask=None, dropout_module=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention_weights = F.softmax(attn_logits, dim=-1)
    if dropout_module is not None:
        attention_weights = dropout_module(attention_weights)
    values = torch.matmul(attention_weights, v)
    return values, attention_weights


class MultiheadAttention(nn.Module):
    """
    Multihead Attention module from "Attention is All You Need"
    This attention is for text processing, not for image processing.
    """

    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, name=None, **kwargs
    ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."
        self.name = name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.bias = bias

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        if self.bias:
            nn.init.constant_(self.q_linear.bias.data, 0.0)
            nn.init.constant_(self.v_linear.bias.data, 0.0)
            nn.init.constant_(self.k_linear.bias.data, 0.0)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias.data, 0)

    def forward(self, q, k, v, mask=None, return_attention=True):
        batch_size = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim)

        # transpose to get dimensions batch_size * num_heads * sl * head_dim

        k = k.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Determine value outputs
        values, attention_weights = scaled_dot_product(
            q, k, v, mask=mask, dropout_module=self.dropout
        )
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, -1, self.embed_dim)
        output = self.out(values)

        if return_attention:
            return output, attention_weights
        else:
            return output


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention for spatial data (e.g. images).
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = torch.div(qkv.shape[1], 3, rounding_mode="floor")
        # ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / torch.sqrt(torch.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


if __name__ == "__main__":
    pass
