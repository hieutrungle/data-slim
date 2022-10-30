import torch
import torch.nn as nn
import torch.nn.functional as F
from . import cus_layers

try:
    import gdn
    import models.custom_layers.cus_layers as cus_layers
except:
    from . import gdn
import numpy as np
import math


class DownSamplingBlock(nn.Module):
    """Downsampling block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

        self.gdn = gdn.GDN(c_in)
        self.conv0 = cus_layers.Conv2dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        x = self.gdn(x)
        x = self.conv0(x)
        return x


class UpSamplingBlock(nn.Module):
    """Upsampling block"""

    def __init__(
        self, c_in, c_out, kernel_size, stride, padding=0, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.conv1 = cus_layers.Conv2dTransposeSame(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x):
        x = self.igdn(x)
        x = self.conv1(x)
        return x


class DownSamplingResBlock2D(nn.Module):
    """Downsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4
        self.gdn = gdn.GDN(c_in)

        self.shortcut = cus_layers.Conv2dSame(
            c_in, c_out, kernel_size=1, stride=stride, bias=False
        )
        self.act = nn.GELU()
        self.conv0 = cus_layers.Conv2dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.conv1 = cus_layers.Conv2dSame(
            c_out, c_hidden, kernel_size=1, stride=1, bias=False
        )
        self.conv2 = cus_layers.Conv2dSame(
            c_hidden, c_out, kernel_size=1, stride=1, bias=False
        )
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.gdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        return x + x_shortcut


class UpSamplingResBlock2D(nn.Module):
    """Upsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.shortcut = cus_layers.Conv2dTransposeSame(
            c_in, c_out, kernel_size=1, stride=stride, bias=False
        )
        self.act = nn.GELU()
        self.conv0 = cus_layers.Conv2dTransposeSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.conv1 = cus_layers.Conv2dTransposeSame(
            c_out, c_hidden, kernel_size=1, stride=1, bias=False
        )
        self.conv2 = cus_layers.Conv2dTransposeSame(
            c_hidden, c_out, kernel_size=1, stride=1, bias=False
        )
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.igdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        return x + x_shortcut


class DownSamplingResBlock3D(nn.Module):
    """Downsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.gdn = gdn.GDN(c_in)
        self.shortcut = cus_layers.Conv3dSame(c_in, c_out, kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.conv1 = cus_layers.Conv3dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv2 = cus_layers.Conv3dSame(c_out, c_hidden, kernel_size=1, stride=1)
        self.conv3 = cus_layers.Conv3dSame(c_hidden, c_out, kernel_size=1, stride=1)
        self.cells = [self.conv1, self.conv2, self.act, self.conv3]

    def forward(self, x):
        x = self.gdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        return x


class UpSamplingResBlock3D(nn.Module):
    """Upsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.shortcut = cus_layers.Conv3dTransposeSame(
            c_in, c_out, kernel_size=1, stride=stride
        )
        self.act = nn.GELU()
        self.conv0 = cus_layers.Conv3dTransposeSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv1 = cus_layers.Conv3dTransposeSame(
            c_out, c_hidden, kernel_size=1, stride=1
        )
        self.conv2 = cus_layers.Conv3dTransposeSame(
            c_hidden, c_out, kernel_size=1, stride=1
        )
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.igdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        return x


class ForwardConv1d(nn.Module):
    """Forward Conv1d"""

    def __init__(
        self, c_in, c_out, c_hidden, kernel_size=1, stride=1, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.gdn = gdn.GDN(c_in)
        self.flatten = nn.Flatten()
        self.conv1d = cus_layers.Conv1dSame(
            c_hidden, c_out, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        # (B, C, ...)
        x = self.gdn(x)
        # (B, C)
        x = self.flatten(x)
        # (B, C, 1)
        x = torch.unsqueeze(x, axis=-1)
        x = self.conv1d(x)
        return x


class ForwardMLP(nn.Module):
    """Forward MLP"""

    def __init__(self, c_in, c_out, c_hidden, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.gdn = gdn.GDN(c_in)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(c_hidden, c_out)

    def forward(self, x):
        # (B, C, ...)
        x = self.gdn(x)
        # (B, C)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PreBlock(nn.Module):
    """Preprocessing block for the encoder."""

    def __init__(self, data_channels, pre_num_channels, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self._conv0 = cus_layers.Conv2dSame(
            data_channels, pre_num_channels, kernel_size=1
        )
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self._conv1 = cus_layers.Conv2dSame(
            pre_num_channels,
            pre_num_channels * 2,
            kernel_size=4,
            stride=1,
        )
        self._conv2 = cus_layers.Conv2dSame(
            pre_num_channels * 2,
            pre_num_channels * 2,
            kernel_size=3,
            stride=1,
        )

    def forward(self, x):
        x = self._conv0(x)
        x = self.act0(x)
        x = self._conv1(x)
        x = self.act1(x)
        x = self._conv2(x)
        return x


class PostBlock(nn.Module):
    """Postprocessing block for the decoder"""

    def __init__(self, post_num_channels, data_channels, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self._conv1 = cus_layers.Conv2dTransposeSame(
            post_num_channels * 2,
            post_num_channels,
            kernel_size=3,
            stride=1,
        )
        self._conv2 = cus_layers.Conv2dTransposeSame(
            post_num_channels,
            data_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        x = self.act0(x)
        x = self._conv1(x)
        x = self.act1(x)
        x = self._conv2(x)
        return x


class EncodingStack(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        num_residual_blocks,
        kernel_size=3,
        stride=2,
        name=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self._num_residual_blocks = num_residual_blocks
        self._layers = nn.ModuleList(
            [DownSamplingResBlock2D(c_in, c_out, kernel_size, stride)]
        )
        for _ in range(self._num_residual_blocks - 1):
            self._layers.append(
                DownSamplingResBlock2D(c_out, c_out, kernel_size, stride)
            )

    def forward(self, x):
        for i in range(self._num_residual_blocks):
            x = self._layers[i](x)
        return x


class DecodingStack(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        num_residual_blocks,
        kernel_size=3,
        stride=2,
        name=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self._num_residual_blocks = num_residual_blocks

        self._layers = nn.ModuleList()
        for _ in range(self._num_residual_blocks - 1):
            self._layers.append(UpSamplingResBlock2D(c_in, c_in, kernel_size, stride))
        self._layers.append(UpSamplingResBlock2D(c_in, c_out, kernel_size, stride))

    def forward(self, x):
        for i in range(self._num_residual_blocks):
            x = self._layers[i](x)
        return x


class TransformerEncodingBlock(nn.Module):
    """
    Transformer Encoding block for text processing
    """

    def __init__(self, embed_dim, num_heads, dropout=0, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.multihead_attention = cus_layers.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.layernorm_0 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        x = q
        attn_out, _ = self.multihead_attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.layernorm_0(x)

        # (batch_size, target_seq_len, embed_dim)
        linear_out = self.ffn(x)
        x = x + self.dropout(linear_out)
        # (batch_size, target_seq_len, embed_dim)
        attention_output_1 = self.layernorm_1(x)

        return attention_output_1


class AttentionBlock(nn.Module):
    """
    Attention for high dimensional data, such as images, 3D data, etc.
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = cus_layers.normalization(channels)
        self.qkv = cus_layers.Conv1dSame(
            channels, channels * 3, kernel_size=channels // 16, stride=1
        )
        self.attention = cus_layers.QKVAttention()
        self.proj_out = cus_layers.Conv1dSame(
            channels, channels, kernel_size=5, stride=1
        )

    def forward(self, x):
        b, c, *spatial = x.shape
        # [B, C, H*W]
        x = x.reshape(b, c, -1)
        # [B, 3C, H*W]
        qkv = self.qkv(self.norm(x))
        # [B, 3C, H*W] -> [B*num_head, 3C/num_head, H*W]
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class AttentionEncodingBlock(nn.Module):
    """
    Attention for high dimensional data, such as images, 3D data, etc.
    Attention + Linear + GELU + Linear
    """

    def __init__(self, channels, num_heads, dropout=0, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.num_heads = num_heads
        self.dropout = dropout

        self.multihead_attention = AttentionBlock(
            channels=channels,
            num_heads=num_heads,
        )
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            # cus_layers.Conv2dSame(channels, channels * 2, kernel_size=5, stride=1),
            nn.GELU(),
            # cus_layers.Conv2dSame(channels * 2, channels, kernel_size=5, stride=1),
            nn.Linear(channels * 2, channels),
        )

        self.norm = cus_layers.normalization(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, C, H, W)
        x = self.multihead_attention(x)
        # (B, C, H, W)
        linear_out = torch.permute(x, (0, 2, 3, 1))
        linear_out = self.ffn(linear_out)
        linear_out = torch.permute(linear_out, (0, 3, 1, 2))

        x = x + self.dropout(linear_out)
        # (B, C, H, W)
        x = self.norm(x)

        return x
