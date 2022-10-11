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


class TransformerEncodingBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.layernorm_0 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # (batch_size, target_seq_len, embed_dim)
        attention_output_0, _ = self.multihead_attention(q, k, v, mask)
        attention_output_0 = self.layernorm_0(attention_output_0 + q)

        # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.ffn(attention_output_0)
        # (batch_size, target_seq_len, embed_dim)
        attention_output_1 = self.layernorm_1(attention_output_0 + ffn_output)

        return attention_output_1
