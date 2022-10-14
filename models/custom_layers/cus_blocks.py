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

        self.multihead_attention = MultiheadAttention(
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

        self.norm = normalization(channels)
        self.qkv = cus_layers.Conv1dSame(
            channels, channels * 3, kernel_size=channels // 16, stride=1
        )
        self.attention = QKVAttention()
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


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
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


class AttentionEncodingBlock(nn.Module):
    """
    Attention for high dimensional data, such as images, 3D data, etc.
    Attention + Conv1D
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

        self.norm = normalization(channels)
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
