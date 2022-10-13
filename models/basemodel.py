import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import torch
import torch.nn as nn
import abc
from torchinfo import summary

try:
    from custom_layers import cus_blocks, cus_layers
except:
    from .custom_layers import cus_blocks, cus_layers


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def summarize_model(self):
        total_params = sum(param.numel() for param in self.parameters())

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        totalParams = trainable_params + non_trainable_params
        print(
            f"\n================================================================="
            f"\n                     {self.name} Summary"
            f"\nTotal params: {totalParams:,}"
            f"\nTrainable params: {trainable_params:,}"
            f"\nNon-trainable params: {non_trainable_params:,}"
            f"\n_________________________________________________________________\n"
        )


class EncodingStack(nn.Module):
    def __init__(self, c_in, c_out, num_residual_blocks, kernel_size=3, stride=2):
        super().__init__()
        self._num_residual_blocks = num_residual_blocks
        self._layers = nn.ModuleList(
            [cus_blocks.DownSamplingResBlock2D(c_in, c_out, kernel_size, stride)]
        )
        for _ in range(self._num_residual_blocks - 1):
            self._layers.append(
                cus_blocks.DownSamplingResBlock2D(c_out, c_out, kernel_size, stride)
            )

    def forward(self, x):
        for i in range(self._num_residual_blocks):
            x = self._layers[i](x)
        return x


class DecodingStack(nn.Module):
    def __init__(self, c_in, c_out, num_residual_blocks, kernel_size=3, stride=2):
        super().__init__()
        self._num_residual_blocks = num_residual_blocks

        self._layers = nn.ModuleList()
        for _ in range(self._num_residual_blocks - 1):
            self._layers.append(
                cus_blocks.UpSamplingResBlock2D(c_in, c_in, kernel_size, stride)
            )
        self._layers.append(
            cus_blocks.UpSamplingResBlock2D(c_in, c_out, kernel_size, stride)
        )

    def forward(self, x):
        for i in range(self._num_residual_blocks):
            x = self._layers[i](x)
        return x


class Encoder(BaseModel):
    def __init__(
        self,
        data_channels,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_residual_blocks,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self._conv0 = cus_layers.Conv2dSame(
            data_channels, pre_num_channels, kernel_size=1
        )
        self.act = nn.GELU()
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
        self._encoding_stack = EncodingStack(
            pre_num_channels * 2,
            num_channels,
            num_residual_blocks=num_residual_blocks,
            kernel_size=3,
            stride=2,
        )
        self._conv3 = cus_layers.Conv2dSame(
            num_channels,
            latent_dim,
            kernel_size=3,
            stride=1,
        )

    def forward(self, x):
        x = self._conv0(x)
        x = self.act(x)
        x = self._conv1(x)
        x = self.act(x)
        x = self._conv2(x)
        x = self._encoding_stack(x)
        x = self._conv3(x)
        return x


class Decoder(BaseModel):
    def __init__(
        self,
        data_channels,
        post_num_channels,
        num_channels,
        latent_dim,
        num_residual_blocks,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.act = nn.GELU()
        self._conv0 = cus_layers.Conv2dTransposeSame(
            latent_dim,
            num_channels,
            kernel_size=3,
            stride=1,
        )
        self._decoding_stack = DecodingStack(
            num_channels,
            post_num_channels * 2,
            num_residual_blocks,
            kernel_size=3,
            stride=2,
        )
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
        x = self._conv0(x)
        x = self._decoding_stack(x)
        x = self.act(x)
        x = self._conv1(x)
        x = self.act(x)
        x = self._conv2(x)
        return x


class TransformerEncoder(BaseModel):
    def __init__(
        self, embed_dim, num_heads, num_layers, dropout=0, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.name = name
        self.num_layers = num_layers

        self._layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self._layers.append(
                cus_blocks.TransformerEncodingBlock(embed_dim, num_heads, dropout)
            )

    def forward(self, x, mask=None):
        for i in range(self.num_layers):
            x = self._layers[i](x, x, x, mask)
        return x


if __name__ == "__main__":
    in_channels = 1
    pre_num_channels = post_num_channels = 32
    num_channels = 96
    num_residual_blocks = 3
    latent_dim = 128
    input_size = (1, 1, 128, 128)
    x = torch.normal(0, 1, input_size)
    model = Encoder(
        in_channels, pre_num_channels, num_channels, latent_dim, num_residual_blocks
    )
    summary(model, x.shape, col_width=30, depth=3, verbose=1)
    y = model(x)

    model = Decoder(
        in_channels, post_num_channels, num_channels, latent_dim, num_residual_blocks
    )
    print()
    summary(model, y.shape, col_width=30, depth=3, verbose=1)
