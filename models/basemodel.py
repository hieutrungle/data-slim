import torch
import torch.nn as nn
import abc
from torchinfo import summary
from models.custom_layers import (
    cus_blocks,
    cus_layers,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, model_type=None, name=None):
        super().__init__()
        self.name = name
        self.model_type = model_type

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

        self._encoding_stack = cus_blocks.EncodingStack(
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
        self._conv0 = cus_layers.Conv2dTransposeSame(
            latent_dim,
            num_channels,
            kernel_size=3,
            stride=1,
        )
        self._decoding_stack = cus_blocks.DecodingStack(
            num_channels,
            post_num_channels * 2,
            num_residual_blocks,
            kernel_size=3,
            stride=2,
        )

    def forward(self, x):
        x = self._conv0(x)
        x = self._decoding_stack(x)
        return x


class TransformerEncoder(BaseModel):
    """
    Transformer Encoder for text processing
    """

    def __init__(
        self, embed_dim, num_heads, num_blocks, dropout=0, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.name = name
        self.num_blocks = num_blocks

        self._layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            self._layers.append(
                cus_blocks.TransformerEncodingBlock(embed_dim, num_heads, dropout)
            )

    def forward(self, x, mask=None):
        for i in range(self.num_blocks):
            x = self._layers[i](x, x, x, mask)
        return x


class AttentionEncoder(BaseModel):
    """
    Attention Stack for image or high-dimensional data (e.g 3D) processing
    """

    def __init__(self, channels, num_heads, num_blocks, dropout=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.name = name
        self.num_blocks = num_blocks

        self._layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            self._layers.append(
                cus_blocks.AttentionEncodingBlock(channels, num_heads, dropout)
            )

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self._layers[i](x)
        return x


class Indentity(BaseModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def forward(self, x):
        return x


if __name__ == "__main__":
    patch_size = 64
    patch_depth = -1  # -1 means no depth
    patch_channels = 1
    pre_num_channels = 8
    num_channels = 32
    latent_dim = 64
    num_embeddings = 128
    num_residual_blocks = 3
    num_transformer_blocks = 2
    num_heads = 4
    dropout = 0.0
    ema_decay = 0.99
    commitment_cost = 0.25
    name = "Compressor"
    post_num_channels = pre_num_channels
    input_size = (1, 1, 64, 64)
    x = torch.normal(0, 1, input_size).to(DEVICE)
    model = Encoder(
        patch_channels, pre_num_channels, num_channels, latent_dim, num_residual_blocks
    ).to(DEVICE)
    summary(model, x.shape, depth=3, verbose=1)
    y = model(x)

    # model = Decoder(
    #     in_channels, post_num_channels, num_channels, latent_dim, num_residual_blocks
    # )
    # print()
    # summary(model, y.shape, depth=3, verbose=1)

    # model = cus_blocks.AttentionBlock(latent_dim, num_heads)
    # print()
    # summary(model, y.shape, depth=3, verbose=1)
    # z = model(y)

    model = AttentionEncoder(latent_dim, num_heads, num_transformer_blocks)
    print()
    summary(model, y.shape, depth=5, verbose=1)
    z = model(y)
