import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utils import logger
from models import basemodel
from models.custom_layers import (
    vector_quantizer,
    preprocessors,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Encoder, self).__init__()
        self._conv_0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_1 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_0(inputs)
        x = F.relu(x)

        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        out_channels=1,
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_0 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)
        x = self._conv_trans_0(x)
        x = F.relu(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQCPVAE(basemodel.BaseModel):
    """Vector-quantized Compression Variational Autoencoder"""

    def __init__(
        self,
        patch_size,
        patch_depth,
        patch_channels,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_blocks,
        num_transformer_blocks,
        num_heads,
        dropout,
        ema_decay,
        commitment_cost,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if patch_depth <= 0:
            self.input_shape = [1, patch_channels, patch_size, patch_size]
        else:
            self.input_shape = [1, patch_channels, patch_size, patch_size, patch_size]

        data_channels = self.input_shape[1]  # in_shape = (B, C, H, W)
        embedding_dim = latent_dim
        self.pre_num_channels = pre_num_channels
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.num_residual_blocks = num_residual_blocks
        self.vq_weight = 1.0

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self._encoder = Encoder(
            data_channels, num_channels, num_residual_blocks, num_channels
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )

        if ema_decay > 0.0:
            self.vq_layer = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, ema_decay
            )
        else:
            self.vq_layer = vector_quantizer.VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )

        self._decoder = Decoder(
            embedding_dim,
            num_channels,
            num_residual_blocks,
            num_channels,
            data_channels,
        )

        logger.log(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        y = self._encoder(x)
        y = self._pre_vq_conv(y)
        return y

    def _decode(self, quantized):
        """Decodes data."""
        x_hat = self._decoder(quantized)
        x_hat = self.data_preprocessor(x_hat, normalize=0)
        return x_hat

    def forward(self, x):
        y = self._encode(x)
        loss, quantized, perplexity, _ = self.vq_layer(y)
        x_hat = self._decode(quantized)
        return loss, x_hat, perplexity

    def set_standardizer_layer(self, mean, variance, eta=1e-6):
        self.data_preprocessor = preprocessors.Standardizer(
            mean,
            variance,
            eta,
            name="data_processor",
        )

    def compress(self, x):
        """Compresses data."""
        y = self._encode(x)  # (B, C, H, W)
        y = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        flattened = torch.reshape(y, [-1, self.latent_dim])

        # (BxHxW, 1)
        encoding_indices = self.vq_layer.get_code_indices(flattened)
        # Preserve spatial shapes of both image and latents.
        y_shape = y.shape[1:]
        return encoding_indices, y_shape

    def decompress(self, encoding_indices, y_shape):
        """Decompresses an image."""
        # (BxHxW, num_embeddings)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.vq_layer._num_embeddings,
            device=encoding_indices.device,
        )
        encodings.scatter_(dim=1, index=encoding_indices, value=1)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = torch.matmul(encodings, self.vq_layer._embedding.weight)
        # Unflatten quantized, (B * H * W, C) -> (B, H, W, C)
        quantized = torch.reshape(quantized, (-1, y_shape[0], y_shape[1], y_shape[2]))
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        x_hat = self._decode(quantized)
        return x_hat


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

    model = VQCPVAE(
        patch_size,
        patch_depth,
        patch_channels,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_blocks,
        num_transformer_blocks,
        num_heads,
        dropout,
        ema_decay,
        commitment_cost,
    )
    print()
    summary(model, y.shape, depth=5, verbose=1)
    z = model(y)
