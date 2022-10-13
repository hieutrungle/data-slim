import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

try:
    from . import basemodel
    from .custom_layers import (
        cus_blocks,
        embedding,
        patcher,
        vector_quantizer,
        preprocessors,
    )
except:
    import basemodel
    from custom_layers import (
        cus_blocks,
        embedding,
        patcher,
        vector_quantizer,
        preprocessors,
    )


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
        self.encoder = basemodel.Encoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_blocks,
            name="Encoder",
        )

        if ema_decay > 0.0:
            self.vq_layer = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, ema_decay
            )
        else:
            self.vq_layer = vector_quantizer.VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )

        self.decoder = basemodel.Decoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_blocks,
            name="Decoder",
        )
        print(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)
        return y

    def _decode(self, quantized):
        """Decodes data."""
        x_hat = self.decoder(quantized)
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
    in_channels = 1
    pre_num_channels = post_num_channels = 32
    num_channels = 96
    latent_dim = 128
    num_embeddings = 256
    num_residual_blocks = 3
    in_shape = (1, 1, 128, 128)
    x = torch.normal(0, 1, in_shape)
    model = VQCPVAE(
        in_shape,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_blocks,
        commitment_cost=0.25,
        decay=0.99,
        name=None,
    )
    model.set_standardizer_layer(1, 1)
    summary(model, x.shape, col_width=25, depth=2, verbose=1)

    encoding_indices, y_shape = model.compress(x)
    print(f"encoding_indices.shape = {encoding_indices.shape}; y_shape: {y_shape}")
    x_hat = model.decompress(encoding_indices, y_shape)
    print(f"x_hat.shape = {x_hat.shape}")
