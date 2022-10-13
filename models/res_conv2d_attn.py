import torch
from utils import logger

try:
    from . import basemodel
    from .custom_layers import (
        embedding,
        patcher,
        vector_quantizer,
        preprocessors,
    )
except:
    import basemodel
    from custom_layers import (
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

        if num_transformer_blocks > 0:

            # Forward Attention
            self.conv_shape = self.encoder(torch.zeros(*self.input_shape)).shape[1:]
            self.mini_patch_size = self.conv_shape[1] // 8

            self.forward_patcher = patcher.Patcher2d(
                self.mini_patch_size, name="forward_patcher"
            )
            self.forward_inverse_patcher = patcher.InversePatcher2d(
                self.mini_patch_size, self.conv_shape, name="forward_inverse_patcher"
            )
            # split the latent space into patches for attention
            # patches_shape = (B, num_patches, patch_dim)
            self.patches_shape = self.forward_patcher(
                torch.zeros((1, *self.conv_shape))
            ).shape
            assert (
                self.patches_shape[1] % num_heads == 0
            ), "num_heads must divide num_patches"
            self.num_heads = num_heads
            self.forward_patch_embedding = embedding.PatchEmbedding(
                c_in=self.patches_shape[2],
                projection_dim=self.patches_shape[2],
                num_patches=self.patches_shape[1],
                name="fordward_patch_embedding",
            )
            self.forward_attention = basemodel.TransformerEncoder(
                embed_dim=self.patches_shape[2],
                num_heads=self.num_heads,
                num_layers=num_transformer_blocks,
                dropout=dropout,
                name=f"forward_attention",
            )

            # Backward attention
            self.backward_patcher = patcher.Patcher2d(
                self.mini_patch_size, name="backward_patcher"
            )
            self.backward_inverse_patcher = patcher.InversePatcher2d(
                self.mini_patch_size, self.conv_shape, name="backward_inverse_patcher"
            )
            self.backward_patch_embedding = embedding.PatchEmbedding(
                c_in=self.patches_shape[2],
                projection_dim=self.patches_shape[2],
                num_patches=self.patches_shape[1],
                name="backward_patch_embedding",
            )
            self.backward_attention = basemodel.TransformerEncoder(
                embed_dim=self.patches_shape[2],
                num_heads=self.num_heads,
                num_layers=num_transformer_blocks,
                name=f"backward_attention",
            )

        else:
            self.forward_patcher = basemodel.Indentity(name="forward_patcher")
            self.forward_inverse_patcher = basemodel.Indentity(
                name="forward_inverse_patcher"
            )
            self.forward_patch_embedding = basemodel.Indentity(
                name="fordward_patch_embedding"
            )
            self.forward_attention = basemodel.Indentity(name=f"forward_attention")

            # Backward attention
            self.backward_patcher = basemodel.Indentity(name="backward_patcher")
            self.backward_inverse_patcher = basemodel.Indentity(
                name="backward_inverse_patcher"
            )
            self.backward_patch_embedding = basemodel.Indentity(
                name="backward_patch_embedding"
            )
            self.backward_attention = basemodel.Indentity(name=f"backward_attention")

        logger.log(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)

        y_patches = self.forward_patcher(y)
        y_embedding = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_attention(y_embedding, mask=None)
        y_attn = self.forward_inverse_patcher(y_attn)
        return y_attn

    def _decode(self, quantized):
        """Decodes data."""
        y_hat_patches = self.backward_patcher(quantized)
        y_hat_embedding = self.backward_patch_embedding(y_hat_patches)
        y_hat_attn = self.backward_attention(y_hat_embedding, mask=None)
        y_hat = self.backward_inverse_patcher(y_hat_attn)

        x_hat = self.decoder(y_hat)
        x_hat = self.data_preprocessor(x_hat, normalize=0)
        return x_hat

    def forward(self, x):
        y_attn = self._encode(x)
        loss, quantized, perplexity, _ = self.vq_layer(y_attn)
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
    pass
