import torch
from utils import logger
from models import basemodel
from models.custom_layers import vector_quantizer, preprocessors, cus_blocks


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
        model_type,
        name=None,
        **kwargs,
    ):
        super().__init__(model_type=model_type, name=name, **kwargs)
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
        self.pre_block = cus_blocks.PreBlock(
            data_channels, pre_num_channels, name="pre_block"
        )
        self.encoder = basemodel.Encoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_blocks,
            name="Encoder",
        )
        self.final_conv_encoder = torch.nn.Conv2d(latent_dim, latent_dim, 1, 1)

        if ema_decay > 0.0:
            self.vq_layer = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, ema_decay
            )
        else:
            self.vq_layer = vector_quantizer.VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )

        self.first_conv_decoder = torch.nn.ConvTranspose2d(latent_dim, latent_dim, 1, 1)
        self.decoder = basemodel.Decoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_blocks,
            name="Decoder",
        )
        self.post_block = cus_blocks.PostBlock(
            pre_num_channels, data_channels, name="post_block"
        )

        if num_transformer_blocks > 0:
            self.forward_attention = basemodel.AttentionEncoder(
                channels=latent_dim,
                num_heads=num_heads,
                num_blocks=num_transformer_blocks,
                dropout=dropout,
                name=f"forward_attention",
            )
            self.backward_attention = basemodel.AttentionEncoder(
                channels=latent_dim,
                num_heads=num_heads,
                num_blocks=num_transformer_blocks,
                name=f"backward_attention",
            )

        else:
            self.forward_attention = basemodel.Indentity(name=f"forward_attention")
            self.backward_attention = basemodel.Indentity(name=f"backward_attention")

        logger.log(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        x = self.pre_block(x)
        y = self.encoder(x)
        y = self.forward_attention(y)
        y = self.final_conv_encoder(y)
        return y

    def _decode(self, quantized):
        """Decodes data."""
        y_hat = self.first_conv_decoder(quantized)
        y_hat = self.backward_attention(y_hat)
        x_hat = self.decoder(y_hat)
        x_hat = self.post_block(x_hat)
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
    pass
