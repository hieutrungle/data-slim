import torch
from utils import logger
from models import basemodel
from models.custom_layers import vector_quantizer, preprocessors, cus_blocks, cus_layers

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


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
            data_channels, pre_num_channels, name="PreBlock"
        )

        self.encoder_stack_0 = cus_blocks.EncodingStack(
            pre_num_channels * 2,
            num_channels,
            num_residual_blocks,
            kernel_size=5,
            stride=2,
            name="Encoder_0",
        )
        self.final_conv_encoder_0 = cus_layers.Conv2dSame(
            num_channels,
            latent_dim,
            kernel_size=3,
            stride=1,
        )

        self.encoder_stack_1 = cus_blocks.EncodingStack(
            num_channels,
            num_channels * 2,
            num_residual_blocks=1,
            kernel_size=5,
            stride=2,
            name="Encoder_1",
        )
        self.final_conv_encoder_1 = cus_layers.Conv2dSame(
            num_channels * 2,
            latent_dim,
            kernel_size=3,
            stride=1,
        )

        if ema_decay > 0.0:
            self.vq_layer_0 = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, ema_decay
            )
            self.vq_layer_1 = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, ema_decay
            )
        else:
            self.vq_layer_0 = vector_quantizer.VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
            self.vq_layer_1 = vector_quantizer.VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )

        self.first_conv_decoder_yz = cus_layers.Conv2dTransposeSame(
            latent_dim,
            latent_dim,
            kernel_size=5,
            stride=2,
        )

        self.first_conv_decoder_1 = cus_layers.Conv2dTransposeSame(
            latent_dim,
            num_channels * 2,
            kernel_size=3,
            stride=1,
        )
        self.decoding_stack_1 = cus_blocks.DecodingStack(
            num_channels * 2,
            num_channels,
            num_residual_blocks=1,
            kernel_size=3,
            stride=2,
        )

        self.first_conv_decoder_0 = cus_layers.Conv2dTransposeSame(
            latent_dim,
            num_channels,
            kernel_size=3,
            stride=1,
        )
        self.decoding_stack_0 = cus_blocks.DecodingStack(
            num_channels,
            pre_num_channels * 2,
            num_residual_blocks,
            kernel_size=3,
            stride=2,
        )
        self.post_block = cus_blocks.PostBlock(
            pre_num_channels, data_channels, name="PostBlock"
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
        y = self.encoder_stack_0(x)
        y_enc = self.final_conv_encoder_0(y)
        z = self.encoder_stack_1(y)
        z_enc = self.final_conv_encoder_1(z)
        return y_enc, z_enc

    def _decode(self, y_quantized, z_quantized):
        """Decodes data."""
        z_hat = self.first_conv_decoder_1(z_quantized)
        z_hat = self.decoding_stack_1(z_hat)
        y_hat = self.first_conv_decoder_0(y_quantized)
        y_hat = z_hat + y_hat
        y_hat = self.decoding_stack_0(y_hat)
        x_hat = self.post_block(y_hat)
        x_hat = self.data_preprocessor(x_hat, normalize=0)
        return x_hat

    def forward(self, x):
        y, z = self._encode(x)
        z_loss, z_quantized, z_perplexity, _ = self.vq_layer_1(z)
        yz = self.first_conv_decoder_yz(z_quantized)
        y = yz + y
        y_loss, y_quantized, y_perplexity, _ = self.vq_layer_0(y)
        x_hat = self._decode(y_quantized, z_quantized)
        return y_loss + z_loss, x_hat, (y_perplexity, z_perplexity)

    def set_standardizer_layer(self, mean, variance, eta=1e-6):
        self.data_preprocessor = preprocessors.Standardizer(
            mean,
            variance,
            eta,
            name="data_processor",
        )

    def compress(self, x):
        """Compresses data."""
        y, z = self._encode(x)  # (B, C, H, W)
        z = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flattened = torch.reshape(z, [-1, self.latent_dim])
        # (BxHxW, 1)
        z_encoding_indices = self.vq_layer_1.get_code_indices(z_flattened)
        # Preserve spatial shapes of both image and latents.
        z_shape = z.shape[1:]

        z_quantized = self.vq_layer_1.get_quantized_from_indices(
            z_encoding_indices, z_shape
        )
        yz = self.first_conv_decoder_yz(z_quantized)
        y = yz + y

        y = y.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        y_flattened = torch.reshape(y, [-1, self.latent_dim])
        # (BxHxW, 1)
        y_encoding_indices = self.vq_layer_0.get_code_indices(y_flattened)
        # Preserve spatial shapes of both image and latents.
        y_shape = y.shape[1:]

        return (y_encoding_indices, z_encoding_indices, y_shape, z_shape)

    def decompress(self, y_encoding_indices, z_encoding_indices, y_shape, z_shape):
        """Decompresses an image."""

        z_quantized = self.vq_layer_1.get_quantized_from_indices(
            z_encoding_indices, z_shape
        )
        y_quantized = self.vq_layer_0.get_quantized_from_indices(
            y_encoding_indices, y_shape
        )
        x_hat = self._decode(y_quantized, z_quantized)
        return x_hat


if __name__ == "__main__":
    pass
