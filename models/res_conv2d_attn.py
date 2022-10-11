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
        in_shape,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_layers,
        num_transformer_layers,
        commitment_cost=0.25,
        decay=0,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        data_channels = in_shape[1]  # in_shape = (B, C, H, W)
        self.pre_num_channels = pre_num_channels
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.num_residual_layers = num_residual_layers
        self.vq_weight = 1.0

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = basemodel.Encoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_layers,
            name="Encoder",
        )

        self.conv_shape = self.encoder(torch.zeros(*in_shape)).shape[1:]
        print(f"conv_shape: {self.conv_shape}")
        self.mini_patch_size = self.conv_shape[1] // 8
        print(f"mini_patch_size: {self.mini_patch_size}")

        self.patcher = patcher.Patcher2d(self.mini_patch_size, name="Patcher")
        self.inverse_patcher = patcher.InversePatcher2d(
            self.mini_patch_size, self.conv_shape, name="InversePatcher"
        )

        # split the latent space into patches for attention
        # patches_shape = (B, num_patches, patch_dim)
        self.patches_shape = self.patcher(torch.zeros((1, *self.conv_shape))).shape
        self.num_heads = self.patches_shape[1] // 16

        # Forward Attention
        self.forward_patch_embedding = embedding.PatchEmbedding(
            c_in=self.patches_shape[2],
            projection_dim=self.patches_shape[2],
            num_patches=self.patches_shape[1],
            name="fordward_patch_embedding",
        )
        # TODO: check multi-head attn
        self.forward_attention = basemodel.TransformerEncoder(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            num_layers=num_transformer_layers,
            name=f"forward_attention",
        )

        if decay > 0.0:
            self.vq_layer = vector_quantizer.VectorQuantizerEMA(
                num_embeddings, latent_dim, commitment_cost, decay
            )
        else:
            self.vq_layer = vector_quantizer.VectorQuantizer(
                num_embeddings, latent_dim, commitment_cost
            )

        # Backward attention
        self.backward_patch_embedding = embedding.PatchEmbedding(
            c_in=self.patches_shape[2],
            projection_dim=self.patches_shape[2],
            num_patches=self.patches_shape[1],
            name="backward_patch_embedding",
        )
        self.backward_attention = basemodel.TransformerEncoder(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            num_layers=num_transformer_layers,
            name=f"backward_attention",
        )

        self.decoder = basemodel.Decoder(
            data_channels,
            pre_num_channels,
            num_channels,
            latent_dim,
            num_residual_layers,
            name="Decoder",
        )
        print(f"Initialization of {self.name} completed!")

    def _encode(self, x):
        """Encodes data."""
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)

        y_patches = self.patcher(y)
        y_embedding = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_attention(y_embedding, mask=None)
        y_attn = self.inverse_patcher(y_attn)
        return y_attn

    def _decode(self, quantized):
        """Decodes data."""
        y_hat_patches = self.patcher(quantized)
        y_hat_embedding = self.backward_patch_embedding(y_hat_patches)
        y_hat_attn = self.backward_attention(y_hat_embedding, mask=None)
        y_hat = self.inverse_patcher(y_hat_attn)

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
        y_attn = self._encode(x)  # (B, C, H, W)
        y_attn = y_attn.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        flattened = torch.reshape(y_attn, [-1, self.latent_dim])

        # (BxHxW, 1)
        encoding_indices = self.vq_layer.get_code_indices(flattened)
        # Preserve spatial shapes of both image and latents.
        y_shape = y_attn.shape[1:]
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
    num_residual_layers = 3
    num_transformer_layers = 2

    in_shape = (1, 1, 128, 128)
    x = torch.normal(0, 1, in_shape)
    model = VQCPVAE(
        in_shape,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_layers,
        num_transformer_layers,
        commitment_cost=0.25,
        decay=0.99,
        name=None,
    )
    model.set_standardizer_layer(1, 1)
    summary(model, x.shape, col_width=25, depth=3, verbose=1)

    encoding_indices, y_shape = model.compress(x)
    print(f"encoding_indices.shape = {encoding_indices.shape}; y_shape: {y_shape}")
    x_hat = model.decompress(encoding_indices, y_shape)
    print(f"x_hat.shape = {x_hat.shape}")


#     @property
#     def metrics(self):
#         return [self.total_loss, self.mse, self.vq_loss]

#     def train_step(self, inputs):
#         x = tf.cast(inputs[0], self.compute_dtype)
#         x_mask = tf.cast(inputs[1], self.compute_dtype)
#         x_mask = tf.ensure_shape(x_mask, x.shape)
#         with tf.GradientTape() as tape:
#             x_hat = self(x)
#             mse = tf.reduce_mean((x * x_mask - x_hat * x_mask) ** 2)
#             total_loss = 2 * mse + self.vq_weight * sum(self.losses)

#         # Backpropagation
#         grads = tape.gradient(total_loss, self.trainable_variables)
#         grads = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in grads]
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

#         # Loss tracking
#         self.total_loss.update_state(total_loss)
#         self.mse.update_state(mse)
#         self.vq_loss.update_state(sum(self.losses))

#         return {m.name: m.result() for m in [self.total_loss, self.mse, self.vq_loss]}

#     def test_step(self, inputs):
#         x = tf.cast(inputs[0], self.compute_dtype)
#         x_mask = tf.cast(inputs[1], self.compute_dtype)
#         x_mask = tf.ensure_shape(x_mask, x.shape)
#         x_hat = self(x)
#         mse = tf.reduce_mean((x * x_mask - x_hat * x_mask) ** 2)
#         total_loss = 2 * mse + self.vq_weight * sum(self.losses)

#         # Loss tracking
#         self.total_loss.update_state(total_loss)
#         self.mse.update_state(mse)
#         self.vq_loss.update_state(sum(self.losses))

#         return {m.name: m.result() for m in [self.total_loss, self.mse, self.vq_loss]}

#     def compile(self, **kwargs):
#         super().compile(
#             loss=None,
#             metrics=None,
#             loss_weights=None,
#             weighted_metrics=None,
#             **kwargs,
#         )
#         self.total_loss = tf.keras.metrics.Mean(name="loss")
#         self.mse = tf.keras.metrics.Mean(name="mse")
#         self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")
