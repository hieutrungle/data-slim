import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layers import (
    cus_layers,
    embedding,
    vector_quantizer,
    preprocessors,
    multihead_attn,
    patcher_data2d,
)

try:
    from . import basemodel
except:
    import basemodel
    
class VQCPVAE(nn.Module):
    """Vector-quantized Compression Variational Autoencoder"""
    def __init__(
        self,
        in_shape,
        latent_dim,
        num_embeddings,
        num_channels,
        num_conv_layers,
        train_variance=0.5,
        name=None,
        **kwargs,
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.train_variance = train_variance
        self.vq_weight = tf.Variable(1.0, trainable=False, dtype=self.compute_dtype)

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = basemodel.Encoder(
            in_shape, latent_dim, num_channels, num_conv_layers, name="Encoder"
        )
        self.conv_shape = self.encoder.model().output_shape[1:]
        self.mini_patch_size = self.conv_shape[0] // 8
        self.patcher = patcher_data2d.Patcher(self.mini_patch_size, name="Patcher")
        self.inverse_patcher = patcher_data2d.InversePatcher(
            self.mini_patch_size, self.conv_shape, name="InversePatcher"
        )

        # split the latent space into patches for attention
        self.patches_shape = self.patcher(tf.random.normal((1, *self.conv_shape))).shape
        self.num_heads = self.patches_shape[1] // 16

        # Forward Attention
        self.forward_patch_embedding = embedding.PatchEmbedding(
            self.patches_shape[1],
            self.patches_shape[2],
            name="fordward_patch_embedding",
        )
        self.forward_attention = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            name=f"forward_attention",
        )

        self.vq_layer = vector_quantizer.VectorQuantizer(
            num_embeddings, latent_dim, name="vector_quantizer"
        )

        # Backward attention
        self.backward_patch_embedding = embedding.PatchEmbedding(
            self.patches_shape[1],
            self.patches_shape[2],
            name="backward_patch_embedding",
        )
        self.backward_attention = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            name=f"backward_attention",
        )

        self.decoder = basemodel.Decoder(
            self.conv_shape, num_channels, num_conv_layers, name="Decoder"
        )

        tmp = tf.random.normal((1, *in_shape))
        self._set_inputs(inputs=tmp, outputs=self(tmp))

        self.reconstruct_data = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, self.in_shape[-1]),
                    dtype=self.compute_dtype,
                ),
            ]
        )(self.reconstruct_data)

        self.compress = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, self.in_shape[-1]),
                    dtype=self.compute_dtype,
                ),
            ]
        )(self.compress)

        print(f"Initialization of {self.name} completed!")
    
    def forward(self, x):
        pass

class VQCPVAE(basemodel.BaseModel):
    """Vector-quantized Compression Variational Autoencoder"""

    def __init__(
        self,
        in_shape,
        latent_dim,
        num_embeddings,
        num_channels,
        num_conv_layers,
        train_variance=0.5,
        name=None,
        **kwargs,
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.train_variance = train_variance
        self.vq_weight = tf.Variable(1.0, trainable=False, dtype=self.compute_dtype)

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = basemodel.Encoder(
            in_shape, latent_dim, num_channels, num_conv_layers, name="Encoder"
        )
        self.conv_shape = self.encoder.model().output_shape[1:]
        self.mini_patch_size = self.conv_shape[0] // 8
        self.patcher = patcher_data2d.Patcher(self.mini_patch_size, name="Patcher")
        self.inverse_patcher = patcher_data2d.InversePatcher(
            self.mini_patch_size, self.conv_shape, name="InversePatcher"
        )

        # split the latent space into patches for attention
        self.patches_shape = self.patcher(tf.random.normal((1, *self.conv_shape))).shape
        self.num_heads = self.patches_shape[1] // 16

        # Forward Attention
        self.forward_patch_embedding = embedding.PatchEmbedding(
            self.patches_shape[1],
            self.patches_shape[2],
            name="fordward_patch_embedding",
        )
        self.forward_attention = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            name=f"forward_attention",
        )

        self.vq_layer = vector_quantizer.VectorQuantizer(
            num_embeddings, latent_dim, name="vector_quantizer"
        )

        # Backward attention
        self.backward_patch_embedding = embedding.PatchEmbedding(
            self.patches_shape[1],
            self.patches_shape[2],
            name="backward_patch_embedding",
        )
        self.backward_attention = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            name=f"backward_attention",
        )

        self.decoder = basemodel.Decoder(
            self.conv_shape, num_channels, num_conv_layers, name="Decoder"
        )

        tmp = tf.random.normal((1, *in_shape))
        self._set_inputs(inputs=tmp, outputs=self(tmp))

        self.reconstruct_data = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, self.in_shape[-1]),
                    dtype=self.compute_dtype,
                ),
            ]
        )(self.reconstruct_data)

        self.compress = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, self.in_shape[-1]),
                    dtype=self.compute_dtype,
                ),
            ]
        )(self.compress)

        print(f"Initialization of {self.name} completed!")

    def __call__(self, x, training=False):
        x = tf.cast(x, self.compute_dtype)
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)

        y_patches = self.patcher(y)
        y_embed = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_attention(y_embed, mask=None)
        y_attn = self.inverse_patcher(y_attn)

        y_hat = self.vq_layer(y_attn)

        y_hat_patches = self.patcher(y_hat)
        y_hat_embed = self.backward_patch_embedding(y_hat_patches)
        y_hat_attn = self.backward_attention(y_hat_embed, mask=None)
        y_hat_attn = self.inverse_patcher(y_hat_attn)

        x_hat = self.decoder(y_hat_attn)
        x_hat_final = self.data_preprocessor(x_hat, normalize=0)
        return x_hat_final

    def call(self, x, training=False):
        return self.__call__(x, training=training)

    def reconstruct_data(self, images):
        return self.__call__(images)

    def set_standardizer_layer(self, mean, variance, eta=1e-6):
        mean = tf.cast(mean, self.compute_dtype)
        variance = tf.cast(mean, self.compute_dtype)
        self.train_variance = variance

        self.data_preprocessor = preprocessors.Standardizer(
            mean,
            variance,
            eta,
            name="data_processor",
        )

    def get_standardizer_layer(self):
        if self.data_preprocessor:
            return self.data_preprocessor
        else:
            print("data preprocessor is not defined")

    def set_vq_weight(self, weight):
        self.vq_weight = weight

    def get_vq_weight(self):
        return self.vq_weight

    @property
    def metrics(self):
        return [self.total_loss, self.mse, self.vq_loss]

    def train_step(self, inputs):
        x = tf.cast(inputs[0], self.compute_dtype)
        x_mask = tf.cast(inputs[1], self.compute_dtype)
        x_mask = tf.ensure_shape(x_mask, x.shape)
        with tf.GradientTape() as tape:
            x_hat = self(x)
            mse = tf.reduce_mean((x * x_mask - x_hat * x_mask) ** 2)
            total_loss = 2 * mse + self.vq_weight * sum(self.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking
        self.total_loss.update_state(total_loss)
        self.mse.update_state(mse)
        self.vq_loss.update_state(sum(self.losses))

        return {m.name: m.result() for m in [self.total_loss, self.mse, self.vq_loss]}

    def test_step(self, inputs):
        x = tf.cast(inputs[0], self.compute_dtype)
        x_mask = tf.cast(inputs[1], self.compute_dtype)
        x_mask = tf.ensure_shape(x_mask, x.shape)
        x_hat = self(x)
        mse = tf.reduce_mean((x * x_mask - x_hat * x_mask) ** 2)
        total_loss = 2 * mse + self.vq_weight * sum(self.losses)

        # Loss tracking
        self.total_loss.update_state(total_loss)
        self.mse.update_state(mse)
        self.vq_loss.update_state(sum(self.losses))

        return {m.name: m.result() for m in [self.total_loss, self.mse, self.vq_loss]}

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.total_loss = tf.keras.metrics.Mean(name="loss")
        self.mse = tf.keras.metrics.Mean(name="mse")
        self.vq_loss = tf.keras.metrics.Mean(name="vq_loss")

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compress(self, x):
        """Compresses data."""
        x = self.data_preprocessor(x)
        y = self.encoder(x)
        y_patches = self.patcher(y)
        y_embed = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_attention(y_embed, mask=None)
        y_attn = self.inverse_patcher(y_attn)

        flattened = tf.reshape(y_attn, [-1, self.latent_dim])

        # (BxHxW, )
        encoding_indices = self.vq_layer.get_code_indices(flattened)

        # Preserve spatial shapes of both image and latents.
        y_shape = tf.shape(y)[1:]
        return encoding_indices, y_shape

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32),
        ]
    )
    def decompress(self, encoding_indices, y_shape):
        """Decompresses an image."""
        # (BxHxW, num_embeddings)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = tf.matmul(encodings, self.vq_layer.embeddings, transpose_b=True)
        # (B, H, W, embedding_dim)
        quantized = tf.reshape(quantized, (-1, y_shape[0], y_shape[1], y_shape[2]))

        y_hat_patches = self.patcher(quantized)
        y_hat_embed = self.backward_patch_embedding(y_hat_patches)
        y_hat_attn = self.backward_attention(y_hat_embed, mask=None)
        y_hat_attn = self.inverse_patcher(y_hat_attn)
        x_hat = self.decoder(y_hat_attn)
        x_hat = self.data_preprocessor(x_hat, normalize=0)
        return x_hat

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_shape": self.in_shape,
                "latent_dim": self.latent_dim,
                "num_embeddings": self.num_embeddings,
                "conv_shape": self.conv_shape,
            }
        )
        return config


if __name__ == "__main__":
    pass
