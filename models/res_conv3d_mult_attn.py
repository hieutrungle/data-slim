from .custom_layers import vector_quantizer
from .custom_layers import preprocessors
from .custom_layers import cus_layers
from .custom_layers import patcher_data3d
from .custom_layers import multihead_attn
import tensorflow as tf
import tensorflow_compression as tfc
import os
from . import basemodel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class Encoder(basemodel.BaseModel):
    """Encoder"""

    def __init__(
        self, in_shape, latent_dim, channel, num_conv_layers, name=None, **kwargs
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.channel = channel
        self.num_conv_layers = num_conv_layers
        self.cells = []
        intial_dim = channel // (2**3)

        first_cell = tf.keras.layers.Conv3D(
            intial_dim, kernel_size=1, strides=1, padding="same", name="enc_block0"
        )

        cells = []
        for i in range(num_conv_layers):
            cell = cus_layers.DownSamplingResBlock3D(
                channel, kernel_size=4, strides=2, name=f"enc_block{i+1}"
            )
            cells.append(cell)

        last_cell = tf.keras.layers.Conv3D(
            latent_dim, kernel_size=1, strides=1, padding="same", name="enc_block_final"
        )

        self.cells = [first_cell, *cells, last_cell]

    def __call__(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "channel": self.channel,
                "num_conv_layers": self.num_conv_layers,
            }
        )
        return config


class ForwardConv1d(tf.keras.layers.Layer):
    """Forward MLP"""

    def __init__(self, in_shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_shape = in_shape

        self.flatten = tf.keras.layers.Flatten(name="enc_flatten")
        self.gdn = tfc.GDN(name="enc_activation")
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=tf.math.reduce_prod(in_shape),
            padding="same",
            name="enc_conv1d",
        )

    def __call__(self, x):
        x = self.gdn(x)
        x = self.flatten(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x)
        return x

    def call(self, x):
        return self.__call__(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_shape": self.in_shape,
            }
        )
        return config


class InverseConv1d(tf.keras.layers.Layer):
    """Inverse MLP"""

    def __init__(self, conv_shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_shape = conv_shape
        # height, width, latent_dim = conv_shape

        self.conv1d = tf.keras.layers.Conv1DTranspose(
            filters=1,
            kernel_size=tf.math.reduce_prod(conv_shape),
            padding="same",
            name="dec_conv1d",
        )
        self.reshape = tf.keras.layers.Reshape(conv_shape, name="dec_reshape")
        self.igdn = tfc.GDN(inverse=True, name="dec_activation")

    def __call__(self, x):

        x = self.dense(x)
        x = self.reshape(x)
        x = self.igdn(x)
        return x

    def call(self, x):
        return self.__call__(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_shape": self.conv_shape,
            }
        )
        return config


class ForwardMLP(tf.keras.layers.Layer):
    """Forward MLP"""

    def __init__(self, in_shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_shape = in_shape

        self.flatten = tf.keras.layers.Flatten(name="enc_flatten")
        self.gdn = tfc.GDN(name="enc_activation")
        self.dense = tf.keras.layers.Dense(
            tf.math.reduce_prod(in_shape), name="enc_dense"
        )

    def __call__(self, x):
        x = self.gdn(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def call(self, x):
        return self.__call__(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_shape": self.in_shape,
            }
        )
        return config


class InverseMLP(tf.keras.layers.Layer):
    """Inverse MLP"""

    def __init__(self, conv_shape, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_shape = conv_shape
        # height, width, latent_dim = conv_shape

        self.dense = tf.keras.layers.Dense(
            tf.math.reduce_prod(conv_shape), name="dec_dense"
        )
        self.reshape = tf.keras.layers.Reshape(conv_shape, name="dec_reshape")
        self.igdn = tfc.GDN(inverse=True, name="dec_activation")

    def __call__(self, x):

        x = self.dense(x)
        x = self.reshape(x)
        x = self.igdn(x)
        return x

    def call(self, x):
        return self.__call__(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "conv_shape": self.conv_shape,
            }
        )
        return config


class Decoder(basemodel.BaseModel):
    """Decoder"""

    def __init__(self, in_shape, channel, num_conv_layers, name=None, **kwargs):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.channel = channel
        self.num_conv_layers = num_conv_layers
        latent_dim = in_shape[-1]

        first_cell = tf.keras.layers.Conv3DTranspose(
            latent_dim, kernel_size=1, strides=1, padding="same", name="dec_block0"
        )

        cells = []
        for i in range(num_conv_layers):
            cell = cus_layers.UpSamplingResBlock3D(
                channel, kernel_size=4, strides=2, name=f"dec_block{i+1}"
            )
            cells.append(cell)

        last_cell = tf.keras.layers.Conv3DTranspose(
            1, kernel_size=1, strides=1, padding="same", name="dec_block_final"
        )

        self.cells = [first_cell, *cells, last_cell]

    def __call__(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channel": self.channel,
                "num_conv_layers": self.num_conv_layers,
            }
        )
        return config


class VQCPVAE(basemodel.BaseModel):
    """Vector-quantized Compression Variational Autoencoder"""

    def __init__(
        self,
        in_shape,
        latent_dim,
        num_embeddings,
        channel,
        num_conv_layers,
        name=None,
        **kwargs,
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.channel = channel
        self.num_conv_layers = num_conv_layers
        self.num_heads = 2
        self.mini_patch_size = 2
        self.train_variance = 0.2

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = Encoder(
            in_shape, latent_dim, channel, num_conv_layers, name="Encoder"
        )
        self.conv_shape = self.encoder.model().output_shape[1:]
        self.patcher = patcher_data3d.Patcher(self.mini_patch_size, name="Patcher")

        self.patches_shape = self.patcher(tf.random.normal((1, *self.conv_shape))).shape

        self.inverse_patcher = patcher_data3d.InversePatcher(
            self.mini_patch_size, self.conv_shape, name="InversePatcher"
        )
        self.forward_patch_embedding = multihead_attn.PatchEncoder(
            self.patches_shape[1],
            self.patches_shape[2],
            name="FordwardPatchEmbedding",
        )
        self.forward_transformer_block = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[2],
            num_heads=self.num_heads,
            name="ForwardTransformerBlock",
        )

        self.forward_mlp = ForwardMLP(self.conv_shape, name="ForwardMLP")
        self.vq_layer = vector_quantizer.VectorQuantizer(
            num_embeddings, latent_dim, name="vector_quantizer"
        )
        self.backward_mlp = InverseMLP(self.conv_shape, name="InverseMLP")

        self.backward_patch_embedding = multihead_attn.PatchEncoder(
            self.patches_shape[1],
            self.patches_shape[2],
            name="BackwardPatchEmbedding",
        )
        self.backward_transformer_block = multihead_attn.TransformerBlock(
            embed_dim=self.patches_shape[-1],
            num_heads=self.num_heads,
            name="BackwardTransformerBlock",
        )
        self.decoder = Decoder(
            self.conv_shape, channel, num_conv_layers, name="Decoder"
        )

        tmp = tf.random.normal((1, *in_shape))
        self._set_inputs(inputs=tmp, outputs=self(tmp))

        self.reconstruct_data = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, None, self.in_shape[-1]), dtype=tf.float32
                ),
            ]
        )(self.reconstruct_data)

        self.compress = tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, None, None, self.in_shape[-1]), dtype=tf.float32
                ),
            ]
        )(self.compress)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructions = self(x)

            reconstruction_loss = tf.reduce_mean((x - reconstructions) ** 2)
            total_loss = reconstruction_loss / self.train_variance + sum(self.losses)

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.losses))

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, x):
        reconstructions = self(x)
        reconstruction_loss = (
            tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
        )
        total_loss = reconstruction_loss + sum(self.losses)

        # Loss tracking
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.losses))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vqvae_loss": sum(self.losses),
        }

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def __call__(self, x, training=False):
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)

        y_patches = self.patcher(y)
        y_attn = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_transformer_block(y_attn)
        y_attn = self.inverse_patcher(y_attn)

        y_mlp = self.forward_mlp(y_attn)
        y_hat = self.vq_layer(y_mlp)
        y_hat_mlp = self.backward_mlp(y_hat)

        y_patches = self.patcher(y_hat_mlp)
        y_attn = self.backward_patch_embedding(y_patches)
        y_attn = self.backward_transformer_block(y_attn)
        y_attn = self.inverse_patcher(y_attn)

        x_hat = self.decoder(y_attn)
        x_hat_final = self.data_preprocessor(x_hat, normalize=0)
        return x_hat_final

    def call(self, x, training=False):
        return self.__call__(x, training=training)

    def reconstruct_data(self, images):
        return self.__call__(images)

    def set_standardizer_layer(self, mean, variance, eta=1e-6):
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

    def compress(self, x):
        """Compresses data."""
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)

        y_patches = self.patcher(y)
        y_attn = self.forward_patch_embedding(y_patches)
        y_attn = self.forward_transformer_block(y_attn)
        y_attn = self.inverse_patcher(y_attn)

        y_mlp = self.forward_mlp(y_attn)

        flattened = tf.reshape(y_mlp, [-1, self.latent_dim])

        # (BxHxW, )
        encoding_indices = self.vq_layer.get_code_indices(flattened)

        # Preserve spatial shapes of both image and latents.
        y_shape = tf.shape(y)[1:]
        y_mlp_shape = tf.shape(y_mlp)[1:]
        # conv_shape = self.conv_shape
        return encoding_indices, y_shape, y_mlp_shape

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(4,), dtype=tf.int32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        ]
    )
    def decompress(self, encoding_indices, y_shape, y_mlp_shape):
        """Decompresses an image."""
        # (BxHxW, num_embeddings)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = tf.matmul(encodings, self.vq_layer.embeddings, transpose_b=True)
        # (B, H, W, embedding_dim)
        quantized = tf.reshape(quantized, (-1, y_mlp_shape[-1]))
        y_hat_mlp = self.backward_mlp(quantized)
        y_patches = self.patcher(y_hat_mlp)
        y_attn = self.backward_patch_embedding(y_patches)
        y_attn = self.backward_transformer_block(y_attn)
        y_attn = self.inverse_patcher(y_attn)
        x_hat = self.decoder(y_attn)
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
                "channel": self.channel,
                "num_conv_layers": self.num_conv_layers,
            }
        )
        return config


if __name__ == "__main__":
    pass
