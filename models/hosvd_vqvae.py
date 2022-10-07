import os

from .custom_layers import cus_layers

from models.custom_layers import cus_layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_compression as tfc
import tensorly as tl
from tensorly import decomposition
from tensorly.decomposition import tucker


try:
    from . import basemodel
    from .custom_layers import vector_quantizer, preprocessors
except:
    import basemodel
    from custom_layers import vector_quantizer, preprocessors


def decompose(tensor, decomposition_model):
    """Decompose tensor into a sum of rank-1 tensors"""
    cp_tensor = decomposition_model.fit_transform(tensor)
    return cp_tensor


class Encoder(basemodel.BaseModel):
    """Encoder"""

    def __init__(
        self, in_shape, latent_dim, num_channels, num_conv_layers, name=None, **kwargs
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        self.cells = []
        intial_dim = num_channels // (2**3)

        first_cell = tf.keras.layers.Conv2D(
            intial_dim, kernel_size=1, strides=1, padding="same", name="enc_block0"
        )

        cells = []
        for i in range(num_conv_layers):
            cell = cus_layers.DownSamplingResBlock2D(
                num_channels=num_channels,
                kernel_size=3,
                strides=2,
                name=f"enc_block{i+1}",
            )
            cells.append(cell)

        last_cell = tf.keras.layers.Conv2D(
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
                "num_channels": self.num_channels,
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

    def __init__(self, in_shape, num_channels, num_conv_layers, name=None, **kwargs):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.num_channels = num_channels
        self.num_conv_layers = num_conv_layers
        latent_dim = in_shape[-1]

        first_cell = tf.keras.layers.Conv2DTranspose(
            latent_dim, kernel_size=1, strides=1, padding="same", name="dec_block0"
        )

        cells = []
        for i in range(num_conv_layers):
            cell = cus_layers.UpSamplingResBlock2D(
                num_channels=num_channels,
                kernel_size=4,
                strides=2,
                name=f"dec_block{i+1}",
            )
            cells.append(cell)

        last_cell = tf.keras.layers.Conv2DTranspose(
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
                "num_channels": self.num_channels,
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
        num_channels,
        num_conv_layers,
        name="Vector_quantized_Compression_Variational_Autoencoder",
        **kwargs,
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.train_variance = 1.0

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = Encoder(
            in_shape, latent_dim, num_channels, num_conv_layers, name="Encoder"
        )
        self.conv_shape = self.encoder.model().output_shape[1:]
        self.forward_mlp = ForwardMLP(self.conv_shape, name="ForwardMLP")
        self.vq_layer = vector_quantizer.VectorQuantizer(
            num_embeddings, latent_dim, name="vector_quantizer"
        )
        self.backward_mlp = InverseMLP(self.conv_shape, name="InverseMLP")
        self.decoder = Decoder(
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

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

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

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def train_step(self, x):
        x = tf.cast(x, self.compute_dtype)
        with tf.GradientTape() as tape:
            reconstructions = self(x)

            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.losses)

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
        x = tf.cast(x, self.compute_dtype)
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
        x = tf.cast(x, self.compute_dtype)
        x = self.data_preprocessor(x, normalize=1)
        y = self.encoder(x)
        y_mlp = self.forward_mlp(y)
        y_hat = self.vq_layer(y_mlp)
        y_hat_mlp = self.backward_mlp(y_hat)
        x_hat = self.decoder(y_hat_mlp)
        x_hat_final = self.data_preprocessor(x_hat, normalize=0)
        return x_hat_final

    def call(self, x, training=False):
        return self.__call__(x, training=training)

    def reconstruct_data(self, images):
        return self.__call__(images)

    def compress(self, x):
        """Compresses data."""
        x = self.data_preprocessor(x)
        y = self.encoder(x)
        y_mlp = self.forward_mlp(y)

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
            tf.TensorSpec(shape=(3,), dtype=tf.int32),
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
        x_hat = self.decoder(y_hat_mlp)
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
