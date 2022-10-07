import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
from .custom_layers import vector_quantizer, preprocessors

try:
    from . import basemodel
except:
    import basemodel


class VQCPVAE(basemodel.BaseModel):
    """Vector-quantized Compression Variational Autoencoder"""

    def __init__(
        self,
        in_shape,
        latent_dim,
        num_embeddings,
        num_channels,
        num_conv_layers,
        name=None,
        **kwargs,
    ):
        super().__init__(in_shape=in_shape, name=name, **kwargs)
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.train_variance = 1.0

        self.data_preprocessor = preprocessors.IdentityDataProcessor()
        self.encoder = basemodel.Encoder(
            in_shape, latent_dim, num_channels, num_conv_layers, name="Encoder"
        )
        self.conv_shape = self.encoder.model().output_shape[1:]
        self.vq_layer = vector_quantizer.VectorQuantizer(
            num_embeddings, latent_dim, name="vector_quantizer"
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
        y_hat = self.vq_layer(y)
        x_hat = self.decoder(y_hat)
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
