import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (beta)  # should be in [0.1, 2] as in the paper

        # Initialize the embeddings which we will quantize
        w_init = tf.random_uniform_initializer()
        # (embedding_dim, num_embeddings)
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # calculate the input shape of te inputs and
        # then flatten the inputs keeping embedding_dim intact
        # Note: embedding_dim = latent_dim
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization
        # (BxHxW, )
        encoding_indices = self.get_code_indices(flattened)
        # (BxHxW, num_embeddings)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        # (B, H, W, embedding_dim)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(x)) ** 2
        )
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes
        # flattened_inputs: (BxHxW, embedding_dim)
        # (BxHxW, embedding_dim) * (embedding_dim, num_embeddings) = (BxHxW, num_embeddings)
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        reduced_flatten = tf.reduce_sum(
            flattened_inputs ** 2, axis=1, keepdims=True)
        reduced_embedding = tf.reduce_sum(self.embeddings ** 2, axis=0)
        # (BxHxW, num_embeddings)
        distances = (
            reduced_flatten + reduced_embedding - 2 * similarity
        )

        # Derive the indices for minimum distance
        # (BxHxW, 1)
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_embeddings": self.num_embeddings,
                "beta": self.beta,
            }
        )
        return config
