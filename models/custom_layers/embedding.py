import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    """**PositionalEmbedding layer**"""

    def __init__(self, sequence_length, vocab_size, embed_dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class PatchEmbedding(tf.keras.layers.Layer):
    """Positional Embedding for images"""

    def __init__(self, num_patches, projection_dim, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # (num_patches, projection_dim)
        embedded_positions = self.position_embedding(positions)

        # (batch_size, num_patches, projection_dim)
        embedded_projection = self.projection(patch)
        return embedded_projection + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
            }
        )
        return config
