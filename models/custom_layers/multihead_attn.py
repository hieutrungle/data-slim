import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attn_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attn_logits = matmul_qk / tf.math.sqrt(dk)
    # scaled_attn_logits has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.

    # add the mask to the scaled tensor.
    # The mask is multiplied with -1e9 (close to negative infinity).
    # This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax.
    # The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.
    if mask is not None:
        scaled_attn_logits = scaled_attn_logits * mask - 1e6 * (1 - mask)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # As the softmax normalization is done on K, its values decide the amount
    # of importance given to Q.
    # (..., seq_len_q, seq_len_k)
    attn_weights = tf.nn.softmax(scaled_attn_logits, axis=-1)

    # The output represents the multiplication of the attention weights and the
    # V (value) vector. This ensures that the tokens you want to focus on are
    # kept as-is and the irrelevant tokens are flushed out.
    # (..., seq_len_q, depth_v)
    output = tf.matmul(attn_weights, v)

    return output, attn_weights


class CustomizedMultiHeadAttention(tf.keras.layers.Layer):
    """
    <img src="https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png" width="500" alt="multi-head attention">
    Multi-head attention consists of four parts:

    -   Linear layers.
    -   Scaled dot-product attention.
    -   Final linear layer.

    Each multi-head attention block gets three inputs; Q (query), K (key),
    V (value). These are put through linear (Dense) layers before the
    multi-head attention function.

    In the diagram above `(K,Q,V)` are passed through sepearte linear (`Dense`)
    layers for each attention head. For simplicity/efficiency the code below
    implements this using a single dense layer with `num_heads` times as many
    outputs. The output is rearranged to a shape of `(batch, num_heads, ...)`
    before applying the attention function.

    The `scaled_dot_product_attention` function defined above is applied in a
    single call, broadcasted for efficiency. An appropriate mask must be used
    in the attention step. The attention output for each head is then
    concatenated (using `tf.transpose`, and `tf.reshape`) and put through a
    final `Dense` layer.

    Instead of one single attention head, Q, K, and V are split into multiple
    heads because it allows the model to jointly attend to information from
    different representation subspaces at different positions. After the split
    each head has a reduced dimensionality, so the total computation cost is
    the same as a single head attention with full dimensionality.

    """

    def __init__(self, *, embed_dim, num_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % self.num_heads == 0

        self.depth = embed_dim // self.num_heads

        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)

        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        seq_len_q = tf.shape(q)[1]

        # seq_len_k = seq_len_v
        q = self.wq(q)  # (batch_size, seq_len_q, embed_dim)
        k = self.wk(k)  # (batch_size, seq_len_k, embed_dim)
        v = self.wv(v)  # (batch_size, seq_len_v, embed_dim)

        # embed_dim = num_heads x depth
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, embed_dim)
        concat_attention = tf.reshape(scaled_attention, (-1, seq_len_q, self.embed_dim))

        # (batch_size, seq_len_q, embed_dim)
        output = self.dense(concat_attention)

        return output, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, *, embed_dim, num_heads, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multihead_attention = CustomizedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            name=f"{name}_multihead_attention",
        )
        self.ffn = tf.keras.Sequential(
            [
                # (batch_size, seq_len, embed_dim*2)
                tf.keras.layers.Dense(embed_dim * 2, activation="gelu"),
                # (batch_size, seq_len, embed_dim)
                tf.keras.layers.Dense(embed_dim),
            ],
            name=f"{name}_ffn",
        )

        self.layernorm_0 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_layernorm_0"
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_layernorm_1"
        )

        self.add_0 = tf.keras.layers.Add(name=f"{name}_add_0")
        self.add_1 = tf.keras.layers.Add(name=f"{name}_add_1")

    def call(self, x, training=False, mask=None):
        # (batch_size, target_seq_len, embed_dim)
        attention_output_0, _ = self.multihead_attention(x, x, x, mask)
        attention_output_0 = self.layernorm_0(self.add_0([attention_output_0, x]))

        # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.ffn(attention_output_0)
        # (batch_size, target_seq_len, embed_dim)
        attention_output_1 = self.layernorm_1(
            self.add_1([attention_output_0, ffn_output])
        )

        return attention_output_1

    # def call(self, x, training=False, mask=None):
    #     return self.__call__(self, x, training, mask)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
