import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from .custom_layers import embedding, multihead_attn


# Transformer Architecture
# https://www.tensorflow.org/images/tutorials/transformer/standard_transformer_architecture.png


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, *, embed_dim, num_heads, latent_dim, dropout_rate=0.1, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.multihead_attention_0 = multihead_attn.CustomizedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            name=f"{name}_multihead_attention_0",
        )

        # (batch_size, seq_len, latent_dim) -> (batch_size, seq_len, embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, activation="gelu"),
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

        self.dropout_0 = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout_0")
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout_1")

    def call(
        self, encoder_input, training=False, look_ahead_mask=None, padding_mask=None
    ):

        # (batch_size, target_seq_len, embed_dim)
        attention_output_0, _ = self.multihead_attention_0(
            encoder_input, encoder_input, encoder_input, padding_mask
        )
        attention_output_0 = self.dropout_0(attention_output_0, training=training)
        attention_output_0 = self.layernorm_0(attention_output_0 + encoder_input)

        # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.ffn(attention_output_0)
        ffn_output = self.dropout_1(ffn_output, training=training)
        # (batch_size, target_seq_len, embed_dim)
        attention_output_1 = self.layernorm_1(ffn_output + attention_output_0)

        return attention_output_1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class Encoder(tf.keras.layers.Layer):
    """
    The `Encoder` consists of:
    1.  Input Embedding
    2.  Positional Encoding
    3.  N encoder layers

    The input is put through an embedding which is summed with the positional
    encoding. The output of this summation is the input to encoder layers.
    The output of the encoder is the input to the final linear layer.
    """

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        latent_dim,
        dropout_rate=0.1,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.enc_layers = list()
        for i in range(num_layers):
            self.enc_layers.append(
                EncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    latent_dim=latent_dim,
                    dropout_rate=dropout_rate,
                    name=f"{name}_layer_{i}",
                )
            )

    def __call__(self, x, training=False, look_ahead_mask=None, padding_mask=None):

        # x.shape = (batch_size, seq_len_input, embed_dim)
        # x.shape is unchanged
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, look_ahead_mask, padding_mask)

        return x

    def call(self, x, training=False, look_ahead_mask=None, padding_mask=None):
        return self.__call__(self, x, training, look_ahead_mask, padding_mask)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class DecoderLayer(tf.keras.layers.Layer):
    """
    Each decoder layer consists of sublayers:

    1.  Masked multi-head attention (with look ahead mask and padding mask)
    2.  Multi-head attention (with padding mask). V (value) and K (key) receive
        the _encoder output_ as inputs. Q (query) receives the _output from the
        masked multi-head attention sublayer._
    3.  Point wise feed forward networks

    Each of these sublayers has a residual connection around it followed by a
    layer normalization. The output of each sublayer is
    `LayerNorm(x + Sublayer(x))`. The normalization is done on the `embed_dim`
    (last) axis.

    There are a number of decoder layers in the model.

    As Q receives the output from decoder's first attention block, and K
    receives the encoder output, the attention weights represent the importance
    given to the decoder's input based on the encoder's output. In other words,
    the decoder predicts the next token by looking at the encoder output and
    self-attending to its own output. See the demonstration above in the scaled
    dot product attention section.

    """

    def __init__(
        self, *, embed_dim, num_heads, latent_dim, dropout_rate=0.1, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.multihead_attention_0 = multihead_attn.CustomizedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            name=f"{name}_multihead_attention_0",
        )
        self.multihead_attention_1 = multihead_attn.CustomizedMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            name=f"{name}_multihead_attention_1",
        )

        # (batch_size, seq_len, latent_dim) -> (batch_size, seq_len, embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, activation="gelu"),
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
        self.layernorm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_layernorm_2"
        )

        self.dropout_0 = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout_0")
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout_1")
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout_2")

    def call(
        self,
        decoder_input,
        encoder_output,
        training=False,
        look_ahead_mask=None,
        padding_mask=None,
    ):
        # encoder_output.shape == (batch_size, input_seq_len, embed_dim)

        # (batch_size, target_seq_len, embed_dim)
        attention_output_0, _ = self.multihead_attention_0(
            decoder_input, decoder_input, decoder_input, look_ahead_mask
        )
        attention_output_0 = self.dropout_0(attention_output_0, training=training)
        attention_output_0 = self.layernorm_0(attention_output_0 + decoder_input)

        attention_output_1, _ = self.multihead_attention_1(
            attention_output_0, encoder_output, encoder_output, padding_mask
        )
        attention_output_1 = self.dropout_1(attention_output_1, training=training)
        attention_output_1 = self.layernorm_1(attention_output_1 + attention_output_0)

        # (batch_size, target_seq_len, embed_dim)
        ffn_output = self.ffn(attention_output_1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        # (batch_size, target_seq_len, embed_dim)
        attention_output_2 = self.layernorm_2(ffn_output + attention_output_1)

        return attention_output_2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class Decoder(tf.keras.layers.Layer):
    """
    The `Decoder` consists of:
    1.  Output Embedding
    2.  Positional Encoding
    3.  N decoder layers

    The target is put through an embedding which is summed with the positional
    encoding. The output of this summation is the input to the decoder layers.
    The output of the decoder is the input to the final linear layer.
    """

    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        latent_dim,
        dropout_rate=0.1,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.dec_layers = list()
        for i in range(num_layers):
            self.dec_layers.append(
                DecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    latent_dim=latent_dim,
                    dropout_rate=dropout_rate,
                    name=f"{name}_layer_{i}",
                )
            )

    def __call__(
        self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None
    ):

        # x.shape == (batch_size, target_seq_len, embed_dim)
        for dec_layer in self.dec_layers:
            x = dec_layer(x, encoder_output, training, look_ahead_mask, padding_mask)

        return x

    def call(
        self, x, encoder_output, training=False, look_ahead_mask=None, padding_mask=None
    ):
        return self.__call__(x, encoder_output, training, look_ahead_mask, padding_mask)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "latent_dim": self.latent_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class MaskingLayer(tf.keras.layers.Layer):
    """**Causal Mask layer**"""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, src_inputs, dst_inputs):
        padding_mask, look_ahead_mask = self.create_masks(src_inputs, dst_inputs)
        return padding_mask, look_ahead_mask

    def create_masks(self, encoder_inputs, decoder_inputs):
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        input_seq_len = tf.shape(encoder_inputs)[1]
        target_seq_len = tf.shape(decoder_inputs)[1]
        look_ahead_mask = self.causal_attention_mask(target_seq_len, input_seq_len)
        padding_mask = self.create_padding_mask(decoder_inputs)
        # look_ahead_mask = tf.minimum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask

    def create_padding_mask(self, seq):
        """
        Mask all the pad tokens in the batch of sequence. It ensures that the
        model does not treat padding as the input. The mask indicates where pad
        value `0` is present: it outputs a `1` at those locations, and a
        0` otherwise.
        """
        seq = 1 - tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def causal_attention_mask(self, nd, ns):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd, dtype=tf.float32)[:, None]
        j = tf.range(ns, dtype=tf.float32)
        nd = tf.cast(nd, tf.float32)
        ns = tf.cast(ns, tf.float32)
        m = i >= j - ns + nd
        return tf.cast(m, self.compute_dtype)


# Create the transformer class
class TimeSeriesTransformer(tf.keras.Model):
    """
    A transformer consists of the decoder and a final linear layer.
    The output of the decoder is the input to the linear layer and its output
    is returned.
    """

    def __init__(
        self,
        num_layers,
        embed_dim,
        latent_dim,
        num_heads,
        input_sequence_length,
        input_vocab_size,
        target_sequence_length,
        target_vocab_size,
        dropout_rate=0.1,
        name="Time_Series_Transformer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.input_sequence_length = input_sequence_length
        self.input_vocab_size = input_vocab_size
        self.target_sequence_length = target_sequence_length
        self.target_vocab_size = target_vocab_size
        self.dropout_rate = dropout_rate

        self.encoder_pos_embedding = embedding.PositionalEmbedding(
            sequence_length=input_sequence_length,
            vocab_size=input_vocab_size,
            embed_dim=embed_dim,
            name="enc_pos_embeding",
        )
        self.encoder_dropout = tf.keras.layers.Dropout(
            dropout_rate, name="enc_pos_embeding_dropout"
        )

        self.encoder = Encoder(
            num_layers,
            embed_dim,
            num_heads,
            latent_dim,
            dropout_rate,
            name=f"enc",
        )

        self.decoder_pos_embedding = embedding.PositionalEmbedding(
            sequence_length=target_sequence_length,
            vocab_size=target_vocab_size,
            embed_dim=embed_dim,
            name="dec_pos_embeding",
        )
        self.decoder_dropout = tf.keras.layers.Dropout(
            dropout_rate, name="dec_pos_embeding_dropout"
        )

        self.mask = MaskingLayer(name="dec_masking_layer")

        self.decoder = Decoder(
            num_layers,
            embed_dim,
            num_heads,
            latent_dim,
            dropout_rate,
            name="dec",
        )

        # self.final_layer = tf.keras.layers.Dense(
        #     target_vocab_size, activation="softmax", name="dense_with_softmax"
        # )

    def __call__(self, encoder_inputs, decoder_inputs, training=False):

        enc_outputs = self.encoder_pos_embedding(encoder_inputs)
        enc_outputs = self.encoder_dropout(enc_outputs, training=training)

        enc_outputs = self.encoder(
            enc_outputs, training=False, look_ahead_mask=None, padding_mask=None
        )

        dec_outputs = self.decoder_pos_embedding(decoder_inputs)
        dec_outputs = self.decoder_dropout(dec_outputs, training=training)

        padding_mask, look_ahead_mask = self.mask(encoder_inputs, decoder_inputs)

        # decoder_output.shape == (batch_size, tar_seq_len, embed_dim)
        dec_outputs = self.decoder(
            dec_outputs,
            enc_outputs,
            training,
            look_ahead_mask,
            padding_mask=None,
        )

        # # (batch_size, tar_seq_len, target_vocab_size)
        # final_output = self.final_layer(decoder_output)

        return dec_outputs

    def call(self, inputs, training=False):
        return self.__call__(inputs, training)

    def model(self):
        enc_inputs = tf.keras.Input(
            shape=(self.input_sequence_length,), dtype="int64", name="enc_inputs"
        )
        dec_inputs = tf.keras.Input(
            shape=(self.target_sequence_length,), dtype="int64", name="dec_inputs"
        )
        inputs = [enc_inputs, dec_inputs]
        return tf.keras.Model(
            inputs=inputs, outputs=self(*inputs, training=False), name=self.name
        )

    def summarize_model(self):
        trainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
        )
        nonTrainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
        )
        totalParams = trainableParams + nonTrainableParams
        print(
            f"\n================================================================="
            f"\n                     {self.name} Summary"
            f"\nTotal params: {totalParams:,}"
            f"\nTrainable params: {trainableParams:,}"
            f"\nNon-trainable params: {nonTrainableParams:,}"
            f"\n_________________________________________________________________\n"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
                "input_sequence_length": self.input_sequence_length,
                "input_vocab_size": self.input_vocab_size,
                "target_sequence_length": self.target_sequence_length,
                "target_vocab_size": self.target_vocab_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


if __name__ == "__main__":
    num_layers = 3
    embed_dim = 128
    latent_dim = 256
    num_heads = 2
    input_sequence_length = 200
    input_vocab_size = 128
    target_sequence_length = 100
    target_vocab_size = 128
    dropout_rate = 0.1
    model = TimeSeriesTransformer(
        num_layers,
        embed_dim,
        latent_dim,
        num_heads,
        input_sequence_length,
        input_vocab_size,
        target_sequence_length,
        target_vocab_size,
        dropout_rate,
    )
    model.model().summary()
