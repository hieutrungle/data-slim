import tensorflow as tf


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch_size, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.patch_size = patch_size

    def call(self, x):
        len_x, len_y = tf.shape(x)[1], tf.shape(x)[2]
        num_tile_x = len_x // self.patch_size
        num_tile_y = len_y // self.patch_size
        num_channels = tf.shape(x)[-1]
        patch_dims = num_channels * (self.patch_size**2)

        patches = tf.reshape(
            x,
            [
                -1,
                num_tile_x,
                self.patch_size,
                num_tile_y,
                self.patch_size,
                num_channels,
            ],
        )
        patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
        patches = tf.reshape(patches, [-1, num_tile_x * num_tile_y, patch_dims])

        return patches

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
            }
        )
        return config


class InversePatcher(tf.keras.layers.Layer):
    def __init__(self, patch_size, original_shape, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.patch_size = patch_size
        self.original_shape = original_shape

    def call(self, x):
        batch_size = tf.shape(x)[0]
        num_patches = tf.shape(x)[1]
        num_tile = tf.exp(
            tf.math.log(tf.cast(num_patches, tf.float32)) / tf.cast(2.0, tf.float32)
        )
        num_tile = tf.cast(num_tile, tf.int32)

        patches = tf.reshape(
            x,
            [
                batch_size,
                num_tile,
                num_tile,
                self.patch_size,
                self.patch_size,
                -1,
            ],
        )

        patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
        patches = tf.reshape(patches, [-1, *self.original_shape])

        return patches

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "original_shape": self.original_shape,
            }
        )
        return config
