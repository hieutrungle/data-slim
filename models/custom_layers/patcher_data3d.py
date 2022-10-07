import tensorflow as tf


class Patcher(tf.keras.layers.Layer):
    def __init__(self, patch_size, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.patch_size = patch_size

    def call(self, x):
        batch_size = tf.shape(x)[0]
        len_x, len_y, len_z = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        num_tile_x = len_x // self.patch_size
        num_tile_y = len_y // self.patch_size
        num_tile_z = len_z // self.patch_size
        last_channel = tf.shape(x)[-1]
        patch_dims = last_channel * (self.patch_size**3)

        patches = tf.reshape(
            x,
            [
                batch_size,
                num_tile_x,
                self.patch_size,
                num_tile_y,
                self.patch_size,
                num_tile_z,
                self.patch_size,
                -1,
            ],
        )
        patches = tf.transpose(patches, [0, 1, 3, 5, 2, 4, 6, 7])
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

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
            tf.math.log(tf.cast(num_patches, tf.float32)) / tf.cast(3.0, tf.float32)
        )
        num_tile = tf.cast(num_tile, tf.int32)

        patches = tf.reshape(
            x,
            [
                batch_size,
                num_tile,
                num_tile,
                num_tile,
                self.patch_size,
                self.patch_size,
                self.patch_size,
                -1,
            ],
        )

        patches = tf.transpose(patches, [0, 1, 4, 2, 5, 3, 6, 7])
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
