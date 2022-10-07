import tensorflow as tf
import tensorflow_compression as tfc


class DownSamplingBlock(tf.keras.layers.Layer):
    """Downsampling block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.gdn = tfc.GDN()

    def call(self, x):
        x = self.gdn(x)
        x = self.conv1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class UpSamplingBlock(tf.keras.layers.Layer):
    """Upsampling block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.igdn = tfc.GDN(inverse=True)

    def call(self, x):
        x = self.igdn(x)
        x = self.conv1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class DownSamplingResBlock2D(tf.keras.layers.Layer):
    """Downsampling res block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels * 4,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="gelu",
        )
        self.conv3 = tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.cells = [self.conv1, self.conv2, self.conv3]
        self.shortcut = tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=1,
            strides=strides,
            padding="same",
        )
        self.gdn = tfc.GDN()

    def call(self, x):
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = self.gdn(x)

        return x + x_shortcut

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class UpSamplingResBlock2D(tf.keras.layers.Layer):
    """Upsampling res block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            num_channels * 4,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="gelu",
        )
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            num_channels,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.cells = [self.conv1, self.conv2, self.conv3]
        self.shortcut = tf.keras.layers.Conv2DTranspose(
            num_channels,
            kernel_size=1,
            strides=strides,
            padding="same",
        )
        self.igdn = tfc.GDN(inverse=True)

    def call(self, x):
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = self.igdn(x)
        return x + x_shortcut

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class DownSamplingResBlock3D(tf.keras.layers.Layer):
    """Downsampling res block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv3D(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            num_channels * 4,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="gelu",
        )
        self.conv3 = tf.keras.layers.Conv3D(
            num_channels,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.cells = [self.conv1, self.conv2, self.conv3]
        self.shortcut = tf.keras.layers.Conv3D(
            num_channels,
            kernel_size=1,
            strides=strides,
            padding="same",
        )
        self.gdn = tfc.GDN()

    def call(self, x):
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        x = self.gdn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
            }
        )
        return config


class UpSamplingResBlock3D(tf.keras.layers.Layer):
    """Upsampling res block"""

    def __init__(self, num_channels, kernel_size, strides, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv3DTranspose(
            num_channels, kernel_size=1, strides=strides, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv3DTranspose(
            num_channels * 4,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="gelu",
        )
        self.conv3 = tf.keras.layers.Conv3DTranspose(
            num_channels,
            kernel_size=1,
            strides=1,
            padding="same",
        )
        self.cells = [self.conv1, self.conv2, self.conv3]
        self.shortcut = tf.keras.layers.Conv3DTranspose(
            num_channels,
            kernel_size=1,
            strides=strides,
            padding="same",
        )
        self.igdn = tfc.GDN(inverse=True)

    def call(self, x):
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        x = self.igdn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
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
