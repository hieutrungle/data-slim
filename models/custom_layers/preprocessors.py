import tensorflow as tf


class ScientificDataProcessor(tf.keras.layers.Layer):
    """Data processor for scientific data."""

    def __init__(
        self, ds_min, ds_max, min_scale=1.1, max_scale=2.7, name=None, **kwargs
    ):
        super().__init__(name=name, trainable=False, **kwargs)
        self._max_value = ds_max
        self._min_value = ds_min
        self._scale_range = max_scale - min_scale
        self._min_scale = min_scale
        self._max_scale = max_scale

    def call(self, x, normalize=1):
        return tf.cond(
            tf.equal(normalize, 1),
            lambda: tf.math.log(self.normalize_minmax(x)),
            lambda: self.denormalize_minmax(tf.math.exp(x)),
        )

    def normalize_minmax(self, x):
        x = (x - self._min_value) / (self._max_value - self._min_value)
        x = x * self._scale_range + self._min_scale
        return x

    def denormalize_minmax(self, x):
        x = (x - self._min_scale) / self._scale_range
        x = x * (self._max_value - self._min_value) + self._min_value
        return x


class ImageDataProcessor(tf.keras.layers.Layer):
    """Data processor for images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)

    def call(self, x, normalize=1):
        return tf.cond(tf.equal(normalize, 1), lambda: x / 255.0, lambda: x * 255.0)


class IdentityDataProcessor(tf.keras.layers.Layer):
    """Identity data processor"""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)

    def call(self, x, normalize=1):
        return x


class Standardizer(tf.keras.layers.Layer):
    """Standardizer."""

    def __init__(self, mean, variance, eta=1e-6, name=None, **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self._mean = mean
        self._variance = variance
        self._sd = tf.sqrt(variance)
        self._eta = eta

    def call(self, x, normalize=1):
        return tf.cond(
            tf.equal(normalize, 1),
            lambda: self.standardize(x),
            lambda: self.destandardize(x),
        )

    def standardize(self, x):
        x = (x - self._mean) / (self._sd + self._eta)
        return x

    def destandardize(self, x):
        x = x * (self._sd + self._eta) + self._mean
        return x
