import torch
import torch.nn as nn


class ScientificDataProcessor(nn.Module):
    """Data processor for scientific data."""

    def __init__(
        self, ds_min, ds_max, min_scale=1.1, max_scale=2.7, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self._max_value = ds_max
        self._min_value = ds_min
        self._scale_range = max_scale - min_scale
        self._min_scale = min_scale
        self._max_scale = max_scale
        self.name = name

    def forward(self, x, normalize=1):
        if normalize:
            x = torch.log(self._normalize_minmax(x))
        else:
            x = self._denormalize_minmax(torch.exp(x))
        return x

    def _normalize_minmax(self, x):
        x = (x - self._min_value) / (self._max_value - self._min_value)
        x = x * self._scale_range + self._min_scale
        return x

    def _denormalize_minmax(self, x):
        x = (x - self._min_scale) / self._scale_range
        x = x * (self._max_value - self._min_value) + self._min_value
        return x


class ImageDataProcessor(nn.Module):
    """Data processor for images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def forward(self, x, normalize=1):
        if normalize:
            x = x / 255.0
        else:
            x = x * 255.0
        return x


class IdentityDataProcessor(nn.Module):
    """Identity data processor"""

    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def forward(self, x, normalize=1):
        return x


class Standardizer(nn.Module):
    """Standardizer."""

    def __init__(self, mean, variance, eta=1e-6, name=None, **kwargs):
        super().__init__(**kwargs)
        self._mean = torch.tensor(mean)
        self._variance = torch.tensor(variance)
        self._sd = torch.sqrt(self._variance)
        self._eta = torch.tensor(eta)
        self.name = name

    def forward(self, x, normalize=1):
        if normalize == 1:
            x = self._standardize(x)
        else:
            x = self._destandardize(x)
        return x

    def _standardize(self, x):
        x = (x - self._mean) / (self._sd + self._eta)
        return x

    def _destandardize(self, x):
        x = x * (self._sd + self._eta) + self._mean
        return x
