import torch
import torch.nn as nn
import math


class Patcher2d(nn.Module):
    def __init__(self, patch_size, name=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.name = name

    def forward(self, x):
        # x shape: (B, C, H, W)
        # side_len = len_x = len_y
        num_channels, side_len = x.shape[1], x.shape[2]
        num_tile = torch.div(side_len, self.patch_size, rounding_mode="floor")
        patch_dims = num_channels * (self.patch_size**2)

        patches = torch.reshape(
            x,
            [
                -1,
                num_channels,
                num_tile,
                self.patch_size,
                num_tile,
                self.patch_size,
            ],
        )
        patches = torch.permute(patches, [0, 2, 4, 3, 5, 1])
        patches = torch.reshape(patches, [-1, num_tile * num_tile, patch_dims])

        return patches


class InversePatcher2d(nn.Module):
    def __init__(self, patch_size, original_shape, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.patch_size = patch_size
        self.original_shape = original_shape

    def forward(self, x):
        batch_size, num_patches = x.shape[0], x.shape[1]
        num_tile = torch.exp(torch.log(torch.tensor(0) + num_patches) / 2.0)
        num_tile = num_tile.type(torch.int32)

        patches = torch.reshape(
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

        patches = torch.permute(patches, [0, 5, 1, 3, 2, 4])
        patches = torch.reshape(patches, [-1, *self.original_shape])

        return patches


class Patcher3d(nn.Module):
    def __init__(self, patch_size, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.patch_size = patch_size

    def forward(self, x):
        # x shape: (B, C, H, W, Z)
        num_channels, side_len = x.shape[1], x.shape[2]
        num_tile = side_len // self.patch_size
        patch_dims = num_channels * (self.patch_size**3)

        patches = torch.reshape(
            x,
            [
                -1,
                num_channels,
                num_tile,
                self.patch_size,
                num_tile,
                self.patch_size,
                num_tile,
                self.patch_size,
            ],
        )
        patches = torch.permute(patches, [0, 2, 4, 6, 3, 5, 7, 1])
        patches = torch.reshape(
            patches, [-1, num_tile * num_tile * num_tile, patch_dims]
        )

        return patches


class InversePatcher3d(nn.Module):
    def __init__(self, patch_size, original_shape, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.patch_size = patch_size
        self.original_shape = original_shape

    def forward(self, x):
        batch_size, num_patches = x.shape[0], torch.tensor(x.shape[1])
        num_tile = torch.exp(torch.log(num_patches) / 3.0)
        num_tile = num_tile.type(torch.int32)

        patches = torch.reshape(
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
        patches = torch.permute(patches, [0, 7, 1, 4, 2, 5, 3, 6])
        patches = torch.reshape(patches, [-1, *self.original_shape])

        return patches
