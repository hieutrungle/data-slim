import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdn
    import cus_conv
except:
    from . import gdn, cus_conv
import numpy as np


class DownSamplingBlock(nn.Module):
    """Downsampling block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

        self.gdn = gdn.GDN(c_in)
        self.conv0 = cus_conv.Conv2dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        x = self.gdn(x)
        x = self.conv0(x)
        return x


class UpSamplingBlock(nn.Module):
    """Upsampling block"""

    def __init__(
        self, c_in, c_out, kernel_size, stride, padding=0, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.conv1 = cus_conv.Conv2dTransposeSame(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            # padding=padding,
            # # output_padding=stride - 1,
        )

    def forward(self, x):
        x = self.igdn(x)
        x = self.conv1(x)
        return x


class DownSamplingResBlock2D(nn.Module):
    """Downsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4
        self.gdn = gdn.GDN(c_in)

        self.shortcut = cus_conv.Conv2dSame(c_in, c_out, kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.conv0 = cus_conv.Conv2dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv1 = cus_conv.Conv2dSame(c_out, c_hidden, kernel_size=1, stride=1)
        self.conv2 = cus_conv.Conv2dSame(c_hidden, c_out, kernel_size=1, stride=1)
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.gdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        return x + x_shortcut


class UpSamplingResBlock2D(nn.Module):
    """Upsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.shortcut = cus_conv.Conv2dTransposeSame(
            c_in, c_out, kernel_size=1, stride=stride
        )
        self.act = nn.GELU()
        self.conv0 = cus_conv.Conv2dTransposeSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv1 = cus_conv.Conv2dTransposeSame(
            c_out, c_hidden, kernel_size=1, stride=1
        )
        self.conv2 = cus_conv.Conv2dTransposeSame(
            c_hidden, c_out, kernel_size=1, stride=1
        )
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.igdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        return x + x_shortcut


class DownSamplingResBlock3D(nn.Module):
    """Downsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.gdn = gdn.GDN(c_in)
        self.shortcut = cus_conv.Conv3dSame(c_in, c_out, kernel_size=1, stride=stride)
        self.act = nn.GELU()
        self.conv1 = cus_conv.Conv3dSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv2 = cus_conv.Conv3dSame(c_out, c_hidden, kernel_size=1, stride=1)
        self.conv3 = cus_conv.Conv3dSame(c_hidden, c_out, kernel_size=1, stride=1)
        self.cells = [self.conv1, self.conv2, self.act, self.conv3]

    def forward(self, x):
        x = self.gdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        return x


class UpSamplingResBlock3D(nn.Module):
    """Upsampling res block"""

    def __init__(self, c_in, c_out, kernel_size, stride, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.kernel_size = kernel_size
        self.stride = stride
        c_hidden = c_out * 4

        self.igdn = gdn.GDN(c_in, inverse=True)
        self.shortcut = cus_conv.Conv3dTransposeSame(
            c_in, c_out, kernel_size=1, stride=stride
        )
        self.act = nn.GELU()
        self.conv0 = cus_conv.Conv3dTransposeSame(
            c_in, c_out, kernel_size=kernel_size, stride=stride
        )
        self.conv1 = cus_conv.Conv3dTransposeSame(
            c_out, c_hidden, kernel_size=1, stride=1
        )
        self.conv2 = cus_conv.Conv3dTransposeSame(
            c_hidden, c_out, kernel_size=1, stride=1
        )
        self.cells = [self.conv0, self.conv1, self.act, self.conv2]

    def forward(self, x):
        x = self.igdn(x)
        x_shortcut = self.shortcut(x)
        for cell in self.cells:
            x = cell(x)
        x = x + x_shortcut
        return x


class ForwardConv1d(nn.Module):
    """Forward Conv1d"""

    def __init__(
        self, c_in, c_out, c_hidden, kernel_size=1, stride=1, name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.gdn = gdn.GDN(c_in)
        self.flatten = nn.Flatten()
        self.conv1d = cus_conv.Conv1dSame(
            c_hidden, c_out, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        # (B, C, ...)
        x = self.gdn(x)
        # (B, C)
        x = self.flatten(x)
        # (B, C, 1)
        x = torch.unsqueeze(x, axis=-1)
        x = self.conv1d(x)
        return x


class ForwardMLP(nn.Module):
    """Forward MLP"""

    def __init__(self, c_in, c_out, c_hidden, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.gdn = gdn.GDN(c_in)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(c_hidden, c_out)

    def forward(self, x):
        # (B, C, ...)
        x = self.gdn(x)
        # (B, C)
        x = self.flatten(x)
        x = self.linear(x)
        return x


def _test_res_down_2d():
    print(f"\nTest DownSamplingResBlock2D")
    input_size = (2, 3, 6, 6)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    downsampling_res_block = DownSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="downsampling_res_block"
    )
    results = downsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 6, 6)  # (B, C, H, W)
    stride = 2
    downsampling_res_block = DownSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="downsampling_res_block"
    )
    results = downsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    input_size = (2, 3, 7, 7)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    downsampling_res_block = DownSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="downsampling_res_block"
    )
    results = downsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 7, 7)  # (B, C, H, W)
    stride = 2
    downsampling_res_block = DownSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="downsampling_res_block"
    )
    results = downsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")


def _test_res_up_2d():
    print(f"\nTest UpSamplingResBlock2D")
    input_size = (2, 3, 6, 6)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    upsampling_res_block = UpSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="upsampling_res_block"
    )
    results = upsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 6, 6)  # (B, C, H, W)
    stride = 2
    upsampling_res_block = UpSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="upsampling_res_block"
    )
    results = upsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    input_size = (2, 3, 7, 7)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    upsampling_res_block = UpSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="upsampling_res_block"
    )
    results = upsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 7, 7)  # (B, C, H, W)
    stride = 2
    upsampling_res_block = UpSamplingResBlock2D(
        a.shape[1], 5, kernel_size=3, stride=stride, name="upsampling_res_block"
    )
    results = upsampling_res_block(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")


def _test_res_2d():
    print(f"\nTest UpSamplingResBlock2D and DownSamplingResBlock2D")
    x = torch.randn(1, 3, 64, 64)  # (B, C, H, W)
    stride = 2
    y = x
    for i in range(3):
        downsampling_res_block = DownSamplingResBlock2D(
            y.shape[1], 5, kernel_size=5, stride=stride, name="downsampling_res_block"
        )
        y = downsampling_res_block(y)
        print(f"y.shape: {y.shape}")

    x_hat = y
    for i in range(3):
        upsampling_res_block = UpSamplingResBlock2D(
            x_hat.shape[1], 3, kernel_size=5, stride=stride, name="upsampling_res_block"
        )
        x_hat = upsampling_res_block(x_hat)
        print(f"x_hat.shape: {x_hat.shape}")

    print(f"input shape: {x.shape}; stride: {stride}; results.shape: {x_hat.shape}")


def _test_res_3d():
    print(f"\nTest UpSamplingResBlock3D and DownSamplingResBlock3D")
    x = torch.randn(1, 3, 64, 64, 64)  # (B, C, H, W)
    stride = 2
    y = x
    for i in range(3):
        downsampling_res_block = DownSamplingResBlock3D(
            y.shape[1], 5, kernel_size=5, stride=stride, name="downsampling_res_block"
        )
        y = downsampling_res_block(y)
        print(f"y.shape: {y.shape}")

    x_hat = y
    for i in range(3):
        upsampling_res_block = UpSamplingResBlock3D(
            x_hat.shape[1], 3, kernel_size=5, stride=stride, name="upsampling_res_block"
        )
        x_hat = upsampling_res_block(x_hat)
        print(f"x_hat.shape: {x_hat.shape}")

    print(f"input shape: {x.shape}; stride: {stride}; results.shape: {x_hat.shape}")


def _test_forward_conv():
    print(f"\nTest Forward1D")
    input_size = (2, 3, 6, 6)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    forward_1d = ForwardConv1d(
        a.shape[1],
        np.prod(input_size[1:]),
        c_hidden=np.prod(input_size[1:]),
        kernel_size=3,
        stride=stride,
        name="forward_1d",
    )
    results = forward_1d(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 6, 6)  # (B, C, H, W)
    stride = 2
    forward_1d = ForwardConv1d(
        a.shape[1],
        np.prod(input_size[1:]),
        c_hidden=np.prod(input_size[1:]),
        kernel_size=3,
        stride=stride,
        name="forward_1d",
    )
    results = forward_1d(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    input_size = (2, 3, 7, 7)
    a = torch.randn(input_size)  # (B, C, H, W)
    stride = 1
    forward_1d = ForwardConv1d(
        a.shape[1],
        np.prod(input_size[1:]),
        c_hidden=np.prod(input_size[1:]),
        kernel_size=3,
        stride=stride,
        name="forward_1d",
    )
    results = forward_1d(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")

    a = torch.randn(2, 3, 7, 7)  # (B, C, H, W)
    stride = 2
    forward_1d = ForwardConv1d(
        a.shape[1],
        np.prod(input_size[1:]),
        c_hidden=np.prod(input_size[1:]),
        kernel_size=3,
        stride=stride,
        name="forward_1d",
    )
    results = forward_1d(a)
    print(f"input shape: {a.shape}; stride: {stride}; results.shape: {results.shape}")


def main():

    _test_res_down_2d()
    _test_res_up_2d()
    _test_res_2d()
    _test_res_3d()
    _test_forward_conv()


if __name__ == "__main__":
    main()
