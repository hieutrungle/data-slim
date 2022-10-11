import torch

try:
    from cus_blocks import *
except:
    from .cus_blocks import *


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
