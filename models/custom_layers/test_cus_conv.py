import unittest
import torch

try:
    import models.custom_layers.cus_layers as cus_layers
except ImportError:
    from . import cus_layers


class test_custom_conv_same_odd_kernel_size(unittest.TestCase):
    def test_odd_stride_2d(self):
        input_size = (2, 3, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 1
        kernel_size = 3
        downsampling_block = cus_layers.Conv2dSame(
            x.shape[1],
            5,
            kernel_size=kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv2dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_even_stride_2d(self):
        input_size = (2, 3, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 2
        kernel_size = 3
        downsampling_block = cus_layers.Conv2dSame(
            x.shape[1],
            5,
            kernel_size=kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv2dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_odd_stride_3d(self):
        input_size = (2, 3, 6, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 1
        kernel_size = 3
        downsampling_block = cus_layers.Conv3dSame(
            x.shape[1],
            5,
            kernel_size=kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv3dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_even_stride_3d(self):
        input_size = (2, 3, 6, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 2
        kernel_size = 3
        downsampling_block = cus_layers.Conv3dSame(
            x.shape[1],
            5,
            kernel_size=kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv3dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )


class test_custom_conv_same_random(unittest.TestCase):
    kernel_size = int(torch.randint(low=1, high=10, size=[1])) * 2 + 1

    def test_odd_stride_2d(self):
        input_size = (2, 3, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 1
        downsampling_block = cus_layers.Conv2dSame(
            x.shape[1],
            5,
            kernel_size=self.kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv2dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=self.kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_even_stride_2d(self):
        input_size = (2, 3, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 2
        downsampling_block = cus_layers.Conv2dSame(
            x.shape[1],
            5,
            kernel_size=self.kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv2dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=self.kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_odd_stride_3d(self):
        input_size = (2, 3, 6, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 1
        downsampling_block = cus_layers.Conv3dSame(
            x.shape[1],
            5,
            kernel_size=self.kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv3dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=self.kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )

    def test_even_stride_3d(self):
        input_size = (2, 3, 6, 6, 6)
        x = torch.randn(input_size)  # (B, C, H, W)
        stride = 2
        downsampling_block = cus_layers.Conv3dSame(
            x.shape[1],
            5,
            kernel_size=self.kernel_size,
            stride=stride,
        )
        y = downsampling_block(x)

        upsampling_block = cus_layers.Conv3dTransposeSame(
            y.shape[1],
            input_size[1],
            kernel_size=self.kernel_size,
            stride=stride,
        )
        x_hat = upsampling_block(y)

        self.assertEqual(
            torch.equal(torch.tensor(x.shape), torch.tensor(x_hat.shape)),
            True,
            f"Transpose should be equal to original. "
            f"Transpose shape: {x_hat.shape}; "
            f"Original shape: {x.shape}; "
            f"Intermediate shape: {y.shape}",
        )


if __name__ == "__main__":
    unittest.main()
