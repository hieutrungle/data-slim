import unittest
import torch
import patcher as pch


class test_patcher_data3d(unittest.TestCase):
    def test_random(self):
        patch_size = 4
        patcher = pch.Patcher3D(patch_size)

        input_size = (2, 64, 8, 8, 8)
        x = torch.randn(input_size)
        original_shape = x.shape[1:]
        patches = patcher(x)
        inverse_patcher = pch.InversePatcher3D(patch_size, original_shape)
        inverse_patches = inverse_patcher(patches)
        self.assertEqual(
            torch.equal(inverse_patches, x),
            True,
            f"Inverse patches should be equal to original data. "
            f"Inverse patches shape: {inverse_patches.shape};"
            f"Original data shape: {x.shape}",
        )


class test_patcher_data2d(unittest.TestCase):
    def test_random(self):
        patch_size = 2
        patcher = pch.Patcher2D(patch_size)

        input_size = (2, 64, 16, 16)
        x = torch.randn(input_size)
        original_shape = x.shape[1:]
        patches = patcher(x)
        inverse_patcher = pch.InversePatcher2D(patch_size, original_shape)
        inverse_patches = inverse_patcher(patches)
        self.assertEqual(
            torch.equal(inverse_patches, x),
            True,
            f"Inverse patches should be equal to original data. "
            f"Inverse patches shape: {inverse_patches.shape};"
            f"Original data shape: {x.shape}",
        )


if __name__ == "__main__":
    unittest.main()
