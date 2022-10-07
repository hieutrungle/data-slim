import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import unittest
import numpy as np
import padder


class test_padder_3d(unittest.TestCase):
    def test_single_batch(self):
        rng = np.random.default_rng(1)
        original_ds = rng.random((1, 50, 50, 50, 1))
        patch_size = 8

        padder3d = padder.Padder3D(patch_size, original_ds.shape)
        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_one_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 50, 3))

        patch_size = 1
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_multiple_batch(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 50, 3))

        patch_size = 8
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_odd_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 50, 3))

        patch_size = 7
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_devisible_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 50, 3))

        patch_size = 10
        padder3D = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3D.pad_data(original_ds)
        ds = padder3D.split_data(ds)

        unsplit_data = padder3D.unsplit_data(ds)
        unpad_ds = padder3D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_patch_size_equal_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 50, 3))

        patch_size = original_ds.shape[1]
        padder3D = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3D.pad_data(original_ds)
        ds = padder3D.split_data(ds)

        unsplit_data = padder3D.unsplit_data(ds)
        unpad_ds = padder3D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_single_batch_uneven_data_size(self):
        rng = np.random.default_rng(1)
        original_ds = rng.random((1, 51, 50, 50, 1))
        patch_size = 8

        padder3d = padder.Padder3D(patch_size, original_ds.shape)
        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_one_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 11, 22, 33, 3))

        patch_size = 1
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_multiple_batch_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 23, 33, 55, 3))

        patch_size = 8
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_odd_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 53, 62, 64, 3))

        patch_size = 7
        padder3d = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3d.pad_data(original_ds)
        ds = padder3d.split_data(ds)

        unsplit_data = padder3d.unsplit_data(ds)
        unpad_ds = padder3d.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_devisible_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 45, 50, 65, 3))

        patch_size = 10
        padder3D = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3D.pad_data(original_ds)
        ds = padder3D.split_data(ds)

        unsplit_data = padder3D.unsplit_data(ds)
        unpad_ds = padder3D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_patch_size_equal_data_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 42, 32, 11, 7))

        patch_size = original_ds.shape[1]
        padder3D = padder.Padder3D(patch_size, original_ds.shape)

        ds = padder3D.pad_data(original_ds)
        ds = padder3D.split_data(ds)

        unsplit_data = padder3D.unsplit_data(ds)
        unpad_ds = padder3D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )


class test_padder_2d(unittest.TestCase):
    def test_single_batch(self):
        rng = np.random.default_rng(1)
        original_ds = rng.random((1, 50, 50, 3))
        patch_size = 8

        padder2D = padder.Padder2D(patch_size, original_ds.shape)
        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_one_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 1))

        patch_size = 1
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_devisible_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 3))

        patch_size = 10
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_patch_size_equal_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 3))

        patch_size = original_ds.shape[1]
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_multiple_batch(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 3))

        patch_size = 8
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_odd_patch_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 50, 50, 3))

        patch_size = 7
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_single_batch_uneven_data_size(self):
        rng = np.random.default_rng(1)
        original_ds = rng.random((1, 31, 50, 3))
        patch_size = 8

        padder2D = padder.Padder2D(patch_size, original_ds.shape)
        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_one_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 21, 50, 1))

        patch_size = 1
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_devisible_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 11, 50, 3))

        patch_size = 10
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_patch_size_equal_data_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 41, 50, 3))

        patch_size = original_ds.shape[1]
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_multiple_batch_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 81, 50, 3))

        patch_size = 8
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )

    def test_odd_patch_size_uneven_data_size(self):
        rng = np.random.default_rng(2)
        original_ds = rng.random((2, 71, 50, 3))

        patch_size = 7
        padder2D = padder.Padder2D(patch_size, original_ds.shape)

        ds = padder2D.pad_data(original_ds)
        ds = padder2D.split_data(ds)

        unsplit_data = padder2D.unsplit_data(ds)
        unpad_ds = padder2D.remove_pad_data(unsplit_data)

        self.assertEqual(
            np.array_equal(original_ds, unpad_ds),
            True,
            f"Should be equal. original_ds shape: {original_ds.shape} - processed_shape: {unpad_ds.shape}",
        )


if __name__ == "__main__":
    unittest.main()
