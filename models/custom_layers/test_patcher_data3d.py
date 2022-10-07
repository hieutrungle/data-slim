import unittest
import tensorflow as tf
import patcher_data3d


class test_patcher_data3d(unittest.TestCase):
    def test_random(self):
        patch_size = 4
        patcher = patcher_data3d.Patcher(patch_size)

        data = tf.random.normal([300, 8, 8, 8, 32])
        original_shape = data.shape[1:]
        patches = patcher(data)
        inverse_patcher = patcher_data3d.InversePatcher(patch_size, original_shape)
        inverse_patches = inverse_patcher(patches)
        self.assertEqual(
            tf.math.reduce_all(tf.math.equal(inverse_patches, data)),
            True,
            f"Inverse patches should be equal to original data. "
            f"Inverse patches shape: {inverse_patches.shape};"
            f"Original data shape: {data.shape}",
        )


if __name__ == "__main__":
    unittest.main()
