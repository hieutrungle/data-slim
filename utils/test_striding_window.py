import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import unittest
import numpy as np
import sliding_window


def _convert_if_not_numpy(x, dtype=np.float32):
    """Convert the input `x` to a numpy array of type `dtype`.
    Args:
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    Returns:
        A numpy array.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=dtype)
    return x


def _num_windows_without_padding_naive(image_size, kernel_size, stride):
    kernel_size = _convert_if_not_numpy(kernel_size)
    if len(kernel_size.shape) == 0:
        width = kernel_size
        height = kernel_size
    elif len(kernel_size.shape) == 1:
        width = kernel_size[0]
        height = kernel_size[1]
    count = 0
    for i in range(0, image_size[0], stride):
        if i + height > image_size[0]:
            break
        for j in range(0, image_size[1], stride):
            if j + width > image_size[1]:
                break
            count += 1
    return count


def _num_windows_with_padding_naive(image_size, kernel_size, stride):
    kernel_size = _convert_if_not_numpy(kernel_size)
    if len(kernel_size.shape) == 0:
        width = kernel_size
        height = kernel_size
    elif len(kernel_size.shape) == 1:
        width = kernel_size[0]
        height = kernel_size[1]
    count = 0
    for i in range(0, image_size[0], stride):
        for j in range(0, image_size[1], stride):
            count += 1
    return count


def _get_padding_naive(image_size, kernel_size, stride):
    kernel_size = _convert_if_not_numpy(kernel_size, dtype=np.int16)
    if len(kernel_size.shape) == 0:
        width = kernel_size
        height = kernel_size
    elif len(kernel_size.shape) == 1:
        width = kernel_size[0]
        height = kernel_size[1]
    num_windows_width = 0
    num_windows_height = 0
    for i in range(0, image_size[0], stride):
        num_windows_width += 1
    for j in range(0, image_size[1], stride):
        num_windows_height += 1

    padding_width = (num_windows_width - 1) * stride + width - image_size[0]
    padding_height = (num_windows_height - 1) * stride + height - image_size[1]
    padding_width = 0 if padding_width < 0 else padding_width
    padding_height = 0 if padding_height < 0 else padding_height
    return [padding_width, padding_height]


class TestStridingWindowWithoutPadding(unittest.TestCase):
    """
    Test that the number of windows without padding
    # Test that the number of windows is correct
    # for a given image size and window size
    # and stride
    """

    def test_num_windows_without_padding_stride_1(self):
        image_size = (10, 10)
        window_size = (3, 3)

        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_stride_2(self):
        image_size = (10, 10)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_stride_1_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)

        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_stride_2_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_stride_3_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)

        stride = 3
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"Expect: {naive_result}",
        )

    def test_num_windows_without_padding_stride_4_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 4
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_random(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = int(rng.integers(1, 100, size=1))
        stride = int(rng.integers(1, 100, size=1))
        sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_without_padding_naive(
            image_size, window_size, stride
        )
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_without_padding_random_loop(self):
        for i in range(100):
            rng = np.random.default_rng(i)
            image_size = rng.integers(1, 200, size=2)
            window_size = int(rng.integers(1, 200, size=1))
            stride = int(rng.integers(1, 200, size=1))
            sw = sliding_window.SlidingWindow(window_size, stride, padding=False)
            result = sw.get_total_num_windows(image_size)
            naive_result = _num_windows_without_padding_naive(
                image_size, window_size, stride
            )
            self.assertEqual(
                result,
                naive_result,
                f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
            )


class TestStridingWindowWithPadding(unittest.TestCase):
    """Test the number of windows with padding"""

    def test_num_windows_with_padding_stride_1(self):
        image_size = (10, 10)
        window_size = (3, 3)
        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_stride_2(self):
        image_size = (10, 10)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_stride_1_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)

        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_stride_2_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_stride_3_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)
        stride = 3
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"Expect: {naive_result}",
        )

    def test_num_windows_with_padding_stride_4_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 4
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_random(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = int(rng.integers(1, 100, size=1))
        stride = int(rng.integers(1, 100, size=1))
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_total_num_windows(image_size)
        naive_result = _num_windows_with_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_num_windows_with_padding_random_loop(self):
        for i in range(100):
            rng = np.random.default_rng(i)
            image_size = rng.integers(1, 200, size=2)
            window_size = int(rng.integers(1, 200, size=1))
            stride = int(rng.integers(1, 200, size=1))
            sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
            result = sw.get_total_num_windows(image_size)
            naive_result = _num_windows_with_padding_naive(
                image_size, window_size, stride
            )
            self.assertEqual(
                result,
                naive_result,
                f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
            )


class TestPadding(unittest.TestCase):
    """Test the number of windows with padding"""

    def test_padding_stride_1(self):
        image_size = (10, 10)
        window_size = (3, 3)
        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        # naive_result = (2, 2)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_stride_2(self):
        image_size = (10, 10)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        # naive_result = (1, 1)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_stride_1_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)

        stride = 1
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_stride_2_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 2
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_stride_3_random_data_size(self):
        rng = np.random.default_rng(1)
        image_size = rng.integers(1, 100, size=2)
        window_size = (3, 3)
        stride = 3
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"Expect: {naive_result}",
        )

    def test_padding_stride_4_random_data_size(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = 3
        stride = 4
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_random(self):
        rng = np.random.default_rng(0)
        image_size = rng.integers(1, 100, size=2)
        window_size = int(rng.integers(1, 100, size=1))
        stride = int(rng.integers(1, 100, size=1))
        sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
        result = sw.get_padding(image_size)
        naive_result = _get_padding_naive(image_size, window_size, stride)
        self.assertEqual(
            result,
            naive_result,
            f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
        )

    def test_padding_random_loop(self):
        for i in range(100):
            rng = np.random.default_rng(i)
            image_size = rng.integers(1, 200, size=2)
            window_size = int(rng.integers(1, 200, size=1))
            stride = int(rng.integers(1, 200, size=1))
            sw = sliding_window.SlidingWindow(window_size, stride, padding=True)
            result = sw.get_padding(image_size)
            naive_result = _get_padding_naive(image_size, window_size, stride)
            self.assertEqual(
                result,
                naive_result,
                f"\nresult: {result}; image_size: {image_size}; window_size: {window_size}; stride: {stride}; Expect: {naive_result}",
            )


def _get_coor_pair(coors):
    all_coor_pairs = []
    row_coors, col_coors = coors
    for i in row_coors:
        for j in col_coors:
            all_coor_pairs.append(np.array([i, j]))
    all_coor_pairs = np.array(all_coor_pairs)
    return all_coor_pairs


class TestGetCoordinateGivenIndex(unittest.TestCase):
    """Given an index of a window, find the corresponding coordinates"""

    rng = np.random.default_rng(10)
    image_size = (1, 2400, 3600, 1)
    k = 128
    s = 120
    image = rng.random(image_size)

    def test_with_padding(self):
        sw = sliding_window.SlidingWindow(
            kernel_size=self.k, stride=self.s, padding=True
        )
        coors = sw.get_window_coors(self.image.shape)
        num_windows = sw.get_total_num_windows(self.image.shape[1:-1])
        total_windows = num_windows * self.image.shape[0]
        coors_from_indices = []
        for i in range(total_windows):
            if i >= num_windows:
                break
            coors_from_indices.append(sw.get_coor_given_index(coors, i % num_windows))
        coors_from_indices = np.array(coors_from_indices)
        all_coor_pairs = _get_coor_pair(coors)
        self.assertEqual(
            np.array_equal(coors_from_indices, all_coor_pairs),
            True,
            f"\n{sw}",
        )

    def test_without_padding(self):
        sw = sliding_window.SlidingWindow(
            kernel_size=self.k, stride=self.s, padding=False
        )
        coors = sw.get_window_coors(self.image.shape)
        num_windows = sw.get_total_num_windows(self.image.shape[1:-1])
        total_windows = num_windows * self.image.shape[0]
        coors_from_indices = []
        for i in range(total_windows):
            if i >= num_windows:
                break
            coors_from_indices.append(sw.get_coor_given_index(coors, i % num_windows))
        coors_from_indices = np.array(coors_from_indices)
        all_coor_pairs = _get_coor_pair(coors)
        self.assertEqual(
            np.array_equal(coors_from_indices, all_coor_pairs),
            True,
            f"\n{sw}",
        )


if __name__ == "__main__":
    unittest.main()
