import numpy as np


class DataNormalizer:

    def __init__(self, min_scale: float = 0.0, max_scale: float = 1.0):
        self.__max_value = 0.0
        self.__min_value = 0.0
        self.__scale_range = max_scale - min_scale
        self.__min_scale = min_scale
        self.__max_scale = max_scale

    def get_max_value(self):
        return self.__max_value

    def get_min_value(self):
        return self.__min_value

    def normalize_scientific_data(self, data):
        data = self.normalize_minmax(data)
        data = self.normalize_log10(data)
        return data

    def denormalize_scientific_data(self, data):
        data = self.denormalize_log10(data)
        data = self.denormalize_minmax(data)
        return data

    def normalize_log10(self, orig_data: np.ndarray) -> np.ndarray:
        # Scale data using log10
        return np.log10(np.array(orig_data))

    def denormalize_log10(self, normalized_data: np.ndarray) -> np.ndarray:
        return np.power(10, np.array(normalized_data))

    # Scale data using min-max scaler
    def normalize_minmax(self, orig_data: np.ndarray) -> np.ndarray:
        orig_data = np.array(orig_data)
        if (self.__max_value == 0 and self.__min_value == 0):
            self.__max_value = orig_data.max()
            self.__min_value = orig_data.min()
        normalized_data = (orig_data - self.__min_value) / \
            (self.__max_value - self.__min_value)
        normalized_data = normalized_data * self.__scale_range + self.__min_scale
        return normalized_data

    def denormalize_minmax(self, normalized_data: np.ndarray) -> np.ndarray:
        normalized_data = np.array(normalized_data)
        denormalized_data = (
            normalized_data - self.__min_scale) / self.__scale_range
        denormalized_data = denormalized_data * (self.__max_value - self.__min_value) \
            + self.__min_value
        return denormalized_data

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)

    def test_normalizer(data: np.ndarray):
        data_normalizer = DataNormalizer()
        print("maximum value before normalization: ", data.max())
        print("minimum value before normalization: ", data.min())

        print("\nNORMALIZING DATA")

        normalized_data = data_normalizer.normalize_log10(data)
        print("maximum normalized value after log10_normalization: ",
              normalized_data.max())
        print("minimum normalized value after log10_normalization: ",
              normalized_data.min())

        normalized_data = data_normalizer.normalize_minmax(normalized_data)
        print("maximum normalized value after minmax_normalization: ",
              normalized_data.max())
        print("minimum normalized value after minmax_normalization: ",
              normalized_data.min())

        print("\nDENORMALIZING DATA")

        denormalized_data = data_normalizer.denormalize_minmax(normalized_data)
        print("maximum denormalized value after minmax_dermalization: ",
              denormalized_data.max())
        print("minimum denormalized value after minmax_dermalization: ",
              denormalized_data.min())

        denormalized_data = data_normalizer.denormalize_log10(
            denormalized_data)
        print("maximum denormalized value after log10_dermalization: ",
              denormalized_data.max())
        print("minimum denormalized value after log10_dermalization: ",
              denormalized_data.min())
