import numpy as np


def compare_replace(x, x_hat, tolerance=1e-2):
    # Compare pixels' values then replace if the difference is greater than tolerence
    (unsatisfied_values, unsatisfied_flatten_indices) = get_unsatisfied_values_indices(
        x, x_hat, tolerance=tolerance)
    x_hat = replace_pixel(x_hat, unsatisfied_values,
                          unsatisfied_flatten_indices[0])
    return x_hat


def replace_pixel(target, replace_values, indices):
    # Replace target values with `replace_values` at coresponding indices
    # indices are flatten -> need to get the original indices
    target_shape = target.shape
    for value, index in zip(replace_values, indices):
        full_index = reverse_flatten_indices(target_shape, index)
        target[full_index] = value
    return target


def reverse_flatten_indices(shape, flatten_index):
    # get the actual index from flatten_index
    full_index = []
    for len in reversed(shape):
        i = flatten_index % len
        flatten_index = flatten_index // len
        full_index.append(i)
    full_index.reverse()
    return tuple(full_index)


def get_unsatisfied_values_indices(x, x_hat, tolerance=1e-2):
    # get values and their coresponding flatten_indices
    # if the differnces are greater than the tolerence
    x = np.array(x).flatten()
    x_hat = np.array(x_hat).flatten()
    x_diff = np.absolute(x - x_hat)
    unsatisfied_values = x[x_diff > tolerance]
    unsatisfied_flatten_indices = np.nonzero(x_diff > tolerance)
    unsatisfied_flatten_indices = [row.tolist()
                                   for row in unsatisfied_flatten_indices]
    return (unsatisfied_values, unsatisfied_flatten_indices)
