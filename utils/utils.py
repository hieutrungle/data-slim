import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import shutil
import errno
import numpy as np
import glob
import random
import logging
import sys
from matplotlib import figure
import matplotlib.pyplot as plt
import gc
import tensorflow as tf
import json
import pandas as pd
from numpy import linalg as LA
import tensorly as tl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string)


def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)


def check_image_size(image, patch_size):
    shape = tf.shape(image)
    return shape[0] >= patch_size and shape[1] >= patch_size and shape[-1] == 3


def crop_image(image, patch_size):
    image = tf.image.random_crop(image, (patch_size, patch_size, 3))
    return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def get_data_info(data: np.ndarray):
    print(f"data shape: {data.shape}")
    print(f"maximum value: {data.max()}")
    print(f"minimum value: {data.min()}\n")


def get_files(train_glob, random_seed=None):
    """Get all file in a given path"""
    random.seed(random_seed)
    files = glob.glob(os.path.join(train_glob, "*"))
    random.shuffle(files)
    return files


def split_train_val_test_tf(
    ds,
    ds_size,
    train_split=0.95,
    val_split=0.025,
    test_split=0.025,
    shuffle=True,
    shuffle_size=10000,
    random_seed=None,
):
    # Use tensorflow to split data into train + val + test, compatible with tf.data API
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=random_seed)

    train_size = int(np.ceil(train_split * ds_size))
    val_size = int(np.ceil(val_split * ds_size))

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def split_train_val_tf(
    ds, ds_size, train_size=0.95, shuffle=True, shuffle_size=10000, random_seed=None
):
    # Use tensorflow to split data into train + val, compatible with tf.data API
    assert train_size <= 1, "Split proportion must be in [0, 1]"
    assert train_size >= 0, "Split proportion must be in [0, 1]"

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=random_seed)

    train_size = int(np.ceil(train_size * ds_size))

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)

    return train_ds, val_ds


def split_train_val_test_pd(
    df, train_split=0.95, val_split=0.025, test_split=0.025, random_seed=None
):
    # Use pandas to split data into train + val + test, compatible with pandas DataFrame
    assert (train_split + test_split + val_split) == 1

    # Only allows for equal validation and test splits
    assert val_split == test_split

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_seed)
    indices_or_sections = [
        int(train_split * len(df)),
        int((1 - val_split - test_split) * len(df)),
    ]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


def split_train_val_pd(df, train_size=0.95, random_seed=None):
    # Use pandas to split data into train + val, compatible with pandas DataFrame
    assert train_size <= 1, "Split proportion must be in [0, 1]"
    assert train_size >= 0, "Split proportion must be in [0, 1]"

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_seed)
    indices_or_sections = [int(train_size * len(df)), len(df)]

    train_ds, val_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds


def split_train_test_np(df, train_size=0.95, random_seed=None):
    assert train_size <= 1, "Split proportion must be in [0, 1]"
    assert train_size >= 0, "Split proportion must be in [0, 1]"

    np.random.seed(random_seed)
    split = np.random.choice(range(df.shape[0]), int(train_size * df.shape[0]))
    train_ds = df[split]
    test_ds = df[~split]

    print(f"train_ds.shape : {train_ds.shape}")
    print(f"test_ds.shape : {test_ds.shape}")
    return train_ds, test_ds


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_raw_data(file: str):
    """read binary files"""
    with open(file, "rb") as f:
        data = f.read()
        logging.info(
            f"Name of the training dataset: {file}; "
            f"Length of file: {len(data):,} bytes"
        )

        if file[-4:] == ".txt":
            data = data.decode().split("\n")
        # If the input file is a standard file,
        # there is a chance that the last line could simply be an empty line;
        # if this is the case, then remove the empty line
        if data[len(data) - 1] == "":
            data.remove("")
            data = np.array(data)
            data = data.astype("float32")
        else:
            # The input file is a binary file
            data = tf.io.decode_raw(data, tf.float32)
    return data


def mkdir_storage(model_dir, resume={}):
    if os.path.exists(os.path.join(model_dir, "summaries")):
        if len(resume) == 0:
            # val = input("The model directory %s exists. Overwrite? (y/n) " % model_dir)
            # print()
            # if val == 'y':
            if os.path.exists(os.path.join(model_dir, "summaries")):
                shutil.rmtree(os.path.join(model_dir, "summaries"))
            if os.path.exists(os.path.join(model_dir, "checkpoints")):
                shutil.rmtree(os.path.join(model_dir, "checkpoints"))

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, "summaries")
    mkdir_if_not_exist(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    mkdir_if_not_exist(checkpoints_dir)
    return summaries_dir, checkpoints_dir


def get_console_logger(name, set_level):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: Line %(lineno)s - %(message)s"
    )
    # set up logging to console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.setLevel(logging.info)
    return logger


def save_data_as_fig(data: float, output_path: str, fig_name: str):
    fig = figure.Figure(figsize=(36, 18))
    ax = fig.subplots(1)
    ax.imshow(data)
    ax.axis("tight")
    ax.axis("off")
    ax.set_title(f"{fig_name}")
    ax.autoscale(False)
    fig.savefig(os.path.join(output_path, fig_name))


def plot_save_reconstruct(x, x_hat, model_name, output_path=None):
    """plot reconstruct and original data"""

    if output_path == None:
        output_path = "./outputs/"
    mkdir_if_not_exist(output_path)

    combined_data = np.array(x)
    # Get the min and max of all your data
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    fig = figure.Figure(figsize=(36, 12))
    i = 0
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)
    for image, name in zip([x_hat, x], ["reconstructed", "original"]):
        im = axes[i].imshow(image, vmin=_min, vmax=_max)
        axes[i].set_adjustable("box")
        axes[i].axis("tight")
        axes[i].axis("off")
        axes[i].set_title(f"{name}")
        axes[i].autoscale(False)
        i += 1
    fig.colorbar(im, ax=axes, location="bottom", fraction=0.08, pad=0.1, shrink=0.8)
    fig.suptitle(
        f"Visualization of the original and reconstructed data of {model_name}"
    )
    fig.savefig(os.path.join(output_path, f"{model_name}_comparison.png"))

    save_data_as_fig(x, output_path, f"{model_name}_original.png")
    save_data_as_fig(x_hat, output_path, f"{model_name}_reconstructed.png")
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close("all")
    gc.collect()


def get_eval_commands(path_to_saved_models, pattern):
    files = get_files(path_to_saved_models)
    for file in files:
        if file.find(pattern) != -1:
            print(
                f"python eval_saved_model.py -V --model_path {os.path.join(path_to_saved_models, file.rpartition('/')[-1])}"
            )


def get_data_statistics(ds):
    if not isinstance(ds, np.ndarray):
        ds = np.concatenate(list(ds), axis=0)
    mean = np.mean(ds)
    var = np.var(ds)
    min_value = np.min(ds)
    max_value = np.max(ds)
    return (mean, var, min_value, max_value)


def get_ocean_statistics(
    da, mask_value, verbose=False, saved_path="da_statistics.csv", is_save=True
):
    # compute data statistics
    # While computing the mean, missing values and land values are ignored.
    da_mask = da.where((da > mask_value))
    df = get_da_statistics(
        da_mask, verbose=verbose, saved_path=saved_path, is_save=is_save
    )
    return df


def get_da_statistics(da, verbose=False, saved_path="da_statistics.csv", is_save=True):
    mkdir_if_not_exist(saved_path)
    mins = da.min(["nlat", "nlon"]).compute()
    maxs = da.max(["nlat", "nlon"]).compute()
    means = da.mean(["nlat", "nlon"]).compute()
    medians = da.median(["nlat", "nlon"]).compute()
    stds = da.std(["nlat", "nlon"]).compute()

    np_mins = mins.to_numpy()
    np_maxs = maxs.to_numpy()
    np_means = means.to_numpy()
    np_medians = medians.to_numpy()
    np_stds = stds.to_numpy()

    columns = ["min", "max", "mean", "median", "std"]
    df = pd.DataFrame([np_mins, np_maxs, np_means, np_medians, np_stds]).T
    df.columns = columns
    if verbose:
        df.describe()
    if is_save:
        df.to_csv(saved_path)
    return df


def getclosest_ij(lats, longs, lat_pt, long_pt):
    # a function to find the index of the point closest pt
    # (in squared distance) to give lat/lon value.

    # find squared distance of every point on grid
    dist_sq = (lats - lat_pt) ** 2 + (longs - long_pt) ** 2
    # 1D index of minimum dist_sq element
    minindex_flattened = dist_sq.argmin()
    # Get 2D index for latvals and lonvals arrays from 1D index
    return np.unravel_index(minindex_flattened, lats.shape)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find_rank_with_tolerance(da, tol=1e-2):
    da = da.numpy()
    sqr_x = np.square(da)
    normsqr_x = np.sum(sqr_x)
    eigsum_thresh = np.square(tol) * normsqr_x / da.ndim

    rank = np.zeros(da.ndim)

    for mode in range(da.ndim):
        unfolded = tl.unfold(da, mode)
        # left side SVD
        gram = np.matmul(unfolded, np.transpose(unfolded))
        d, v = LA.eig(gram)
        # Reverse sort
        eigvec = np.msort(d)[::-1]
        eigsum = np.cumsum(eigvec[::-1], axis=0)[::-1]
        rank[mode] = np.count_nonzero(eigsum > eigsum_thresh)
    rank = rank.astype(int)
    print("rank = ", rank)


def evaluate_from_tucker_core(cp_tensor, original_data, mask, verbose=False):

    da = original_data
    rec_da = tl.tucker_to_tensor(cp_tensor)
    da = da * mask
    rec_da = rec_da * mask
    mse = np.mean(np.square(da - rec_da))
    error = tl.norm(da - rec_da) / tl.norm(da)
    abs_error = tl.max(tl.abs(rec_da - da))
    print(f"mse = {mse}")
    print(f"error: {error}, asb_error: {abs_error}")


def plot_results(x, x_hat):
    fig = plt.figure(figsize=(36, 12))
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)
    min_value = np.min(x)
    max_value = np.max(x)
    for i, (image, name) in enumerate(zip([x_hat, x], ["reconstructed", "original"])):
        im = axes[i].imshow(image, vmin=min_value, vmax=max_value, cmap="seismic")
        axes[i].set_adjustable("box")
        axes[i].axis("tight")
        axes[i].axis("off")
        axes[i].set_title(f"{name}")
        axes[i].autoscale(False)
    fig.colorbar(
        im,
        ax=axes,
        location="bottom",
        fraction=0.08,
        pad=0.1,
        shrink=0.8,
        cmap="seismic",
    )
    fig.suptitle(f"Visualization of the original and reconstructed data")

    diff = x_hat - x
    diff_abs = np.abs(diff)

    fig = plt.figure(figsize=(12, 12))
    axes = fig.subplots(nrows=1, ncols=1, sharey=True)
    im = plt.imshow(diff_abs, cmap="seismic")
    fig.colorbar(
        im,
        ax=axes,
        location="bottom",
        fraction=0.08,
        pad=0.1,
        shrink=0.8,
        cmap="seismic",
    )
    fig.suptitle(
        f"Visualization of the difference between original and reconstructed data"
        f"\n(mse: {np.mean(diff**2):.2f}); max_error: {np.max(diff_abs):.2f} at {np.where(diff_abs == np.max(diff_abs))}"
    )
    plt.show()
