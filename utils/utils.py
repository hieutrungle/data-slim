import os
import shutil
import numpy as np
import glob
import random
from matplotlib import figure
import matplotlib.pyplot as plt
import gc
import json
import pandas as pd
from models import res_conv2d_attn
from utils import logger
from pathlib import Path
from skimage.io.collection import alphanumeric_key
import torch
import argparse
import errno

DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


def get_data_info(data: np.ndarray):
    logger.log(f"data shape: {data.shape}")
    logger.log(f"maximum value: {data.max()}")
    logger.log(f"minimum value: {data.min()}\n")


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

    logger.log(f"train_ds.shape : {train_ds.shape}")
    logger.log(f"test_ds.shape : {test_ds.shape}")
    return train_ds, test_ds


def get_raw_data(file: str):
    """read binary files"""
    with open(file, "rb") as f:
        data = f.read()
        logger.log(
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
    return data


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_filenames(data_path, postfix=".nc", random_seed=None):

    if os.path.isfile(data_path):
        filenames = [data_path]
    elif os.path.isdir(data_path):
        filenames = sorted(
            glob.glob(os.path.join(data_path, f"*{postfix}")), key=alphanumeric_key
        )
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    return filenames


def get_netcdf_data_stats(data_path):
    if os.path.isfile(data_path):
        # data_path is a single file, look into the parent
        # directory to find the statistics file
        filename = Path(data_path)
        parent_folder = filename.parent.absolute()
        filename = glob.glob(os.path.join(parent_folder, "*.csv"))[0]
    elif os.path.isdir(data_path):
        filename = glob.glob(os.path.join(data_path, "*.csv"))[0]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)

    df = pd.read_csv(filename, index_col=0)
    stats = {}
    for col in ["mean", "median", "std"]:
        stats[col] = df[col].mean()
    return stats


def get_binary_data_stats(data):
    stats = {}
    stats["mean"] = np.nanmean(data)
    stats["median"] = np.nanmedian(data)
    stats["std"] = np.nanstd(data)
    return stats


def get_data_stats(data_type, data_path=None):
    if data_type == "netcdf":
        stats = get_netcdf_data_stats(data_path)
    elif data_type == "binary":
        data = get_raw_data(data_path)
        stats = get_binary_data_stats(data)
    return stats


def get_filenames_and_fillna_value(data_path, postfix=".nc"):
    filenames = get_filenames(data_path, postfix=postfix)
    try:
        stats = get_netcdf_data_stats(data_path)
        fillna_value = stats["mean"]
    except:
        logger.log("No statistics file found. Using default nan_values = 0.")
        fillna_value = 0
    return filenames, fillna_value


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


class NpEncoder(json.JSONEncoder):
    # json format for saving numpy array
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    min_value = np.min(diff_abs)
    max_value = np.max(diff_abs)

    fig = plt.figure(figsize=(12, 12))
    axes = fig.subplots(nrows=1, ncols=1, sharey=True)
    im = plt.imshow(diff_abs, vmin=min_value, vmax=max_value, cmap="seismic")
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


def model_defaults():
    """
    Defaults for model.
    """
    return dict(
        patch_size=64,
        patch_depth=-1,  # -1 means no depth
        patch_channels=1,
        pre_num_channels=32,
        num_channels=96,
        latent_dim=128,
        num_embeddings=256,
        num_residual_blocks=3,
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.0,
        ema_decay=0.99,
        commitment_cost=0.25,
        model_type="hierachical",
        name="",
    )


def train_defaults():
    """
    Defaults for training.
    """
    return dict(
        epochs=100,
        lr=4e-4,
        warm_up_portion=0.15,
        weight_decay=1e-4,  # optimizer
        log_interval=2_500,
        save_interval=10,
        train_verbose=False,
    )


def data_defaults():
    """
    Defaults for data.
    """
    return dict(
        data_type="netcdf",
        data_height=2400,
        data_width=3600,
        data_depth=-1,  # -1 means no depth
        data_channels=1,
        batch_size=8,
    )


def get_data_defaults():
    """
    Defaults for getting data.
    """
    return dict(
        start_time=0,  # time dimension start index
        end_time=3,  # time dimension end index
        start_pos_x=524,
        start_pos_y=234,
        end_pos_x=2541,
        end_pos_y=2054,
    )


def create_model(
    patch_size,
    patch_depth,
    patch_channels,
    pre_num_channels,
    num_channels,
    latent_dim,
    num_embeddings,
    num_residual_blocks,
    num_transformer_blocks,
    num_heads,
    dropout,
    ema_decay,
    commitment_cost,
    model_type,
    name,
):
    if patch_depth <= 0:
        input_shape = [1, patch_channels, patch_size, patch_size]
    else:
        input_shape = [1, patch_channels, patch_size, patch_size, patch_size]

    return res_conv2d_attn.VQCPVAE(
        patch_size,
        patch_depth,
        patch_channels,
        pre_num_channels,
        num_channels,
        latent_dim,
        num_embeddings,
        num_residual_blocks,
        num_transformer_blocks,
        num_heads,
        dropout,
        ema_decay,
        commitment_cost,
        model_type,
        name,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def log_args_and_device_info(args):
    """Logs arguments to the console."""
    logger.log(f"{'*'*23} {str(args.command).upper()} BEGIN {'*'*23}")
    if args.verbose == True:
        message = "\n"
        for k, v in args.__dict__.items():
            if isinstance(v, str):
                message += f"{k} = '{v}'\n"
            else:
                message += f"{k} = {v}\n"

        # Additional Info when using cuda
        message += f"\nUsing device: {str(DEVICE)}\n"
        if DEVICE.type != "cpu":
            for i in range(NUM_GPUS):
                mem_allot = round(torch.cuda.memory_allocated(i) / 1024**3, 1)
                mem_cached = round(torch.cuda.memory_reserved(i) / 1024**3, 1)
                message += f"{str(torch.cuda.get_device_name(i))}\n"
                message += "Memory Usage: " + "\n"
                message += "Allocated: " + str(mem_allot) + " GB" + "\n"
                message += "Cached: " + str(mem_cached) + " GB" + "\n"
        logger.log(f"{message}")
        logger.log(f"Pytorch version: {torch.__version__}\n")


def configure_args(args):
    if args.data_depth == -1:
        args.data_shape = (1, args.data_height, args.data_width, args.data_channels)
    else:
        args.data_shape = (
            1,
            args.data_depth,
            args.data_height,
            args.data_width,
            args.data_channels,
        )
    args.prefix_folder = (
        f"-patch_size_{args.patch_size}"
        f"-pre_num_channels_{args.pre_num_channels}-num_channels_{args.num_channels}"
        f"-latent_dim_{args.latent_dim}-num_embeddings_{args.num_embeddings}"
        f"-num_residual_blocks_{args.num_residual_blocks}-num_transformer_blocks_{args.num_transformer_blocks}"
    )
    if args.command == "train":
        args.model_path = args.model_path + f"-{args.model_type}-" + args.prefix_folder
    if args.name == "":
        args.name = args.model_type + "-" + args.model_path.rpartition("/")[-1]
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["NUM_CHANNELS"] = str(args.num_channels)
    os.environ["PRE_NUM_CHANNELS"] = str(args.pre_num_channels)


def get_checkpoint(args):
    # Get checkpoint to resume training
    checkpoint = {}
    ckpt_files = glob.glob(os.path.join(args.model_path, "checkpoints", "*.pt"))
    weight_path = ""
    if len(ckpt_files) > 0:
        if args.iter == -1:
            logger.log(f"Using the best model.")
            # Get the best model path.
            sorted_ckpt_files = {}
            for ckpt_file in ckpt_files:
                sorted_ckpt_files[ckpt_file] = float(
                    ckpt_file.split("-")[-1].split("=")[-1].rpartition(".")[0]
                )
            sorted_ckpt_files = dict(
                sorted(sorted_ckpt_files.items(), key=lambda item: item[1])
            )
            best_ckpt = list(sorted_ckpt_files.keys())[0]
            # get epoch of the best model path.
            resume_epoch = best_ckpt.split("/")[-1].split("-")[1].split("=")[1]
            checkpoint["resume_epoch"] = int(resume_epoch)
            weight_path = best_ckpt

        elif args.iter > 0:
            checkpoint["resume_epoch"] = args.iter
            logger.log(f"Using model at epoch {checkpoint['resume_epoch']}.")
            for ckpt_file in ckpt_files:
                resume_epoch = ckpt_file.split("/")[-1].split("-")[1]
                if int(resume_epoch.split("=")[1]) == args.iter:
                    weight_path = ckpt_file
                    break
    checkpoint["weight_path"] = weight_path
    return checkpoint


def load_model_with_checkpoint(model, weight_path, verbose=False):
    is_weight_loaded = False
    if os.path.isfile(weight_path):
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        is_weight_loaded = True
        if verbose:
            logger.log(f"weight path: {weight_path}")
        logger.log(f"Model weights successfully loaded.\n")
    else:
        logger.log(f"No pretrained model found at {weight_path}")
    return model, is_weight_loaded


def check_memory(check_name: str):
    free_mem, used_mem = torch.cuda.mem_get_info()
    logger.log(
        f"{check_name}: "
        f"Free memory: {free_mem / 1024**3} GB; Used memory: {used_mem / 1024**3} GB"
    )


def save_data_as_fig(
    da: float, da_min: float, da_max: float, output_path: str, fig_name: str
):
    fig = figure.Figure(figsize=(36, 18))
    ax = fig.subplots(1)
    ax.imshow(da, vmin=da_min, vmax=da_max, cmap="seismic")
    ax.axis("tight")
    ax.axis("off")
    ax.set_title(f"{fig_name}")
    ax.autoscale(False)
    fig.savefig(os.path.join(output_path, fig_name))


def save_reconstruction(x, x_hat, file_name, output_path=None):
    """save reconstruct and original data"""

    if output_path == None:
        output_path = os.path.join("./outputs/", file_name)
    mkdir_if_not_exist(output_path)

    # Get the min and max of all your data
    _min, _max = np.amin(x), np.amax(x)

    plt.rcParams.update({"font.size": 22})
    fig = figure.Figure(figsize=(36, 12))
    i = 0
    axes = fig.subplots(nrows=1, ncols=2, sharey=True)
    for image, name in zip([x, x_hat], ["original", "reconstructed"]):
        im = axes[i].imshow(image, vmin=_min, vmax=_max, cmap="seismic")
        axes[i].set_adjustable("box")
        axes[i].axis("tight")
        axes[i].axis("off")
        axes[i].set_title(f"{name}")
        axes[i].autoscale(False)
        i += 1
    fig.colorbar(im, ax=axes, location="bottom", fraction=0.08, pad=0.1, shrink=0.9)
    fig.suptitle(f"Visualization of the original and reconstructed data of {file_name}")
    fig.savefig(os.path.join(output_path, f"{file_name}_comparison.png"))

    save_data_as_fig(x, _min, _max, output_path, f"{file_name}_original.png")
    save_data_as_fig(x_hat, _min, _max, output_path, f"{file_name}_reconstructed.png")
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close("all")
    gc.collect()


def get_first_key(dictionary):
    for key in dictionary:
        return key
    raise IndexError


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")
