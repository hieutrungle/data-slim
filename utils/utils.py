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
from . import logger
from pathlib import Path
from skimage.io.collection import alphanumeric_key
import torch
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_filenames(data_path, prefix=".nc", random_seed=None):
    if data_path.endswith(".nc"):
        filenames = [data_path]
    else:
        filenames = sorted(
            glob.glob(os.path.join(data_path, "*" + prefix)), key=alphanumeric_key
        )
    # random.seed(random_seed)
    # random.shuffle(filenames)
    return filenames


def get_data_statistics(data_path):
    if data_path.endswith(".nc"):
        filename = Path(data_path)
        parent_folder = filename.parent.absolute()
        filename = glob.glob(os.path.join(parent_folder, "*.csv"))[0]
    else:
        filename = glob.glob(os.path.join(data_path, "*.csv"))[0]

    df = pd.read_csv(filename, index_col=0)
    stats = {}
    for col in ["mean", "median", "std"]:
        stats[col] = df[col].mean()
    return stats


def get_filenames_and_fillna_value(data_path, prefix=".nc"):
    filenames = get_filenames(data_path, prefix=prefix)
    try:
        stats = get_data_statistics(data_path)
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


def model_defaults():
    """
    Defaults for model.
    """
    return dict(
        patch_size=64,
        patch_depth=-1,  # -1 means no depth
        patch_channels=1,
        pre_num_channels=8,
        num_channels=16,
        latent_dim=64,
        num_embeddings=128,
        num_residual_blocks=3,
        num_transformer_blocks=2,
        num_heads=4,
        dropout=0.0,
        ema_decay=0.99,
        commitment_cost=0.25,
        name="Compressor",
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
        log_interval=10,
        save_interval=10,
        train_verbose=False,
    )


def data_defaults():
    """
    Defaults for data.
    """
    return dict(
        data_height=2400,
        data_width=3600,
        data_depth=-1,  # -1 means no depth
        data_channels=1,
        batch_size=8,
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
    # return {k: args.__dict__[k] for k in keys}


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
            message += k + " = " + str(v) + "\n"
        # Additional Info when using cuda
        if DEVICE.type == "cuda":
            message += f"\nUsing device: {str(DEVICE)}\n"
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
    args.model_path = args.model_path + args.prefix_folder
    args.name = args.model_path.rpartition("/")[-1]


def get_resume_checkpoint(args):
    resume_checkpoint = {}
    ckpt_files = glob.glob(os.path.join(args.model_path, "checkpoints", "*.pt"))
    weight_path = ""
    if args.iter == -1:
        logger.log(f"Resume training using the best model.")
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
        resume_checkpoint["resume_epoch"] = int(resume_epoch)
        weight_path = best_ckpt
    elif args.iter > 0:
        resume_checkpoint["resume_epoch"] = args.iter
        logger.log(f"Resume training at epoch {resume_checkpoint['resume_epoch']}.")
        for ckpt_file in ckpt_files:
            resume_epoch = ckpt_file.split("/")[-1].split("-")[1]
            if int(resume_epoch.split("=")[1]) == args.iter:
                weight_path = ckpt_file
                break
    resume_checkpoint["weight_path"] = weight_path
    return resume_checkpoint
