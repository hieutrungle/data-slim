import sys
import os
import numpy as np
import os
import json
import gc
import time
from utils import utils, logger
from torch.utils.data import DataLoader
import torch

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if DEVICE != "cpu":
#     NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
# else:
#     NUM_GPUS = 0
IS_CHECKING_MEMORY = False
DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


def compress_step(model, x, batch_size=4):
    """Compress step."""
    x = DataLoader(x, batch_size=batch_size, shuffle=False)
    if model.model_type.lower().find("hierachical") != -1:
        z_tensor, y_tensor = [], []
        model_device = next(model.parameters()).device
        for i, da in enumerate(x):
            compressed = model.compress(da.to(model_device).type(torch.float))
            y_tensor.append(compressed[0].detach().cpu())
            z_tensor.append(compressed[1].detach().cpu())
        y_tensor = torch.cat(y_tensor, axis=0).cpu()
        z_tensor = torch.cat(z_tensor, axis=0).cpu()
        tensors = (y_tensor, z_tensor, *compressed[2:])

    elif model.model_type.lower().find("res_1") != -1:
        tensor = []
        model_device = next(model.parameters()).device
        for i, da in enumerate(x):
            compressed = model.compress(da.to(model_device).type(torch.float))
            tensor.append(compressed[0].detach().cpu())
        tensor = torch.cat(tensor, axis=0).cpu()
        tensors = (tensor, *compressed[1:])

    else:
        raise ValueError("Invalid model type.")

    return tensors


def compress(model, x, mask=None, verbose=False, is_benchmarking=False):
    """Compress data."""
    if IS_CHECKING_MEMORY:
        utils.check_memory("Before compression")

    if mask is None:
        mask = torch.ones_like(x)

    start_time = time.perf_counter()
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    tensors = compress_step(model, x, batch_size=batch_size)
    encoding_time = time.perf_counter() - start_time

    if IS_CHECKING_MEMORY:
        utils.check_memory("After compression")

    x_hat = None
    if verbose or is_benchmarking:
        # decoding to get stats
        packed = tensors
        start_time = time.perf_counter()
        x_hat = decompress(model, packed, mask)
        decoding_time = time.perf_counter() - start_time

    if verbose:
        logger.info(f"Encoding time: {encoding_time:0.2f} seconds")
        results = get_metrics(x * mask, x_hat, packed)
        results.update(
            {
                "model_name": model.name,
                "encoding_time": encoding_time,
                "decoding_time": decoding_time,
            }
        )
        save_results(**results)

    gc.collect()
    if x_hat is not None:
        return tensors, x_hat
    else:
        return tensors


def save_compressed(output_file, tensors):
    tensors = [np.array(tensor) for tensor in tensors]
    np.savez_compressed(output_file, *tensors)


def load_compressed(input_file):
    with np.load(input_file, allow_pickle=True) as f:
        tensors = [f[key] for key in f.files]
    return tensors


def decompress_step(model, tensors, batch_size=4):
    """Decompress step."""
    model_device = next(model.parameters()).device
    x_hat = []

    if model.model_type.lower().find("hierachical") != -1:
        z_shape = np.array(tensors[3])
        num_hidden_z_tensor = int(np.prod(z_shape[:-1]) * batch_size)
        # make data for fast data loading
        z_quantized = DataLoader(
            tensors[1].to(model_device).type(torch.int64),
            batch_size=num_hidden_z_tensor,
            shuffle=False,
        )

        y_shape = np.array(tensors[2])
        num_hidden_y_tensor = int(np.prod(y_shape[:-1]) * batch_size)
        # make data for fast data loading
        y_quantized = DataLoader(
            tensors[0].to(model_device).type(torch.int64),
            batch_size=num_hidden_y_tensor,
            shuffle=False,
        )

        for i, da in enumerate(zip(y_quantized, z_quantized)):
            # da = da.to(model_device).type(torch.int64)
            decompressed = model.decompress(*da, *tensors[2:])
            x_hat.append(decompressed.detach().cpu())
        x_hat = torch.cat(x_hat, axis=0).cpu()

    elif model.model_type.lower().find("res_1") != -1:
        y_shape = np.array(tensors[1])
        num_hidden_y_tensor = int(np.prod(y_shape[:-1]) * batch_size)
        # make data for fast data loading
        y_quantized = DataLoader(
            tensors[0].to(model_device).type(torch.int64),
            batch_size=num_hidden_y_tensor,
            shuffle=False,
        )

        for i, da in enumerate(y_quantized):
            da = da.type(torch.int64)
            decompressed = model.decompress(da, *tensors[1:])
            x_hat.append(decompressed.detach().cpu())
        x_hat = torch.cat(x_hat, axis=0).cpu()

    else:
        raise ValueError("Invalid model type.")

    return x_hat


def decompress(model, tensors, mask=None, verbose=False):
    """Decompress data."""
    if IS_CHECKING_MEMORY:
        utils.check_memory("Before decompression")
    if not isinstance(tensors[0], torch.Tensor):
        tensors = [torch.tensor(tensor) for tensor in tensors]
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    x_hat = decompress_step(model, tensors, batch_size=batch_size)
    if mask is not None:
        x_hat = x_hat * mask
    if IS_CHECKING_MEMORY:
        utils.check_memory("After decompression")
    gc.collect()
    return x_hat


def save_reconstructed(x_hat, output_file):
    # Write reconstructed data to file.
    if output_file is not None:
        with open(output_file, "wb") as f:
            f.write(np.array(x_hat).tobytes())


def get_metrics(x, x_hat, packed, redundancy=None):
    # Cast to float in order to compute metrics.
    x = x.numpy()
    x_hat = x_hat.numpy()
    max_val = np.amax(x) - np.amin(x)
    mse = (np.square(x - x_hat)).mean()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)

    output_file = "./outputs/compressed.npz"
    np.savez_compressed(output_file, *packed)
    compressed_size = round(os.stat(output_file).st_size / 1024, 2)  # in KB
    os.remove(output_file)

    if redundancy is not None:
        logger.info(f"redundancy: {redundancy:,} KB")

        compressed_size = compressed_size + redundancy
    original_size = round(len(np.array(x).tobytes()) / 1024, 2)  # in KB
    compression_ratio = compressed_size / original_size
    bit_rate = 32 * compression_ratio

    logger.info(f"Compressed size: {compressed_size:,} KB")
    logger.info(f"Original size: {original_size:,} KB")
    logger.info(f"Mean squared error: {mse:0.6f}")
    logger.info(f"PSNR (dB): {psnr:0.2f}")
    logger.info(f"bit_rate: {bit_rate:0.6f}")
    logger.info(f"Compression ratio: {1-compression_ratio:0.4f}")
    return {
        "compression_ratio": 1 - compression_ratio,
        "bit_rate": bit_rate,
        "psnr": psnr,
        "mse": mse,
    }


def update_dict_cond(results, key, value):
    if not (key.startswith("__") or key == "results"):
        # if isinstance(value, tf.Tensor):
        #     value = np.array(value)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if not isinstance(value, str):
            value = np.round(value, 4)
        results.update({key: value})


def save_results(
    model_name,
    **kwargs,
):
    results = {}
    for k, v in locals().items():
        if k == "kwargs":
            for k_kwarg, v_kwarg in v.items():
                update_dict_cond(results, k_kwarg, v_kwarg)
        else:
            update_dict_cond(results, k, v)

    folder = os.path.join("./outputs/", model_name)
    utils.mkdir_if_not_exist(folder)
    with open(os.path.join(folder, f"results.txt"), "a") as f:
        f.write(f"{json.dumps(results, cls=utils.NpEncoder)}\n")
