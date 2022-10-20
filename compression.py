import sys
import os
import utils.utils as utils
import numpy as np
import os
import json
import copy
import gc
import time
from utils import utils, logger
from torch.utils.data import DataLoader
import torch
import gzip
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
IS_CHECKING_MEMORY = False
BATCH_SIZE = 128


def compress_step(model, x, batch_size=4):
    """Compress step."""

    # make tf data for fast data loading

    x = DataLoader(x, batch_size=batch_size, shuffle=False)
    tensor = []
    model_divice = next(model.parameters()).device
    for i, da in enumerate(x):
        compressed = model.compress(da.to(model_divice).type(torch.float))
        tensor.append(compressed[0].detach().cpu())
    tensor = torch.cat(tensor, axis=0).cpu()
    tensors = (tensor, *compressed[1:])
    return tensors


def compress(model, x, mask=None, verbose=False):
    """Compress data."""
    if IS_CHECKING_MEMORY:
        utils.check_memory("Before compression")
    print("Compressing...")
    start_time = time.perf_counter()
    tensors = compress_step(model, x, batch_size=BATCH_SIZE)
    encoding_time = time.perf_counter() - start_time
    logger.info(f"Encoding time: {encoding_time:0.2f} seconds")
    logger.info(f"Compression completed!")
    if IS_CHECKING_MEMORY:
        utils.check_memory("After compression")

    if verbose:
        packed = tensors
        start_time = time.perf_counter()
        x_hat = decompress(model, packed, mask)
        decoding_time = time.perf_counter() - start_time

        results = get_metrics(x * mask, x_hat, packed)
        results.update(
            {
                "model_name": model.name,
                "encoding_time": encoding_time,
                "decoding_time": decoding_time,
            }
        )
        save_results(**results)
        return tensors, x_hat
    gc.collect()
    return tensors


def save_compressed(tensors, output_file):
    tensors = [np.array(tensor) for tensor in tensors]
    np.savez_compressed(output_file, *tensors)


def decompress_step(model, packed, batch_size=4):
    """Decompress step."""
    tensors = packed
    y_shape = tensors[1]
    num_hidden_tensor = int(np.prod(y_shape[:-1]) * batch_size)
    # make data for fast data loading
    y_quantized = DataLoader(
        tensors[0].type(torch.int64),
        batch_size=num_hidden_tensor,
        shuffle=False,
    )

    x_hat = []
    model_divice = next(model.parameters()).device
    for i, da in enumerate(y_quantized):
        da = da.to(model_divice).type(torch.int64)
        decompressed = model.decompress(da, *tensors[1:])
        x_hat.append(decompressed.detach().cpu())
    x_hat = torch.cat(x_hat, axis=0).cpu()
    return x_hat


def decompress(model, x, mask=None, verbose=False):
    """Decompress data."""
    if IS_CHECKING_MEMORY:
        utils.check_memory("Before decompression")
    print("Decompressing...")
    start_time = time.perf_counter()
    x_hat = decompress_step(model, x, batch_size=BATCH_SIZE)
    if mask is not None:
        x_hat = x_hat * mask
    decoding_time = time.perf_counter() - start_time
    logger.info(f"Decoding time: {decoding_time:0.2f} seconds")
    if IS_CHECKING_MEMORY:
        utils.check_memory("After decompression")
    gc.collect()
    return x_hat


def save_reconstructed(x_hat, output_file):
    # Write reconstructed data to file.
    if output_file is not None:
        with open(output_file, "wb") as f:
            f.write(x_hat.numpy().tobytes())


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
