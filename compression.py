import sys
import os
import tensorflow_compression as tfc
import utils.utils as utils
import numpy as np
import os
import json
import copy
import gc
import time
from utils import utils, logger

# TODO: convert to pytorch


def compress_step(model, x):
    """Compress step."""

    # make tf data for fast data loading
    x = tf.cast(x, dtype=tf.float32)
    x = tf.data.Dataset.from_tensor_slices(x)
    x = x.batch(16)
    x = x.prefetch(tf.data.AUTOTUNE)

    tensor = []
    for i, data in enumerate(x):
        compressed = model.compress(tf.cast(data, tf.float32))
        tensor.append(compressed[0])
    tensor = tf.concat(tensor, axis=0)
    tensors = tf.tuple([tensor, *compressed[1:]])
    return tensors


def compress(model, x, mask=None, verbose=False):
    """Compress data."""

    print("Compressing...")
    start_time = time.perf_counter()
    tensors = compress_step(model, x)
    encoding_time = time.perf_counter() - start_time
    logger.info(f"Encoding time: {encoding_time:0.2f} seconds")
    logger.info(f"Compression completed!")

    if verbose:
        packed = tfc.PackedTensors()
        packed.pack(tensors)
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
    gc.collect()
    tf.keras.backend.clear_session()
    return tensors


def save_compressed(tensors, output_file):
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(output_file, "wb") as f:
        f.write(packed.string)


def decompress_step(model, packed, batch_size=4):
    """Decompress step."""
    dtypes = [t.dtype for t in model.decompress.input_signature]
    tensors = packed.unpack(dtypes)
    y_shape = tensors[1]
    num_hidden_tensor = int(np.prod(y_shape[:-1]) * batch_size)

    # make tf data for fast data loading
    y_quantized = tf.cast(tensors[0], dtype=tf.int64)
    y_quantized = tf.data.Dataset.from_tensor_slices(y_quantized)
    y_quantized = y_quantized.batch(num_hidden_tensor)
    y_quantized = y_quantized.prefetch(tf.data.AUTOTUNE)

    x_hat = []
    for i, data in enumerate(y_quantized):
        decompressed = model.decompress(tf.cast(data, tf.int32), *tensors[1:])
        x_hat.append(decompressed)
    x_hat = tf.concat(x_hat, axis=0)
    return x_hat


def decompress(model, x, mask=None, verbose=False):
    """Decompress data."""

    print("Decompressing...")
    start_time = time.perf_counter()
    batch_size = 16
    x_hat = decompress_step(model, x, batch_size)
    if mask is not None:
        x_hat = x_hat * mask
    decoding_time = time.perf_counter() - start_time
    logger.info(f"Decoding time: {decoding_time:0.2f} seconds")
    gc.collect()
    tf.keras.backend.clear_session()
    return x_hat


def save_reconstructed(x_hat, output_file):
    # Write reconstructed data to file.
    if output_file is not None:
        with open(output_file, "wb") as f:
            f.write(x_hat.numpy().tobytes())


def get_metrics(x, x_hat, packed, redundancy=None):
    # Cast to float in order to compute metrics.
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    max_val = tf.reduce_max(x) - tf.reduce_min(x)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)

    compressed_size = len(packed.string)  # in bytes
    if redundancy is not None:
        logger.info(f"redundancy: {redundancy:,} bytes")

        compressed_size = compressed_size + redundancy
    original_size = len(np.array(x).tobytes())  # in bytes
    compression_ratio = compressed_size / original_size
    bit_rate = 32 * compression_ratio

    logger.info(f"Compressed size: {compressed_size:,} bytes")
    logger.info(f"Original size: {original_size:,} bytes")
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
        if isinstance(value, tf.Tensor):
            value = np.array(value)
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

    folder = "./outputs/"
    utils.mkdir_if_not_exist(folder)
    with open(
        os.path.join(folder, f"results-{model_name.split('-')[0]}.txt"), "a"
    ) as f:
        f.write(f"{json.dumps(results, cls=utils.NpEncoder)}\n")
