import sys
import os
from torchinfo import summary
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cpu":
    NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
else:
    NUM_GPUS = 0
os.environ["DEVICE"] = str(DEVICE.type)
os.environ["NUM_GPUS"] = str(NUM_GPUS)

import argparse
import netcdf_utils
import data_io
from models import res_conv2d_attn, hierachical_res_2d, hier_mbconv
from utils import logger, utils
import compression
import train
import time
import shutil
from pathlib import Path
import glob
import errno
import numpy as np
import data_retrival


def main(args):
    logger.configure(dir="./tmp_logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)

    # Model Initialization
    start_time = time.perf_counter()
    if args.model_type.lower().find("hierachical") != -1:
        model = hierachical_res_2d.VQCPVAE(
            **utils.args_to_dict(args, utils.model_defaults().keys())
        )
    elif args.model_type.lower().find("hier_mbconv") != -1:
        model = hier_mbconv.VQCPVAE(
            **utils.args_to_dict(args, utils.model_defaults().keys())
        )
    elif args.model_type.lower().find("res_1") != -1:
        model = res_conv2d_attn.VQCPVAE(
            **utils.args_to_dict(args, utils.model_defaults().keys())
        )
    else:
        raise ValueError(f"Invalid model type ({args.model_type}).")
    model = model.to(torch.device(DEVICE))

    data_path = (
        args.data_dir
        if args.data_dir != ""
        else Path(args.input_path).parent.absolute()
    )
    if data_path == "":
        raise ValueError("No input path or data path provided.")
    stats = utils.get_data_stats(args.data_type, data_path)
    model.set_standardizer_layer(stats["mean"], stats["std"] ** 2, 1e-6)
    logger.log(
        f"Model initialization time: {time.perf_counter() - start_time:0.4f} seconds\n"
    )
    if args.verbose:
        logger.log(
            summary(
                model,
                model.input_shape,
                depth=3,
                col_names=(
                    "input_size",
                    "output_size",
                    "num_params",
                ),
                verbose=args.verbose,
            )
        )

    if args.command == "train":

        # Get data.
        start_time = time.perf_counter()
        dataio = data_io.Dataio(args.batch_size, args.patch_size, args.data_shape)
        train_ds, test_ds = dataio.get_train_test_data_loader(
            args.data_dir, args.ds_name, local_test=args.local_test
        )
        logger.log(f"I/O time: {time.perf_counter() - start_time:0.4f} seconds\n")

        # Resume parameters.
        resume_checkpoint = {}
        if args.resume:
            resume_checkpoint = utils.get_checkpoint(args)
            model, is_weight_loaded = utils.load_model_with_checkpoint(
                model, resume_checkpoint["weight_path"], args.verbose
            )
            if not is_weight_loaded:
                logger.log("Training from scratch.\n")
                resume_checkpoint = {}

        if args.verbose:
            dataio.log_training_parameters()

        model = train.train(
            model=model,
            train_ds=train_ds,
            model_path=args.model_path,
            test_ds=test_ds,
            resume_checkpoint=resume_checkpoint,
            **utils.args_to_dict(args, utils.train_defaults().keys()),
            args=args,
            dataio=dataio,
        )

    elif (
        args.command == "compress"
        or args.command == "decompress"
        or args.command == "get_data"
    ):

        # Load model weights.
        model, is_weight_loaded = utils.load_model_with_checkpoint(
            model, args.model_path, args.verbose
        )
        if not is_weight_loaded:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), args.model_path
            )
        model.name = args.model_path.split("/")[-3]
        model.eval()

        #  Data IO
        dataio = data_io.Dataio(args.batch_size, args.patch_size, args.data_shape)
        dataio.batch_size = int(dataio.get_num_batch_per_time_slice())

        if args.command == "compress":
            #  Load data
            ds = dataio.get_compression_data_loader(args.input_path, args.ds_name)
            if args.verbose:
                dataio.log_training_parameters()

            logger.log("Compressing...")
            start_time = time.perf_counter()
            compress_loop(args, model, ds, dataio)
            logger.info(f"Compression completed!")
            logger.log(
                f"Total compression time: {time.perf_counter() - start_time:0.4f} seconds"
            )

        # elif args.command == "decompress":
        else:
            # Load compressed data (mask included)
            filenames = utils.get_filenames(args.input_path, postfix=".npz")

            # If command is get_data, we only select the files that are within the specified time and space range
            if args.command == "get_data":
                lower_pos_x = min(args.start_pos_x, args.end_pos_x)
                higher_pos_x = max(args.start_pos_x, args.end_pos_x)
                lower_pos_y = min(args.start_pos_y, args.end_pos_y)
                higher_pos_y = max(args.start_pos_y, args.end_pos_y)
                times = (args.start_time, args.end_time)
                # pad_fronts = dataio.get_padded_dims()[1:-1]
                # pad_fronts = np.array([pads[0] for pads in pad_fronts])
                lower_coors = (lower_pos_y, lower_pos_x)
                # lower_coors = list(np.array(lower_coors) + pad_fronts)
                upper_coors = (higher_pos_y, higher_pos_x)
                # upper_coors = list(np.array(upper_coors) + pad_fronts)
                filenames = data_retrival.get_desired_filenames(filenames, times)
            metadata_file = utils.get_filenames(args.input_path, postfix=".nc")
            filenames.extend(metadata_file)

            logger.log(f"Decompressing...")
            start_time = time.perf_counter()
            if args.command == "decompress":
                decompress_loop(args, model, filenames, dataio)
            else:
                get_data(args, model, filenames, dataio, lower_coors, upper_coors)
            logger.info(f"Decompression completed!")
            logger.log(
                f"Total decompression time: {time.perf_counter() - start_time:0.4f} seconds"
            )

    else:
        raise ValueError(
            f"Unknown command: {args.command}. Options: train, compress, decompress, get_data."
        )


def compress_loop(args, model, ds, dataio):
    if args.output_path is None:
        output_path = os.path.join("./outputs", "".join(args.model_path.split("/")[-3]))
    else:
        output_path = args.output_path
    utils.mkdir_if_not_exist(output_path)

    for i, (x, mask) in enumerate(ds):
        output_filename = args.input_path.split("/")[-1].rpartition(".")[0] + f"_{i}"
        output_file = os.path.join(output_path, output_filename)

        # Save mask and stats and metadata
        if i == 0:
            logger.log("Making metadata...")
            meta_time = time.perf_counter()
            # Mask
            mask_path = os.path.join(output_path, "mask")
            compression.save_compressed(mask_path, [mask])
            # Stats
            stat_folder = Path(args.input_path).parent.absolute()
            stat_path = glob.glob(os.path.join(stat_folder, "*.csv"))[0]
            stat_path = Path(stat_path)
            stat_filename = "stats.csv"
            shutil.copy(stat_path, os.path.join(output_path, stat_filename))
            # Metadata
            metadata_output_file = output_filename[:-1] + "metadata.nc"
            metadata_output_file = os.path.join(output_path, metadata_output_file)
            if args.verbose:
                logger.log(f"Saving mask to {mask_path}")
                logger.log(f"Saving statistics to {stat_path}")
                logger.log(f"Saving metadata to {metadata_output_file}")
            netcdf_utils.create_dataset_with_only_metadata(
                args.input_path, metadata_output_file, args.ds_name, args.verbose
            )
            logger.log(
                f"Metadata completed in {time.perf_counter() - meta_time:0.4f} seconds"
            )

        if args.verbose:
            logger.log(f"\nCompressing {args.input_path}_{i}")
            tensors, x_hat = compression.compress(model, x, mask, args.verbose)

            # Save images of original data and reconstructed data for comparison.
            x = x * mask
            x = torch.permute(x, (0, 2, 3, 1)).detach().cpu().numpy()
            x = dataio.revert_partition(x)

            x_hat = x_hat * mask
            x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
            x_hat = dataio.revert_partition(x_hat)
            utils.save_reconstruction(x[0], x_hat[0], output_filename, output_file)
        else:
            tensors = compression.compress(model, x, mask, args.verbose)

        compression.save_compressed(output_file, tensors)


def decompress_loop(args, model, filenames, dataio):

    if args.output_path is None:
        output_path = os.path.join(
            "./outputs",
            "reconstructed_" + "".join(args.model_path.split("/")[-3]),
        )
    else:
        output_path = args.output_path
    utils.mkdir_if_not_exist(output_path)

    mask_filename = [f for f in filenames if f.find("mask") != -1][0]
    metadata_filename = [f for f in filenames if f.find("metadata") != -1][0]
    filenames.remove(mask_filename)
    filenames.remove(metadata_filename)
    ncfile = os.path.join(
        output_path,
        metadata_filename.split("/")[-1].rpartition("_")[0] + "-reconstruction.nc",
    )
    shutil.copy(metadata_filename, ncfile)

    mask = compression.load_compressed(mask_filename)[0]
    total_writing_time = 0
    for i, filename in enumerate(filenames):
        tensors = compression.load_compressed(filename)
        x_hat = compression.decompress(model, tensors, mask, args.verbose)
        x_hat = x_hat * mask
        x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
        x_hat = dataio.revert_partition(x_hat)
        x_hat = x_hat[0, ::-1, :, 0]

        writing_time = time.perf_counter()
        netcdf_utils.write_data_to_netcdf(
            ncfile,
            x_hat,
            args.ds_name,
            time_idx=i,
            verbose=args.verbose,
        )
        total_writing_time += time.perf_counter() - writing_time
    logger.log(f"Total writing time: {total_writing_time:0.4f} seconds")


# // TODO: get data based on num_tiles
# // TODO: using the tile index is faster than using the coor index
# // TODO: need to take into account the padding at the front and back when applaying dataio padding at the beginning
# TODO: numerical error when decompressing x_hat and x_hat_test, (error < 1e-3)
# (x_hat_test is the decompressed data on the whole original data)
def get_data(args, model, filenames, dataio, lower_coors, upper_coors):
    if args.output_path is None:
        output_path = os.path.join(
            "./outputs",
            "get_data_" + "".join(args.model_path.split("/")[-3]),
        )
    else:
        output_path = args.output_path
    utils.mkdir_if_not_exist(output_path)

    # Prepare the mask and metadata
    mask_filename = [f for f in filenames if f.find("mask") != -1][0]
    metadata_filename = [f for f in filenames if f.find("metadata") != -1][0]
    filenames.remove(mask_filename)
    filenames.remove(metadata_filename)
    ncfile = os.path.join(
        output_path,
        metadata_filename.split("/")[-1].rpartition("_")[0] + "-reconstruction.nc",
    )
    shutil.copy(metadata_filename, ncfile)
    mask = compression.load_compressed(mask_filename)[0]

    # The original data is padded to be divisible by the patch size.
    # We need to add the padding to the given coordinates.
    pad_fronts = dataio.get_padded_dims()[1:-1]
    pad_fronts = np.array([pads[0] for pads in pad_fronts])
    coors = (lower_coors, upper_coors)
    lower_coors = list(np.array(lower_coors) + pad_fronts)
    upper_coors = list(np.array(upper_coors) + pad_fronts)

    # Get the corresponding tiles given the coordinates.
    num_tiles = dataio.get_num_tiles()
    lower_tiles = []
    upper_tiles = []
    for i in range(len(num_tiles)):
        lower_tiles.append(lower_coors[i] // dataio.patch_size)
        if (upper_coors[i] % dataio.patch_size) == 0:
            upper_tiles.append((upper_coors[i] // dataio.patch_size))
        else:
            upper_tiles.append((upper_coors[i] // dataio.patch_size) + 1)
        upper_tiles[i] = min(upper_tiles[i], num_tiles[i])
    ranges = [upper_coors[i] - lower_coors[i] for i in range(len(lower_coors))]

    # Get mask corresponding to the calculated tiles.
    mask = mask.reshape((*num_tiles, *mask.shape[len(num_tiles) :]))
    mask = mask[
        lower_tiles[0] : upper_tiles[0],
        lower_tiles[1] : upper_tiles[1],
        ...,
    ]
    num_x_hat_tiles = mask.shape[: len(mask.shape) // 2]
    mask = mask.reshape((-1, 1, *mask.shape[-2:]))

    total_writing_time = 0
    total_decompress_time = 0
    model.eval()
    with torch.no_grad():
        for i, filename in enumerate(filenames):
            tensors = compression.load_compressed(filename)

            # Get data corresponding to the calculated tiles.
            num_latents = len(tensors) // 2
            for j in range(num_latents):
                da, da_shape = tensors[j], tensors[j + num_latents]
                da = da.reshape((*num_tiles, *da_shape[:-1]))
                da = da[
                    lower_tiles[0] : upper_tiles[0],
                    lower_tiles[1] : upper_tiles[1],
                    ...,
                ]
                da = da.reshape((-1, 1))
                tensors[j] = da

            # Decompress and reshape the data.
            decompress_time = time.perf_counter()
            x_hat = compression.decompress(model, tensors, mask, args.verbose)
            x_hat = x_hat * mask
            x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
            total_decompress_time += time.perf_counter() - decompress_time

            new_shape = [num_tile * dataio.patch_size for num_tile in num_x_hat_tiles]
            dataio.data_shape = [1] + new_shape + [1]
            x_hat = dataio.revert_partition(x_hat)
            x_hat = x_hat[0, ::-1, :, 0]

            # Because the original data is padded to be divisible by the patch size,
            # the decompressed data is also padded to be divisible by the patch size.
            # Therefore, there are some extra data points at the begining and the end of the decompressed data.
            # we need to remove the residual points from the reconstructed data.
            residual_fronts = [
                lower_coors[i] - lower_tiles[i] * dataio.patch_size
                for i in range(len(lower_coors))
            ]
            x_hat = x_hat[
                residual_fronts[0] : residual_fronts[0] + ranges[0],
                residual_fronts[1] : residual_fronts[1] + ranges[1],
            ]
            writing_time = time.perf_counter()
            netcdf_utils.write_data_to_netcdf(
                ncfile,
                x_hat,
                args.ds_name,
                time_idx=i,
                coors=coors,
                verbose=args.verbose,
            )
            total_writing_time += time.perf_counter() - writing_time
    logger.log(f"Total writing time: {total_writing_time:0.4f} seconds")
    logger.log(f"Total decompress time: {total_decompress_time:0.4f} seconds")


def get_default_arguments():
    defaults = dict(
        command="",
        data_dir="",
        ds_name="SST",
        model_path="./saved_models/model",
        use_fp16=False,
        verbose=True,
        resume="",
        iter=-1,  # -1 means resume from the best model
        local_test=False,
        input_path="",  # a file if compress, a folder if decompress
        output_path="",  # a folder if compress, a folder if decompress
    )
    defaults.update(utils.model_defaults())
    defaults.update(utils.train_defaults())
    defaults.update(utils.data_defaults())
    defaults.update(utils.get_data_defaults())
    return defaults


class Args:
    def __init__(self, *args, **kwargs):
        for item in kwargs:
            setattr(self, item, kwargs[item])

    def __repr__(self):
        message = "\n"
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                message += f"{k} = '{v}'\n"
            else:
                message += f"{k} = {v}\n"
        return message


def create_argparser():
    """Parses command line arguments."""
    defaults = get_default_arguments()
    parser = argparse.ArgumentParser()
    utils.add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    if args.command.lower() in ["compress", "decompress"]:
        if args.input_path == "":
            raise ValueError("Must specify an input path.")
        if args.output_path == "":
            if args.command.lower() == "compress":
                args.output_path = os.path.join(
                    "outputs", args.input_path.rpartition(".")[0] + "_compression"
                )
            elif args.command.lower() == "decompress":
                args.output_path = os.path.join(
                    "outputs", args.input_path.rpartition(".")[0] + "_decompression"
                )

    return args


if __name__ == "__main__":
    # Using main as a function, need to predefine 'args'
    # defaults = get_default_arguments()
    # args = Args(**defaults)

    # Using CLI
    args = create_argparser()
    main(args)
