import sys
import os
from torchinfo import summary
import torch
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
import matplotlib.pyplot as plt
import data_retrival

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


def main():
    args = create_argparser()
    logger.configure(dir="./tmp_logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)
    logger.log(f"NUM_GPUS: {NUM_GPUS}")
    logger.log(f"DEVICE: {DEVICE}")

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
        raise ValueError("Invalid model type.")

    model = model.to(torch.device(DEVICE))
    stats = None
    if args.data_dir != "":
        try:
            stats = utils.get_data_statistics(args.data_dir)
            stat_dir = args.data_dir
        except Exception as e:
            logger.log(f"No statistics file available at data_dir: {args.data_dir}")
            logger.error(e)
    if args.input_path != "":
        try:
            stats = utils.get_data_statistics(args.input_path)
            stat_dir = args.input_path
        except Exception as e:
            logger.log(f"No statistics file available at input_path: {args.input_path}")
            logger.error(e)
    if stats is None:
        raise ValueError("No statistics file available.")
    mean = stats["mean"]
    std = stats["std"]
    model.set_standardizer_layer(mean, std**2, 1e-6)
    logger.log(f"Using statistics from {stat_dir}")
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
                lower_coors = (lower_pos_y, lower_pos_x)
                upper_coors = (higher_pos_y, higher_pos_x)
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


def get_data(args, model, filenames, dataio, lower_coors, upper_coors):
    if args.output_path is None:
        output_path = os.path.join(
            "./outputs",
            "get_data_" + "".join(args.model_path.split("/")[-3]),
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

    num_pad_tiles = dataio.get_num_pad_tiles()
    lower_tiles = []
    upper_tiles = []
    for i in range(len(num_pad_tiles)):
        lower_tiles.append(lower_coors[i] // dataio.patch_size)
        if (upper_coors[i] % dataio.patch_size) == 0:
            upper_tiles.append((upper_coors[i] // dataio.patch_size))
        else:
            upper_tiles.append((upper_coors[i] // dataio.patch_size) + 1)
        upper_tiles[i] = min(upper_tiles[i], num_pad_tiles[i])
    # upper_tiles = min((upper_coors // dataio.patch_size) + 1, num_pad_tiles)
    print(f"num_pad_tiles: {num_pad_tiles}")
    print(f"lower_coors: {lower_coors}, upper_coors: {upper_coors}")
    print(f"lower_tiles: {lower_tiles}, upper_tiles: {upper_tiles}")
    coors = (lower_coors, upper_coors)
    ranges = [upper_coors[i] - lower_coors[i] for i in range(len(lower_coors))]
    print(f"ranges: {ranges}")
    # // TODO: get data based on num_pad_tiles
    # // TODO: using the tile index is faster than using the coor index
    # TODO: need to take into account the padding at the front and back when applaying dataio padding at the beginning
    # initial dataio params
    print(f"\ninitial dataio params")
    dataio.print_instance_attributes()
    dataio.padder.print_instance_attributes()
    print(f"\n\n")
    total_writing_time = 0
    for i, filename in enumerate(filenames):
        tensors = compression.load_compressed(filename)

        # tensors = compression.load_compressed(filename)
        x_hat_test = compression.decompress(model, tensors, mask, args.verbose)
        x_hat_test = x_hat_test * mask
        x_hat_test = torch.permute(x_hat_test, (0, 2, 3, 1)).detach().cpu().numpy()
        x_hat_test = dataio.revert_partition(x_hat_test)
        x_hat_test = x_hat_test[0, ::-1, :, 0]
        plt.figure()
        plt.imshow(x_hat_test)

        num_latents = len(tensors) // 2
        for i in range(num_latents):
            da, da_shape = tensors[i], tensors[i + num_latents]
            da = da.reshape((*num_pad_tiles, *da_shape[:-1]))
            da = da[
                lower_tiles[0] : upper_tiles[0], lower_tiles[1] : upper_tiles[1], ...
            ]
            # number of tiles in each dimension
            # final_num_pad_tiles = da.shape[:num_latents]
            da = da.reshape((-1, 1))
            tensors[i] = da
            # print(f"tensor {i}: {da.shape}")

        # print(f"final_num_pad_tiles: {final_num_pad_tiles}")

        # print(f"mask: {mask.shape}")
        mask = mask.reshape((*num_pad_tiles, *mask.shape[-2:]))
        mask = mask[
            lower_tiles[0] : upper_tiles[0], lower_tiles[1] : upper_tiles[1], ...
        ]
        num_pad_tiles = mask.shape[:num_latents]
        print(f"num_pad_tiles: {num_pad_tiles}")

        mask = mask.reshape((-1, 1, *mask.shape[-2:]))
        # print(f"mask: {mask.shape}")
        # sys.exit()
        x_hat = compression.decompress(model, tensors, mask, args.verbose)
        x_hat = x_hat * mask
        x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()

        new_shape = [num_tile * dataio.patch_size for num_tile in num_pad_tiles]
        dataio.data_shape = [1] + new_shape + [1]
        dataio.print_instance_attributes()
        dataio.padder.print_instance_attributes()
        print(f"x_hat.shape: {x_hat.shape}")
        # sys.exit()
        x_hat = dataio.revert_partition(x_hat)
        x_hat = x_hat[0, ::-1, :, 0]
        print(f"x_hat.shape: {x_hat.shape}")

        pad_fronts = [
            lower_coors[i] - lower_tiles[i] * dataio.patch_size
            for i in range(len(lower_coors))
        ]
        print(f"pad_fronts: {pad_fronts}")
        print(f"x_hat.shape: {x_hat.shape}")
        plt.figure()
        plt.imshow(x_hat)

        x_hat = x_hat[
            pad_fronts[0] : pad_fronts[0] + ranges[0],
            pad_fronts[1] : pad_fronts[1] + ranges[1],
        ]
        print(f"x_hat.shape: {x_hat.shape}")

        x_hat_test_rear = x_hat_test[
            lower_tiles[0] * dataio.patch_size : upper_tiles[0] * dataio.patch_size,
            0 : lower_tiles[1] * dataio.patch_size,
        ]
        plt.figure()
        plt.imshow(x_hat_test_rear)

        x_hat_test_rear = x_hat_test[
            0 : lower_tiles[0] * dataio.patch_size,
            lower_tiles[1] * dataio.patch_size : upper_tiles[1] * dataio.patch_size,
        ]
        plt.figure()
        plt.imshow(x_hat_test_rear)

        # x_hat_test = x_hat_test[
        #     lower_tiles[0] * dataio.patch_size : upper_tiles[0] * dataio.patch_size,
        #     lower_tiles[1] * dataio.patch_size : upper_tiles[1] * dataio.patch_size,
        # ]
        # x_hat_test = x_hat_test[
        #     lower_tiles[0] * dataio.patch_size : upper_tiles[0] * dataio.patch_size,
        #     lower_tiles[1] * dataio.patch_size : upper_tiles[1] * dataio.patch_size,
        # ]
        x_hat_test = x_hat_test[
            lower_coors[0] : upper_coors[0],
            lower_coors[1] : upper_coors[1],
        ]

        print(f"x_hat_test.shape: {x_hat_test.shape}")
        x_hat_diff = x_hat_test - x_hat
        plt.figure()
        plt.imshow(x_hat_test)
        plt.figure()
        plt.imshow(x_hat)
        plt.figure()
        plt.imshow(x_hat_diff)
        plt.show()

        sys.exit()
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
        sys.exit()

    logger.log(f"Total writing time: {total_writing_time:0.4f} seconds")


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        command="",
        data_dir="",
        ds_name="SST",
        model_path="./saved_models/model",
        use_fp16=False,
        verbose=False,
        resume="",
        iter=-1,  # -1 means resume from the best model
        local_test=False,
        input_path="",  # a file if compress, a folder if decompress
        output_path="",  # a folder if compress, a folder if decompress
    )
    defaults.update(utils.model_defaults())
    defaults.update(utils.train_defaults())
    defaults.update(utils.data_defaults())
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
    elif args.command.lower() in ["get_data"]:

        args.start_time = 0  # time dimension start index
        args.end_time = 10
        args.start_pos_x = 3000
        args.start_pos_y = 500
        args.end_pos_x = 270
        args.end_pos_y = 2000

    return args


if __name__ == "__main__":
    main()
