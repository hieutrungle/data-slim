import sys
import os
from torchinfo import summary
import torch
import argparse
import netcdf_utils
import data_io
from models import hier_mbconv, res_conv2d_attn, hierachical_res_2d
from utils import logger, utils
import compression
import train
import time
import shutil
from pathlib import Path
import glob
import errno
import matplotlib.pyplot as plt

torch.manual_seed(41)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


def main():
    args = create_argparser().parse_args()
    logger.configure(dir="./tmp_logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)

    input_shape = [1, args.patch_channels, args.patch_size, args.patch_size]
    model = hier_mbconv.VQCPVAE(
        **utils.args_to_dict(args, utils.model_defaults().keys())
    )
    logger.log(
        summary(
            model,
            input_shape,
            depth=3,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
            ),
            verbose=args.verbose,
        )
    )

    model.eval()

    a = torch.randn(input_shape)
    a = a.to(DEVICE)
    outputs = model(a)
    print(outputs)

    print(f"testing decompression")
    compressed = model.compress(a)
    # print(f"compressed: {compressed}")
    decoding_outputs = model.decompress(compressed)
    print(decoding_outputs)
    print(f"decoding_outputs.shape: {decoding_outputs.shape}")

    sys.exit()

    # Model Initialization
    start_time = time.perf_counter()
    if args.model_type.lower().find("hierachical") != -1:
        model = hierachical_res_2d.VQCPVAE(
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
                depth=1,
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

    elif args.command == "compress" or args.command == "decompress":

        # Load model weights.
        model, is_weight_loaded = utils.load_model_with_checkpoint(
            model, args.model_path, args.verbose
        )
        if not is_weight_loaded:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), args.model_path
            )
        model.name = args.model_path.split("/")[-3]

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

        else:
            # Load compressed data (mask included)
            filenames = utils.get_filenames(args.input_path, postfix=".npz")
            metadata_file = utils.get_filenames(args.input_path, postfix=".nc")
            filenames.extend(metadata_file)
            logger.log(f"Decompressing...")
            start_time = time.perf_counter()
            decompress_loop(args, model, filenames, dataio)
            logger.info(f"Compression completed!")
            logger.log(
                f"Total compression time: {time.perf_counter() - start_time:0.4f} seconds"
            )

    else:
        raise ValueError(
            f"Unknown command: {args.command}. Options: train, compress, decompress."
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

    if defaults["command"].lower() in ["compress", "decompress"]:
        if defaults["input"] == "":
            raise ValueError("Must specify an input file.")
        if defaults["output"] == "":
            if defaults["command"].lower() == "compress":
                defaults["output"] = defaults["input"] + ".tfci"
            elif defaults["command"].lower() == "decompress":
                defaults["output"] = defaults["input"] + ".f32"
    return parser


if __name__ == "__main__":
    main()
