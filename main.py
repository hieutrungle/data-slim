import enum
import sys
import os
from torchinfo import summary
import torch
import argparse
import data_io
from models import res_conv2d_attn, simple_model
from utils import logger, utils
import compression
import train
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


def main():
    args = create_argparser().parse_args()
    logger.configure(dir="./logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)

    # Model Initialization
    start_time = time.perf_counter()
    model = res_conv2d_attn.VQCPVAE(
        **utils.args_to_dict(args, utils.model_defaults().keys())
    )
    model = model.to(torch.device(DEVICE))
    try:
        stats = utils.get_data_statistics(args.data_dir)
        mean = stats["mean"]
        std = stats["std"]
        model.set_standardizer_layer(mean, std**2, 1e-6)
    except Exception as e:
        logger.log("No statistics file available. Cannot use Stadardization.")
        logger.error(e)
    logger.log(
        f"Model initialization time: {time.perf_counter() - start_time:0.4f} seconds\n"
    )
    if args.verbose:
        logger.log(
            summary(
                model, model.input_shape, col_width=30, depth=4, verbose=args.verbose
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
            raise FileNotFoundError(f"{args.model_path} is not exists.")
        model.name = args.model_path.split("/")[-3]

        if args.command == "compress":

            #  Load data
            dataio = data_io.Dataio(args.batch_size, args.patch_size, args.data_shape)
            ds = dataio.get_compression_data_loader(args.input_path, args.ds_name)
            if args.verbose:
                dataio.log_training_parameters()

            logger.log("Compressing...")
            start_compression_time = time.perf_counter()
            compress_loop(args, model, ds, dataio)

            logger.info(f"Compression completed!")
            logger.log(
                f"Total Compression-Decompression time: {time.perf_counter() - start_compression_time:0.4f} seconds"
            )

        else:
            # Load mask
            x = torch.rand([4096, 1])
            input_path = "random_data.tfci"
            output_path = os.path.join(output_path, input_path + ".f32")
            x_hat = compression.decompress(model, x, args.verbose)
            compression.save_reconstructed(x_hat, output_path)

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

        compression.save_compressed(tensors, output_file)
        # save mask
        if i == 0:
            mask_path = os.path.join(output_path, "mask")
            compression.save_compressed([mask], mask_path)


def perform_decompress():
    pass


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        command="",
        data_dir="../data/tccs/ocean/SST_modified",
        ds_name="SST",
        model_path="./saved_models/model",
        use_fp16=False,
        verbose=False,
        resume="",
        iter=-1,  # -1 means resume from the best model
        local_test=False,
        input_path="",
        output_path="",
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
