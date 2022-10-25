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

    # Model
    start_time = time.perf_counter()
    model = res_conv2d_attn.VQCPVAE(
        **utils.args_to_dict(args, utils.model_defaults().keys())
    )
    # model = simple_model.VQCPVAE(
    #     **utils.args_to_dict(args, utils.model_defaults().keys())
    # )
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
        f"Model initialization time: {time.perf_counter() - start_time:0.4f} seconds"
    )
    logger.log(
        summary(model, model.input_shape, col_width=30, depth=4, verbose=args.verbose)
    )

    # sys.exit()

    if args.command == "train":

        # Get data.
        start_time = time.perf_counter()
        dataio = data_io.Dataio(
            args.batch_size,
            args.patch_size,
            args.data_shape,
        )
        train_ds, test_ds = dataio.get_train_test_data_loader(
            args.data_dir,
            local_test=args.local_test,
        )
        logger.log(f"I/O time: {time.perf_counter() - start_time:0.4f} seconds")

        # Resume parameters.
        resume_checkpoint = {}
        if args.resume:
            resume_checkpoint = utils.get_checkpoint(args)
            if os.path.isfile(resume_checkpoint["weight_path"]):
                logger.log(f"weight path: {resume_checkpoint['weight_path']}")
                checkpoint = torch.load(
                    resume_checkpoint["weight_path"], map_location=DEVICE
                )
                model.load_state_dict(checkpoint)
                logger.log(f"Model weights successfully loaded.\n")
            else:
                logger.log(
                    f"No pretrained model found at {resume_checkpoint['weight_path']}, training from scratch.\n"
                )
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
    # Need path to model weights.
    # Need to initilize correct model w.r.t. the weights.
    # Input file.
    elif args.command == "compress" or args.command == "decompress":
        if os.path.isfile(args.model_path):
            logger.log(f"weight path: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint)
            logger.log(f"Model weights successfully loaded.\n")
        else:
            logger.error(
                f"No pretrained model found at {args.model_path}!\nCannot do {args.command}."
            )
            raise FileNotFoundError(f"{args.model_path} is not exists.")
        model.name = args.model_path.split("/")[-3]
        print(f"Model name: {model.name}")
        # sys.exit()
        if args.command == "compress":
            dataio = data_io.Dataio(
                args.batch_size,
                args.patch_size,
                args.data_shape,
            )
            dataio.batch_size = int(dataio.get_num_batch_per_time_slice())
            filenames, fillna_value = utils.get_filenames_and_fillna_value(
                args.input_file
            )

            ds = dataio.create_disjoint_generator(
                filenames, fillna_value, name="compression", shuffle=False
            )
            dataio.log_training_parameters()
            ds = dataio.get_data_loader(
                ds,
                num_workers=4 * NUM_GPUS,
            )

            output_folder = os.path.join(
                "./outputs", "".join(args.model_path.split("/")[-3])
            )
            utils.mkdir_if_not_exist(output_folder)

            start_compression_time = time.perf_counter()

            for i, (x, mask) in enumerate(ds):
                output_file = (
                    args.input_file.split("/")[-1].rpartition(".")[0] + f"_{i}.f32"
                )
                output_path = os.path.join(
                    output_folder,
                    args.input_file.split("/")[-1].rpartition(".")[0] + f"_{i}.f32",
                )
                print(f"\nCompressing {args.input_file}_{i}")
                tensors, x_hat = compression.compress(model, x, mask, args.verbose)
                compression.save_compressed(tensors, output_path)

                x = x * mask
                x = torch.permute(x, (0, 2, 3, 1)).detach().cpu().numpy()
                x = dataio.revert_partition(x)

                x_hat = x_hat * mask
                x_hat = torch.permute(x_hat, (0, 2, 3, 1)).detach().cpu().numpy()
                x_hat = dataio.revert_partition(x_hat)
                utils.save_reconstruction(x[0], x_hat[0], output_file, output_folder)

            logger.log(
                f"Total Compression-Decompression time: {time.perf_counter() - start_compression_time:0.4f} seconds"
            )

        else:
            # Load mask
            x = torch.rand([4096, 1])
            input_file = "random_data.tfci"
            output_file = os.path.join(output_folder, input_file + ".f32")
            x_hat = compression.decompress(model, x, args.verbose)
            compression.save_reconstructed(x_hat, output_file)

    else:
        raise ValueError(
            f"Unknown command: {args.command}. Options: train, compress, decompress."
        )


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        command="",
        data_dir="../data/tccs/ocean/SST_modified",
        model_path="./saved_models/model",
        use_fp16=False,
        verbose=False,
        resume="",
        iter=-1,  # -1 means resume from the best model
        local_test=False,
        input_file="",
        output_file="",
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
