import sys
import os
from torchinfo import summary
import torch
import argparse
from pathlib import Path
from skimage.io.collection import alphanumeric_key
import glob
import logging.config
import torchvision
from torchvision import transforms

# import train
import data_io
import numpy as np

from models import res_conv2d_attn
import tmp_train

# import compression
# from utils import utils
import matplotlib.pyplot as plt


def main(args):

    # Invoke subcommand.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"*********{args.command.upper()} BEGIN***********")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    num_gpus = len(available_gpus)

    if args.verbose == True:
        message = "\n"
        for k, v in args.__dict__.items():
            message += k + " = " + str(v) + "\n"
        # Additional Info when using cuda
        if device.type == "cuda":
            message += f"\nUsing device: {str(device)}\n"
            for i in range(num_gpus):
                mem_allot = round(torch.cuda.memory_allocated(i) / 1024**3, 1)
                mem_cached = round(torch.cuda.memory_cached(i) / 1024**3, 1)
                message += f"{str(torch.cuda.get_device_name(i))}\n"
                message += "Memory Usage: " + "\n"
                message += "Allocated: " + str(mem_allot) + " GB" + "\n"
                message += "Cached: " + str(mem_cached) + " GB" + "\n"
        logger.info(f"{message}")
        logger.info(f"Pytorch version: {torch.__version__}\n")

    if args.command == "train":
        args.prefix_folder = (
            f"-latent_dim_{args.latent_dim}-num_embeddings_{args.num_embeddings}"
            f"-batch_size_{args.batch_size}-data_patch_size_{args.data_patch_size}"
            f"-model_patch_size_{args.model_patch_size}"
        )
        args.model_path = args.model_path + args.prefix_folder
        args.train_path = args.train_path + args.prefix_folder
        args.model_name = args.model_path.rpartition("/")[-1]

        # Get data.
        dataio = data_io.Dataio(
            args.batch_size,
            args.data_patch_size,
            args.model_patch_size,
            args.data_shape,
        )
        filenames, fillna_value = dataio.get_filenames_and_fillna_value(args.data_path)
        filenames = filenames[:2]
        split = int(len(filenames) * 0.99)
        train_files = filenames[:split]
        logger.info(f"number of train_files: {len(train_files)}")
        train_dataset = dataio.create_overlapping_generator(
            train_files, fillna_value=fillna_value, name="train", shuffle=True
        )

        logger.info(f"train_dataset: {train_dataset}")
        train_ds = dataio.get_data_loader(
            train_dataset,
            drop_last=True,
            shuffle=True,
            num_workers=4 * num_gpus,
            pin_memory=True,
        )
        # for i, (data, mask) in enumerate(train_ds):
        #     print(f"{i}: {data.shape}, {mask.shape}")
        #     if i >= 5:

        #         # fig = plt.figure(figsize=(10, 10))
        #         # plt.imshow(data[0])

        #         # fig = plt.figure(figsize=(10, 10))
        #         # plt.imshow(mask[0])
        #         break

        # # fig = plt.figure(figsize=(10, 10))
        # # plt.imshow(train_dataset.da[0])

        # # fig = plt.figure(figsize=(10, 10))
        # # plt.imshow(train_dataset.mask[0])

        # # plt.show()
        # print("\n\n")

        # create test_ds
        test_files = filenames[split:]
        logger.info(f"number of test_files: {len(test_files)}")
        test_dataset = dataio.create_disjoint_generator(
            test_files, fillna_value=fillna_value, name="test", shuffle=False
        )
        logger.info(f"test_dataset: {test_dataset}")
        test_ds = dataio.get_data_loader(
            test_dataset,
            num_workers=4 * num_gpus,
        )

        dataio.log_training_parameters()
        # sys.exit()

        # Get model.
        input_shape = [
            args.batch_size,
            1,
            args.model_patch_size,
            args.model_patch_size,
        ]
        model = res_conv2d_attn.VQCPVAE(
            input_shape=input_shape,
            pre_num_channels=args.pre_num_channels,
            num_channels=args.num_channels,
            latent_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            num_residual_layers=args.num_residual_layers,
            num_transformer_layers=args.num_transformer_layers,
            commitment_cost=0.25,
            decay=0.99,
            name=args.model_name,
        )

        try:
            stats = dataio.get_data_statistics(args.data_path)
            mean = stats["mean"]
            std = stats["std"]
            model.set_standardizer_layer(mean, std**2, 1e-6)
        except Exception as e:
            logger.info("No statistics file available. Cannot use Stadardization.")
            logger.error(e, exc_info=True)

        # if args.verbose:
        #     summary(model, input_shape, col_width=25, depth=3, verbose=1)

        # Resume parameters.
        resume_checkpoint = {}
        if args.resume:
            if args.iter == -1:
                resume_checkpoint["resume_epoch"] = args.iter
                logger.info(f"Resume training using the best model")
                weight_path = os.path.join(
                    args.model_path, "checkpoints", "only_weights", "best"
                )
            elif args.iter > 0:
                resume_checkpoint["resume_epoch"] = args.iter
                logger.info(
                    f"Resume training at {resume_checkpoint['resume_epoch']} epoch"
                )
                weight_path = os.path.join(
                    args.model_path, "checkpoints", f"model_{args.iter:06d}"
                )
            try:
                logger.info(f"weight path: {weight_path}")
                model.load_weights(weight_path)
                logger.info(f"Model weights successfully loaded.")
            except Exception as e:
                logger.error(e, exc_info=True)
                logger.info(f"Cannot load model weights. Training from scratch.")
                resume_checkpoint = {}

        tmp_train.train(
            args,
            model,
            train_ds,
            dataio,
            args.epochs_til_ckpt,
            resume_checkpoint,
            test_ds,
        )

        sys.exit()

        # Train.
        trainer = train.Trainer(args)
        trainer.train(
            args,
            model,
            ds.repeat(),
            dataio,
            args.epochs_til_ckpt,
            resume_checkpoint,
            test_ds=test_ds.repeat(),
        )

    elif args.command == "compress" or args.command == "decompress":

        # Load model and create dataio
        model = tf.keras.models.load_model(args.model_path)
        print(f"Model {model.name} loaded.")
        arg_names = args.model_path.split("/")[-4].split("-")
        for arg in arg_names:
            if arg.find("batch_size") != -1:
                batch_size = int(arg.split("_")[-1])
            elif arg.find("data_patch_size") != -1:
                data_patch_size = int(arg.split("_")[-1])
            elif arg.find("model_patch_size") != -1:
                model_patch_size = int(arg.split("_")[-1])

        if args.command == "compress":
            dataio = data_io.Dataio(
                batch_size,
                data_patch_size,
                model_patch_size,
                args.data_shape,
            )
            dataio.batch_size = dataio.get_num_batch_per_time_slice()
            filenames, fillna_value = dataio.get_filenames_and_fillna_value(
                args.input_file
            )
            tf_gen = dataio.create_compression_generator(
                filenames, fillna_value, name="compression", shuffle=False
            )
            ds = dataio.get_compression_tfdata(tf_gen, name="compression")
            ds = dataio.configure_for_performance(ds, shuffle=False)
            dataio.log_training_parameters()

            output_folder = os.path.join("./outputs", "".join(arg_names))
            utils.mkdir_if_not_exist(output_folder)
            mask = dataio.get_mask(tf_gen)
            for i, da in enumerate(ds):
                output_file = os.path.join(
                    output_folder,
                    args.input_file.split("/")[-1].rpartition(".")[0] + f"_{i}.tfci",
                )
                print(f"Compressing {args.input_file}_{i} to {output_file}")
                tensors = compression.compress(model, da, mask, args.verbose)
                compression.save_compressed(tensors, output_file)

        else:
            # Load mask
            x = torch.random.normal([4096, 1])
            input_file = "random_data.tfci"
            output_file = os.path.join(output_folder, input_file + ".f32")
            x_hat = compression.decompress(model, x, args.verbose)
            compression.save_reconstructed(x_hat, output_file)

    else:
        print(f"Command {args.command} not recognized.")

    return


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Process some integers.")

    # General arguments
    parser.add_argument(
        "--data_shape",
        nargs="*",
        type=int,  # any type/callable can be used here
        default=[],
        help="Shape of the original data.",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        default=False,
        help="Report progress and metrics when training or compressing.",
    )
    parser.add_argument(
        "--model_path",
        default="./saved_models/hosvd_vqcpvae",
        help="Path where to save/load the trained model.",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="What to do:"
        "'train': loads training data and trains (or continues "
        "to train) a new model."
        "'compress': reads an image file and writes a compressed binary file."
        "'decompress': eads a binary file and reconstructs the image"
        "input and output filenames need to be provided for the latter "
        "two options."
        "Invoke '<command> -h' for more information.",
    )

    # 'train' subcommand.
    train_args = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model. Note that this "
        "model trains on a continuous stream of patches drawn from "
        "the training image dataset. An epoch is always defined as "
        "the same number of batches given by --steps_per_epoch. "
        "The purpose of validation is mostly to evaluate the "
        "rate-distortion performance of the model using actual "
        "quantization rather than the differentiable proxy loss. "
        "Note that when using custom training images, the validation "
        "set is simply a random sampling of patches from the "
        "training set.",
    )
    train_args.add_argument(
        "--train_verbose",
        action="store_true",
        default=False,
        help="Report progress and metrics when training.",
    )
    train_args.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to data folder",
    )
    train_args.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training. Default: False",
    )
    train_args.add_argument("--iter", type=int, default=0, help="resume iteration")
    train_args.add_argument(
        "--epochs_til_ckpt",
        type=int,
        default=5,
        help="number of epoch interval to save checkpoints",
    )
    train_args.add_argument(
        "--vq_anneal_portion",
        type=float,
        default=0.6,
        help="The portions epochs that vq loss is annealed",
    )
    train_args.add_argument(
        "--random_seed", type=int, default=None, help="Random seed. Default: None"
    )
    train_args.add_argument(
        "--pre_num_channels",
        type=int,
        default=32,
        help="Number of channel per preactivate layer.",
    )
    train_args.add_argument(
        "--num_channels",
        type=int,
        default=96,
        help="Number of channel per intermediate layer.",
    )
    train_args.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension.",
    )
    train_args.add_argument(
        "--num_embeddings",
        type=int,
        default=128,
        help="number of embeddings of the vector quantizer layer.",
    )
    train_args.add_argument(
        "--num_residual_layers",
        type=int,
        default=3,
        help="Number of residual layers.",
    )
    train_args.add_argument(
        "--num_transformer_layers",
        type=int,
        default=2,
        help="Number of transformer layers.",
    )
    train_args.add_argument(
        "--train_path",
        default="./tmp/cpvae",
        help="Path where to log training metrics for TensorBoard and back up "
        "intermediate model checkpoints.",
    )
    train_args.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate."
    )
    train_args.add_argument(
        "--learning_rate_min",
        type=float,
        default=1e-6,
        help="minimum learning rate for scheduler",
    )
    train_args.add_argument(
        "--warm_up",
        type=float,
        default=0.15,
        help="warm up for linear learning rate schduler",
    )
    train_args.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay for optimizer"
    )
    train_args.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    train_args.add_argument(
        "--data_patch_size",
        type=int,
        default=150,
        help="Size of image patches.",
    )
    train_args.add_argument(
        "--model_patch_size",
        type=int,
        default=128,
        help="Size of image patches.",
    )
    train_args.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Train up to this number of epochs. (One epoch is here defined as "
        "the number of steps given by --steps_per_epoch, not iterations "
        "over the full training dataset.)",
    )
    train_args.add_argument(
        "--preprocess_threads",
        type=int,
        default=-1,
        help="Number of CPU threads to use for parallel decoding of training "
        "images. Default: tf.data.AUTOTUNE",
    )
    train_args.add_argument(
        "--precision_policy",
        type=str,
        default=None,
        help="Policy for `tf.keras.mixed_precision` training.",
    )
    train_args.add_argument(
        "--check_numerics",
        action="store_true",
        help="Enable TF support for catching NaN and Inf in tensors.",
    )

    # 'compress' subcommand.
    compress_args = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.",
    )
    compress_args.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="maximum absolute error",
    )

    # 'decompress' subcommand.
    decompress_args = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
        "a PNG file.",
    )

    # Arguments for both 'compress' and 'decompress'.
    for arg, ext in ((compress_args, ".tfci"), (decompress_args, ".f32")):
        arg.add_argument("--input_file", help="Input filename.")
        arg.add_argument(
            "--output_file",
            nargs="?",
            help=f"Output filename (optional). If not provided, appends '{ext}' to "
            f"the input filename.",
        )

    # Parse arguments.
    args = parser.parse_args()
    if args.command is None:
        parser.print_usage()
        sys.exit(2)

    return args


if __name__ == "__main__":
    logging.config.fileConfig("logging.init", disable_existing_loggers=False)
    main(parse_args())
    # args = parse_args()
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # logger.info(f"*********{args.command.upper()} BEGIN***********")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if args.verbose == True:
    #     message = "\n"
    #     for k, v in args.__dict__.items():
    #         message += k + " = " + str(v) + "\n"
    #     # Additional Info when using cuda
    #     if device.type == "cuda":
    #         message += f"\nUsing device: {str(device)}\n"
    #         message += f"{str(torch.cuda.get_device_name(0))}\n"
    #         message += "Memory Usage: " + "\n"
    #         message += (
    #             "Allocated: "
    #             + str(round(torch.cuda.memory_allocated(0) / 1024**3, 1))
    #             + " GB"
    #             + "\n"
    #         )
    #         message += (
    #             "Cached: "
    #             + str(round(torch.cuda.memory_reserved(0) / 1024**3, 1))
    #             + " GB"
    #             + "\n"
    #         )
    #     logger.info(f"{message}")
    #     logger.info(f"Pytorch version: {torch.__version__}\n")

    # if args.command == "train":
    #     args.prefix_folder = (
    #         f"-latent_dim_{args.latent_dim}-num_embeddings_{args.num_embeddings}"
    #         f"-batch_size_{args.batch_size}-data_patch_size_{args.data_patch_size}"
    #         f"-model_patch_size_{args.model_patch_size}"
    #     )
    #     args.model_path = args.model_path + args.prefix_folder
    #     args.train_path = args.train_path + args.prefix_folder
    #     args.model_name = args.model_path.rpartition("/")[-1]

    #     # Get data.
    #     dataio = data_io.Dataio(
    #         args.batch_size,
    #         args.data_patch_size,
    #         args.model_patch_size,
    #         args.data_shape,
    #     )
    #     filenames, fillna_value = dataio.get_filenames_and_fillna_value(args.data_path)
    #     filenames = filenames[:2]
    #     split = int(len(filenames) * 0.99)
    #     train_files = filenames[:split]
    #     logger.info(f"number of train_files: {len(train_files)}")
    #     train_dataset = dataio.create_overlapping_generator(
    #         train_files, fillna_value=fillna_value, name="train", shuffle=True
    #     )

    #     logger.info(f"train_dataset: {train_dataset}")
    #     # train_dataloader = dataio.get_data_loader(
    #     #     train_dataset, drop_last=True, shuffle=True, num_workers=0
    #     # )
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         drop_last=True,
    #         # timeout=60,
    #         num_workers=2,
    #         prefetch_factor=3,
    #         # pin_memory_device="cuda",
    #     )
    #     # torch.utils.data.get_worker_info()
    #     # print(torch.utils.data.get_worker_info())
    #     for i, (data, mask) in enumerate(train_dataloader):
    #         print(f"{i}: {data.shape}, {mask.shape}")
    #         fig = plt.figure(figsize=(10, 10))
    #         plt.imshow(data[0])

    #         fig = plt.figure(figsize=(10, 10))
    #         plt.imshow(mask[0])
    #         break

    #     # fig = plt.figure(figsize=(10, 10))
    #     # plt.imshow(train_dataset.da[0])

    #     # fig = plt.figure(figsize=(10, 10))
    #     # plt.imshow(train_dataset.mask[0])

    #     plt.show()

    #     # create test_ds
    #     test_files = filenames[split:]
    #     logger.info(f"number of test_files: {len(test_files)}")
    #     test_dataset = dataio.create_disjoint_generator(
    #         test_files, fillna_value=fillna_value, name="test", shuffle=False
    #     )
    #     logger.info(f"test_dataset: {test_dataset}")
    #     # dataio.log_training_parameters()

    #     sys.exit()

    logging.shutdown()
