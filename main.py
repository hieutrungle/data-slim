import sys
import os
from torchinfo import summary
import torch
import argparse
import data_io
from models import res_conv2d_attn
import train
from utils import logger, utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])


def main():
    args = create_argparser().parse_args()
    logger.configure(dir="./logs")
    utils.configure_args(args)
    utils.log_args_and_device_info(args)

    if args.command == "train":

        # Get data.
        dataio = data_io.Dataio(
            args.batch_size,
            args.patch_size,
            args.data_shape,
        )
        train_ds, test_ds = dataio.get_train_test_data_loader(
            args.data_dir,
            local_test=args.local_test,
        )

        # Model
        model = res_conv2d_attn.VQCPVAE(
            **utils.args_to_dict(args, utils.model_defaults().keys())
        )
        try:
            stats = utils.get_data_statistics(args.data_dir)
            mean = stats["mean"]
            std = stats["std"]
            model.set_standardizer_layer(mean, std**2, 1e-6)
        except Exception as e:
            logger.log("No statistics file available. Cannot use Stadardization.")
            logger.error(e)

        # Resume parameters.
        resume_checkpoint = {}
        if args.resume:
            resume_checkpoint = utils.get_resume_checkpoint(args)
            if os.path.isfile(resume_checkpoint["weight_path"]):
                logger.log(f"weight path: {resume_checkpoint['weight_path']}")
                checkpoint = torch.load(
                    resume_checkpoint["weight_path"], map_location=DEVICE
                )
                model.load_state_dict(checkpoint)
                logger.log(f"Model weights successfully loaded.\n")
            else:
                logger.log(f"No pretrained model found, training from scratch.\n")
                resume_checkpoint = {}

        if args.verbose:
            logger.log(
                summary(model, model.input_shape, col_width=30, depth=4, verbose=1)
            )

        model = train.train(
            model=model,
            train_ds=train_ds,
            model_path=args.model_path,
            test_ds=test_ds,
            resume_checkpoint=resume_checkpoint,
            **utils.args_to_dict(args, utils.train_defaults().keys()),
            args=args,
        )
    # Need path to model weights.
    # Need to initilize correct model w.r.t. the weights.
    # Input file.
    elif args.command == "compress":
        pass
    elif args.command == "decompress":
        pass
    else:
        raise ValueError(
            f"Unknown command: {args.command}. Options: train, compress, decompress."
        )


def create_argparser():
    """Parses command line arguments."""
    defaults = dict(
        command="train",
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
