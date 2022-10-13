import sys
import os
from torchinfo import summary
import torch
import argparse
import data_io
from models import res_conv2d_attn
import tmp_train
from utils import logger, utils
import glob

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
        train_ds, test_ds = dataio.get_train_test_data_loader(args.data_dir)

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
            ckpt_files = glob.glob(os.path.join(args.model_path, "checkpoints", "*.pt"))

            if args.iter == -1:
                logger.log(f"Resume training using the best model")
                # Get the best model path.
                sorted_ckpt_files = {}
                for ckpt_file in ckpt_files:
                    sorted_ckpt_files[ckpt_file] = float(
                        ckpt_file.split("-")[-1].split("=")[-1].rpartition(".")[0]
                    )
                sorted_ckpt_files = dict(
                    sorted(sorted_ckpt_files.items(), key=lambda item: item[1])
                )
                best_ckpt = list(sorted_ckpt_files.keys())[0]
                # get epoch of the best model path.
                resume_epoch = best_ckpt.split("/")[-1].split("-")[1].split("=")[1]
                resume_checkpoint["resume_epoch"] = int(resume_epoch)
                weight_path = best_ckpt
            elif args.iter > 0:
                resume_checkpoint["resume_epoch"] = args.iter
                logger.log(
                    f"Resume training at {resume_checkpoint['resume_epoch']} epoch"
                )
                for ckpt_file in ckpt_files:
                    resume_epoch = ckpt_file.split("/")[-1].split("-")[1]
                    if int(resume_epoch.split("=")[1]) == args.iter:
                        weight_path = ckpt_file
                        break

            if os.path.isfile(weight_path):
                logger.log(f"weight path: {weight_path}")
                checkpoint = torch.load(weight_path, map_location=DEVICE)
                model.load_state_dict(checkpoint)
                logger.log(f"Model weights successfully loaded.")
            else:
                logger.log(f"No pretrained model found, training from scratch.")
                resume_checkpoint = {}

        if args.verbose:
            logger.log(
                summary(model, model.input_shape, col_width=25, depth=3, verbose=1)
            )

        model = tmp_train.train(
            model,
            train_ds,
            dataio,
            args.model_path,
            test_ds=test_ds,
            resume_checkpoint=resume_checkpoint,
            **utils.args_to_dict(args, utils.train_defaults().keys()),
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
