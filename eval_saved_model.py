import logging.config
import argparse
from absl import app
from absl.flags import argparse_flags
import gc
import utils.utils as utils
import compression
import tensorflow as tf
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def evaluate_trained_model(model_path, data_path, data_shape, verbose=True):
    gc.collect()
    tf.keras.backend.clear_session()
    model_name = model_path.rpartition("/")[-1]
    print(f"\n****** model_name: {model_name} ******")
    saved_folder = "./outputs/"
    saved_folder = os.path.join(saved_folder, model_name)
    utils.mkdir_if_not_exist(saved_folder)

    input_file = data_path
    output_file = input_file.rpartition("/")[-1]
    output_file = os.path.join(saved_folder, output_file + ".tfci")
    print(f"output_file: {output_file}")

    compression.compress(
        model_path, data_path, output_file, data_shape, tolerance=1e-1, verbose=verbose
    )

    compression.compress(
        model_path, data_path, output_file, data_shape, tolerance=1e-2, verbose=verbose
    )

    input_file = output_file
    test_ds_pred = compression.decompress(model_path, input_file, data_shape=data_shape)


def main(args):
    logging.config.fileConfig("logging.init", disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    logger.info(f"********* TESTING: {args.model_path} ***********")

    if args.verbose == True:
        message = "\n"
        for k, v in args.__dict__.items():
            message += k + " = " + str(v) + "\n"
        logger.info(f"{message}")
        logger.info(
            f"{tf.config.list_physical_devices()}; "
            f"Tennsorflow version: {tf.__version__}\n"
        )

    try:
        evaluate_trained_model(
            args.model_path, args.data_path, args.data_shape, verbose=args.verbose
        )
        logger.info("Testing Completed.")
    except Exception as e:
        logger.error(f"{e}", exc_info=True)
        logger.info("Testing Failed.")


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Report progress and metrics when training or compressing.",
    )
    # Model arguments
    parser.add_argument(
        "--model_path", required=True, help="Path where to save/load the trained model."
    )
    parser.add_argument(
        "--data_shape",
        nargs="*",
        type=int,  # any type/callable can be used here
        default=[],
        help="Shape of the original data.",
    )
    parser.add_argument("--data_path", required=True, help="Path where to data.")
    #
    # Parse arguments.
    args = parser.parse_args(argv[1:])

    return args


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
