import sys
import os

# from torchinfo import summary
import torch

import argparse
import data_io
from models import res_conv2d_attn, hierachical_res_2d, hier_mbconv
from utils import logger, utils, timer
import compression

# import train
import time
import shutil
from pathlib import Path
import glob
import errno
import numpy as np
import data_retrival
import matplotlib.pyplot as plt
import json
import copy
import utils.straight_through_pixels as stp
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torchvision.transforms as transforms
import training
import torch.optim as optim
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch


def run_main():

    args = create_argparser()
    utils.configure_args(args)
    if not args.xpu:
        print("You need to choose running on XPU.")
        sys.exit()

    logger.configure(dir="./tmp_logs")
    utils.log_args_and_device_info(args)
    [
        print(f"[{i}]: {torch.xpu.get_device_properties(i)}")
        for i in range(torch.xpu.device_count())
    ]

    if args.tf32:
        print("doing TF32 training")
        torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.TF32)
    elif args.bf32:
        args.bf16 = 1
        print("doing BF32 training")
        torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.BF32)
    else:
        torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)

    if args.world_size == -1:
        mpi_world_size = int(os.environ.get("PMI_SIZE", -1))

        if mpi_world_size > 0:
            os.environ["MASTER_ADDR"] = args.dist_url  #'127.0.0.1'
            os.environ["MASTER_PORT"] = args.dist_port  #'29500'
            os.environ["RANK"] = os.environ.get("PMI_RANK", -1)
            os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", -1)
            args.rank = int(os.environ.get("PMI_RANK", -1))
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = 1
    utils.log_args_and_device_info(args)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)
    # if args.command == "train":

    #     pass
    # else:
    #     pass


def main_worker(ngpus_per_node, args):
    if args.distributed:
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        init_method = "tcp://" + args.dist_url + ":" + args.dist_port
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

        local_rank = os.environ["MPI_LOCALRANKID"]
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ.keys():
            local_rank = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        args.xpu = local_rank
        print(
            "world_size:{}, rank:{}, local_rank:{}".format(
                args.world_size, args.rank, local_rank
            )
        )

    print("Use XPU: {}".format(args.xpu))
    args.xpu = "xpu:{}".format(args.xpu)


def get_dataset(args, dataio):
    world_size = dist.get_world_size()

    # dataio = data_io.Dataio(args.batch_size, args.patch_size, args.data_shape)
    filenames, fillna_value = utils.get_filenames_and_fillna_value(args.data_path)
    split = int(len(filenames) * 0.99)
    train_files = filenames[:split]

    # Train data
    logger.log(f"number of train_files: {len(train_files)}")
    train_ds = dataio.create_overlapping_generator(
        train_files,
        args.ds_name,
        args.da_name,
        fillna_value=fillna_value,
        name="train",
        shuffle=True,
    )
    # Test data
    test_files = filenames[split:]
    logger.log(f"number of test_files: {len(test_files)}")
    test_ds = dataio.create_disjoint_generator(
        test_files,
        args.ds_name,
        args.da_name,
        fillna_value=fillna_value,
        name="test",
        shuffle=False,
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, shuffle=True, drop_last=True
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=world_size, shuffle=False, drop_last=False
    )
    batch_size = int(args.batch_size / float(world_size))
    logger.log(f"Batch size: {batch_size}")
    logger.log(f"World size: {world_size}")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        sampler=test_sampler,
        batch_size=batch_size,
    )

    return train_loader, test_loader


def get_default_arguments():
    defaults = dict(
        command="",
        data_path="",
        ds_name="SST",
        da_name="",  # if empty, da_name=ds_name, da is the data in ds
        model_path="./saved_models/model",
        use_fp16=False,
        verbose=False,
        resume="",
        iter=-1,  # -1 means resume from the best model
        local_test=False,
        input_path="",  # a file if compress, a folder if decompress
        output_path="",  # a folder if compress, a folder if decompress
        benchmark=False,  # test model on benchmark dataset
        tolerance=1e-1,  # tolerance for compression
        straight_through_weight=1,  # weight on traight through value
        dist_port="29500",
        dist_url="127.0.0.1",
        multiprocessing_distributed=True,
        dist_backend="ccl",
        num_devices=-1,
        world_size=-1,
        rank=-1,
        xpu=False,
        tf32=0,
        bf32=0,
        bf16=0,
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

    if args.da_name == "":
        args.da_name = args.ds_name

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
    run_main()
