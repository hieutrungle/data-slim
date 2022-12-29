import torch
import numpy as np
from utils import logger, utils
import argparse
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

def get_data(top_rights, bottom_lefts, times, output_path):
    print("Getting data...")
    pass

if __name__ == "__main__":
    get_data()
    