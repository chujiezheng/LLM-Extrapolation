
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List
import logging
from tqdm import tqdm
import random
import math
from transformers import AutoTokenizer
import re
from torch.utils.data import Dataset
from datasets import load_dataset, disable_caching
disable_caching()

import pynvml
pynvml.nvmlInit()

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def logging_cuda_memory_usage():
    n_gpus = pynvml.nvmlDeviceGetCount()
    for i in range(n_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info("GPU {}: {:.2f} GB / {:.2f} GB".format(i, meminfo.used / 1024 ** 3, meminfo.total / 1024 ** 3))
