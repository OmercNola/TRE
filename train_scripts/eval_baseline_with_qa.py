import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from utils.utils import *
from utils.saver import *
from model.model import *
from utils.logger import *
from torch import distributed as dist
from transformers import get_linear_schedule_with_warmup, AdamW


def eval_bl_with_qa():
    pass

