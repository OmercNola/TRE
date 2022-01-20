import pathlib
import platform
import os
import random
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from xml.dom import minidom
import json
import torch
from torch import nn
import itertools
import argparse
import warnings
import time
from datetime import datetime, timedelta
from model import create_pretrained_model_and_tokenizer
from logger import create_config_for_wandb
import boolq
from boolq import train_boolq, eval_boolq
from dataloaders import create_dataloader
from train import train_tre_new_questions_with_markers
from eval import eval_tre_new_questions_with_markers
from utils import results_tracker
from pathlib import Path
from pprint import pprint
import wandb
torch.set_printoptions(profile="full")
parser = argparse.ArgumentParser(description='TRE')
parser.add_argument('--device', type=torch.device,
                    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='device type')
"============================================================================"
"Train settings"
parser.add_argument('--eval', type=bool, default=True,
                    help='eval mode ? if False then training mode')
parser.add_argument('--eval_during_training', type=bool, default=True,
                    help='eval during training ?')
parser.add_argument('--save_model_during_training', type=bool, default=True,
                    help='save model during training ? ')
parser.add_argument('--save_model_every', type=int, default=500,
                    help='when to save the model - number of batches')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size (default: 2)') # 6 is good for 3 3090 GPU'S, 8 for 8 GPU'S..
parser.add_argument('--print_loss_every', type=int, default=50,
                    help='when to print the loss - number of batches')
parser.add_argument('--print_eval_every', type=int, default=50,
                    help='when to print f1 scores during eval - number of batches')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--boolq_pre_trained_model_path', type=str,
                    default='models/model_boolq_with_markers_epoch_10_.pt',
                    help='this is a pre trained model on boolq dataset, with acc (0.82)')
parser.add_argument('--checkpoint_path', type=str,
                    default='models/model_epoch_8_iter_1910_.pt',
                    help='checkpoint path for evaluation or proceed training')
"============================================================================"
"Hyper-parameters"
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.00001)')
parser.add_argument('--max_grad_norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--dropout_p', type=float, default=0.2,
                    help='dropout_p (default: 0.1)')
"============================================================================"
"Model settings"
parser.add_argument('--output_size', type=int, default=2,
                    help='output_size (default: 2)')
parser.add_argument('--Max_Len', type=int, default=4096,
                    help='Max_Len (default: 4096)')
parser.add_argument('--Size_of_longfor', type=str, default='base',
                    help='Size_of_longformer (default: "base")')
"============================================================================"

if __name__ == '__main__':
    __file__ = 'main.py'

    os.environ['OMP_NUM_THREADS'] = '1'
    print('Available devices ', torch.cuda.device_count())

    "================================================================================="
    args = parser.parse_known_args()[0]

    # login to W&B:
    wandb.login()

    # Ensure deterministic behavior
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # not relly sure what it is, needs to check !!!!
    torch.backends.cudnn.deterministic = True
    "================================================================================="
    # create model and tokenizer (after markers adition):
    model, tokenizer = create_pretrained_model_and_tokenizer(args)
    model.to(args.device)

    # parallel:
    model = nn.DataParallel(model)
    "================================================================================="
    "BOOLQ WITH MARKERS"
    # # # Datasets:
    # dataset_boolq = load_dataset("boolq")
    # # Dataloaders:
    # train_dataloader = DataLoader(dataset_boolq['train'], batch_size=args.batch_size, shuffle=True)
    #
    # # # Training
    # #train_boolq(model, args, train_dataloader, tokenizer, num_epochs=10)
    #
    # # Evaluation
    # test_dataloader = DataLoader(dataset_boolq['validation'], batch_size=4, shuffle=False)
    # PATH = Path('models/model_boolq_with_markers_epoch_10_.pt')
    # model.load_state_dict(torch.load(PATH))
    # model.to(args.device)
    # eval_boolq(model, args, test_dataloader, tokenizer)
    "================================================================================="
    "Temporal Relation Classification"

    # boolq is a yes/no QA dataset, load the pretrained model:
    PATH = Path(args.boolq_pre_trained_model_path)
    model.load_state_dict(torch.load(PATH))

    # load checkpoint:
    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)

    # Dataloaders:
    train_dataloader = create_dataloader(args, 'train')
    val_dataloader = create_dataloader(args, 'val')
    test_dataloader = create_dataloader(args, 'test')

    # config for the experiment:
    config_for_wandb = create_config_for_wandb(args, 'MTRES')

    # tell wandb to get started
    with wandb.init(project="tre", entity='omerc', config=config_for_wandb):

        wandb.log({"seed": args.seed})

        """Training"""
        if not args.eval:
            train_tre_new_questions_with_markers(
                model, args, train_dataloader, test_dataloader,
                tokenizer, checkpoint_path=checkpoint_path
            )

        """Evaluation"""
        if args.eval:
            tracker = results_tracker()
            eval_tre_new_questions_with_markers(
                model, args, val_dataloader,
                tokenizer, tracker, checkpoint_path=checkpoint_path
            )
    "================================================================================="
