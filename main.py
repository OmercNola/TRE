import platform
import os
import random
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from xml.dom import minidom
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, BertTokenizer
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, AdamW
from datasets import load_dataset
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
import itertools
import argparse
import warnings
import time
from datetime import datetime, timedelta
from longformer import create_longformer
import boolq
from boolq import train_boolq, eval_boolq
from data import TRE_training_data_with_markers, TRE_test_data_with_markers
from data import TRE_validation_data_with_markers
from train_and_eval import train_tre_new_questions_with_markers
from train_and_eval import eval_tre_new_questions_with_markers, results_tracker
from pathlib import Path
torch.set_printoptions(profile="full")
parser = argparse.ArgumentParser(description='TRE')
parser.add_argument('--device', type=torch.device,
                    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='device type')
"============================================================================"
"Train settings"
parser.add_argument('--eval', type=bool, default=False,
                    help='eval mode ? if False then training mode')
parser.add_argument('--eval_during_training', type=bool, default=True,
                    help='eval during training ?')
parser.add_argument('--save_model_during_training', type=bool, default=True,
                    help='save model during training ? ')
parser.add_argument('--save_model_every', type=int, default=500,
                    help='when to save the model - number of batches')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size (default: 2)')
parser.add_argument('--print_loss_every', type=int, default=50,
                    help='when to print the loss - number of batches')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
"============================================================================"
"Hyper-parameters"
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.00001)')
parser.add_argument('--max-grad-norm', type=float, default=50,
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
    torch.manual_seed(args.seed)
    "================================================================================="
    "MODEL AND TOKENIZER"
    # load tokenizer, add new tokens, and save tokenizer:
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    print(f'len of tokenizer before adding new tokens: {len(tokenizer)}')
    special_tokens_dict = {
        'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print(f'len of tokenizer after adding new tokens: {len(tokenizer)}')
    model_ = AutoModel.from_pretrained("allenai/longformer-base-4096")
    model_.resize_token_embeddings(len(tokenizer))
    model = create_longformer(model_, args).to(args.device)
    model = nn.DataParallel(model)
    "================================================================================="
    "BOOLQ"
    # Datasets:
    # dataset_boolq = load_dataset("boolq")
    # Dataloaders:
    # train_dataloader = DataLoader(dataset_boolq['train'], batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(dataset_boolq['validation'], batch_size=4, shuffle=False)

    # Training / evaluation:
    # train_boolq(num_epochs=8)
    # eval_boolq()
    "================================================================================="
    "BOOLQ WITH MARKERS"
    # model_.resize_token_embeddings(len(tokenizer))
    # model = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
    # model = nn.DataParallel(model)
    #
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

    """this is a trained model on boolq dataset, with acc (0.82)"""
    # boolq is a yes/no QA dataset, load the pretrained model:
    PATH = Path('models/model_boolq_with_markers_epoch_10_.pt')
    model.load_state_dict(torch.load(PATH))

    """if you want to evaluate or proceed training,
       change this path, if set to None it will ignor checkpoint"""
    # checkpoint_path = Path('models/model_epoch_9_.pt')
    checkpoint_path = None

    # Dataloaders:
    train_dataloader = DataLoader(
        TRE_training_data_with_markers,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        TRE_validation_data_with_markers,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        TRE_test_data_with_markers,
        batch_size=args.batch_size,
        shuffle=True
    )

    """Training"""
    if not args.eval:
        eval_scores = train_tre_new_questions_with_markers(
            model, args, train_dataloader, test_dataloader,
            tokenizer, num_epochs=args.epochs,
            checkpoint_path=checkpoint_path
        )

        # df = pd.DataFrame(eval_scores)
        # df.to_csv(Path('models/results.csv'))

    """Evaluation"""
    if args.eval:
        tracker = results_tracker()
        (f1_macro, f1_micro) = eval_tre_new_questions_with_markers(
            model, args, test_dataloader,
            tokenizer, tracker, checkpoint_path=checkpoint_path
        )
    "================================================================================="
