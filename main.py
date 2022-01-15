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
from longformer import Longformer
import boolq
from boolq import train_boolq, eval_boolq
from data_preprocess import TRE_validation_data, TRE_training_data, TRE_test_data, TRE_training_data_with_markers, TRE_test_data_with_markers
from data_preprocess import TRE_training_data_for_vague, TRE_training_data_for_equal, percents_equal, percents_vague
from TRE_base_on_boolq import train_TRE, eval_TRE, train_TRE_with_markers
from TRE_base_on_boolq import eval_TRE_with_markers, train_TRE_New_questions_with_markers, eval_TRE_New_questions_with_markers
from TRE_base_on_boolq_4_different_models import train_TRE_diff_models, eval_TRE_one_model, eval_TRE_diff_models
from pathlib import Path

torch.set_printoptions(profile="full")
parser = argparse.ArgumentParser(description='TRE')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate (default: 0.00001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--output_size', type=int, default=2,
                    help='output_size (default: 2)')
parser.add_argument('--dropout_p', type=float, default=0.1,
                    help='dropout_p (default: 0.1)')
parser.add_argument('--Max_Len', type=int, default=4096,
                    help='Max_Len (default: 4096)')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size (default: 2)')
parser.add_argument('--Size_of_longfor', type=str, default='base',
                    help='Size_of_longformer (default: "base")')
parser.add_argument('--device', type=torch.device,
                    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='device type')

if __name__ == '__main__':
    __file__ = 'main.py'
    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    print('Available devices ', torch.cuda.device_count())
    print(torch.cuda.current_device())

    with warnings.catch_warnings():
        "====================================================================================================="
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", RuntimeError)
        warnings.simplefilter("ignore", EnvironmentError)
        args = parser.parse_known_args()[0]
        "====================================================================================================="
        "MODEL AND TOKENIZER"
        # load tokenizer, add new tokens, and save tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        print(f'len of tokenizer before adding new tokens: {len(tokenizer)}')
        special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        print(f'len of tokenizer after adding new tokens: {len(tokenizer)}')
        # path_for_tokenizer = Path().resolve().parent / 'models/tokenizer/'
        # tokenizer.save_pretrained(path_for_tokenizer)
        # tokenizer = AutoTokenizer.from_pretrained(path_for_tokenizer)
        model_ = AutoModel.from_pretrained("allenai/longformer-base-4096")
        #model_.resize_token_embeddings(len(tokenizer))
        # model = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
        "====================================================================================================="
        "BOOL-Q"
        # Datasets:
        # dataset_boolq = load_dataset("boolq")
        # Dataloaders:
        # train_dataloader = DataLoader(dataset_boolq['train'], batch_size=1, shuffle=True)
        # test_dataloader = DataLoader(dataset_boolq['validation'], batch_size=4, shuffle=False)

        # Training / evaluation:
        # train_boolq(num_epochs=8)
        # eval_boolq()
        "====================================================================================================="
        "BOOL-Q WITH MARKERS"
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
        "====================================================================================================="
        "TRE 1 model"
        model_.resize_token_embeddings(len(tokenizer))
        model = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
        model = nn.DataParallel(model)
        # PATH = Path('models/model_boolq_with_markers_epoch_10_.pt')
        PATH = Path('models/model_with_markers_epoch_6_.pt')
        model.load_state_dict(torch.load(PATH))

        train_dataloader = DataLoader(TRE_training_data_with_markers, batch_size=args.batch_size, shuffle=True)
        # val_dataloader = DataLoader(TRE_validation_data, batch_size=2, shuffle=True)
        test_dataloader = DataLoader(TRE_test_data_with_markers, batch_size=2, shuffle=False)

        # Training / evaluation:
        # train_TRE_New_questions_with_markers(model, args, train_dataloader, tokenizer, num_epochs=5)
        eval_TRE_New_questions_with_markers(model, args, train_dataloader, tokenizer)
        "====================================================================================================="
        "TRE 4 models"
        # PATH = Path().resolve().parent / 'models/model_tre_Aq_Timebank_BEFORE_epoch_4_.pt'
        # model.load_state_dict(torch.load(PATH))

        # train_dataloader = DataLoader(TRE_training_data, batch_size=1, shuffle=True)
        # train_dataloader_vague = DataLoader(TRE_training_data_for_vague, batch_size=1, shuffle=True)
        # train_dataloader_equal = DataLoader(TRE_training_data_for_equal, batch_size=1, shuffle=True)
        # train_dataloader_equal = DataLoader([i for i in TRE_training_data_for_equal if i[1][4].strip() == 'EQUAL'], batch_size=1, shuffle=True)

        # val_dataloader = DataLoader(TRE_validation_data, batch_size=2, shuffle=True)
        # test_dataloader = DataLoader(TRE_test_data, batch_size=1, shuffle=False)
        # test_dataloader = DataLoader([i for i in TRE_test_data if i[1][4].strip() == 'EQUAL'], batch_size=1, shuffle=False)

        # Training / evaluation:
        # train_TRE_diff_models(model, args, train_dataloader, tokenizer, 'BEFORE', num_epochs=8)
        # train_TRE_diff_models(model, args, train_dataloader, tokenizer, 'AFTER', num_epochs=4)
        # train_TRE_diff_models(model, args, train_dataloader_equal, tokenizer, 'EQUAL', percents_equal, num_epochs=4)
        # train_TRE_diff_models(model, args, train_dataloader_vague, tokenizer, 'VAGUE', percents_vague, num_epochs=10)

        # eval_TRE_one_model(model, args, test_dataloader, tokenizer, 'BEFORE')

        # model_before = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
        # PATH = Path().resolve().parent / 'models/model_tre_Aq_Timebank_BEFORE_epoch_4_.pt'
        # model_before.load_state_dict(torch.load(PATH))
        #
        # model_vague = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
        # PATH = Path().resolve().parent / 'models/model_tre_Aq_Timebank_VAGUE_epoch_10_.pt'
        # model_vague.load_state_dict(torch.load(PATH))
        #
        # model_equal = Longformer(model_, args.output_size, args.dropout_p, args.Size_of_longfor, args.Max_Len).to(args.device)
        # PATH = Path().resolve().parent / 'models/model_tre_Aq_Timebank_EQUAL_epoch_4_.pt'
        # model_equal.load_state_dict(torch.load(PATH))
        #
        # models_list = [model_before, model_vague, model_equal]
        # eval_TRE_diff_models(models_list, args, test_dataloader, tokenizer)