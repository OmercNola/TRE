import os
from sys import platform
import random
import numpy as np
import torch
from torch import nn
import argparse
from eval import eval_tre_new_questions_with_markers
from transformers import get_linear_schedule_with_warmup, AdamW
from model.model import create_pretrained_model_and_tokenizer
from utils.logger import *
from utils.saver import *
from utils.utils import *
from datasets_and_loaders.datasets import *
from datasets_and_loaders.dataloaders import create_dataloader
from utils.utils import results_tracker
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import torch.multiprocessing as mp
import wandb
import time
from tqdm import tqdm
def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0
def train(model, args, train_dataloader, test_dataloader, tokenizer):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param train_dataloader:
    :type train_dataloader:
    :param tokenizer:
    :type tokenizer:
    :param args.epochs:
    :type args.epochs:
    :param checkpoint_path:
    :type checkpoint_path:
    :return:
    :rtype:
    """

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    if is_master():
        wandb.watch(model, criterion, log="all", log_freq=10)

    # loss progress counters
    total_loss_for_print = 0
    total_loss_for_save = 0

    # set epoch_start to 1, it we have checkpoint, will load it from there.
    epoch_start = 1

    # training mode:
    model.train()

    # start time:
    t0 = time.time()

    # total nuber of batches counter:
    batches_overall = 0

    if is_master():
        epoch_itrator = tqdm(range(epoch_start, args.epochs+1, 1))
    else:
        epoch_itrator = range(epoch_start, args.epochs+1, 1)

    for epoch in epoch_itrator:

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for batch_counter, instances in enumerate(train_dataloader, start=1):

            batches_overall += 1

            passages = instances[0]
            first_words = instances[1][0]
            second_words = instances[1][1]
            word_labels = instances[1][2]

            zip_object = zip(passages, first_words, second_words, word_labels)
            for passage, first_word, second_word, Label in zip_object:

                if args.ignor_vague_lable_in_training:
                    if Label.strip() == 'VAGUE':
                        continue

                question_1 = question_1_for_regular_markers(
                    first_word, second_word) + tokenizer.sep_token
                question_2 = question_2_for_regular_markers(
                    first_word, second_word) + tokenizer.sep_token

                questions_list = [
                    ('question_1', question_1),
                    ('question_2', question_2)
                ]

                for question_name, question in questions_list:

                    label = get_label(question_name, Label)

                    # tokenize question and text as a pair, Roberta
                    encodings = tokenizer(
                        question,
                        passage,
                        max_length=args.Max_Len,
                        padding='max_length',
                        truncation=True
                    )

                    input_ids = encodings['input_ids']
                    attention_mask = encodings['attention_mask']

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_labels.append(label)

                    # compute loss and update weights every args.batch_size:
                    if len(batch_input_ids) == args.batch_size:

                        batch_input_ids = torch.tensor(
                            batch_input_ids, requires_grad=False).to(args.device)
                        batch_attention_mask = torch.tensor(
                            batch_attention_mask, requires_grad=False).to(args.device)
                        batch_labels = torch.tensor(
                            batch_labels, requires_grad=False).to(args.device)

                        # zero gradients before update:
                        optimizer.zero_grad()

                        # forward pass:
                        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

                        # extract loss
                        loss = criterion(outputs, batch_labels)

                        # compute gradients:
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        # update parameters
                        optimizer.step()

                        # Update the learning rate.
                        scheduler.step()

                        # save training loss:
                        total_loss_for_print += loss.item()
                        total_loss_for_save += loss.item()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            # Print and save progress once in a while...
            if is_master():
                if batch_counter % args.print_loss_every == 0:
                    # just print:
                    print_training_progress(
                        t0, len(train_dataloader),
                        epoch, batch_counter, total_loss_for_print
                    )
                    # save in wandb:

                    train_log(total_loss_for_print, epoch, batches_overall)
                    total_loss_for_print = 0

                # save the model once in a while:
                if batch_counter % args.save_model_every == 0:

                    # save:
                    if args.save_model_during_training:
                        save_model_checkpoint(
                            args, model, optimizer,
                            scheduler, len(train_dataloader),
                            batch_counter, epoch,
                            total_loss_for_save
                        )
                        total_loss_for_save = 0

        if is_master():
            # evaluate at the end of the epoch:
            if args.eval_during_training:
                tracker = results_tracker()
                eval_tre_new_questions_with_markers(
                    model, args, test_dataloader,
                    tokenizer, tracker, checkpoint_path=None,
                    batches_overall=batches_overall
                )
def main(args, init_distributed=False):

    """
    :param args:
    :type args:
    :param init_distributed:
    :type init_distributed:
    :return:
    :rtype:
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()
        args.device = torch.device("cuda")

    if init_distributed:
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )
        # dist.all_reduce(torch.zeros(1).cuda())
        args.device = torch.device("cuda", args.rank)

    # create model and tokenizer (after markers adition):
    model, tokenizer = create_pretrained_model_and_tokenizer(args)
    model = nn.DataParallel(model, device_ids=[args.rank])
    model.to(args.device)
    "================================================================================="
    if is_master():
        # config for the experiment:
        config_for_wandb = create_config_for_wandb(args, 'MTRES')
        # tell wandb to get started:
        wandb.init(project="tre", entity='omerc', config=config_for_wandb)
        wandb.config.update(args)
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
    "=========================  Temporal Relation Classification  ===================="
    "================================================================================="
    # boolq is a yes/no QA dataset, load the pretrained model:
    PATH = Path(args.boolq_pre_trained_model_path)
    model.load_state_dict(torch.load(PATH))
    "=================================================================="
    # prepare checkpoint path:
    checkpoint_path = None
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
    # if there is a checkpoint, load it:
    if checkpoint_path is not None:
        (model, _, _, _, _, _) = \
            load_model_checkpoint(
                checkpoint_path, model,
                None, None
            )
    "=================================================================="
    # Parallel
    if torch.cuda.is_available():
        # if we have more than 1 gpu:
        if args.world_size > 1:

            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = DDP(
                model,
                device_ids=[args.rank],
                output_device=args.rank,
                find_unused_parameters=True,
                broadcast_buffers=False
            ).cuda()

            ddp = True
            train_dataloader = create_dataloader(args, 'train', ddp)
            val_dataloader = create_dataloader(args, 'val', ddp)
            test_dataloader = create_dataloader(args, 'test', ddp)

        # if we have just 1 gpu:
        else:
            model = model.to(args.device)
            ddp = False
            train_dataloader = create_dataloader(args, 'train', ddp)
            val_dataloader = create_dataloader(args, 'val', ddp)
            test_dataloader = create_dataloader(args, 'test', ddp)
    "=================================================================="
    """Training"""
    if not args.eval:
        train(model, args, train_dataloader, val_dataloader, tokenizer)
    "=================================================================="
    """Evaluation"""
    if args.eval:
        tracker = results_tracker()
        eval_tre_new_questions_with_markers(
            model, args, test_dataloader,
            tokenizer, tracker, checkpoint_path=checkpoint_path
        )
    "=================================================================="
def distributed_main(device_id, args):
    """
    :param device_id:
    :type device_id:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    args.device_id = device_id
    if args.rank is None:
        args.rank = args.start_rank + device_id
    main(args, init_distributed=True)
if __name__ == '__main__':
    __file__ = 'main.py'
    "============================================================================"
    parser = argparse.ArgumentParser(description='TRE')
    "============================================================================"
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='device type')
    "============================================================================"
    "Train settings 1"
    parser.add_argument('--eval', type=bool, default=False,
                        help='eval mode ? if False then training mode')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle')
    parser.add_argument('--eval_during_training', type=bool, default=True,
                        help='eval during training ?')
    parser.add_argument('--save_model_during_training', type=bool, default=True,
                        help='save model during training ? ')
    parser.add_argument('--save_table_of_results_after_eval', type=bool, default=False,
                        help='save table of results (with text) after eval ?')
    parser.add_argument('--save_model_every', type=int, default=1000,
                        help='when to save the model - number of batches')
    parser.add_argument('--ignor_vague_lable_in_training', type=bool, default=True,
                        help='if True - ignor vague lable in training')
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='batch_size (default: 2)')  # 6 is good for 3 3090 GPU'S, 8 for 8 GPU'S..
    parser.add_argument('--checkpoint_path', type=str,
                        default=None,  # 'models/fast-butterfly-49_epoch_1_iter_3184_.pt',
                        help='checkpoint path for evaluation or proceed training ,'
                             'if set to None then ignor checkpoint')
    "============================================================================"
    "Train settings 2"
    parser.add_argument('--print_loss_every', type=int, default=50,
                        help='when to print the loss - number of batches')
    parser.add_argument('--print_eval_every', type=int, default=50,
                        help='when to print f1 scores during eval - number of batches')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--boolq_pre_trained_model_path', type=str,
                        default='models/model_boolq_with_markers_epoch_10_.pt',
                        help='this is a pre trained model on boolq dataset, with acc (0.82)')
    "============================================================================"
    "Hyper-parameters"
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta 1 for AdamW. default=0.9')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='beta 2 for AdamW. default=0.999')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight_decay for AdamW. default=0.001')
    parser.add_argument('--max_grad_norm', type=float, default=40,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--dropout_p', type=float, default=0.15,
                        help='dropout_p (default: 0.1)')
    parser.add_argument('--sync-bn', action='store_true',
                        default=False, help='sync batchnorm')
    parser.add_argument('--num_workers', type=int,
                        default=6, help='num_workers')
    "============================================================================"
    "Model settings"
    parser.add_argument('--output_size', type=int, default=2,
                        help='output_size (default: 2)')
    parser.add_argument('--Max_Len', type=int, default=4096,
                        help='Max_Len (default: 4096)')
    parser.add_argument('--Size_of_longfor', type=str, default='base',
                        help='Size_of_longformer (default: "base")')
    "============================================================================"
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Available devices ', torch.cuda.device_count())
    "================================================================================="
    args = parser.parse_known_args()[0]
    "================================================================================="
    # login to W&B:
    wandb.login()
    "================================================================================="
    # Ensure deterministic behavior
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # not relly sure what it is, needs to check !!!!
    torch.backends.cudnn.deterministic = True
    "================================================================================="
    # Distributed
    args.world_size = torch.cuda.device_count()
    # args.world_size = args.gpus * args.nodes
    if args.world_size == 1:
        args.device_id = 0
        main(args)
    if args.world_size > 1:
        args.batch_size = int(args.batch_size / args.world_size)
        port = random.randint(10000, 20000)
        args.init_method = f"tcp://localhost:{port}" if platform != "win32" else 'tcp://127.0.0.1:23456'
        args.rank = None
        args.start_rank = 0
        args.backend = 'nccl' if platform != "win32" else 'gloo'
        mp.spawn(fn=distributed_main, args=(args,), nprocs=args.world_size,)
    else:
        args.device = torch.device("cpu")
        main(args)

