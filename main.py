import os
import wandb
import time
import random
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
import platform
from utils.utils import *
from utils.saver import *
from model.model import *
from utils.logger import *
import torch.multiprocessing as mp
from torch import distributed as dist
from data.dataloaders import create_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup, AdamW
def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0
def train(model, args, train_loader, train_sampler, test_loader, tokenizer,):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param train_loader:
    :type train_loader:
    :param train_sampler:
    :type train_sampler:
    :param test_loader:
    :type test_loader:
    :param tokenizer:
    :type tokenizer:
    :return:
    :rtype:
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )

    # Create the learning rate scheduler.
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    if is_master() and args.use_wandb_logger:
        wandb.watch(model, criterion, log="all", log_freq=50, log_graph=(False))

    best_f1_macro, best_f1_micro = 0, 0

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

    if is_master() and not platform.platform().startswith('Win'):
        # the progress bar doesnt work very good in Windows and pycharm
        epoch_itrator = tqdm(range(epoch_start, args.epochs+1, 1), position=0, leave=True)
    else:
        epoch_itrator = range(epoch_start, args.epochs+1, 1)

    is_distributed = args.world_size > 1

    for epoch in epoch_itrator:

        if is_master():
            print(f'training... epoch {epoch}')

        if is_distributed:
            # the next line is for shuffling the data every epoch (if shuffle is True)
            # it has tp be before creating the dataloaer
            train_sampler.set_epoch(epoch)
            # the next line is to ensure that all ranks start
            # the epoch together after evaluation
            dist.barrier()

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for batch_counter, instances in enumerate(train_loader, start=1):

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

                q_1 = question_1(args, first_word, second_word) + tokenizer.sep_token
                q_2 = question_2(args, first_word, second_word) + tokenizer.sep_token
                questions_list = [('question_1', q_1), ('question_2', q_2)]

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

                    # compute loss and update weights every args.single_rank_batch_size:
                    if len(batch_input_ids) == args.single_rank_batch_size:

                        # see this post for understanding the next lines
                        # https://discuss.pytorch.org/t/multiprocessing-barrier-blocks-all-processes/80345
                        # we have to keep this lines if we want to use dist.barrier()
                        # otherwise it will hung forever...
                        signal = torch.tensor([1], device=args.device)
                        work = dist.all_reduce(signal, async_op=True)
                        work.wait()
                        if signal.item() < args.world_size:
                            continue

                        batch_input_ids = torch.tensor(
                            batch_input_ids, requires_grad=False, device=args.device)
                        batch_attention_mask = torch.tensor(
                            batch_attention_mask, requires_grad=False, device=args.device)
                        batch_labels = torch.tensor(
                            batch_labels, requires_grad=False, device=args.device)

                        # zero gradients before update:
                        #optimizer.zero_grad(set_to_none=True)
                        optimizer.zero_grad()

                        # forward pass:
                        try:
                            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                        except RuntimeError as exception:
                            if "out of memory" in str(exception):
                                print("WARNING: out of memory")
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                            else:
                                raise exception

                        # extract loss
                        loss = criterion(outputs, batch_labels)

                        # compute gradients:
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem:
                        if args.use_clip_grad_norm:
                            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        # update parameters
                        optimizer.step()

                        # Update the learning rate.
                        if (args.use_scheduler) and (scheduler is not None):
                            scheduler.step()

                        # save training loss:
                        total_loss_for_print += loss.item()
                        total_loss_for_save += loss.item()

                        batch_input_ids = []
                        batch_attention_mask = []
                        batch_labels = []

            # Print and save progress once in a while...
            if batch_counter % args.print_loss_every == 0:

                total_loss_for_print = total_loss_for_print / args.print_loss_every

                # just print:
                print_training_progress(
                    args, t0, len(train_loader),
                    epoch, batch_counter,
                    total_loss_for_print
                )

                if is_master():
                    # save in wandb:
                    if args.use_wandb_logger:
                        train_log(args, total_loss_for_print, epoch, batches_overall)

                total_loss_for_print = 0

            # save the model once in a while:
            if batch_counter % args.save_model_every == 0:

                if is_master():
                    if args.save_model_during_training:
                        save_model_checkpoint(
                            args, model, optimizer,
                            scheduler, len(train_loader),
                            batch_counter, epoch,
                            total_loss_for_save
                        )
                        total_loss_for_save = 0

        # # at the end of the epoch:

        # compute the correct ave loss:
        if batch_counter < args.print_loss_every:
            total_loss_for_print = total_loss_for_print / batch_counter

        elif batch_counter % args.print_loss_every != 0:
            total_loss_for_print = total_loss_for_print /\
                                       (batch_counter % args.print_loss_every)

        elif batch_counter % args.print_loss_every == 0:
            total_loss_for_print = 0

        # if total_loss_for_print == 0 then we dont need to print or save
        # because we already did
        if total_loss_for_print != 0:
            # print:
            print_training_progress(
                args, t0, len(train_loader),
                epoch, batch_counter, total_loss_for_print
            )
            # save wandb:
            if is_master() and args.use_wandb_logger:
                train_log(args, total_loss_for_print, epoch, batches_overall)

            total_loss_for_print = 0

        # see this post for understanding the next lines
        # https://discuss.pytorch.org/t/multiprocessing-barrier-blocks-all-processes/80345
        # we have to keep this lines if we want to use dist.barrier()
        if signal.item() >= args.world_size:
            dist.all_reduce(torch.tensor([0], device=args.device))

        # ensure that all ranks start evaluation together
        dist.barrier()

        # evaluate at the end of the epoch:
        if args.eval_during_training:
            macro, micro = eval(model, args, test_loader, tokenizer, epoch=epoch)

        best_f1_macro = max(macro, best_f1_macro)
        best_f1_micro = max(micro, best_f1_micro)

    if is_master() and args.use_wandb_logger and args.eval_during_training:
        wandb.log({"best_f1_macro": best_f1_macro, "best_f1_micro": best_f1_micro})
def train_baseline(model, args, train_loader, train_sampler, test_loader, tokenizer,):
    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param train_loader:
    :type train_loader:
    :param train_sampler:
    :type train_sampler:
    :param test_loader:
    :type test_loader:
    :param tokenizer:
    :type tokenizer:
    :return:
    :rtype:
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        weight_decay=args.weight_decay
    )

    # Create the learning rate scheduler.
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    if is_master() and args.use_wandb_logger:
        wandb.watch(model, criterion, log="all", log_freq=50, log_graph=(False))

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

    if is_master() and not platform.platform().startswith('Win'):
        # the progress bar doesnt work very good in Windows and pycharm
        epoch_itrator = tqdm(range(epoch_start, args.epochs + 1, 1), position=0, leave=True)
    else:
        epoch_itrator = range(epoch_start, args.epochs + 1, 1)

    is_distributed = args.world_size > 1

    for epoch in epoch_itrator:

        if is_master():
            print(f'training... epoch {epoch}')

        if is_distributed:
            # the next line is for shuffling the data every epoch (if shuffle is True)
            # it has tp be before creating the dataloaer
            train_sampler.set_epoch(epoch)
            # the next line is to ensure that all ranks start
            # the epoch together after evaluation
            dist.barrier()

        for batch_counter, instances in enumerate(train_loader, start=1):

            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []

            batches_overall += 1

            passages = instances[0]
            word_labels = instances[1][2]

            for passage, Label in zip(passages, word_labels):

                label = get_label_for_baseline(Label)

                # tokenize question and text as a pair, Roberta
                encodings = tokenizer(
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

            # see this post for understanding the next lines
            # https://discuss.pytorch.org/t/multiprocessing-barrier-blocks-all-processes/80345
            # we have to keep this lines if we want to use dist.barrier()
            # otherwise it will hung forever...
            signal = torch.tensor([1], device=args.device)
            work = dist.all_reduce(signal, async_op=True)
            work.wait()
            if signal.item() < args.world_size:
                continue

            batch_input_ids = torch.tensor(
                batch_input_ids, requires_grad=False, device=args.device)
            batch_attention_mask = torch.tensor(
                batch_attention_mask, requires_grad=False, device=args.device)
            batch_labels = torch.tensor(
                batch_labels, requires_grad=False, device=args.device)

            # zero gradients before update:
            #optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()

            # forward pass:
            try:
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            # extract loss
            loss = criterion(outputs, batch_labels.squeeze(1))

            # compute gradients:
            loss.backward()

            # This is to help prevent the "exploding gradients" problem:
            if args.use_clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # update parameters
            optimizer.step()

            # Update the learning rate.
            if (args.use_scheduler) and (scheduler is not None):
                scheduler.step()

            # save training loss:
            total_loss_for_print += loss.item()
            total_loss_for_save += loss.item()



            # Print and save progress once in a while...
            if batch_counter % args.print_loss_every == 0:

                total_loss_for_print = total_loss_for_print / args.print_loss_every

                # just print:
                print_training_progress(
                    args, t0, len(train_loader),
                    epoch, batch_counter,
                    total_loss_for_print
                )

                if is_master():
                    # save in wandb:
                    if args.use_wandb_logger:
                        train_log(args, total_loss_for_print, epoch, batches_overall)

                total_loss_for_print = 0

            # save the model once in a while:
            if batch_counter % args.save_model_every == 0:

                if is_master():
                    if args.save_model_during_training:
                        save_model_checkpoint(
                            args, model, optimizer,
                            scheduler, len(train_loader),
                            batch_counter, epoch,
                            total_loss_for_save
                        )
                        total_loss_for_save = 0

                # evaluate:
                if args.eval_during_training:
                    eval(model, args, test_loader, tokenizer, batches_overall=batches_overall)

        # # at the end of the epoch:

        # compute the correct ave loss:
        if batch_counter < args.print_loss_every:
            total_loss_for_print = total_loss_for_print / batch_counter

        elif batch_counter % args.print_loss_every != 0:
            total_loss_for_print = total_loss_for_print / \
                                   (batch_counter % args.print_loss_every)

        elif batch_counter % args.print_loss_every == 0:
            total_loss_for_print = 0

        # if total_loss_for_print == 0 then we dont need to print or save
        # because we already did
        if total_loss_for_print != 0:
            # print:
            print_training_progress(
                args, t0, len(train_loader),
                epoch, batch_counter, total_loss_for_print
            )
            # save wandb:
            if is_master() and args.use_wandb_logger:
                train_log(args, total_loss_for_print, epoch, batches_overall)

            total_loss_for_print = 0

        # see this post for understanding the next lines
        # https://discuss.pytorch.org/t/multiprocessing-barrier-blocks-all-processes/80345
        # we have to keep this lines if we want to use dist.barrier()
        if signal.item() >= args.world_size:
            dist.all_reduce(torch.tensor([0], device=args.device))

        # ensure that all ranks start evaluation together
        dist.barrier()

        # evaluate at the end of the epoch:
        if args.eval_during_training:
            eval_baseline(
                model, args, test_loader, tokenizer, batches_overall=batches_overall
            )
def eval(model, args, test_loader, tokenizer, epoch=None):

    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param test_loader:
    :type test_loader:
    :param tokenizer:
    :type tokenizer:
    :param tracker:
    :type tracker:
    :param checkpoint_path:
    :type checkpoint_path:
    :param batches_overall:
    :type batches_overall:
    :return:
    :rtype:
    """

    # the evaluation is currently done only on master (rank 0),
    # the next line ensures localy evaluation (withot ddp comunication).
    # without this line it will hung forever.
    # see this post: https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522
    # if hasattr(model, "module"):
    #     model = model.module

    # evaluation mode:
    model.eval()

    # check if Distributed mode:
    is_distributed = args.world_size > 1

    # create Tracker:
    tracker = results_tracker()

    # reset Tracker:
    tracker.reset()

    # create wandb table for traking the rsults:
    if args.use_wandb_logger and args.save_table_of_results_after_eval:
        table = wandb.Table(
            columns=[
                'passage', 'passage length',
                'word_1', 'word_2',
                'ans_1', 'ans_2',
                'pred_label', 'real_label',
                'correct answer'
            ]
        )

    if is_master():
        print(f'evaluation in progress...')

    for batch_counter, instances in enumerate(test_loader, start=1):

        passages = instances[0]
        first_words, second_words = instances[1][0], instances[1][1]
        word_labels = instances[1][2]

        zip_object = zip(passages, first_words, second_words, word_labels)
        for passage, first_word, second_word, Label in zip_object:

            # get the questions:
            q_1 = question_1(
                args, first_word, second_word) + tokenizer.sep_token
            q_2 = question_2(
                args, first_word, second_word) + tokenizer.sep_token

            questions_list = [
                ('question_1', q_1),
                ('question_2', q_2)
            ]

            # 2 Questions for each instance:
            results = []
            for question_name, question in questions_list:

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

                input_ids = torch.tensor(
                    [input_ids], requires_grad=False).to(args.device)
                attention_mask = torch.tensor(
                    [attention_mask], requires_grad=False).to(args.device)

                # ensure no gradients for eval:
                with torch.no_grad():

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                    # our prediction:
                    pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

                    # move to cpu and numpy:
                    pred = pred.clone().detach().cpu().numpy()[0]

                    # results:
                    results.append([question_name, pred])

            # now we 2 questions ready, we update results tracker:
            ans1, ans2 = results[0][1], results[1][1]

            pred_label = tracker.update(Label, ans1, ans2)

            # raw_data for logging in wandb:
            if args.use_wandb_logger and args.save_table_of_results_after_eval:

                passage_length = len(passage)

                if Label.strip() == 'SIMULTANEOUS':
                    correct_answer = pred_label == 'EQUAL'
                else:
                    correct_answer = pred_label == Label.strip()

                real_label = Label.strip()

                # add raw_data to the wandb table:
                table.add_data(
                    passage, passage_length, first_word, second_word,
                    ans1, ans2, pred_label, real_label, correct_answer
                )

        if batch_counter % args.print_eval_every == 0:

            # if is_master() and (not is_distributed):

            # get f1 macro and f1 micro results:
            macro, micro = tracker.f1_macro_and_micro()

            eval_precent = (batch_counter / len(test_loader)) * 100
            print(f'[eval Rank: {args.rank}] f1 macro: {macro}, f1 micro: {micro}, '
                  f'evaluation percent: {eval_precent:.3f}')


    # if we are in Distributed mode, then we need to collect the results from
    # all processes:
    if is_distributed:
        # tracker.get_list_of_values() gives us list of [tracker.TP_BEFORE, tracker.TN_BEFORE... etc]
        # make a tensor for all_reduce:
        tensor = torch.tensor(tracker.get_list_of_values(),
                              dtype=torch.int64, device=args.device)
        # here we sum up the values from all processes:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # convert tensor to numpy:
        list_of_values_after_all_reduce = tensor.clone().detach().cpu().numpy()
        # update the tracker with reduce results before computing F1 scores:
        tracker.update_values_from_list(list_of_values_after_all_reduce)

    # F1 scores at the end of the evaluation:
    macro, micro = tracker.f1_macro_and_micro()

    # print results and log to wandb:
    if is_master():

        if args.use_wandb_logger:
            eval_log(args, macro, micro, epoch)

        print_eval_results(macro, micro, batch_counter, len(test_loader))

        if args.use_wandb_logger and args.save_table_of_results_after_eval:
            wandb.log({f'results table {wandb.run.name}': table})

    return macro, micro
def eval_baseline(model, args, test_loader, tokenizer, batches_overall=None):

    """
    :param model:
    :type model:
    :param args:
    :type args:
    :param test_loader:
    :type test_loader:
    :param tokenizer:
    :type tokenizer:
    :param tracker:
    :type tracker:
    :param checkpoint_path:
    :type checkpoint_path:
    :param batches_overall:
    :type batches_overall:
    :return:
    :rtype:
    """

    # the evaluation is currently done only on master (rank 0),
    # the next line ensures localy evaluation (without DDP comunication).
    # without this line it will hung forever.
    # see this post: https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522
    if hasattr(model, "module"):
        model = model.module

    # evaluation mode:
    model.eval()

    # check if Distributed mode:
    is_distributed = args.world_size > 1

    # create Tracker:
    tracker = baseline_results_tracker()

    # reset Tracker:
    tracker.reset()

    for batch_counter, instances in enumerate(test_loader, start=1):

        passages = instances[0]
        word_labels = instances[1][2]

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for passage, Label in zip(passages, word_labels):

            # tokenize question and text as a pair, Roberta
            encodings = tokenizer(
                passage,
                max_length=args.Max_Len,
                padding='max_length',
                truncation=True
            )

            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(Label.strip())

        batch_input_ids = torch.tensor(
            batch_input_ids, requires_grad=False, device=args.device)
        batch_attention_mask = torch.tensor(
            batch_attention_mask, requires_grad=False, device=args.device)

        # forward pass:
        with torch.no_grad():

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            # prediction:
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # move to cpu and numpy:
            preds = preds.clone().detach().cpu().numpy()

            for pred, label in zip(preds, batch_labels):
                tracker.update(pred, label)

        if batch_counter % args.print_eval_every == 0:

            if is_master() and (not is_distributed):

                # get f1 macro and f1 micro results:
                macro, micro = tracker.f1_macro_and_micro()

                eval_precent = (batch_counter / len(test_loader)) * 100
                print(f'f1 macro: {macro}, f1 micro: {micro}, '
                      f'evaluation percent: {eval_precent:.3f}')

    # if we are in Distributed mode, then we need to collect the results from
    # all processes:
    if is_distributed:
        # tracker.get_list_of_values() gives us list of [tracker.TP_BEFORE, tracker.TN_BEFORE... etc]
        # make a tensor for all_reduce:
        tensor = torch.tensor(tracker.get_list_of_values(),
                              dtype=torch.int64, device=args.device)
        # here we sum up the values from all processes:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # convert tensor to numpy:
        list_of_values_after_all_reduce = tensor.clone().detach().cpu().numpy()
        # update the tracker with reduce results before computing F1 scores:
        tracker.update_values_from_list(list_of_values_after_all_reduce)

    # F1 scores at the end of the evaluation:
    macro, micro = tracker.f1_macro_and_micro()

    # log to wandb:
    if is_master():

        if args.use_wandb_logger:
            wandb.log({"batches_overall": batches_overall,
                       "f1 macro": macro, "f1 micro": micro})

        eval_precent = (batch_counter / len(test_loader)) * 100
        print(f'f1 macro: {macro}, f1 micro: {micro}, '
              f'evaluation percent: {eval_precent:.3f}')
def main(args, init_distributed=False):

    """
    :param args:
    :type args:
    :param init_distributed:
    :type init_distributed:
    :return:
    :rtype:
    """
    "================================================================================="
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(args.device_id)
    #     torch.cuda.empty_cache()
    #     torch.cuda.init()
    #     args.device = torch.device("cuda")

    try:
        print(f'rank: {args.rank}')
        if init_distributed:
            dist.init_process_group(
                backend=args.backend,
                init_method=args.init_method,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(seconds=30)
            )
            args.device = torch.device("cuda", args.rank)
    except Exception as e:
        print(e)

    # if args.rank != 0:
    #     print(args.rank)
    #     if init_distributed:
    #         dist.init_process_group(
    #             backend=args.backend,
    #             init_method=args.init_method,
    #             world_size=args.world_size,
    #             rank=args.rank,
    #             timeout=timedelta(seconds=60)
    #         )
    #         args.device = torch.device("cuda", args.rank)

    print(f'rank: {args.rank} after init')
    "================================================================================="
    if is_master() and args.use_wandb_logger:
        # config for the experiment:
        config_for_wandb = create_config_for_wandb(args, 'MTRES')
        # tell wandb to get started:
        wandb.init(project="tre", entity='omerc', config=config_for_wandb)
        # change the run name:
        if args.run_name is not None:
            args.run_name = args.run_name + f' {args.part_of_train_data}'
            try:
                wandb.run.name = args.run_name
                wandb.run.save()
            except:
                pass
        # update general info of the run:
        wandb.config.update(args)
    "================================================================================="
    if args.use_baseline_model:
        # create baseline model and tokenizer (after markers adition):
        model, tokenizer = \
            create_baesline_pretrained_model_and_tokenizer(args)
        model.to(args.device)
    else:
        # create our model and tokenizer (after markers adition):
        model, tokenizer = create_pretrained_model_and_tokenizer(args)
    "================================================================================="
    # if no checkpoint and not a baseline model - load the pretrained boolq model:
    if (args.checkpoint_path is None) and (not args.use_baseline_model):
        PATH = Path(args.boolq_pre_trained_model_path)
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
    "================================================================================="
    # if there is a checkpoint and not a baseline model - load it:
    if (args.checkpoint_path is not None) and (not args.use_baseline_model):
        (model, _, _, _, _, _) = \
            load_model_checkpoint(
                args, Path(args.checkpoint_path), model,
                None, None
            )
        model.to(args.device)
    "================================================================================="
    """Parallel"""
    is_distributed = args.world_size > 1
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
            )
            model.to(args.device)

            # Dataloaders:
            train_loader, train_sampler = create_dataloader(args, 'train', is_distributed)
            val_dataloader, _ = create_dataloader(args, 'val', is_distributed)
            test_loader, _ = create_dataloader(args, 'test', is_distributed)

        # if we have just 1 gpu:
        elif args.world_size == 1:
            model = model.to(args.device)

            # Dataloaders:
            train_loader, train_sampler = create_dataloader(args, 'train')
            val_dataloader, _ = create_dataloader(args, 'val')
            test_loader, _ = create_dataloader(args, 'test')
    "================================================================================="
    """Training"""
    if not args.eval:
        if args.use_baseline_model:
            train_baseline(
                model, args, train_loader,
                train_sampler, test_loader, tokenizer)
        else:
            train(
                model, args, train_loader,
                train_sampler, test_loader, tokenizer)
        # finish the session:
        if is_master() and args.use_wandb_logger:
            wandb.finish()
    "================================================================================="
    """Evaluation"""
    if args.eval:
        if args.use_baseline_model:
            eval_baseline(
                model, args, test_loader, tokenizer)
        else:
            eval(
                model, args, test_loader, tokenizer)
        # finish the session:
        if is_master() and args.use_wandb_logger:
            wandb.finish()
    "================================================================================="
    """cleanup"""
    print(f'rank: {args.rank}')
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
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
        args.local_rank = args.rank
    main(args, init_distributed=True)
if __name__ == '__main__':
    __file__ = 'main.py'
    "================================================================================="
    parser = argparse.ArgumentParser(description='TRE')
    "================================================================================="
    parser.add_argument('--device', type=torch.device,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='device type')
    "================================================================================="
    "Train settings 1"
    parser.add_argument('--eval', type=bool, default=False,
                        help='eval mode ? if False then training mode')
    parser.add_argument('--use_baseline_model', type=bool, default=False,
                        help='if True - uses baseline model, else our model')
    parser.add_argument('--use_wandb_logger', type=bool, default=False,
                        help='use wandb logger ?')
    parser.add_argument('--wandb_log_training_data', type=bool, default=False,
                        help='for correct comparsion between runs with diff size of train data')
    parser.add_argument('--run_name', type=str, default='ours',
                        help='if None then wandb random name,'
                             ' else itself + args.part_of_train_data')
    parser.add_argument('--use_E_markers', type=bool, default=False,
                        help='if True then use ([E1] word1 [/E1]) / ([E2] word2 [/E2]) markers, '
                             'else use (@ word @) markers')
    parser.add_argument('--eval_during_training', type=bool, default=True,
                        help='eval during training ?')
    parser.add_argument('--save_model_during_training', type=bool, default=False,
                        help='save model during training ? ')
    parser.add_argument('--save_table_of_results_after_eval', type=bool, default=False,
                        help='save table of results (with text) after eval ?')
    parser.add_argument('--save_model_every', type=int, default=600,
                        help='when to save the model - number of batches')
    parser.add_argument('--ignor_vague_lable_in_training', type=bool, default=True,
                        help='if True - ignors vague lable in training')
    parser.add_argument('--short_passage', type=bool, default=True,
                        help='if True then cut the passage after the first "." after second verb')
    parser.add_argument('--boolq_pre_trained_model_path', type=str,
                        default='models/pretrained_boolq_with_markers.pt',
                        help='this is a pre trained model on boolq dataset, with acc (0.82)')
    parser.add_argument('--print_loss_every', type=int, default=25,
                        help='when to print the loss - number of batches')
    parser.add_argument('--print_eval_every', type=int, default=50,
                        help='when to print f1 scores during eval - number of batches')
    parser.add_argument('--checkpoint_path', type=str,
                        default=None,
                        #'models/fluent-rain-249_epoch_3_iter_1200_.pt', #'models/fast-butterfly-49_epoch_1_iter_3184_.pt',
                        help='checkpoint path for evaluation or proceed training ,'
                             'if set to None then ignor checkpoint')
    "================================================================================="
    "Hyper-parameters"
    parser.add_argument('--world_size', type=int, default=None,
                        help='if None - will be number of devices')
    parser.add_argument('--start_rank', default=0, type=int,
                        help='we need to pass diff values if we are using multiple machines')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')  # every 2 instances are using 1 "3090 GPU"
    parser.add_argument('--part_of_train_data', type=float, default=None,
                        help='amount of train instances for training, (between 1 and 12736)')
    parser.add_argument("--parts_of_train_data", nargs="+",
                        default=[12736])
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate (default: 0.00001) took from longformer paper')
    parser.add_argument('--dropout_p', type=float, default=0.25,
                        help='dropout_p (default: 0.1)')
    parser.add_argument('--use_scheduler', type=bool, default=False,
                        help='use linear scheduler with warmup ?')
    parser.add_argument('--num_warmup_steps', type=int, default=50,
                        help='number of warmup steps in the scheduler, '
                             '(just if args.use_scheduler is True)')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta 1 for AdamW. default=0.9')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='beta 2 for AdamW. default=0.999')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay for AdamW. default=0.0001')
    parser.add_argument('--use_clip_grad_norm', type=bool, default=False,
                        help='clip grad norm to args.max_grad_norm')
    parser.add_argument('--max_grad_norm', type=float, default=40,
                        help='max norm for gradients cliping '
                             '(just if args.use_clip_grad_norm is True)')
    parser.add_argument('--sync-bn', action='store_true', default=True,
                        help='sync batchnorm')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers in dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='prefetch factor in dataloader')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    "================================================================================="
    "Model settings"
    parser.add_argument('--output_size', type=int, default=2,
                        help='output_size (default: 2)')
    parser.add_argument('--Max_Len', type=int, default=4096,
                        help='Max_Len (default: 4096)')
    parser.add_argument('--Size_of_longfor', type=str, default='base',
                        help='Size_of_longformer (default: "base")')
    "================================================================================="
    args = parser.parse_known_args()[0]
    "================================================================================="
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # os.environ['MASTER_ADDR'] = '192.168.1.102' #'127.0.0.1'#
    # os.environ['MASTER_PORT'] = '20546'
    # os.environ['GLOO_SOCKET_IFNAME'] = 'Wi-Fi'
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"  # set to DETAIL for runtime logging

    print(f'Available devices: {torch.cuda.device_count()}\n')
    "================================================================================="
    # Ensure deterministic behavior
    torch.use_deterministic_algorithms(True)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    "================================================================================="
    """Distributed:"""

    # multiple nodes:
    # args.world_size = args.gpus * args.nodes

    # single node:
    if (args.world_size is None):
        args.world_size = torch.cuda.device_count()

    # check platform:
    IsWindows = platform.platform().startswith('Win')

    for part in args.parts_of_train_data:

        args.part_of_train_data = part

        # login to W&B:
        if args.use_wandb_logger:
            wandb.login()

        # if single GPU:
        if args.world_size == 1:
            # on nvidia 3090:
            args.batch_size = 2
            args.single_rank_batch_size = 2
            args.device_id = 0
            args.rank = 0
            main(args)

        # DDP for multiple GPU'S:
        elif args.world_size > 1:

            print(f'world_size: {args.world_size}')

            args.local_world_size = torch.cuda.device_count()

            # for nvidia 3090 or titan rtx (24GB each)
            args.batch_size = args.local_world_size * 2

            args.single_rank_batch_size = int(args.batch_size / args.local_world_size)

            port = random.randint(10000, 20000)
            args.init_method = f'tcp://127.0.0.1:{port}'
            # args.init_method = f'tcp://192.168.1.102:{port}'
            # args.init_method = 'env://'

            # we will set the rank in distributed main function
            args.rank = None

            # 'nccl' is the fastest, but doesnt woek in windows.
            args.backend = 'gloo' if IsWindows else 'nccl'

            # open args.local_world_size new process in each node:
            mp.spawn(fn=distributed_main, args=(args,), nprocs=args.local_world_size,)

        else:
            args.device = torch.device("cpu")
            args.single_rank_batch_size = args.batch_size
            main(args)
    "================================================================================="
