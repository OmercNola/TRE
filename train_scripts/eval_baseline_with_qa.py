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


def eval_bl_with_qa(model, args, test_loader, tokenizer, epoch=None):
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
    # see this post:
    # https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522
    #if hasattr(model, "module"):
    #    model = model.module

    # evaluation mode:
    model.eval()

    # check if Distributed mode:
    is_distributed = args.world_size > 1

    # create Tracker:
    tracker = baseline_results_tracker()

    # reset Tracker:
    tracker.reset()
    temp_counter = 0
    for batch_counter, instances in enumerate(test_loader, start=1):
        
        print(f'batch_counter: {batch_counter}')
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        passages = instances[0]
        first_words = instances[1][0]
        second_words = instances[1][1]
        word_labels = instances[1][2]

        zip_object = zip(passages, first_words, second_words, word_labels)
        for passage, first_word, second_word, Label in zip_object:

            question = question_baseline(args,
                                         first_word,
                                         second_word)
            question += tokenizer.sep_token

            label = get_label_for_baseline(Label)

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
            batch_labels.append(Label.strip())

        batch_input_ids = torch.tensor(
            batch_input_ids, requires_grad=False, device=args.device)
        batch_attention_mask = torch.tensor(
            batch_attention_mask, requires_grad=False, device=args.device)

        # forward pass:
        with torch.no_grad():

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask)

            # prediction:
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            # move to cpu and numpy:
            preds = preds.clone().detach().cpu().numpy()

            for pred, label in zip(preds, batch_labels):
                tracker.update(pred, label)
                temp_counter += 1
                print(f'temp_counter: {temp_counter}')

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

    # print results and log to wandb:
    if is_master():

        if args.use_wandb_logger:
            eval_log(args, macro, micro, epoch)

        print_eval_results(macro, micro, batch_counter, len(test_loader))

        if args.use_wandb_logger and args.save_table_of_results_after_eval:
            wandb.log({f'results table {wandb.run.name}': table})

    return macro, micro
