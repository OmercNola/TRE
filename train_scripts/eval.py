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

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask)

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
            print(
                f'[eval Rank: {args.rank}] f1 macro: {macro}, f1 micro: {micro}, '
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
