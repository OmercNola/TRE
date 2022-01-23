from logger import (train_log, save_model_checkpoint,
                    load_model_checkpoint, print_training_progress)
from utils import (question_1_for_markers, question_2_for_markers,
                   get_label, results_tracker)
from eval import eval_tre_new_questions_with_markers
from torch import nn
import torch
from transformers import get_linear_schedule_with_warmup, AdamW
import time
from datetime import timedelta, datetime
import random
from pathlib import Path
import wandb
from tqdm import tqdm
"============================================================================="
# train:
def train_tre_new_questions_with_markers(
        model, args, train_dataloader, test_dataloader,
        tokenizer, checkpoint_path=None):
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

    print('training tre with markers...')

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create the learning rate scheduler.
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # loss progress counters
    total_loss_for_print = 0
    total_loss_for_save = 0

    # set epoch_start to 1, it we have checkpoint, will load it from there.
    epoch_start = 1

    # if there is a checkpoint, load it:
    if checkpoint_path is not None:
        (model, optimizer, scheduler, epoch_start, _, _) = \
            load_model_checkpoint(
                checkpoint_path, model,
                optimizer, scheduler
            )

    # training mode:
    model.train()

    # start time:
    t0 = time.time()

    # total nuber of batches counter:
    batches_overall = 0

    for epoch in tqdm(range(epoch_start, args.epochs+1, 1)):

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

                # ignor vague, like other papers do:
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

                        # calculate loss
                        loss.backward()

                        # This is to help prevent the "exploding gradients" problem:
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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

        # evaluate at the end of the epoch:
        if args.eval_during_training:
            tracker = results_tracker()
            eval_tre_new_questions_with_markers(
                model, args, test_dataloader,
                tokenizer, tracker, checkpoint_path=None,
                batches_overall=batches_overall
            )

        # save at the end of the epoch:
        if args.save_model_during_training:
            save_model_checkpoint(
                args, model, optimizer,
                scheduler, len(train_dataloader),
                batch_counter, epoch,
                total_loss_for_save
            )
            total_loss_for_save = 0