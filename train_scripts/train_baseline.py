import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from utils.utils import *
from utils.saver import *
from model.model import *
from utils.logger import *
from torch import distributed as dist
from train_scripts.eval_baseline_with_qa import eval_bl_with_qa
from transformers import get_linear_schedule_with_warmup, AdamW


def train_baseline(
    model,
    args,
    train_loader,
    train_sampler,
    test_loader,
    tokenizer,
):
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

    # Tell wandb to watch what the model gets up to: gradients, weights, and
    # more!
    if is_master() and args.use_wandb_logger:
        wandb.watch(
            model,
            criterion,
            log="all",
            log_freq=50,
            log_graph=(False))

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
        epoch_itrator = tqdm(
            range(
                epoch_start,
                args.epochs + 1,
                1),
            position=0,
            leave=True)
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
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()

            # forward pass:
            try:
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask)
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
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

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
                        train_log(
                            args, total_loss_for_print, epoch, batches_overall)

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
                    eval(
                        model,
                        args,
                        test_loader,
                        tokenizer,
                        batches_overall=batches_overall)

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
            eval_baseline(model, args, test_loader, tokenizer,
                          batches_overall=batches_overall)
