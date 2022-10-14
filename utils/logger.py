from __future__ import absolute_import, division, print_function
import wandb
from datetime import datetime, timedelta
import time
# create config dict from args:


def create_config_for_wandb(args, dataset):
    """
    :param args:
    :type args:
    :param dataset:
    :type dataset:
    :return:
    :rtype:
    """

    config = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dropout_p=args.dropout_p,
        max_grad_norm=args.max_grad_norm,
        dataset=dataset
    )

    return config
# log traing statistics


def train_log(args, loss, epoch, batches_overall):
    """
    :param loss:
    :type loss:
    :param epoch:
    :type epoch:
    :param batches_overall:
    :type batches_overall:
    :return:
    :rtype:
    """
    if args.wandb_log_training_data:

        if args.eval_during_training:
            wandb.log({"epoch": epoch, "loss": loss})
        else:
            wandb.log({"epoch": epoch, "loss": loss}, step=batches_overall)
# log eval statistics


def eval_log(args, macro, micro, epoch):
    """
    :param args:
    :type args:
    :param macro:
    :type macro:
    :param micro:
    :type micro:
    :param batches_overall:
    :type batches_overall:
    :param epoch:
    :type epoch:
    :return:
    :rtype:
    """
    if epoch is None:
        wandb.log({"f1 macro": macro, "f1 micro": micro})
    elif not args.wandb_log_training_data:
        wandb.log({"f1 macro": macro, "f1 micro": micro}, step=epoch)
    else:
        wandb.log({"f1 macro": macro, "f1 micro": micro})
# print the training:


def print_training_progress(
        args,
        start_time,
        length_of_data_loader,
        epoch,
        batch_counter,
        total_loss):
    """
    """
    delta = timedelta(seconds=time.time() - start_time)
    print(
        f'[train Rank: {args.rank}]: '
        f'epoch: {epoch}, '
        f'loss: {total_loss:.2f}, '
        f'train time: {delta - timedelta(microseconds=delta.microseconds)}, '
        f'epoch progress: {round((batch_counter / length_of_data_loader) * 100, 2)}%')


def print_eval_results(macro, micro, batch_counter, len_test_loader):
    """
    :param macro:
    :type macro:
    :param micro:
    :type micro:
    :param batch_counter:
    :type batch_counter:
    :param len_test_loader:
    :type len_test_loader:
    :return:
    :rtype:
    """
    eval_precent = (batch_counter / len_test_loader) * 100
    print(f'f1 macro: {macro}, f1 micro: {micro}, '
          f'evaluation percent: {eval_precent:.3f}\n')


"============================================================================="
