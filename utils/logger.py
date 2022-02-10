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
# save traing statistics
def train_log(loss, epoch, batches_overall):
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
    wandb.log({"epoch": epoch, "loss": loss}, step=batches_overall)
# print the training:
def print_training_progress(
        args, start_time, length_of_data_loader, epoch, batch_counter, total_loss):
    """
    """
    delta = timedelta(seconds=time.time() - start_time)
    print(f'[train Rank: {args.rank}]: '
          f'epoch: {epoch}, '
          f'loss: {total_loss:.2f}, '
          f'train time: {delta - timedelta(microseconds=delta.microseconds)}, '
          f'epoch progress: {round((batch_counter / length_of_data_loader) * 100, 2)}%')
"============================================================================="