import wandb
import torch
from datetime import datetime, timedelta
import time
from pathlib import Path
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
        learning_rate=args.lr,
        seed=args.seed,
        dropout_p=args.dropout_p,
        max_grad_norm=args.max_grad_norm,
        dataset=dataset
    )

    return config
# save traing statistics
def train_log(loss, epoch, batch_counter, batches_overall):
    """
    :param loss:
    :type loss:
    :param example_ct:
    :type example_ct:
    :param epoch:
    :type epoch:
    :return:
    :rtype:
    """
    wandb.log(
        {"epoch": epoch, "loss": loss,
        "batch_counter": batch_counter},
        step=batches_overall
    )
# print the training:
def print_training_progress(
        start_time, length_of_data_loader, epoch, batch_counter, total_loss):
    """
    """
    print(f'Epoch:{epoch}, '
          f'loss:{total_loss:.2f}, '
          f'Training time:{timedelta(seconds=time.time() - start_time)}, '
          f'Epoch percent: {round((batch_counter / length_of_data_loader) * 100, 2)}')
# save checkpoit:
def save_model_checkpoint(
        args, model, optimizer, scheduler,
        length_of_data_loader, batch_counter,
        epoch, loss):
    """
    """
    epoch_percent = round((batch_counter / length_of_data_loader) * 100, 2)
    PATH = Path(f"models/model_epoch_{epoch}_iter_{batch_counter}_.pt")

    torch.save({
        'epoch': epoch,
        'epoch percent': epoch_percent,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss / args.save_model_every
    }, PATH)

    print(f'checkpoint has been saved !')
    print(f'Epoch percent: {epoch_percent}')
# load checkpoint:
def load_model_checkpoint(path_, model, optimizer=None, scheduler=None):
    """
    """
    # load the checkpoint:
    checkpoint = torch.load(path_)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    loss = checkpoint['loss']

    epoch_percent = checkpoint['epoch percent']

    return model, optimizer, scheduler, epoch, loss, epoch_percent
"============================================================================="