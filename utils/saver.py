from __future__ import absolute_import, division, print_function
import wandb
import torch
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
# save checkpoit:
def save_model_checkpoint(
        args, model, optimizer, scheduler,
        length_of_data_loader, batch_counter,
        epoch, loss):
    """
    """
    epoch_percent = round((batch_counter / length_of_data_loader) * 100, 2)
    PATH = Path(f"models/{wandb.run.name}_epoch_{epoch}_iter_{batch_counter}_.pt")

    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None

    torch.save({
            'epoch': epoch,
            'epoch percent': epoch_percent,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'loss': loss / args.save_model_every
        }, PATH)

    print(f'checkpoint has been saved !')
    print(f'Epoch percent: {epoch_percent}')
# load checkpoint:
def load_model_checkpoint(args, path_, model, optimizer=None, scheduler=None):
    """
    """

    print('loading checkpoint..')

    # load the checkpoint:
    checkpoint = torch.load(path_, map_location=f'cuda:{args.rank}')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(e)

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    epoch_percent = checkpoint['epoch percent']

    return model, optimizer, scheduler, epoch, loss, epoch_percent