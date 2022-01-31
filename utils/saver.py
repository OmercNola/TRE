import wandb
import torch
from pathlib import Path
# save checkpoit:
def save_model_checkpoint(
        args, model, optimizer, scheduler,
        length_of_data_loader, batch_counter,
        epoch, loss):
    """
    """
    epoch_percent = round((batch_counter / length_of_data_loader) * 100, 2)
    PATH = Path(f"models/{wandb.run.name}_epoch_{epoch}_iter_{batch_counter}_.pt")

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

    print('loading checkpoint..')

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