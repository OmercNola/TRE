from datasets_and_loaders.data import \
    (TRE_train_dataset,
     TRE_val_dataset,
     TRE_test_dataset)
from torch.utils.data import DataLoader
def create_dataloader(args, train_val_test):
    """
    :param args:
    :type args:
    :param train_val_test:
    :type train_val_test:
    :return:
    :rtype:
    """
    if train_val_test == 'train':

        dataloader = DataLoader(
            TRE_train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

    elif train_val_test == 'val':

        dataloader = DataLoader(
            TRE_val_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

    elif train_val_test == 'test':

        dataloader = DataLoader(
            TRE_test_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

    return dataloader