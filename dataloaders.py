from data import TRE_training_data_with_markers, TRE_test_data_with_markers
from data import TRE_validation_data_with_markers
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
            TRE_training_data_with_markers,
            batch_size=args.batch_size,
            shuffle=True
        )

    elif train_val_test == 'val':

        dataloader = DataLoader(
            TRE_validation_data_with_markers,
            batch_size=args.batch_size,
            shuffle=True
        )

    elif train_val_test == 'test':

        dataloader = DataLoader(
            TRE_test_data_with_markers,
            batch_size=args.batch_size,
            shuffle=True
        )

    return dataloader