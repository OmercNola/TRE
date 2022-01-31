from __future__ import absolute_import, division, print_function
from datasets_and_loaders.data import \
    (TRE_train_dataset,
     TRE_val_dataset,
     TRE_test_dataset)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
"=================================================================="
def create_dataloader(args, train_val_test, ddp=False):
    """
    :param args:
    :type args:
    :param train_val_test:
    :type train_val_test:
    :return:
    :rtype:
    """

    if train_val_test == 'train':

        train_dataset = TRE_train_dataset()

        if ddp:

            train_sampler = DistributedSampler(
                train_dataset,
                shuffle=args.shuffle,
            )

            dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=train_sampler,
                pin_memory=True
            )

        # not ddp:
        else:
            dataloader = DataLoader(
                train_dataset,
                shuffle=args.shuffle,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True
            )

    elif train_val_test == 'val':

        val_dataset = TRE_val_dataset()

        if ddp:

            val_sampler = DistributedSampler(
                val_dataset,
                shuffle=args.shuffle,
            )

            dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=val_sampler,
                pin_memory=True
            )

        # not ddp:
        else:
            dataloader = DataLoader(
                val_dataset,
                shuffle=args.shuffle,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True
            )

    elif train_val_test == 'test':

        test_dataset = TRE_test_dataset()

        if ddp:

            test_sampler = DistributedSampler(
                test_dataset,
                shuffle=args.shuffle,
            )

            dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sampler=test_sampler,
                pin_memory=True
            )

        # not ddp:
        else:
            dataloader = DataLoader(
                test_dataset,
                shuffle=args.shuffle,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True
            )

    return dataloader