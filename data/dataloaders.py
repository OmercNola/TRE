from __future__ import absolute_import, division, print_function
from data.datasets import \
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

        train_dataset = TRE_train_dataset(args)

        if ddp:

            train_sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=args.shuffle,
            )

            dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                prefetch_factor=args.prefetch_factor,
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
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    elif train_val_test == 'val':

        val_dataset = TRE_val_dataset(args)

        if ddp:

            val_sampler = DistributedSampler(
                dataset=val_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )

            dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
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
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    elif train_val_test == 'test':

        test_dataset = TRE_test_dataset(args)

        if ddp:

            test_sampler = DistributedSampler(
                test_dataset,
                shuffle=False,
            )

            dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
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
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    return dataloader