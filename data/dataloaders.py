from __future__ import absolute_import, division, print_function
from data.datasets import \
    (TRE_train_dataset,
     TRE_val_dataset,
     TRE_test_dataset)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.sampler import DistributedEvalSampler
import numpy as np
import random
import torch
"=================================================================="
def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32
    np.random.seed(1)
    random.seed(1)
def create_dataloader(args, train_val_test, is_distributed=False):
    """
    :param args:
    :type args:
    :param train_val_test:
    :type train_val_test:
    :return:
    :rtype:
    """

    sampler = None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if train_val_test == 'train':

        train_dataset = TRE_train_dataset(args)

        if is_distributed:

            sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=args.shuffle,
                seed=args.seed
            )

            g = torch.Generator()
            g.manual_seed(1)

            dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                prefetch_factor=args.prefetch_factor,
                sampler=sampler,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g
            )

        # not distributed:
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

        if is_distributed:

            sampler = DistributedEvalSampler(
                dataset=val_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
                seed=args.seed
            )

            dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                sampler=sampler,
                pin_memory=True
            )

        # not distributed:
        else:
            dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    elif train_val_test == 'test':

        test_dataset = TRE_test_dataset(args)

        if is_distributed:

            sampler = DistributedEvalSampler(
                test_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
                seed=args.seed
            )

            dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.single_rank_batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                sampler=sampler,
                pin_memory=True
            )

        # not distributed:
        else:
            dataloader = DataLoader(
                test_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.batch_size,
                prefetch_factor=args.prefetch_factor,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=True
            )

    return dataloader, sampler