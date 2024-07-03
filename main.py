import os
import wandb
import time
import random
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from pathlib import Path
import platform
from utils.utils import *
from utils.saver import *
from model.model import *
from utils.logger import *
from train_scripts.train import train
from train_scripts.eval import eval
from train_scripts.train_baseline import train_baseline
from train_scripts.eval_baseline import eval_baseline
from train_scripts.train_baseline_with_qa import train_bl_with_qa
from train_scripts.eval_baseline_with_qa import eval_bl_with_qa
import torch.multiprocessing as mp
from torch import distributed as dist
from data.dataloaders import create_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup, AdamW
from ipdb import set_trace


def main(args, init_distributed=False):
    """
    :param args:
    :type args:
    :param init_distributed:
    :type init_distributed:
    :return:
    :rtype:
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.empty_cache()
        torch.cuda.init()
        args.device = torch.device("cuda", args.device_id)

    try:
        print(f'rank: {args.rank}')
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(seconds=120)
        )
    except Exception as e:
        print(e)

    print(f'rank: {args.rank} after init')

    if is_master() and args.use_wandb_logger:
        # config for the experiment:
        config_for_wandb = create_config_for_wandb(args, 'MTRES')
        # tell wandb to get started:
        wandb.init(project="tre", entity='omerc', config=config_for_wandb)
        # change the run name:
        if args.run_name is not None:
            args.run_name = args.run_name + f' {args.part_of_train_data}'
            try:
                wandb.run.name = args.run_name
                wandb.run.save()
            except BaseException:
                pass
        # update general info of the run:
        wandb.config.update(args)

    if args.use_baseline_model:
        # create baseline model and tokenizer (after markers adition):
        model, tokenizer = \
            create_baesline_pretrained_model_and_tokenizer(args)
        model.to(args.device)
    else:
        # create our model and tokenizer (after markers adition):
        model, tokenizer = create_roberta_pretrained_model_and_tokenizer(args)
        model.to(args.device)

    # if no checkpoint and not a baseline model - load the pretrained boolq
    # model:
    # if (args.checkpoint_path is None) and (not args.use_baseline_model):
    #    try:
    #        PATH = Path(args.boolq_pre_trained_model_path)
    #        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    #        model.load_state_dict(checkpoint['model_state_dict'])
    #        model.to(args.device)
    #    except Exception as e:
    #        print(e)

    # if there is a checkpoint and not a baseline model - load it:
    if (args.checkpoint_path is not None) and (not args.use_baseline_model):
        try:
            (model, _, _, _, _, _) = \
                load_model_checkpoint(
                    args, Path(args.checkpoint_path), model,
                    None, None
            )
            model.to(args.device)
        except Exception as e:
            print(e)
    print(f'rank: {args.rank}')

    """Parallel"""
    is_distributed = args.world_size > 1
    if torch.cuda.is_available():
        # if we have more than 1 gpu:
        if args.world_size > 1:

            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = DDP(
                model,
                device_ids=[args.device_id],
                output_device=args.device_id,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
            model.to(args.device)

            # Dataloaders:
            train_loader, train_sampler = \
                    create_dataloader(args, 'train', is_distributed, dataset=args.dataset)
            val_dataloader, _ = create_dataloader(args, 'val', is_distributed, dataset=args.dataset)
            test_loader, _ = create_dataloader(args, 'test', is_distributed, dataset=args.dataset)

        # if we have just 1 gpu:
        elif args.world_size == 1:
            model = model.to(args.device)

            # Dataloaders:
            train_loader, train_sampler = create_dataloader(args, 'train', dataset=args.dataset)
            val_dataloader, _ = create_dataloader(args, 'val', dataset=args.dataset)
            test_loader, _ = create_dataloader(args, 'test', dataset=args.dataset)

    else:
        # Dataloaders:
        train_loader, train_sampler = create_dataloader(args, 'train', dataset=args.dataset)
        val_dataloader, _ = create_dataloader(args, 'val', dataset=args.dataset)
        test_loader, _ = create_dataloader(args, 'test', dataset=args.dataset)

    set_trace()

    """Training"""
    if not args.eval:
        if args.use_baseline_model:
            train_baseline(
                model, args, train_loader,
                train_sampler, test_loader, tokenizer)
        else:
            train(
                model, args, train_loader,
                train_sampler, test_loader, tokenizer)
        # finish the session:
        if is_master() and args.use_wandb_logger:
            wandb.finish()

    """Evaluation"""
    if args.eval:
        if args.use_baseline_model:
            eval_baseline(
                model, args, test_loader, tokenizer)
        else:
            eval(
                model, args, test_loader, tokenizer)
        # finish the session:
        if is_master() and args.use_wandb_logger:
            wandb.finish()

    """cleanup"""
    print(f'rank: {args.rank}')
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def distributed_main(device_id, args):
    """
    :param device_id:
    :type device_id:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    args.device_id = device_id
    if args.rank is None:
        args.rank = args.start_rank + device_id
        args.local_rank = args.rank
    main(args, init_distributed=True)


if __name__ == '__main__':
    __file__ = 'main.py'
    parser = argparse.ArgumentParser(description='TRE')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'),
        help='device type')

    "Train settings 1"
    parser.add_argument('--dataset', type=str, default='tb_dense',
                        help='dataset to use, can be matres or tb_dense')
    parser.add_argument('--eval', type=bool, default=False,
                        help='eval mode ? if False then training mode')
    parser.add_argument('--use_baseline_model', type=bool, default=False,
                        help='if True - uses baseline model, else our model')
    parser.add_argument('--use_wandb_logger', type=bool, default=False,
                        help='use wandb logger ?')
    parser.add_argument(
        '--save_table_of_results_after_eval',
        type=bool,
        default=True,
        help='save table of results (with text) after eval ?')
    parser.add_argument(
        '--wandb_log_training_data',
        type=bool,
        default=False,
        help='for correct comparsion between runs with diff size of train data')
    parser.add_argument('--run_name', type=str, default='ours',
                        help='if None then wandb random name,'
                             ' else itself + args.part_of_train_data')
    parser.add_argument(
        '--use_E_markers',
        type=bool,
        default=False,
        help='if True then use ([E1] word1 [/E1]) / ([E2] word2 [/E2]) markers, '
        'else use (@ word @) markers')
    parser.add_argument('--eval_during_training', type=bool, default=True,
                        help='eval during training ?')
    parser.add_argument(
        '--save_model_during_training',
        type=bool,
        default=False,
        help='save model during training ? ')
    parser.add_argument('--save_model_every', type=int, default=600,
                        help='when to save the model - number of batches')
    parser.add_argument(
        '--ignor_vague_lable_in_training',
        type=bool,
        default=True,
        help='if True - ignors vague lable in training')
    parser.add_argument(
        '--short_passage',
        type=bool,
        default=True,
        help='if True then cut the passage after the first "." after second verb')
    parser.add_argument(
        '--boolq_pre_trained_model_path',
        type=str,
        default='models/pretrained_boolq_with_markers.pt',
        help='this is a pre trained model on boolq dataset, with acc (0.82)')
    parser.add_argument('--print_loss_every', type=int, default=25,
                        help='when to print the loss - number of batches')
    parser.add_argument(
        '--print_eval_every',
        type=int,
        default=50,
        help='when to print f1 scores during eval - number of batches')
    parser.add_argument('--checkpoint_path', type=str,
                        default=None,
                        help='checkpoint path for evaluation or proceed training ,'
                             'if set to None then ignor checkpoint')

    "Hyper-parameters"
    parser.add_argument('--world_size', type=int, default=None,
                        help='if None - will be number of devices')
    parser.add_argument(
        '--start_rank',
        default=0,
        type=int,
        help='we need to pass diff values if we are using multiple machines')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--epochs', type=int, default=6,
                        help='number of epochs')
    # every 2 instances are using 1 "3090 GPU"
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument(
        '--data_augmentation',
        action='store_true',
        default=True)
    parser.add_argument(
        '--part_of_train_data',
        type=float,
        default=None,
        help='amount of train instances for training, (between 1 and 12736)')
    parser.add_argument("--parts_of_train_data", nargs="+",
                        default=[12736])
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00001,
        help='learning rate (default: 0.00001) took from longformer paper')
    parser.add_argument('--dropout_p', type=float, default=0.25,
                        help='dropout_p (default: 0.1)')
    parser.add_argument('--use_scheduler', type=bool, default=False,
                        help='use linear scheduler with warmup ?')
    parser.add_argument('--num_warmup_steps', type=int, default=50,
                        help='number of warmup steps in the scheduler, '
                             '(just if args.use_scheduler is True)')
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='beta 1 for AdamW. default=0.9')
    parser.add_argument('--beta_2', type=float, default=0.999,
                        help='beta 2 for AdamW. default=0.999')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay for AdamW. default=0.0001')
    parser.add_argument('--use_clip_grad_norm', type=bool, default=False,
                        help='clip grad norm to args.max_grad_norm')
    parser.add_argument('--max_grad_norm', type=float, default=40,
                        help='max norm for gradients cliping '
                             '(just if args.use_clip_grad_norm is True)')
    parser.add_argument('--sync-bn', action='store_true', default=True,
                        help='sync batchnorm')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='number of workers in dataloader')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='prefetch factor in dataloader')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    "Model settings"
    parser.add_argument('--output_size', type=int, default=2,
                        help='output_size (default: 2)')
    parser.add_argument('--Max_Len', type=int, default=514,
                        help='Max_Len (default: 514)')
    parser.add_argument('--model_size', type=str, default='base',
                        help='Size_of_longformer (default: "base")')

    args = parser.parse_known_args()[0]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # os.environ['MASTER_ADDR'] = '192.168.1.101' #'127.0.0.1'#
    # os.environ['MASTER_PORT'] = '20546'
    # os.environ['GLOO_SOCKET_IFNAME'] = 'Wi-Fi'
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"  # set to DETAIL for runtime logging

    print(f'Available devices: {torch.cuda.device_count()}\n')

    # Ensure deterministic behavior
    torch.use_deterministic_algorithms(True)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    """Distributed:"""
    # multiple nodes:
    # args.world_size = args.gpus * args.nodes

    # single node:
    if (args.world_size is None):
        args.world_size = torch.cuda.device_count()

    # check platform:
    IsWindows = platform.platform().startswith('Win')

    for part in args.parts_of_train_data:

        args.part_of_train_data = part

        # login to W&B:
        if args.use_wandb_logger:
            wandb.login()

        # if single GPU:
        if args.world_size == 1:
            # on nvidia 3090:
            args.batch_size = 2
            args.single_rank_batch_size = 2
            args.device_id = 0
            args.rank = 0
            port = random.randint(10000, 20000)
            args.init_method = f'tcp://127.0.0.1:{port}'

            # 'nccl' is the fastest, but doesnt woek in windows.
            args.backend = 'gloo' if IsWindows else 'nccl'

            main(args)

        # DDP for multiple GPU'S:
        elif args.world_size > 1:

            print(f'world_size: {args.world_size}')

            args.local_world_size = torch.cuda.device_count()

            # for nvidia 3090 or titan rtx (24GB each)
            args.batch_size = args.local_world_size * 22 

            args.single_rank_batch_size = int(
                args.batch_size / args.local_world_size)

            port = random.randint(10000, 20000)
            args.init_method = f'tcp://127.0.0.1:{port}'
            # args.init_method = f'tcp://192.168.1.101:{port}'
            # args.init_method = 'env://'

            # we will set the rank in distributed main function
            args.rank = None

            # 'nccl' is the fastest, but doesnt woek in windows.
            args.backend = 'gloo' if IsWindows else 'nccl'

            # open args.local_world_size new process in each node:
            mp.spawn(
                fn=distributed_main, args=(
                    args,), nprocs=args.local_world_size,)

        else:
            args.device = torch.device("cpu")
            args.single_rank_batch_size = args.batch_size
            args.rank = 0
            main(args)
