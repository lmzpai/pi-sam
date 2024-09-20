import os,sys
import argparse
import random
import torch
import numpy as np

from segment_anything.utils import dist
from segment_anything.utils.logger import setup_logger
from segment_anything.runner import Runner


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-Itr-SAM', add_help=False)

    parser.add_argument("--output", type=str, default='work_dir', 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--decoder_type", type=str, default="pi", 
                        help="The type of mask decoder to load, in ['ori', 'hq', 'pi']")
    parser.add_argument("--ckpt_root", default=None, 
                        help="The checkpoint of the original sam mask decoder")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=200, type=int)
    parser.add_argument('--max_epoch_num', default=100, type=int)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--log_print_fre', default=20, type=int)
    parser.add_argument('--train_used_dataset', default=None)
    parser.add_argument('--valid_used_dataset', default=None)
    parser.add_argument('--test_used_dataset', default=None)

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--no_validate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore_model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(args):
    dist.init_distributed_mode()
    torch.cuda.set_device(dist.get_local_rank())
    torch.cuda.empty_cache()

    rank = dist.get_rank()
    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output, 'info.txt'), distributed_rank=rank, color=False, name="detr")
    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('rank: {}'.format(rank))
    logger.info('local rank: {}'.format(dist.get_local_rank()))

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    runner = Runner(logger, args)
    if runner.training:
        runner.train()
    else:
        runner.test()





if __name__ == "__main__":
    args = get_args_parser()
    train_used_dataset = set(["ALL"])
    valid_used_dataset = set(["DIS5K-VD"])
    test_used_dataset = set(["ALL"])
    args.train_used_dataset = train_used_dataset
    args.valid_used_dataset = valid_used_dataset
    args.test_used_dataset = test_used_dataset
    sam_ckpts = dict(vit_b = 'sam_vit_b_01ec64.pth',
                     vit_l = 'sam_vit_l_0b3195.pth',
                     vit_h = 'sam_vit_h_4b8939.pth')
    mask_dec_ckpts = dict(vit_b = 'sam_vit_b_maskdecoder.pth',
                          vit_l = 'sam_vit_l_maskdecoder.pth',
                          vit_h = 'sam_vit_h_maskdecoder.pth')
    args.sam_ckpt = os.path.join(args.ckpt_root, sam_ckpts[args.model_type])
    args.decoder_ckpt = os.path.join(args.ckpt_root, mask_dec_ckpts[args.model_type])
    main(args)
    