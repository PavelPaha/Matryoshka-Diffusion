import argparse
import logging
import os

from torch.utils.tensorboard import SummaryWriter

from run.configs.conf_64_64 import nested_unet_config, diffusion_config

from run.train import train

import torch

from argparse import Namespace


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleNestedDiffusion model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument("--gradient_clip_norm", type=float, default=2.0)
    return parser.parse_args()

def get_test_args():

    args = Namespace()
    
    # Основные параметры обучения
    args.batch_size = 32
    args.epochs = 5
    args.fp16 = False
    args.gradient_clip_norm = 0.5
    args.lr = 1.0e-03

    args.num_levels = 2
    
    args.output_dir = "./output"
    args.log_freq = 5
    args.save_freq = 1000
    
    # args.loss_factor = 1.0
    args.num_gradient_accumulations = 4
    
    return args


def get_logger(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    return logger


def main():
    args = get_test_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    train(
        torch.device("cuda:1"),
        args,
        nested_unet_config,
        diffusion_config,
        logger=get_logger(args)
    )


# if __name__ == "__main__":
#     main()

main()