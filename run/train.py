import argparse
import logging
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from configs.diffusion import *
from configs.unet import NestedUNetConfig
from datasets.flickr30k.dataset import FlickrDataset
from helpers.collator import DiffusionCollator
from helpers.lm import create_lm
from models.diffusion import NestedDiffusion
from models.nested_unet import NestedUNet


def get_dataset():
    IMAGE_SIZE = (64, 64)
    transforms = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    root = "./data/flickr30k/Images"
    caps = "./data/flickr30k/captions.txt"
    return FlickrDataset(root, caps, transforms=transforms)

def get_dataloader(batch_size: int, tokenizer):

    collator = DiffusionCollator(tokenizer)

    train_dataset = get_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,  # Избегает повторной инициализации workers
        prefetch_factor=2,  # Предзагрузка данных для следующих батчей
        collate_fn=collator,
    )

    return train_loader



def get_model(unet_config: NestedUNetConfig, diffusion_config: DiffusionConfig, device):
    unet_model = NestedUNet(
        input_channels=3,
        output_channels=3,
        config=unet_config,
    ).to(device)
    return NestedDiffusion(vision_model=unet_model, config=diffusion_config).to(device)


def train_batch(
    model: NestedDiffusion, sample: dict, optimizer, scheduler, args, logger=None, grad_scaler=None
):
    model.train()
    lr = scheduler.get_last_lr()[0]

    if args.fp16 and grad_scaler is not None:
        with torch.cuda.autocast():
            total_loss, times, x_t_multi_res, predictions = model.get_loss(sample)
            loss = total_loss.mean()
        loss_val = loss.item()
        if np.isnan(loss_val):
            optimizer.zero_grad()
            return loss_val, total_loss, times, x_t_multi_res, predictions
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        total_loss, times, x_t_multi_res, predictions = model.get_loss(sample)
        loss = total_loss.mean()
        loss_val = loss.item()
        if np.isnan(loss_val):
            optimizer.zero_grad()
            return loss_val, total_loss, times, x_t_multi_res, predictions
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad()

    if logger is not None:
        logger.add_scalar("train/Loss", loss_val, global_step=logger.batch_num)
        logger.add_scalar("train/LR", lr, global_step=logger.batch_num)
        if getattr(model.config, "use_double_loss", False):
            for i, weight in enumerate(model.config.multi_res_weights):
                resolution_loss = (total_loss[i] / weight).mean().item()
                logger.add_scalar(
                    f"train/Loss_resolution_{i}",
                    resolution_loss,
                    global_step=logger.batch_num,
                )

    return loss_val, total_loss, times, x_t_multi_res, predictions


def train(
    device,
    args,
    unet_config: NestedUNetConfig,
    diffusion_config: DiffusionConfig,
    logger=None,
):
    logging.info(f"Using device: {device}")

    logging.info("Getting LM")
    tokenizer, encoder = create_lm()
    logging.info("Getting Dataloader")
    train_loader = get_dataloader(args.batch_size, tokenizer)
    
    best_loss = float("inf")
    exp_avg_loss = 0
    batch_num = 0
    
    diffusion_model = get_model(unet_config, diffusion_config, device)
    
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    grad_scaler = torch.amp.GradScaler() if args.fp16 else None

    logging.info("Starting training...")
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        for i, sample in enumerate(train_loader):
            batch_num += 1
            start_time = time.time()

            sample["lm_outputs"] = encoder(sample["lm_outputs"])

            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.to(device)

            loss_val, losses, times, x_t, predictions = train_batch(
                diffusion_model,
                sample,
                optimizer,
                scheduler,
                args,
                logger,
                grad_scaler=grad_scaler,
            )

            exp_avg_loss = (
                loss_val if batch_num == 1 else 0.99 * exp_avg_loss + 0.01 * loss_val
            )

            elapsed = time.time() - start_time
            if batch_num % args.log_freq == 0:
                logging.info(
                    f"Batch {batch_num} | Loss: {loss_val:.4f} | Avg Loss: {exp_avg_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s"
                )

            # if batch_num % args.save_freq == 0:
            #     model_path = os.path.join(args.output_dir, f"model_{batch_num:06d}.pt")
            #     logging.info(f"Saving model to {model_path}")
            #     torch.save(
            #         {
            #             "batch_num": batch_num,
            #             "model_state_dict": diffusion_model.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "scheduler_state_dict": scheduler.state_dict(),
            #             "loss": loss_val,
            #             "exp_avg_loss": exp_avg_loss,
            #         },
            #         model_path,
            #     )
            #     if exp_avg_loss < best_loss:
            #         best_loss = exp_avg_loss
            #         best_model_path = os.path.join(args.output_dir, "best_model.pt")
            #         torch.save(
            #             {
            #                 "batch_num": batch_num,
            #                 "model_state_dict": diffusion_model.state_dict(),
            #                 "optimizer_state_dict": optimizer.state_dict(),
            #                 "loss": loss_val,
            #                 "exp_avg_loss": exp_avg_loss,
            #             },
            #             best_model_path,
            #         )
