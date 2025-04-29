import logging
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from configs.diffusion import *
from datasets.flickr30k.dataset import FlickrDataset
from helpers.collator import DiffusionCollator
from helpers.lm import create_lm
from helpers.lr_scaler import LRScaler
from models.diffusion import NestedDiffusion


def get_dataset():
    IMAGE_SIZE = (64, 64)
    transforms = T.Compose(
        [
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    root = "./data/flickr30k/Images"
    caps = "./data/flickr30k/captions.txt"
    return FlickrDataset(root, caps, transforms=transforms)


def get_dataloader(batch_size: int, tokenizer):

    collator = DiffusionCollator(tokenizer, max_cap_length=128)

    train_dataset = get_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collator,
    )

    return train_loader



def train_batch(
    model: NestedDiffusion,
    sample: dict,
    optimizer,
    scheduler,
    args,
    logger=None,
    accumulate_gradient: bool = False,
    num_grad_accumulations: int = 1,
):
    model.train()
    lr = scheduler.get_last_lr()[0]

    total_loss, times, x_t_multi_res, predictions = model.get_loss(sample)
    loss = total_loss.mean()
    
    if num_grad_accumulations != 1:
        loss = loss / num_grad_accumulations
    
    loss_val = loss.item() * (num_grad_accumulations if num_grad_accumulations != 1 else 1)  # Scale back for logging
    
    if np.isnan(loss_val):
        optimizer.zero_grad()
        return loss_val, total_loss, times, x_t_multi_res, predictions
    
    loss.backward()
    
    if not accumulate_gradient:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if logger is not None and not accumulate_gradient:
        logger.add_scalar("train/Loss", loss_val, global_step=logger.step)
        logger.add_scalar("train/LR", lr, global_step=logger.step)

    return loss_val, total_loss, times, x_t_multi_res, predictions


def train_loop(
    device,
    args,
    diffusion_model: NestedDiffusion,
    logger=None,
):
    logging.info(f"Using device: {device}")

    logging.info("Getting LM")
    tokenizer, encoder = create_lm()
    logging.info("Getting Dataloader")
    train_loader = get_dataloader(args.batch_size, tokenizer)

    best_loss = float("inf")
    exp_avg_loss = 0
    exp_avg_loss_var = 0
    step = 0
    accumulation_counter = 0
    total_loss_val = 0
    wt = 0.01
    CLIP = 3

    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(), lr=args.lr, weight_decay=0.05
    )
    lr_scaler = LRScaler()
    scheduler = lr_scaler.get_lr_scheduler(args.warmup_steps, optimizer)

    logging.info(f"Starting training with {args.warmup_steps} warmup steps")
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(train_loader):
            # Handle gradient accumulation counter
            accumulation_counter = (accumulation_counter + 1) % args.num_gradient_accumulations
            is_last_in_loader = (i == len(train_loader) - 1)
            
            # Only increment step when we're doing an optimizer step
            if accumulation_counter == 0 or is_last_in_loader:
                step += 1
            
            accumulate_gradient = (accumulation_counter != 0 and not is_last_in_loader)

            if logger:
                logger.step = step
            start_time = time.time()

            batch["lm_outputs"] = encoder(batch["lm_outputs"])
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            loss_val, losses, times, x_t, predictions = train_batch(
                model=diffusion_model,
                sample=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                logger=logger,
                accumulate_gradient=accumulate_gradient,
                num_grad_accumulations=args.num_gradient_accumulations,
            )

            if np.isnan(loss_val):
                continue

            if step != 1:
                # E[(x-E[x])^2] = E[x^2] - E[x]^2
                std_loss = np.sqrt(max(1e-8, exp_avg_loss_var))  # Avoid sqrt of negative numbers
                delta_loss = loss_val - exp_avg_loss
                clipped_loss = exp_avg_loss + std_loss * CLIP * np.tanh(
                    delta_loss / (std_loss * CLIP + 1e-8)  # Avoid division by zero
                )
                exp_avg_loss = exp_avg_loss * (1.0 - wt) + wt * clipped_loss
                exp_avg_loss_var = (
                    exp_avg_loss_var * (1.0 - wt) + wt * (clipped_loss - exp_avg_loss) ** 2
                )
            else:
                std_loss = loss_val
                exp_avg_loss = loss_val
                exp_avg_loss_var = loss_val**2
            
            total_loss_val += loss_val
            
            elapsed = time.time() - start_time
            
            # Only log when an optimizer step is performed
            if not accumulate_gradient and step % args.log_freq == 0:
                logging.info(
                    f"Step {step} | Loss: {loss_val:.4f} | Avg Loss: {exp_avg_loss:.4f} | "
                    f"StdDev: {std_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s"
                )
                
            # Only save when an optimizer step is performed
            if not accumulate_gradient and step > 0 and step % args.save_freq == 0:
                model_path = os.path.join(args.output_dir, f"model_{step:06d}.pt")
                logging.info(f"Saving model to {model_path}")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": diffusion_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss_val,
                        "exp_avg_loss": exp_avg_loss,
                    },
                    model_path,
                )
                if exp_avg_loss < best_loss:
                    best_loss = exp_avg_loss
                    best_model_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": diffusion_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss_val,
                            "exp_avg_loss": exp_avg_loss,
                        },
                        best_model_path,
                    )
