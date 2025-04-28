import torch
import torch.nn as nn
from argparse import Namespace

from typing import Optional
import numpy as np

def train_batch(
    model: torch.nn.Module,
    sample: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: Optional[torch.utils.tensorboard.SummaryWriter],
    args: Namespace,
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulate_gradient: bool = False,
    num_grad_accumulations: int = 1,
    ema_model: Optional[nn.Module] = None,
    loss_factor: float = 1.0,
):
    model.train()
    lr = scheduler.get_last_lr()[0]
    
    # Updates the scale for next iteration
    if args.fp16:
        with torch.cuda.autocast(dtype=torch.bfloat16):
            # Изменено: упрощенная модель возвращает другие значения
            total_loss, times, x_t_multi_res, predictions = model.get_loss(sample)
            
            # Средняя потеря по батчу
            loss = total_loss.mean() * loss_factor
            loss_val = loss.item()

            if np.isnan(loss_val):
                optimizer.zero_grad()
                # Возвращаем измененный набор значений
                return loss_val, total_loss, times, x_t_multi_res, predictions

            if num_grad_accumulations != 1:
                loss = loss / num_grad_accumulations
                
        # Остальной процесс тот же
        grad_scaler.scale(loss).backward()

        if not accumulate_gradient:
            grad_scaler.unscale_(optimizer)
            # Изменено: прямое обращение к параметрам модели
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            ).item()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema_model is not None:
                # Изменено: обновление EMA для упрощенной модели
                ema_model.update(model.model)
    else:
        # Изменено: упрощенная модель возвращает другие значения
        total_loss, times, x_t_multi_res, predictions = model.get_loss(sample)
        
        # Средняя потеря по батчу
        loss = total_loss.mean() * loss_factor
        loss_val = loss.item()
        
        if np.isnan(loss_val):
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            # Возвращаем измененный набор значений
            return loss_val, total_loss, times, x_t_multi_res, predictions

        loss.backward()
        if num_grad_accumulations != 1:
            loss = loss / num_grad_accumulations
            
        if not accumulate_gradient:
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            ).item()
            optimizer.step()
            if ema_model is not None:
                # Изменено: обновление EMA для упрощенной модели
                ema_model.update(model.model)

    # Логирование
    if logger is not None and not accumulate_gradient:
        logger.add_scalar("train/Loss", loss_val)
        logger.add_scalar("lr", lr)
        
        # Добавлено: можно логировать потери для разных разрешений
        if hasattr(model.config, 'use_double_loss') and model.config.use_double_loss:
            for i, weight in enumerate(model.config.multi_res_weights):
                if i < len(total_loss):
                    resolution_loss = (total_loss[i] / weight).mean().item()
                    logger.add_scalar(f"train/Loss_resolution_{i}", resolution_loss)

    if not accumulate_gradient:
        optimizer.zero_grad()
        scheduler.step()

    # Возвращаем измененный набор значений
    return loss_val, total_loss, times, x_t_multi_res, predictions
