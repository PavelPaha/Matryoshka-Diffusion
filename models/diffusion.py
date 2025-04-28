import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.diffusion import *


class NestedDiffusion(nn.Module):
    def __init__(self, 
                 vision_model,
                 config: DiffusionConfig = DiffusionConfig()):
        super().__init__()
        self.model = vision_model
        self.config = config
        
        # Создаем расписание шума
        self._setup_noise_schedule()
        
        self.loss_fn = nn.MSELoss(reduction="none")
        logging.info(f"SimpleNestedDiffusion initialized with config: {config}")
        
    def _setup_noise_schedule(self):
        """Настраивает расписание шума для диффузии"""
        n_steps = self.config.num_diffusion_steps
        
        if self.config.schedule_type == ScheduleType.DDPM:
            # Линейное расписание DDPM
            betas = np.concatenate(([0], np.linspace(
                self.config.beta_start, 
                self.config.beta_end, 
                num=n_steps
            )))
            log_alphas = np.log(1.0 - betas)
            gammas = np.exp(np.cumsum(log_alphas))
            
        elif self.config.schedule_type == ScheduleType.DEEPFLOYD:
            # Квадратное косинусное расписание (DeepFloyd/StableDiffusion)
            def alpha_bar(time_step: float) -> float:
                return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
                
            betas = [0]
            for i in range(n_steps):
                t1 = i / n_steps
                t2 = (i + 1) / n_steps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
                
            betas = np.asarray(betas)
            log_alphas = np.log(1.0 - betas)
            gammas = np.exp(np.cumsum(log_alphas))
        
        # Регистрация буферов
        self.register_buffer("gammas", torch.tensor(gammas).float())
        
    def get_noise_levels(self, t: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Получает уровни шума для текущего и предыдущего временных шагов"""
        batch_size = x.shape[0]
        time = t * torch.ones(batch_size, dtype=torch.long, device=x.device)
        
        # Получаем gamma для текущего и предыдущего шагов
        g = self._expand_to_dims(self.gammas[time + 1], x)
        g_prev = self._expand_to_dims(self.gammas[time], x)
        
        return g, g_prev
    
    def _expand_to_dims(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Расширяет тензор времени до размерности входного тензора"""
        return t.view(-1, 1, 1, 1).expand_as(x)
    
    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Добавляет шум к изображению согласно расписанию"""
        g, _ = self.get_noise_levels(t, x)
        return g.sqrt() * x + (1 - g).sqrt() * noise
    
    def get_multi_resolution_images(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Создает изображения с разным разрешением для nested архитектуры"""
        scales = self.model.nest_ratio + [1]
        
        images_multi_res = [images]
        for i in range(1, len(scales)):
            ratio = scales[0] // scales[i]
            images_multi_res.append(F.avg_pool2d(images, ratio))
        
        return images_multi_res
    
    def get_multi_resolution_noise(self, noise: torch.Tensor) -> List[torch.Tensor]:
        """Создает шум с разным разрешением для nested архитектуры"""
        scales = self.model.nest_ratio + [1]
        
        noise_multi_res = [noise]
        for i in range(1, len(scales)):
            ratio = scales[0] // scales[i]
            # Для низких разрешений используем новый шум
            noise_low = torch.randn_like(F.avg_pool2d(noise, ratio))
            noise_multi_res.append(noise_low)
        
        return noise_multi_res
    
    def get_prediction_x0(self, x_t: torch.Tensor, pred: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Предсказывает чистое изображение x0 из зашумленного и предсказания модели"""
        if self.config.prediction_type == PredictionType.DDPM:
            # Если модель предсказывает шум
            x0 = (x_t - pred * (1 - g).sqrt()) / g.sqrt()
        else:  # PredictionType.V_PREDICTION
            # Если модель предсказывает v-параметр
            x0 = x_t * g.sqrt() - pred * (1 - g).sqrt()
            
        # Применяем ограничение значений
        x0 = torch.clamp(x0, -1.0, 1.0)
        return x0
    
    def get_prediction_target(self, x0: torch.Tensor, eps: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Получает целевое значение для обучения модели"""
        if self.config.prediction_type == PredictionType.DDPM:
            return eps
        else:  # PredictionType.V_PREDICTION
            return g.sqrt() * eps - (1 - g).sqrt() * x0
    
    def get_loss(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Рассчитывает функцию потерь для обучения модели"""
        images = sample["images"]
        lm_outputs = sample["lm_outputs"]  # Эмбеддинги текста
        lm_mask = sample["lm_mask"]        # Маска для текста
        
        # 1. Получаем изображения разного разрешения
        images_multi_res = self.get_multi_resolution_images(images)
        
        # 2. Выбираем случайные временные шаги и генерируем шум
        batch_size = images.shape[0]
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=images.device)
        
        # 3. Создаем шум для каждого разрешения
        noise_multi_res = self.get_multi_resolution_noise(torch.randn_like(images))
        
        # 4. Применяем шум к изображениям разного разрешения
        x_t_multi_res = []
        gamma_multi_res = []
        for img, noise in zip(images_multi_res, noise_multi_res):
            g, _ = self.get_noise_levels(t, img)
            x_t = self.add_noise(img, noise, t)
            x_t_multi_res.append(x_t)
            gamma_multi_res.append(g)
        
        # 5. Получаем предсказания модели для разных разрешений
        predictions = self.model(x_t_multi_res, t, lm_outputs, lm_mask)
        
        # 6. Рассчитываем целевые значения для функции потерь
        targets = []
        for img, noise, g in zip(images_multi_res, noise_multi_res, gamma_multi_res):
            target = self.get_prediction_target(img, noise, g)
            targets.append(target)
        
        # 7. Вычисляем функцию потерь для каждого разрешения
        losses = []
        for pred, target, weight in zip(predictions, targets, self.config.multi_res_weights):
            loss = self.loss_fn(pred, target).mean(dim=(1, 2, 3)) * weight
            losses.append(loss)
        
        # Суммируем потери, если используем double_loss
        total_loss = losses[0]
        if self.config.use_double_loss and len(losses) > 1:
            for loss in losses[1:]:
                total_loss = total_loss + loss
        
        return total_loss, t, x_t_multi_res, predictions
    
    def sample(self, 
               text_embeddings: torch.Tensor, 
               text_mask: torch.Tensor, 
               image_size: int = 64, 
               batch_size: int = 1, 
               guidance_scale: float = 1.0,
               num_inference_steps: int = None) -> torch.Tensor:
        """
        Генерирует изображения из шума с использованием текстовых подсказок
        
        Args:
            text_embeddings: Эмбеддинги текста
            text_mask: Маска текста
            image_size: Размер генерируемого изображения
            batch_size: Размер батча
            guidance_scale: Коэффициент для classifier-free guidance
            num_inference_steps: Количество шагов для генерации, если None - используется полное число
        
        Returns:
            Сгенерированные изображения
        """
        self.eval()
        device = text_embeddings.device
        
        # Определяем количество шагов
        if num_inference_steps is None:
            num_inference_steps = self.config.num_diffusion_steps
        
        # Рассчитываем временные шаги для генерации
        step_ratio = (self.config.num_diffusion_steps + 1) / (num_inference_steps + 1)
        timesteps = np.arange(0, num_inference_steps + 1)
        timesteps = (timesteps * step_ratio).round()[::-1].astype(np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        # Создаем начальный шум
        input_channels = self.model.input_channels
        scales = self.model.nest_ratio + [1]
        x_t_multi_res = []
        
        # Инициализируем шум для каждого разрешения
        for scale in scales:
            size = image_size // (scales[0] // scale)
            noise = torch.randn(batch_size, input_channels, size, size, device=device)
            x_t_multi_res.append(noise)
        
        # Реализуем Classifier-Free Guidance если guidance_scale > 1
        if guidance_scale > 1.0:
            # Дублируем текстовые эмбеддинги для условной и безусловной генерации
            uncond_embeddings = torch.zeros_like(text_embeddings)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            
            uncond_mask = torch.zeros_like(text_mask)
            text_mask = torch.cat([uncond_mask, text_mask], dim=0)
        
        # Генерация изображения шаг за шагом
        for i in range(len(timesteps) - 1):
            # Текущий временной шаг
            t = timesteps[i]
            
            with torch.no_grad():
                # Получаем предсказание модели
                if guidance_scale > 1.0:
                    # Дублируем входы для CFG
                    x_t_doubled = [torch.cat([x, x], dim=0) for x in x_t_multi_res]
                    t_doubled = torch.cat([t.unsqueeze(0)] * 2 * batch_size)
                    
                    # Получаем предсказания для условного и безусловного случая
                    predictions = self.model(x_t_doubled, t_doubled, text_embeddings, text_mask)
                    
                    # Разделяем предсказания
                    uncond_preds = []
                    cond_preds = []
                    for pred in predictions:
                        uncond_pred, cond_pred = pred.chunk(2)
                        uncond_preds.append(uncond_pred)
                        cond_preds.append(cond_pred)
                    
                    # Применяем guidance scale
                    predictions = [uncond_pred + guidance_scale * (cond_pred - uncond_pred) 
                                  for uncond_pred, cond_pred in zip(uncond_preds, cond_preds)]
                else:
                    predictions = self.model(x_t_multi_res, t.unsqueeze(0).expand(batch_size), 
                                            text_embeddings, text_mask)
            
            # Шаг к t-1
            next_t = timesteps[i + 1]
            
            # Обновляем каждое разрешение
            for j, (x_t, pred) in enumerate(zip(x_t_multi_res, predictions)):
                # Получаем уровни шума
                g, g_prev = self.get_noise_levels(t.unsqueeze(0), x_t)
                
                # Предсказываем x0
                x0_pred = self.get_prediction_x0(x_t, pred, g)
                
                # Если не последний шаг, добавляем шум
                if next_t > 0:
                    # Вычисляем x_{t-1}
                    eps = (x_t - x0_pred * g.sqrt()) / (1 - g).sqrt()
                    x_t_prev = g_prev.sqrt() * x0_pred + (1 - g_prev).sqrt() * eps
                    
                    # Добавляем немного случайного шума (только для DDPM, не для DDIM)
                    if i < len(timesteps) - 2:  # Не добавляем шум на последнем шаге
                        noise = torch.randn_like(x_t_prev)
                        sigma = ((1 - g_prev) / (1 - g) * (1 - g/g_prev)).sqrt()
                        x_t_prev = x_t_prev + sigma * noise
                else:
                    # На последнем шаге просто используем предсказанное чистое изображение
                    x_t_prev = x0_pred
                
                # Обновляем значение для следующего шага
                x_t_multi_res[j] = x_t_prev
        
        # Возвращаем только изображение высокого разрешения
        return torch.clamp(x_t_multi_res[0], -1, 1)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
