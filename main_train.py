import argparse
import logging
import os

import numpy as np
import torch

from run.configs.conf_64_64 import diffusion_config, nested_unet_config
from run.train import train_loop
from run.utils import get_logger, get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")

    # Основные параметры обучения
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Размер батча для обучения"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Количество эпох обучения"
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=2.0,
        help="Максимальная норма градиентов",
    )
    parser.add_argument(
        "--lr", type=float, default=5.0e-05, help="Начальная скорость обучения"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=150,
        help="Количество шагов разогрева learning rate",
    )

    # Параметры модели
    parser.add_argument(
        "--num_levels",
        type=int,
        default=2,
        help="Количество уровней вложенной архитектуры",
    )

    # Параметры логирования и сохранения
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Директория для сохранения результатов",
    )
    parser.add_argument(
        "--log_freq", type=int, default=5, help="Частота логирования (в шагах)"
    )
    parser.add_argument(
        "--save_freq", type=int, default=200, help="Частота сохранения модели (в шагах)"
    )

    # Оптимизация
    parser.add_argument(
        "--num_gradient_accumulations",
        type=int,
        default=4,
        help="Количество шагов накопления градиентов",
    )

    # Аппаратные параметры
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7" if torch.cuda.is_available() else "cpu",
        help="Устройство для обучения (cuda:X или cpu)",
    )

    # Дополнительные параметры
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Параметр регуляризации весов"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed для воспроизводимости результатов"
    )

    # Параметры генерации
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Количество шагов семплирования при генерации",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Размер генерируемых изображений"
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="Размер батча для генерации изображений",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device(args.device)
    model = get_model(nested_unet_config, diffusion_config, device)

    train_loop(
        torch.device(args.device),
        args,
        model,
        logger=get_logger(args),
    )


if __name__ == "__main__":
    main()
