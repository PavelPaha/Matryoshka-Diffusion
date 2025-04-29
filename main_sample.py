import argparse
import logging
import os
import torch
from run.configs.conf_64_64 import diffusion_config, nested_unet_config
import numpy as np

from run.sample import sample_images
from run.utils import get_model


def parse_sampling_args():
    """
    Парсинг аргументов командной строки для генерации изображений

    Returns:
        argparse.Namespace: Объект с аргументами
    """
    parser = argparse.ArgumentParser(description="Diffusion Model Image Generation")

    # Параметры модели и чекпоинта
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к чекпоинту модели"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        nargs="+",
        help="Промпт(ы) для генерации изображений",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Количество шагов семплирования",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Размер генерируемых изображений"
    )

    # Вывод
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated",
        help="Директория для сохранения сгенерированных изображений",
    )

    # Аппаратные параметры
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7" if torch.cuda.is_available() else "cpu",
        help="Устройство для генерации (cuda:X или cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed для воспроизводимости результатов"
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

    args = parse_sampling_args()
    device = torch.device(args.device)
    
    model = get_model(nested_unet_config, diffusion_config, device)
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    logging.info("Model loaded")

    sample_images(
        model=model,
        prompts=args.prompts,
        device=device,
        num_inference_steps=args.num_inference_steps,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
