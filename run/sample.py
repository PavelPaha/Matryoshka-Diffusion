import logging
import os

import torch

from helpers.lm import create_lm
from models.diffusion import NestedDiffusion


def sample_images(
    model: NestedDiffusion,
    prompts: list,
    device,
    num_inference_steps=50,
    image_size=64,
    output_dir=None,
):

    model.eval()

    images = []

    tokenizer, encoder = create_lm()
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = encoder(tokenized["input_ids"]).to(device)
    text_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        samples = model.sample(
            text_embeddings,
            text_mask,
            image_size,
            len(prompts),
            num_inference_steps=num_inference_steps,
        )

    for j, sample in enumerate(samples):
        # img_tensor = (sample + 1) / 2.0
        # img_tensor = img_tensor.clamp(0, 1)

        img_tensor = sample

        images.append(img_tensor)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            from torchvision.utils import save_image

            save_path = os.path.join(
                output_dir,
                f"sample_{j:04d}.png",
            )
            save_image(img_tensor, save_path)
    
    logging.info("Images generated")
    return images
