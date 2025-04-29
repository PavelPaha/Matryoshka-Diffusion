import random

import torch


class DiffusionCollator:
    def __init__(
        self,
        tokenizer,
        max_cap_length=64,
        random_caption=True,
    ):
        self.max_cap_length = max_cap_length
        self.tokenizer = tokenizer
        self.random_caption = random_caption

    def __call__(self, batch):
        batch_images = []
        batch_captions = []

        for item in batch:
            captions = item["captions"]
            caption_idx = (
                random.randint(0, len(captions) - 1) if self.random_caption else 0
            )
            batch_captions.append(captions[caption_idx])

            batch_images.append(item["image"])

        images = torch.stack(batch_images)

        tokenized = self.tokenizer(
            batch_captions,
            padding="max_length",
            max_length=self.max_cap_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "images": images,
            "lm_outputs": tokenized.input_ids,
            "lm_mask": tokenized.attention_mask,
        }
