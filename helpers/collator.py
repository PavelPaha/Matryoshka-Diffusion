import random

import torch

import torch
import torch.nn.functional as F


class ImagePyramidCreator():
    def __init__(self, num_levels, scale_factor, device):
        self.device = device
        self.num_levels = num_levels
        self.scale_factor = scale_factor
    
    def __call__(self, x: torch.Tensor):
        pyramid = [x.to(self.device)]
        current = pyramid[-1]
        
        for _ in range(self.num_levels - 1):
            current = F.avg_pool2d(
                current,
                kernel_size=self.scale_factor,
                stride=self.scale_factor
            )
            pyramid.append(current.to(self.device))
        
        return pyramid
        


class DiffusionCollator:
    def __init__(self, tokenizer, pyramid_creator: ImagePyramidCreator, max_cap_length=64, random_caption=True,):
        self.max_cap_length = max_cap_length
        self.tokenizer = tokenizer
        self.random_caption = random_caption
        self.pyramid_creator = pyramid_creator
        

    def __call__(self, batch):
        pyramid_batches = [[] for _ in range(self.num_levels)]
        batch_captions = []

        for item in batch:
            captions = item["captions"]
            caption_idx = (
                random.randint(0, len(captions) - 1) if self.random_caption else 0
            )
            batch_captions.append(captions[caption_idx])


            image_pyramid = self.pyramid_creator(
                item["image"], 
                num_levels=self.num_levels,
                scale_factor=self.scale_factor
            )
            for level, img in enumerate(image_pyramid):
                pyramid_batches[level].append(img)

        
        pyramid_tensors = [torch.stack(level_batch) for level_batch in pyramid_batches]
        images = torch.stack(pyramid_tensors, dim=0)
        
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
