import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, root_dir, caps_file, tokenizer, transforms=None, max_length=64):
        self.root_dir = root_dir
        self.df = pd.read_csv(caps_file)
        self.transforms = transforms
        
        self.tokenizer = tokenizer
        self.max_cap_length = max_length
        
        # Group captions by image name
        self.image_captions = {}
        for _, row in self.df.iterrows():
            image_name = row['image']
            if isinstance(row['caption'], float):
                continue

            caption = row['caption'].strip()
            if image_name not in self.image_captions:
                self.image_captions[image_name] = []
            self.image_captions[image_name].append(caption)
        
        # Create a list of unique image names
        self.images = list(self.image_captions.keys())
        
    def __len__(self):
        # Return the number of unique images
        return len(self.images)
  
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
        
        captions = self.image_captions[img_name]
        tokenized_captions = []
        
        for caption in captions:
            # Токенизация каждой подписи на лету
            encoding = self.tokenizer(
                caption,
                truncation=True,
                return_tensors='pt'
            )

            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].squeeze(0)
                        
            tokenized_captions.append(encoding)
        
        return img, tokenized_captions
