import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from functools import lru_cache


class FlickrDataset(Dataset):
    def __init__(self, root_dir, caps_file, transforms=None, cache_size=300):
        self.root_dir = root_dir
        self.df = pd.read_csv(caps_file)
        self.transforms = transforms
        
        # Группировка подписей по имени изображения
        self.image_captions = {}
        for _, row in self.df.iterrows():
            image_name = row['image']
            if isinstance(row['caption'], float):
                continue
            
            caption = row['caption'].strip()
            if image_name not in self.image_captions:
                self.image_captions[image_name] = []
            self.image_captions[image_name].append(caption)
        
        # Создание списка уникальных имен изображений
        self.images = list(self.image_captions.keys())
    
    def __len__(self):
        return len(self.images)

    @lru_cache(maxsize=300)  # Кэширование последних 300 изображений
    def _load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = self._load_image(img_path)
        
        captions = self.image_captions[img_name]
        
        return {
            "image": img,
            "captions": captions
        }
