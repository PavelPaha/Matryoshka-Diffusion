import os

import numpy as np
from PIL import Image
import skimage
import skimage.io
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def read_rgb_image(path_to_image):
    image = skimage.img_as_ubyte(skimage.io.imread(path_to_image))
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if image.shape[-1] == 1:
        image = image * np.ones((1,1,3), dtype=np.uint8)
    return image


class TinyImagenetTrainRAM(Dataset):
    def __init__(self, root, transform=transforms.ToTensor()):
        super().__init__()

        self.root = root
        self.classes = sorted(
            [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))])
        self.class_to_idx = {item: index for index, item in enumerate(self.classes)}

        self.transform = transform
        self.images, self.targets = [], []
        for index, item in tqdm(enumerate(self.classes), total=len(self.classes), desc=self.root):
            path = os.path.join(root, item, 'images')
            for name in sorted(os.listdir(path)):
                image = read_rgb_image(os.path.join(path, name))
                assert image.shape == (64, 64, 3), image.shape
                self.images.append(Image.fromarray(image))
                self.targets.append(index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        target = self.targets[index]
        return image, target

class TinyImagenetValRAM(Dataset):
    def __init__(self, root, transform=transforms.ToTensor()):
        super().__init__()

        self.root = root
        with open(os.path.join(root, 'val_annotations.txt')) as f:
            annotations = []
            for line in f:
                img_name, class_label = line.split('\t')[:2]
                annotations.append((img_name, class_label))

        self.classes = sorted(list(set([class_label for _, class_label in annotations])))
        
        assert len(self.classes) == 200, len(self.classes)
        assert all(self.classes[i] < self.classes[i+1] for i in range(len(self.classes)-1)), 'classes should be ordered'
        assert all(isinstance(elem, type(annotations[0][1])) for elem in self.classes), 'your just need to reuse class_labels'

        # 2. self.class_to_idx - dict from class label to class index
        self.class_to_idx = {item: index for index, item in enumerate(self.classes)}

        self.transform = transform

        self.images, self.targets = [], []
        for img_name, class_name in tqdm(annotations, desc=root):
            img_name = os.path.join(root, 'images', img_name)
            # 3. load image and store it in self.images (your may want to use tiny_img_dataset.read_rgb_image)
            # store the class index in self.targets
            # YOUR CODE
            image = read_rgb_image(img_name)
            
            assert image.shape == (64, 64, 3), image.shape
            self.images.append(Image.fromarray(image))
            self.targets.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # take image and its target label from "self.images" and "self.targets", 
        # transform the image using self.transform and return the transformed image and its target label
        
        # YOUR CODE
        image = self.images[index]
        image = self.transform(image)
        target = self.targets[index]

        return image, target
