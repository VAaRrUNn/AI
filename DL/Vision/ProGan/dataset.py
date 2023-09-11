import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os
import random
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 4
dataset_dirs = (r"D:\material\Machine_Deep\dataset\new_datasets\d0\celeba_hq\train\female",
                r"D:\material\Machine_Deep\dataset\new_datasets\d0\celeba_hq\train\male")


class CelebHQ(Dataset):
    def __init__(self,
                 dataset_dirs=dataset_dirs):
        super(CelebHQ, self).__init__()
        male_dir, female_dir = dataset_dirs

        # o for male, 1 for female
        self.male_images = [(os.path.join(dataset_dirs[1], image), 0)
                            for image in os.listdir(dataset_dirs[1])]
        self.female_images = [(os.path.join(dataset_dirs[0], image), 1)
                              for image in os.listdir(dataset_dirs[0])]
        self.images = self.male_images + self.female_images
        random.shuffle(self.images)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Tuple[torch.tensor, int]:

        # 0 is for male and 1 for female

        return (self.transform(Image.open(self.images[index][0])), self.images[index][1])

dataset = CelebHQ()
dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
