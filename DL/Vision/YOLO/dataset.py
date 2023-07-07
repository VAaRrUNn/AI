# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:51:07 2023

@author: sanat
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

dataset_dir = ["D:\\yolo\\YOLO\\dataset\\images",
               "D:\\yolo\\YOLO\\dataset\\labels"]
training_images = len(os.listdir(dataset_dir[0]))

transformation = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=(300, 300))])


class VOCdataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 csv_file,
                 transform = None,
                 S=7,
                 B=2,
                 C=20):

        super().__init__()

        self.S = S
        self.B = B
        self.C = C
        self.dataset_dir = dataset_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, x) -> Tuple[torch.Tensor, List[str]]:
        image_path = os.path.join(self.dataset_dir[0], self.annotations.iloc[x, 0])
        image = Image.open(image_path)
        boxes = []

        label_path = os.path.join(
            self.dataset_dir[1], self.annotations.iloc[x, 1])
        with open(label_path, 'r') as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", '').split()]

                boxes.append([class_label, x, y, width, height])

        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image), boxes

        label_matrix = torch.zeros(self.S, self.S, self.C + 5)
        # The thing here is the labels we are having rn are relative to the image not
        # to the cell. We have to make them relative to the cell

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

    def __len__(self) -> int:
        return len(self.annotations)


# # Making a test object
# csv_file = os.path.join("D:\\yolo\\YOLO\\dataset\\", "train.csv")
# test_obj = VOCdataset(dataset_dir, csv_file, transform=transformation)


# # The images are irregular meaning they are not of fixed shape so
# # when you are trying to batchify them say batch_size = some other value > 1
# # there there will be error
# train_dataloader = DataLoader(test_obj,
#                               batch_size=13,
#                               shuffle=True)

# img, label = next(iter(train_dataloader))
