import os
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image

BATCH_SIZE = 2

# here A is for Horse and B is for the zebra
dataset_dirs = [r"D:\material\Machine_Deep\dataset\new_datasets\d3\horse2zebra\horse2zebra\trainA",
                r"D:\material\Machine_Deep\dataset\new_datasets\d3\horse2zebra\horse2zebra\trainB"]

img = os.path.join(dataset_dirs[0], os.listdir(dataset_dirs[0])[0])

transform = transforms.Compose([
    transforms.ToTensor(),
])


class unpairedImageDataset(Dataset):
    def __init__(self,
                 dataset_dirs,
                 transform=None):
        super(unpairedImageDataset, self).__init__()

        self.A = [os.path.join(dataset_dirs[0], x)
                  for x in os.listdir(dataset_dirs[0])]
        self.B = [os.path.join(dataset_dirs[1], x)
                  for x in os.listdir(dataset_dirs[1])]

        # For testing if it's working or not
        self.A = self.A[:2]
        self.B = self.B[:2]

        self.transform = transform

    def __len__(self):
        return min(len(self.A), len(self.B))

    def __getitem__(self, index):
        if self.transform:
            imgA = self.transform(Image.open(self.A[index]))
            imgB = self.transform(Image.open(self.B[index]))
            return (imgA, imgB)

        return (self.A[index], self.B[index])


dataset = unpairedImageDataset(dataset_dirs=dataset_dirs,
                               transform=transform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

