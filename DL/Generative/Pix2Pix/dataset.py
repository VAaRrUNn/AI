import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import os
from model import *
import torch
from PIL import Image


images_dir = r"D:\material\Machine_Deep\Deep Learning\code\Specialization\Gan's\mine\Unet\semantic_drone_dataset\original_images"
labels_dir = r"D:\material\Machine_Deep\Deep Learning\code\Specialization\Gan's\mine\Unet\RGB_color_image_masks"


class segmentationDataset(Dataset):
    """
    Custom dataset to load and process images
    """

    def __init__(self, images_dir,
                 labels_dir):

        self.images = os.listdir(images_dir)
        self.labels = os.listdir(labels_dir)

        self.images = [os.path.join(images_dir, image)
                       for image in self.images]
        self.labels = [os.path.join(labels_dir, label)
                       for label in self.labels]

        self.resizing = transforms.Compose([
            transforms.Resize((572, 572)),
        ])

        self.toTensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img, label = Image.open(img), Image.open(label)
        img = self.resizing(self.toTensor(img))
        label = self.resizing(self.toTensor(label))
        return (img, label)


dataset = segmentationDataset(images_dir, labels_dir)

# dataloader
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
unet = UNet().to(device)
gen = Generator(unet=unet).to(device)
disc = Discriminator().to(device)
n_epochs = 1
losses = []
pixel_dis_lambda = 10
loss_fn = nn.BCEWithLogitsLoss()
gen_opt = torch.optim.Adam(params=gen.parameters(), lr=0.001)
disc_opt = torch.optim.Adam(params=disc.parameters(), lr=0.001)

