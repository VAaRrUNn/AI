from tqdm.auto import tqdm
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import torch.nn as nn
from time import time


# -------------------------- UNET -----------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 upsample=False):

        super(CNNBlock, self).__init__()
        if not upsample:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
                nn.Conv2d(out_channels, out_channels, kernel_size,
                          stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
            )

        if upsample:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
            )
        self.skip_connection_layer = []

    def forward(self, x, skip_connections=False):
        out = self.layers(x)
        if skip_connections:
            self.skip_connection_layer.append(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=0,
                 blocks=4, segmented_image_channels=3):

        super(UNet, self).__init__()

        self.downsample = nn.ModuleList()
        conv_in = in_channels
        conv_out = out_channels
        for i in range(blocks):
            self.downsample.append(
                CNNBlock(conv_in, conv_out,
                         kernel_size, stride, padding))
            conv_in = conv_out
            conv_out *= 2

        self.skip_connection_layers = []

        self.upsample = nn.ModuleList()

        # saving for use in upsampling in forward
        self.conv_in = out_channels * (2**(blocks-1))
        self.conv_out = conv_in  # // 2

        conv_in = self.conv_in
        conv_out = self.conv_out
        for i in range(blocks-1):
            self.upsample.append(
                CNNBlock(conv_out*2, conv_out,
                         kernel_size, stride, padding, upsample=True))
            conv_out //= 2

        # this is for the last layer i.e, from 512 channels -> segmented image channels
        self.upsample.append(
            CNNBlock(conv_out*2, segmented_image_channels,
                     kernel_size, stride, padding, upsample=True))

    def forward(self, x):

        out = x
        # print(out.shape)
        # downsampling
        for i in range(len(self.downsample)):
            block = self.downsample[i]
            out = block(out, skip_connections=True)
            # self.skip_connection_layers.append(
            #     *block.skip_connection_layer
            # )
            # print(out.shape)
            self.skip_connection_layers.append(out)
            out = nn.MaxPool2d(2, 2)(out)

        # print(out.shape)
        # print("-------------------")
        # upsampling

        skip_len = len(self.skip_connection_layers)
        conv_in = self.conv_in
        conv_out = self.conv_out
        for i in range(len(self.upsample)):
            block = self.upsample[i]
            out = nn.ConvTranspose2d(conv_in, conv_out,
                                     kernel_size=2,
                                     stride=2, padding=0)(out)
            out = torch.cat(
                (out, self.skip_connection_layers[skip_len - 1 - i]), dim=1)
            # print(out.shape)
            out = block(out)
            conv_in = conv_out
            conv_out //= 2

        return out

# -------------------------- UNET -----------------------------------

# testing UNET
# a = torch.randn((1, 3, 572, 572))
# unet = UNet()
# print(unet(a).shape)


# ------------------------ DATA LOADING -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# hard coded rn
images_dir = r"D:\material\Machine_Deep\github_repos\DL\resources\CNN_architectures\semantic_drone_dataset\original_images"
labels_dir = r"D:\material\Machine_Deep\github_repos\DL\resources\CNN_architectures\RGB_color_image_masks"


class segmentationDataset(Dataset):
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


def calculate_time():
    """
    Make a function so that use can use it as a wrapper function
    to calculate time :).
    """
    pass

# -------------------------------------------------------------


# -------------------------------HYPTERPARAMETERS-------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
n_epochs = 1
losses = []
# loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


def train(n_epochs,
          dataloader,
          optimizer,
          losses,
          model,
          device):

    for epoch in tqdm(range(n_epochs)):
        i = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = torch.mean((predictions - labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(i)
            i += 1


# THE ERROR WAS IN THE LOSS FUNCTION 
# THE LOSS SHOULD BE A SINGLE VLAUE NOT A ARRAY/MATRIX OF VALUES


# train(n_epochs,
#       dataloader,
#         optimizer,
#         losses,
#         model,
#         device)

# training demonstration
start = time()
imgs, labels = next(iter(dataloader))
print(imgs.shape)
model.train()
predictions = model(imgs)
print(f"shape of predictions -> {predictions.shape}")
loss = torch.mean((predictions - labels))
print("backward pass")
loss.backward()
print("optimizer")
optimizer.step()
print(f"the loss is {loss.item()}")
end = time()
print(f"total time taken -> {end-start}")


