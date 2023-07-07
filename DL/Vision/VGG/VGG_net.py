# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:52:54 2023

@author: sanat
"""

import numpy as np
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

VGG_types = {
    'VGG_mine': [32, 32, 'M', 64, 64, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layer(VGG_types["VGG_mine"])
        self.fcs = nn.Sequential(
            # Here 512 is no of channels, 7X7 is the size of image left
            # nn.Flatten will produce the shape -> (batch_size, channels * height * width)
            # or you can have also done x.reshape(x.shape[0], -1)
            nn.Flatten(),
            
            # this one is for other than VGG_mine
            # nn.Linear(512*7*7, 4096),
            
            # this is for VGG_mine
            nn.Linear(64*7*7, 4096),
            
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fcs(x)
        return x
    
    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False,
                                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU()]
                in_channels = x
                
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                
        return nn.Sequential(*layers)
        

model = VGG_net(in_channels=1, num_classes=10)
                            
## Getting the dataset
training_set = datasets.FashionMNIST(root='.', train=True, download=False,
                                     transform=transforms.ToTensor())         
test_set = datasets.FashionMNIST(root='.', train=False, download=False,
                                 transform=transforms.ToTensor())

## Making dataloader
train_dataloader = DataLoader(training_set, batch_size=64)

test_dataloader = DataLoader(test_set, batch_size=64)


optimizer = optim.Adam(params=model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

def train_step(model: torch.nn.Module,
               optimizer: torch.optim,
               loss_fn: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               epochs: int):
    
    train_loss = []
    model.train()
    for epoch in tqdm(range(epochs)):
        for imgs, labels in train_dataloader:
            prediction = model(imgs)
            
            loss = loss_fn(prediction, labels)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    train_loss /= len(train_dataloader)
    return train_loss
            

train_loss = train_step(model, optimizer, loss_fn, train_dataloader, 1)

img, label = next(iter(train_dataloader))
print(model(img).shape)

plt.plot(np.arange(len(train_loss)), train_loss)
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        