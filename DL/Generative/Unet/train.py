from dataset import *
from typing import List, Tuple
import torch
from tqdm.auto import tqdm


def train(n_epochs: int,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim,
          losses: List,
          model: torch.nn.Module,
          device: torch.device
          ):

    for epoch in tqdm(range(n_epochs)):
        # i = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = torch.mean((predictions - labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(i)
            # i += 1


# THE ERROR WAS IN THE LOSS FUNCTION
# THE LOSS SHOULD BE A SINGLE VLAUE NOT A ARRAY/MATRIX OF VALUES
