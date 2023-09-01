from .dataset import train_dataloader
from .model import model
import tqdm.auto as tqdm
import torch
import yaml
import torch.optim as optim
import torch.nn as nn
from .utils import accuracy_fn


# Configuration file
config_path = "../config/default_config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
SAVE_DIR = config["training"]["save_dir"]
LOG_DIR = config["training"]["log_dir"]
LEARNING_RATE = config["training"]["lr"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_(*,
           model,
           optimizer,
           loss_fn,
           dataloader,
           epochs,
           device,
           return_losses=True):

    model.to(device)
    model.train()
    losses = []
    acc = []

    for _ in tqdm(range(epochs)):

        for images, labels in tqdm(dataloader):

            labels = labels.type(torch.float32)
            images, labels = images.to(device), labels.to(device)

            logits = model(images).squeeze()
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            pred = torch.sigmoid(logits)
            pred = torch.where(pred >= 0.5, 1, 0)
            losses.append(loss.item() / BATCH_SIZE)
            acc.append(accuracy_fn(labels, pred))

    return losses, acc


optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()


def train():
    losses, acc = train_(model=model,
                         optimizer=optimizer,
                         loss_fn=loss_fn,
                         dataloader=train_dataloader,
                         epochs=EPOCHS,
                         device=DEVICE)
    return losses, acc
