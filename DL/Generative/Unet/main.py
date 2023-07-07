from model import *
from dataset import *
from train import *
from time import time

train(n_epochs,
      dataloader,
        optimizer,
        losses,
        model,
        device)


def calculate_time():
    """
    Make a function so that use can use it as a wrapper function
    to calculate time :).
    """
    pass


# print("im on dataset.py")
# start = time()
# imgs, labels = next(iter(dataloader))
# print(imgs.shape)
# model.train()
# predictions = model(imgs)
# print(f"shape of predictions -> {predictions.shape}")
# loss = torch.mean((predictions - labels))
# print("backward pass")
# loss.backward()
# print("optimizer")
# optimizer.step()
# print(f"the loss is {loss.item()}")
# end = time()
# print(f"total time taken -> {end-start}")
