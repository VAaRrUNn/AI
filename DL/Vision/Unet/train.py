from tqdm.auto import tqdm
from dataset_dataloader import *
from model_ import *

def train(n_epochs,
          dataloader,
          optimizer,
          loss_fn,
          model,
          device):

    model.train()
    losses = []

    for epoch in tqdm(range(n_epochs)):
        i = 1
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            i += 1
            if i == 5:
                break
    return losses