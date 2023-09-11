from dataset import *
from typing import List, Tuple
import torch
from tqdm.auto import tqdm


def train(n_epochs: int,
          dataloader: torch.utils.data.DataLoader,
          gen_opt: torch.optim,
          disc_opt: torch.optim,
          loss_fn: torch.nn.Module,  # check this :)
          losses: List,
          gen: torch.nn.Module,
          disc: torch.nn.Module,
          pixel_dis_lambda: float, # check how to specify for 2 or more things i.e., int, float. but we don' specify like -> int, float (some kind or error while runtime not compile time)
          device: torch.device,
          ):
    
    gen_losses, disc_losses = [], []
    for epoch in tqdm(range(n_epochs)):
        gen.train()
        disc.train()

        batch_no = 0
        for s_images, r_images in tqdm(dataloader):
            s_images, r_images = s_images.to(device), r_images.to(device)

            # Discriminator
            disc_opt.zero_grad()
            fake_images = gen(s_images)
            disc_fake_pred = disc(fake_images)
            disc_real_pred = disc(r_images)

            disc_loss = loss_fn(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_loss += loss_fn(disc_real_pred, torch.ones_like(disc_real_pred))

            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Generator
            gen_opt.zero_grad()
            # make sure to recalculate disc_fake_pred, without recalculating this error.
            disc_fake_pred = disc(fake_images)
            gen_loss = loss_fn(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss += pixel_dis_lambda * torch.mean(torch.abs(fake_images - r_images))

            gen_loss.backward()
            gen_opt.step()

            # stats
            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

            batch_no += 1
            if (batch_no>=115):
                break
        
        # stat per mini-batch
        # gen_losses = sum(gen_losses)/len(gen_losses)
        # disc_losses = sum(disc_losses)/len(disc_losses)
    
    losses.extend((gen_losses, disc_losses))
    return losses


# THE ERROR WAS IN THE LOSS FUNCTION
# THE LOSS SHOULD BE A SINGLE VLAUE NOT A ARRAY/MATRIX OF VALUES


# -----------------------------------------------------------------------


# IDK ABOUT WHETHER THE BCELOGITLOSS ONE OF CORRECT IS NOT 
# BUT HERE IS SOME DEMONSTRATION OR TESTING OF THE SAME 
# THE ANSWERS ARE NOT EXACTLY SAME BUT RELLY CLOSE 
# I..E, 0.5075 AND 0.5385

# loss_fn = nn.BCEWithLogitsLoss()
# output = torch.randn((1, 1, 34, 34))
# predictions = torch.randn((1, 1, 34, 34))
# loss = loss_fn(predictions, output)
# print(loss.item())
# print("------------")

# predictions = torch.tensor([
#     0.5, 0.2, 0.6, 0.1
# ]).view(2, 2)

# output = torch.tensor([
#     1.0, 1, 1, 1
# ]).view(2, 2)

# print(predictions, output)
# loss = loss_fn(predictions, output)
# print(loss.item())
# print("demomstration of the loss calculation..??..")

# print(f"The predictions are -> {predictions}")
# print(f"The Outputs are -> {output}")
# print(f"The softmax are -> {torch.softmax(predictions, dim=1)}")

# l1 = loss_fn(torch.tensor([0.5]), torch.tensor([1.0]))
# l2 = loss_fn(torch.tensor([0.5]), torch.tensor([1.0]))
# l3 = loss_fn(torch.tensor([0.6]), torch.tensor([1.0]))
# l4 = loss_fn(torch.tensor([0.1]), torch.tensor([1.0]))

# print("the respective losses are:\n")
# tl = 0.0
# for a in [l1, l2, l3, l4]:
#     tl += a.item()
#     print(f"l1 {a.item()}")
# print(f"mine loss -> {tl/4}")
# print(f"the actual -> {loss_fn(predictions, output).item()}")