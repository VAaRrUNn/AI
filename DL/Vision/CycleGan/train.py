# from model import *
import torch
from tqdm.auto import tqdm

CYCLE_LOSS_LAMBDA = 10

# If some error occurs.. btw the errors tell to turn it on anyways :)
# torch.autograd.set_detect_anomaly(True)


def train_(img_a: torch.Tensor,
           img_b: torch.Tensor,
           loss_fn: torch.nn.Module,
           gen_a: torch.nn.Module,
           gen_b: torch.nn.Module,
           disc_a: torch.nn.Module,
           gen_opt_a: torch.optim,
           gen_opt_b: torch.optim,
           disc_opt_a: torch.optim,):
    """
    Utility function to train a GAN.
    """

    disc_opt_a.zero_grad()
    gen_opt_a.zero_grad()
    gen_opt_b.zero_grad()

    fake_images = gen_a(img_a)
    disc_a_fake_pred = disc_a(fake_images)

    # to reuse this prediction again. we have to clone() it
    # cause backprop will just change it inplace
    fake_pred = disc_a_fake_pred.detach().clone()
    disc_a_real_pred = disc_a(img_b)

    disc_loss_a = loss_fn(
        disc_a_fake_pred, torch.zeros_like(disc_a_fake_pred))
    disc_loss_a += loss_fn(
        disc_a_real_pred, torch.ones_like(disc_a_real_pred))

    disc_loss_a.backward(retain_graph=True)
    disc_opt_a.step()

    # print(f"The disc loss -> {disc_loss_a.item()}")

    # Generator training
    # fake_images = gen_a(img_a)
    # disc_a_fake_pred = disc_a(fake_images)
    gen_loss_a = loss_fn(
        fake_pred, torch.ones_like(fake_pred))
    # Cycle loss for gen a
    fake_original_images = gen_b(fake_images)
    cycle_loss_a = CYCLE_LOSS_LAMBDA * \
        torch.mean(torch.abs(fake_original_images - img_a))

    gen_loss_a += cycle_loss_a

    # print(f"The gen loss -> {gen_loss_a.item()}")

    gen_loss_a.backward(retain_graph=False)

    gen_opt_a.step()
    gen_opt_b.step()
    disc_loss_a.detach()

    gen_loss_a.detach()


def train(n_epochs: int,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          gen_a: torch.nn.Module,
          gen_b: torch.nn.Module,
          disc_a: torch.nn.Module,
          disc_b: torch.nn.Module,
          gen_opt_a: torch.optim,
          gen_opt_b: torch.optim,
          disc_opt_a: torch.optim,
          disc_opt_b: torch.optim,
          device: torch.device
          ):

    for epoch in tqdm(range(n_epochs)):
        for img_a, img_b in tqdm(dataloader):
            img_a, img_b = img_a.to(device), img_b.to(device)

            # Training GAN A
            # print(f"training generator A....")
            train_(img_a=img_a,
                   img_b=img_b,
                   loss_fn=loss_fn,
                   gen_a=gen_a,
                   gen_b=gen_b,
                   disc_a=disc_a,
                   gen_opt_a=gen_opt_a,
                   gen_opt_b=gen_opt_b,
                   disc_opt_a=disc_opt_a)

            # print(f"Generator Trained")
            # Training GAN B
            train_(img_a=img_b,
                   img_b=img_a,
                   loss_fn=loss_fn,
                   gen_a=gen_b,
                   gen_b=gen_a,
                   disc_a=disc_b,
                   gen_opt_a=gen_opt_b,
                   gen_opt_b=gen_opt_a,
                   disc_opt_a=disc_opt_b)
