from model import *
from utils import *
from tqdm.auto import tqdm


def train(n_epochs: int,
          gen: nn.Module,
          disc: nn.Module,
          dataloader: torch.utils.data.DataLoader,
          gen_opt: torch.optim,
          disc_opt: torch.optim,
          device: torch.device,
          loss_fn: torch.nn.Module,
          steps: int = NUM_BLOCKS):

    gen.train()
    disc.train()
    # loss per step + epoch
    gen_losses_se, disc_losses_se = [], []

    # loss per epoch
    gen_losses_e, disc_losses_e = [], []

    # loss per batch
    gen_losses_b, disc_losses_b = [], []
    for epoch in tqdm(range(n_epochs)):
        for images, labels in tqdm(dataloader):

            images, labels = images.to(device), labels.to(device)
            noise = get_noise(images.shape[0], Z_DIM, 1, 1).to(device)
            alpha = 0.2
            for step in tqdm(range(steps)):

                # training discriminator
                disc_opt.zero_grad()
                fake_images = gen(noise, step=step, alpha=alpha)
                disc_pred_fake = disc(fake_images, step=step)

                # first resize the real_image to match the shape of fake_image
                images = F.interpolate(
                    images, fake_images.shape[2:], mode="nearest")
                disc_pred_real = disc(images, step=step)

                disc_loss = loss_fn(
                    disc_pred_real, torch.ones_like(disc_pred_real))
                disc_loss += loss_fn(disc_pred_fake,
                                     torch.ones_like(disc_pred_fake))

                disc_losses_se.append(disc_loss.item())
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                # training generator
                gen_opt.zero_grad()
                fake_images = gen(noise, step=step, alpha=alpha)
                disc_pred_fake = disc(fake_images, step=step)

                gen_loss = loss_fn(
                    disc_pred_fake, torch.ones_like(disc_pred_fake))
                gen_losses_se.append(gen_loss.item())
                gen_loss.backward()
                gen_opt.step()

            gen_losses_b.append(sum(gen_losses_se)/steps)
            disc_losses_b.append(sum(disc_losses_se)/steps)

        gen_losses_e.append(sum(gen_losses_b)/len(gen_losses_b))
        gen_losses_e.append(sum(disc_losses_b)/len(disc_losses_b))

    return (gen_losses_se, disc_losses_se,
            gen_losses_e, disc_losses_e,
            gen_losses_b, disc_losses_b)
