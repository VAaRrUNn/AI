from model import *
CYCLE_LOSS_LAMBDA = 10


def train_(img_a: torch.Tensor,
           img_b: torch.Tensor,
           loss_fn: torch.nn.Module,
           gen_a: torch.nn.Module,
           gen_b: torch.nn.Module,
           disc_a: torch.nn.Module,
           gen_opt_a: torch.optim,
           gen_opt_b: torch.optim,
           disc_opt_a: torch.optim,):

    disc_opt_a.zero_grad()
    gen_opt_a.zero_grad()

    fake_images = gen_a(img_a)
    disc_a_fake_pred = disc_a(fake_images)
    disc_a_real_pred = disc_a(img_b)

    disc_loss_a = loss_fn(
        disc_a_fake_pred, torch.zeros_like(disc_a_fake_pred))
    disc_loss_a += loss_fn(
        disc_a_real_pred, torch.zeros_like(disc_a_real_pred))

    disc_loss_a.backward(retain_graph=True)

    # Generator training
    gen_loss_a = loss_fn(
        disc_a_fake_pred, torch.ones_like(disc_a_fake_pred))
    # Cycle loss for gen a
    fake_original_images = gen_b(fake_images)
    cycle_loss_a = CYCLE_LOSS_LAMBDA * \
        torch.mean(torch.abs(fake_original_images - img_a))

    gen_loss_a += cycle_loss_a

    gen_loss_a.backward()
    gen_opt_a.step()
