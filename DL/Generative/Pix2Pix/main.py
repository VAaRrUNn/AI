from model import *
from dataset import *
from train import *
from time import time


def calculate_time():
    """
    Make a function so that use can use it as a wrapper function
    to calculate time :).
    """
    pass


losses = train(n_epochs,
      dataloader,
      gen_opt,
      disc_opt,
      loss_fn,
      losses,
      gen,
      disc,
      pixel_dis_lambda,
      device,
      )

# visualization

# saving model weights
torch.save(obj=gen.state_dict(), f="pix2pix_1_gen.pth")
torch.save(obj=disc.state_dict(), f="pix2pix_1_disc.pth")
