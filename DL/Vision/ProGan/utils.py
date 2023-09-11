from time import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch


def calculate_time(func):
    def wrap(*args, **kwargs):
        start = time()
        out = func(*args, **kwargs)
        end = time()
        print(f"Time taken : {(end-start):.6f}")
        return out
    return wrap


def visualize(images, nrow=30):
    plt.figure(figsize=(20, 20))
    images = make_grid(images, nrow=1)
    plt.imshow(images.permute(1, 2, 0))
    plt.axis("off")
    plt.grid("off")

def get_noise(*args):
  return torch.randn(args)
