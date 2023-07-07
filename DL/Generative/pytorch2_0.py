from time import time
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST, Caltech101
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
torch.manual_seed(0)  # Set for testing purposes, please do not change!


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.grid(False)
    plt.axis(False)
    plt.show()


def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []

    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim *
                                2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan,
                                kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def combine_vectors(x, y):
    return torch.concat((x.float(), y.float()), dim=1)


def calculate_time(func):
    def wrap(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f"Time elapsed: {(end-start):6f}")
    return wrap


n_epochs = 1
batch_size = 128
lr = 0.002
beta_1 = 0.5
beta_2 = 0.999
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10
z_dim = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

dataloader = DataLoader(
    MNIST('.', download="True",
          transform=transform),
          batch_size = batch_size,
          shuffle=True
)


gen = Generator(z_dim + 10).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))

# this is just hard coded which should not be the case... i.e., the im_chan for the Discriminator
disc = Discriminator(num_classes + 1).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# loss
loss_fn = nn.BCEWithLogitsLoss()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

import torch

# gen_compiler = torch.compile(gen)
# disc_compiler = torch.compiler(disc)

@calculate_time
def train():
    gen_loss, disc_loss = [], []
    n_epochs = 10

    for epoch in tqdm(range(n_epochs)):
        batch = 0
        for images, labels in tqdm(dataloader):
            # moving to right device
            images, labels = images.to(device), labels.to(device)

            # discriminator training
            disc_opt.zero_grad()
            noise = get_noise(images.shape[0], z_dim).to(device)
            one_hot_labels = F.one_hot(labels, num_classes)
            final_noise = combine_vectors(noise, one_hot_labels)

            # for fake_images
            fake_image = gen(final_noise)
            fake_image_label = F.one_hot(labels, num_classes)[:, :, None, None]
            fake_image_label = fake_image_label.repeat(1, 1, 28, 28)
            fake_image = combine_vectors(fake_image, fake_image_label)

            fake_predictions = disc(fake_image)
            fake_loss = loss_fn(
                fake_predictions, torch.zeros_like(fake_predictions))

            # for real_images
            real_image = images
            real_image_label = F.one_hot(labels, num_classes)[:, :, None, None]
            real_image_label = real_image_label.repeat(1, 1, 28, 28)
            real_image = combine_vectors(real_image, real_image_label)

            real_predictions = disc(real_image)
            real_loss = loss_fn(
                real_predictions, torch.ones_like(real_predictions))

            loss = (real_loss + fake_loss) / 2

            disc_loss.append(loss.item())
            loss.backward()
            disc_opt.step()

            # generator training
            gen_opt.zero_grad()
            noise = get_noise(images.shape[0], z_dim).to(device)
            one_hot_labels = F.one_hot(labels, num_classes)
            final_noise = combine_vectors(noise, one_hot_labels)

            # for fake_images
            fake_image = gen(final_noise)
            fake_image_label = F.one_hot(labels, num_classes)[:, :, None, None]
            fake_image_label = fake_image_label.repeat(1, 1, 28, 28)
            fake_image = combine_vectors(fake_image, fake_image_label)

            fake_predictions = disc(fake_image)
            loss = loss_fn(fake_predictions, torch.ones_like(fake_predictions))

            gen_loss.append(loss.item())
            loss.backward()
            gen_opt.step()

            batch += 1
            if batch >=10:
                break

        print(f"The Discriminator loss is {sum(gen_loss)/len(gen_loss)}")
        print(f"The Generator loss is {sum(disc_loss)/len(disc_loss)}")
        with torch.no_grad():
            images_to_show = 25
            noise = get_noise(images_to_show, z_dim)

            # generating images for 2nd class i.e., for digit 1
            noise = combine_vectors(noise, F.one_hot(torch.ones(
                images_to_show, dtype=torch.int64), num_classes)).to(device)
            images = gen(noise)
            # images = images.to("cpu")
            show_tensor_images(images, images_to_show)

train()