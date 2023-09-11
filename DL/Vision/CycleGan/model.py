from dataset import *
import torch
import torch.nn as nn

BLOCKS = 2

# WARNING THIS UNET WORKS FOR SAY SHAPE OF 256, 260 WELL BUT NOT WORKING FOR 258?
# CAUSE THER IS A H, W DIMENSION MISSING I.E., IT ACCEPTS SOMETHING (BATCH_SIZE, CHANNELS, X, X),
# BUT GOT (BATCH_SIZE, CHANNELS, X+1, X+1)
# THE ERROR IS THROWN BY TORCH.CAT (SKIP CONNECTION) I.E, CAUSED BY THIS MISMATCH IN H, AND W
# THIS IS A CAUSE OF CONV LAYERS I.E., DIVIDING SOMETHING LIKE 5/2 -> 2.5(FLOAT)
# RATHER THAN 4/2 -> 2 (INT).
# AS WE ARE CONVERTING FLOAT TO INT, THIS CONVERSION/ ERROR IS THERE


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0,
                 upsample=False):

        super(CNNBlock, self).__init__()
        if not upsample:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
                nn.Conv2d(out_channels, out_channels, kernel_size,
                          stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
            )

        if upsample:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.01),
            )
        self.skip_connection_layer = []

    def forward(self, x, skip_connections=False):
        out = self.layers(x)
        # if skip_connections:
        #     self.skip_connection_layer.append(out)
        return out


class Generator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 blocks=4,
                 output_image_channels=3):

        super(Generator, self).__init__()

        self.downsample = nn.ModuleList()
        conv_in = in_channels
        conv_out = out_channels
        for i in range(blocks):
            self.downsample.append(
                CNNBlock(conv_in, conv_out,
                         kernel_size, stride, padding))
            conv_in = conv_out
            conv_out *= 2

        self.skip_connection_layers = []

        self.upsample = nn.ModuleList()

        # saving for use in upsampling in forward
        self.conv_in = out_channels * (2**(blocks-1))
        self.conv_out = conv_in  # // 2

        conv_in = self.conv_in
        conv_out = self.conv_out
        for i in range(blocks-1):
            self.upsample.append(
                CNNBlock(conv_out*2, conv_out,
                         kernel_size, stride, padding, upsample=True))
            conv_out //= 2

        # this is for the last layer i.e, from 512 channels -> segmented image channels
        self.upsample.append(
            CNNBlock(conv_out*2, output_image_channels,
                     kernel_size, stride, padding, upsample=True))

    def forward(self, x):

        out = x
        # print(out.shape)
        # downsampling
        for i in range(len(self.downsample)):
            block = self.downsample[i]
            out = block(out, skip_connections=True)
            # self.skip_connection_layers.append(
            #     *block.skip_connection_layer
            # )
            # print(out.shape)
            self.skip_connection_layers.append(out)
            out = nn.MaxPool2d(2, 2)(out)

        # print("The shape of output is :")
        # print(out.shape)
        # print("-------------------")
        # upsampling

        skip_len = len(self.skip_connection_layers)
        conv_in = self.conv_in
        conv_out = self.conv_out
        for i in range(len(self.upsample)):
            block = self.upsample[i]
            out = nn.ConvTranspose2d(conv_in, conv_out,
                                     kernel_size=2,
                                     stride=2, padding=0)(out)

            # print(out.shape)
            out = torch.cat(
                (out, self.skip_connection_layers[skip_len - 1 - i]), dim=1)
            # print(out.shape)
            out = block(out)
            conv_in = conv_out
            conv_out //= 2

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 im_channels: int = 3):
        super(Discriminator, self).__init__()

        slope = 0.2
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 0),
            nn.LeakyReLU(slope),
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.LeakyReLU(slope),
            nn.Conv2d(64, 32, 3, 1, 0),
            nn.LeakyReLU(slope),
            nn.Conv2d(32, 32, 4, 1, 0),
            nn.LeakyReLU(slope),
            nn.Conv2d(32, 1, 3, 1, 0),
            nn.LeakyReLU(slope),
        )

    def forward(self, x):
        return self.layer(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

gen_a = Generator(blocks=BLOCKS).to(device=device)
gen_b = Generator(blocks=BLOCKS).to(device=device)

disc_a = Discriminator().to(device=device)
disc_b = Discriminator().to(device=device)

gen_opt_a = torch.optim.Adam(params=gen_a.parameters(), lr=0.001)
disc_opt_a = torch.optim.Adam(params=disc_a.parameters(), lr=0.001)

gen_opt_b = torch.optim.Adam(params=gen_b.parameters(), lr=0.001)
disc_opt_b = torch.optim.Adam(params=disc_b.parameters(), lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()

# one loop
# a = torch.zeros(1, 3, 256, 256)
# gen_opt.zero_grad()
# loss_fn = nn.BCEWithLogitsLoss()
# noise = torch.zeros(2, 3, 256, 256)
# real = torch.randn(2, 3, 256, 256)
# fake = gen(noise)
# fake_pred = disc(fake)
# real_pred = disc(real)
# loss = loss_fn( fake_pred, torch.zeros_like(fake_pred) ) + loss_fn(real_pred, torch.ones_like(real_pred))

# loss.backward(retain_graph=True)
# print("success")
# gen_opt.step()
