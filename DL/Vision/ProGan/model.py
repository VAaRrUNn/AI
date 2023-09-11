import torch
import torch.nn as nn
import torch.nn.functional as F

FACTORS = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
NUM_BLOCKS = 9
Z_DIM = 512
IMAGE_CHANNELS = 3
IM_CHANNELS = 1024


class EqualizedConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 gain=2):
        super(EqualizedConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initializing
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # + self.bias.view(-1, self.bias.shape[0], 1, 1)
        return self.conv(x * self.scale)


class PixelNorm(nn.Module):

    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epislon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epislon)


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pixel_normalization=False,
                 gain=2,
                 model="Generator"):
        super(ConvBlock, self).__init__()

        self.model = model
        self.conv1_gen = EqualizedConvLayer(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            gain=gain)

        self.conv1_disc = EqualizedConvLayer(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             gain=gain)

        self.conv2_gen = EqualizedConvLayer(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            gain=gain)

        self.conv2_disc = EqualizedConvLayer(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             gain=gain)

        self.leaky = nn.LeakyReLU(0.01)
        self.apply_px = pixel_normalization
        self.px = PixelNorm()

    def forward(self, x):
        if self.model == "Generator":
            out = self.leaky(self.conv1_gen(x))
            out = self.px(out) if self.apply_px else out
            out = self.leaky(self.conv2_gen(out))
            out = self.px(out) if self.apply_px else out
            return out

        if self.model == "Discriminator":
            out = self.leaky(self.conv1_disc(x))
            out = self.px(out) if self.apply_px else out
            out = self.leaky(self.conv2_disc(out))
            out = self.px(out) if self.apply_px else out
            return out


# we have 7 blocks for 512/ 572 dimensional image


class Generator(nn.Module):

    def __init__(self,
                 in_channels=512,
                 im_channels=3,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 gain=2,
                 pixel_normalization=True,
                 num_blocks=9):  # num_blocks is 8 for 512 image, 9->1024, and so on..
        """
        in_channels -> nosie channels
        """
        super(Generator, self).__init__()

        self.num_blocks = num_blocks
        # 4, 2, 1 for doubling, and 4, 1, 0 for 1->4
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=4,
                               stride=1,
                               padding=0),
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(0.01),
        )

        self.conv_blocks, self.to_rgb = nn.ModuleList(), nn.ModuleList()

        self.to_rgb.append(EqualizedConvLayer(in_channels=in_channels,
                                              out_channels=im_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        for i in range(num_blocks-1):
            conv_in = int(in_channels * FACTORS[i])
            # make sure it be int otherwise error :(
            conv_out = int(in_channels * FACTORS[i+1])
            self.conv_blocks.append(ConvBlock(in_channels=conv_in,
                                              out_channels=conv_out,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              pixel_normalization=pixel_normalization,
                                              gain=gain))
            self.to_rgb.append(EqualizedConvLayer(in_channels=conv_out,
                                                  out_channels=im_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))

    def fade_in(self, upscaled, out, alpha):
        return torch.tanh(alpha * upscaled + (1-alpha) * out)

    def forward(self, x, step=0, alpha=0.5):

        if step > self.num_blocks:
            raise IndexError(
                f"The step can't be greater than {self.num_blocks}")

        out = self.initial_layer(x)

        if step == 0:
            return self.to_rgb[step](out)

        for i in range(step):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.conv_blocks[i](upscaled)

        upscaled = self.to_rgb[step-1](upscaled)
        out = self.to_rgb[step](out)

        return self.fade_in(upscaled, out, alpha)


class Discriminator(nn.Module):

    def __init__(self,
                 noise_channels=512,  # the channels/ z_dim of noise
                 in_channels=1024,  # The input size i.e., for what image shape are you replicating
                 im_channels=3,  # the channels of this input image
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 gain=2,
                 pixel_normalization=False,
                 num_blocks=9):

        self.num_blocks = num_blocks
        super(Discriminator, self).__init__()
        self.blocks, self.from_rgb = nn.ModuleList(), nn.ModuleList()

        FACTORS = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
        self.from_rgb.append(nn.Conv2d(in_channels=im_channels,
                                       out_channels=int(
                                           noise_channels * FACTORS[num_blocks-1]),
                                       kernel_size=1,
                                       stride=1,
                                       padding=0))

        for i in range(num_blocks-1, 0, -1):
            conv_in = int(noise_channels * FACTORS[i])
            conv_out = int(noise_channels * FACTORS[i-1])
            self.blocks.append(ConvBlock(in_channels=conv_in,
                                         out_channels=conv_out,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         pixel_normalization=pixel_normalization,
                                         model="Discriminator"))
            self.from_rgb.append(EqualizedConvLayer(in_channels=im_channels,
                                                    out_channels=conv_out))

        conv_in = noise_channels
        conv_out = noise_channels

        # shrinks size by half
        self.downsample = nn.MaxPool2d(kernel_size=2,
                                       stride=2)

        self.last_from_rgb = EqualizedConvLayer(
            in_channels=im_channels,
            out_channels=conv_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.last_layer = nn.Sequential(
            EqualizedConvLayer(in_channels=conv_in,
                               out_channels=conv_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            EqualizedConvLayer(in_channels=conv_out,
                               out_channels=conv_out,
                               kernel_size=4,
                               stride=1,
                               padding=0),
            nn.Flatten(),
            nn.Linear(in_features=conv_out, out_features=1),
        )

    def forward(self, x, step=0, alpha=0.5):
        out = x
        if step == 0:
            out = self.last_from_rgb(out)
            out = self.last_layer(out)
            return out
        out = self.from_rgb[self.num_blocks - 1 - step](out)
        while(step > 0):
            # print(out.shape)
            out = self.blocks[self.num_blocks - 1 - step](out)
            # print(out.shape)
            out = self.downsample(out)
            step -= 1

        out = self.last_layer(out)
        return out


gen = Generator(in_channels=Z_DIM,
                im_channels=IMAGE_CHANNELS,
                num_blocks=NUM_BLOCKS)

disc = Discriminator(noise_channels=Z_DIM,
                     in_channels=IM_CHANNELS,
                     im_channels=IMAGE_CHANNELS,
                     num_blocks=NUM_BLOCKS)
