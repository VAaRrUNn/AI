import torch
import torch.nn as nn



# -------------------------- UNET -----------------------------------

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
        if skip_connections:
            self.skip_connection_layer.append(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64,
                 kernel_size=3, stride=1, padding=0,
                 blocks=4, segmented_image_channels=3):

        super(UNet, self).__init__()

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
            CNNBlock(conv_out*2, segmented_image_channels,
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
            out = torch.cat(
                (out, self.skip_connection_layers[skip_len - 1 - i]), dim=1)
            # print(out.shape)
            out = block(out)
            conv_in = conv_out
            conv_out //= 2

        return out