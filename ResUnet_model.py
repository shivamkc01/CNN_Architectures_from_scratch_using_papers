"""
Implementation of original Road Extraction by Deep Residual U-Net paper from scratch using pytorch
use it for learning purpose.

Programmed by Shivam Chhetry
** 14-Aug-2021
"""

import torch
import torch.nn as nn

class batchnorm_relu(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class Decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = Residual_block_encode1_encode2(in_channel+out_channel, out_channel)
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        # print(x.shape, skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x


class ResUnet_model(nn.Module):
    def __init__(self):
        super(ResUnet_model, self).__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = Residual_block_encode1_encode2(64, 128, stride=2)
        self.r3 = Residual_block_encode1_encode2(128, 256, stride=2)

        """ Bridge """
        self.r4 = Residual_block_encode1_encode2(256, 512, stride=2)

        """ Decoder """
        self.d1 = Decoder_block(512, 256)
        self.d2 = Decoder_block(256, 128)
        self.d3 = Decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3"""
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)
        # print(d3.shape)

        """ output """
        output = self.output(d3)
        output = self.sigmoid(output)

        return output



class Residual_block_encode1_encode2(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        """ Convolutional layer"""
        self.b1 = batchnorm_relu(in_channel=in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.conv1(x)
        x = self.b2(x)
        x = self.conv2(x)
        s = self.s(inputs)
        skip2 = x + s
        return skip2


if __name__ == '__main__':
    inputs = torch.randn((4, 3, 256, 256))
    model = ResUnet_model()
    y = model(inputs)
    print(y.shape)
