"""
Implementing Original U-Net: Convolutional Networks for
Biomedical Image Segmentation form scratch using PyTorch.

Programmed by Shivam Chhetry
** 12-08-2021
"""
# imports
import torch
import torch.nn as nn


# This we are creating a double convolutional network base
def double_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


""" To help in concatenating with skip connections"""


def crop_tensor_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    '''Here [2] mean our img shape is [1, 512, 56, 56],
      so we are dealing with image height '''
    delta = tensor_size - target_size  # Here we are assuming tensor_size is always > target_size
    delta = delta // 2
    return tensor[:, :, delta: tensor_size - delta, delta:tensor_size - delta]


class UNet(nn.Module):
    def __init__(self, ):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        '''
        we have 5 double convolution downward->
            64, 128, 256, 512, 1024
        '''
        """ Down-Sampling (Contracting path) """
        self.down_conv_1 = double_conv(1, 64)  # (input_channel, out_channel) -> (1,64)
        self.down_conv_2 = double_conv(64, 128)  # (input_channel, out_channel) -> (64, 128)
        self.down_conv_3 = double_conv(128, 256)  # (input_channel, out_channel) -> (128, 256)
        self.down_conv_4 = double_conv(256, 512)  # (input_channel, out_channel) -> (256, 512)
        self.down_conv_5 = double_conv(512, 1024)  # (input_channel, out_channel) -> (512, 1024)

        """ Up-Sampling (expansive path) """
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2
                                             )
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2,
                                             )
        self.up_conv_4 = double_conv(128, 64)
        self.out_conv = nn.Conv2d(
            in_channels=64,
            out_channels=2,  # according to the paper.
            kernel_size=1,
        )

    def forward(self, img):
        # (batch_size, channel, height, width)
        """Encoder"""
        x1 = self.down_conv_1(img)  # skip
        # print(x1.size()) for checking/testing
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)  # skip
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)  # skip
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)  # skip
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        x10 = self.max_pool_2x2(x9)
        # print(x9.size())  # for checking/testing
        """
                x.size =  torch.Size([1, 512, 56, 56])
                x7.size = torch.Size([1, 512, 64, 64])
                both image is having different height and width so we can't concatenate
                directly we need to do either add padding to "x" to make it 64, 64 or 
                crop "x7" to make it 56, 56 then we concatenate.
                        Here x7 is a skip connections.
                Above I have created a crop function because in U-Net paper they use crop
        """

        """Decoder"""
        x = self.up_trans_1(x9)
        y = crop_tensor_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))  # -> torch.Size([1, 512, 52, 52])
        x = self.up_trans_2(x)
        y = crop_tensor_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        x = self.up_trans_3(x)
        y = crop_tensor_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        x = self.up_trans_4(x)
        y = crop_tensor_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))  # -> torch.Size([1, 2, 388, 388])
        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    print(model(image))
