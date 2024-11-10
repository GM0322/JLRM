import torch
import numpy as np
from collections import OrderedDict

class GroupUNet(torch.nn.Module):
    """ U-Net implementation.

    Based on https://github.com/mateuszbuda/brain-segmentation-pytorch/
    and modified in agreement with their licence:

    -----

    MIT License

    Copyright (c) 2019 mateuszbuda

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_features=32,
        drop_factor=0.0,
        do_center_crop=False,
        num_groups=32,
    ):
        # set properties of UNet
        super(GroupUNet, self).__init__()

        self.do_center_crop = do_center_crop
        kernel_size = 3 if do_center_crop else 2

        self.encoder1 = self._conv_block(
            in_channels,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(
            base_features,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._conv_block(
            base_features * 2,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(
            base_features * 4,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._conv_block(
            base_features * 8,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            base_features * 16,
            base_features * 8,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder4 = self._conv_block(
            base_features * 16,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder3 = self._conv_block(
            base_features * 8,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=kernel_size,
            stride=2,
        )
        self.decoder2 = self._conv_block(
            base_features * 4,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_features * 2, base_features, kernel_size=kernel_size, stride=2
        )
        self.decoder1 = self._conv_block(
            base_features * 2,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))

        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self._center_crop(dec4, enc4.shape[-2], enc4.shape[-1])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop(dec3, enc3.shape[-2], enc3.shape[-1])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2.shape[-2], enc2.shape[-1])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1.shape[-2], enc1.shape[-1])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    def _conv_block(
        self, in_channels, out_channels, num_groups, drop_factor, block_name
    ):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )

    def _center_crop(self, layer, max_height, max_width):
        if self.do_center_crop:
            _, _, h, w = layer.size()
            xy1 = (w - max_width) // 2
            xy2 = (h - max_height) // 2
            return layer[
                :, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)
            ]
        else:
            return layer

class guass_filter_model(torch.nn.Module):
    def __init__(self):
        super(guass_filter_model, self).__init__()
        ks = 11
        self.filter = torch.nn.Conv2d(1, 1, kernel_size=(ks, ks), padding=(ks//2, ks//2), bias=False).cuda()
        x = np.array([i-ks//2 for i in range(ks)], dtype=np.float32)
        Ix, Iy = np.meshgrid(x, x)
        sigma = 1.0
        kernel = 1 / (2 * np.pi * sigma) * np.exp(-(Ix ** 2 + Iy ** 2) / (2 * sigma ** 2))
        kernel = torch.from_numpy(kernel / np.sum(kernel)).view(1, 1, ks, ks).cuda()
        kernel = torch.nn.Parameter(kernel)
        self.filter.weight = kernel

    def forward(self,x):
        out = []
        for i in range(x.shape[1]):
            out.append(self.filter(x[:,i:i+1,...]))
        return torch.cat(out,dim=1)

class basicnetwork(torch.nn.Module):
    def __init__(self,in_channels=2, out_channels=1,base_features=4, drop_factor=0.0,num_groups=4):
        super().__init__()
        kernel_size = 3
        self.encoder1 = self._conv_block(
            in_channels,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_1",
        )
        self.encoder2 = self._conv_block(
            base_features,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_2",
        )
        self.encoder3 = self._conv_block(
            base_features * 2,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_3",
        )
        self.encoder4 = self._conv_block(
            base_features * 4,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="encoding_4",
        )
        self.bottleneck = self._conv_block(
            base_features * 8,
            base_features * 16,
            num_groups,
            drop_factor=drop_factor,
            block_name="bottleneck",
        )

        self.upconv4 = torch.nn.Conv2d(
            base_features * 16,
            base_features * 8,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
        )
        self.decoder4 = self._conv_block(
            base_features * 16,
            base_features * 8,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_4",
        )
        self.upconv3 = torch.nn.Conv2d(
            base_features * 8,
            base_features * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
        )
        self.decoder3 = self._conv_block(
            base_features * 8,
            base_features * 4,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_3",
        )
        self.upconv2 = torch.nn.Conv2d(
            base_features * 4,
            base_features * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
        )
        self.decoder2 = self._conv_block(
            base_features * 4,
            base_features * 2,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_2",
        )
        self.upconv1 = torch.nn.Conv2d(
            base_features * 2, base_features, kernel_size=kernel_size,
            stride=1,padding=kernel_size//2,
        )
        self.decoder1 = self._conv_block(
            base_features * 2,
            base_features,
            num_groups,
            drop_factor=drop_factor,
            block_name="decoding_1",
        )

        self.outconv = torch.nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=kernel_size//2,
        )

    def forward(self, x):

        enc1 = self.encoder1(x)

        enc2 = self.encoder2(enc1)

        enc3 = self.encoder3(enc2)

        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        # dec4 = self._center_crop(dec4, enc4.shape[-2], enc4.shape[-1])
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        # dec3 = self._center_crop(dec3, enc3.shape[-2], enc3.shape[-1])
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        # dec2 = self._center_crop(dec2, enc2.shape[-2], enc2.shape[-1])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        # dec1 = self._center_crop(dec1, enc1.shape[-2], enc1.shape[-1])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.outconv(dec1)

    def _conv_block(self, in_channels, out_channels, num_groups, drop_factor, block_name):
        return torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        block_name + "conv1",
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_1",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu1", torch.nn.ReLU(True)),
                    (block_name + "dr1", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                    (
                        block_name + "conv2",
                        torch.nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        block_name + "bn_2",
                        torch.nn.GroupNorm(num_groups, out_channels),
                    ),
                    (block_name + "relu2", torch.nn.ReLU(True)),
                    (block_name + "dr2", torch.nn.Dropout(p=drop_factor)),
                ]
            )
        )

class network(torch.nn.Module):
    def __init__(self,in_channel=2,out_channel=1):
        super().__init__()
        self.pre = torch.nn.Conv2d(in_channel,16,kernel_size=(3,3),padding=(1,1),stride=(2,2))
        self.post = torch.nn.Conv2d(16,out_channel,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.conv = basicnetwork(16,16)
        # self.conv = GroupUNet(16,16)
        self.relu = torch.nn.ReLU()

    def forward(self,input,train=True):
        x = self.pre(input)
        out = 1.1*self.relu(self.post(self.conv(x)))
        if train:
            out[out>1.0] = 1.0
        else:
            out[out>0.5] = 1.0
            out[out<=0.5] = 0.0
        gfilter = guass_filter_model().cuda(input.device)
        out_ = gfilter(out)
        return out_,out

class network2(torch.nn.Module):
    def __init__(self,in_channel=2,out_channel=1):
        super().__init__()
        self.pre = torch.nn.Conv2d(in_channel,16,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.post = torch.nn.Conv2d(16,out_channel,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.conv = basicnetwork(16,16)
        # self.conv = GroupUNet(16,16)
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        x = self.pre(input)
        out = 1.1*self.relu(self.post(self.conv(x)))
        out[out>1.0] = 1.0
        gfilter = guass_filter_model().cuda(input.device)
        out_ = gfilter(out)
        return out_,out

class basic2network(torch.nn.Module):
    def __init__(self,in_channels=16,out_channels=16):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.depth = 4
        for i in range(self.depth):
            self.layers.append(self.conv_block(in_channels*2**i,in_channels*2**(i+1)))
        self.layers.append((self.conv_block(in_channels*2**(self.depth),in_channels*2**(self.depth))))
        for i in range(self.depth):
            self.layers.append(self.conv_block(in_channels*2**(4-i+1),in_channels*2**(4-i-1)))
        self.outConv = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False)

    def forward(self,input):
        input = [input]
        for i in range(self.depth):
            input.append(self.layers[i](input[i]))
        out = self.layers[self.depth](input[self.depth])
        for i in range(self.depth):
            out = torch.cat([input[self.depth-i],out],dim=1)
            out = self.layers[i+self.depth+1](out)
        return self.outConv(out)

    def conv_block(self,in_channels,out_channels):
        conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels,affine=False),
            torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            torch.nn.ReLU(),
        )
        return conv
class convBlock(torch.nn.Module):
    def __init__(self,channels=16):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channels,channels,kernel_size=(3,3),padding=(1,1),stride=(1,1)),
            torch.nn.BatchNorm2d(channels, affine=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(channels, affine=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            torch.nn.ReLU(),
        )
    def forward(self,input):
        return self.conv(input)

class resassignnetwork(torch.nn.Module):
    def __init__(self,in_channel=6,out_channel=1):
        super().__init__()
        self.pre = torch.nn.Conv2d(in_channel,16,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.post = torch.nn.Conv2d(16,2,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        # self.conv = basic2network(16,16)
        self.conv = convBlock(16)
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        # x.data.cpu().numpy().tofile('1.raw')
        # input.data.cpu().numpy().tofile('0.raw')
        x = self.pre(input)
        out = self.relu(self.post(self.conv(x)))
        # out[out > 1.0] = 1.0
        # out[out <=1.0] = 0.0
        out[out>=0.1] = 1.0
        gfilter = guass_filter_model().cuda(input.device)
        out_ = gfilter(out)
        return out,out_

class resassignnetwork2(torch.nn.Module):
    def __init__(self,in_channel=2,out_channel=1):
        super().__init__()
        self.pre = torch.nn.Conv2d(in_channel,16,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.post = torch.nn.Conv2d(16,1,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.conv = basic2network(16,16)
        # self.conv = convBlock(16)
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        # x.data.cpu().numpy().tofile('1.raw')
        # input.data.cpu().numpy().tofile('0.raw')
        x = self.pre(input)
        out = self.post(self.conv(x))
        # out[out > 1.0] = 1.0
        # out[out <=1.0] = 0.0
        out[out>=0.5] = 1.0
        out[out<0.1] = 0.0
        gfilter = guass_filter_model().cuda(input.device)
        out_ = gfilter(out)
        # out_ = out
        return out,out_
