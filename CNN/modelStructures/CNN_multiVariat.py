import torch
from torch import nn


class CNN_Model(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, input_size, anz_channals):
        super(CNN_Model, self).__init__()
        kernel_size = 3
        OUT_CHANNELS = 8
        # chanal = 1 bc no colors;
        self.conv_layer1 = nn.Conv2d(in_channels=anz_channals, out_channels=OUT_CHANNELS, kernel_size=kernel_size,
                                     padding=1, dilation=1)
        self.conv_layer2 = nn.Conv2d(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS * 2, kernel_size=kernel_size,
                                     padding=1, dilation=1)
        # self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size-1, stride=1) #dimensions reduziert -1
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)  # dimensions / 2; wenn kernelsize 2

        self.conv_layer3 = nn.Conv2d(in_channels=OUT_CHANNELS * 2, out_channels=OUT_CHANNELS * 2 * 2,
                                     kernel_size=kernel_size, padding=1, dilation=1)
        self.conv_layer4 = nn.Conv2d(in_channels=OUT_CHANNELS * 2 * 2, out_channels=OUT_CHANNELS * 2 * 2 * 2,
                                     kernel_size=kernel_size, padding=1, dilation=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size-1, stride=1) #dimensions reduziert -1
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size - 1, stride=2)  # int(dimensions / 2); wenn kernelsize 2
        # final out_channel_dim * reduced img dim (by kernel, pooling)
        '''
        in dieser architektur: 
            conv_layer: padding=1; dilation=1; stride=1 img-dim stay the same
            max_pool:   dimension wird um 1 reduziert. Bei Pool mit kernel_size=2 und stride=2 halbiert
        '''
        finalOutChannels = OUT_CHANNELS * 2 * 2 * 2
        new_img_dimensions = int(int(input_size / 2) / 2)
        self.fc1 = nn.Linear(finalOutChannels * new_img_dimensions * new_img_dimensions, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    # Progresses data across layers
    def forward(self, x):
        # print('lvl: 1')
        # print(x.size())
        # torch.Size([1=batch_size, anz_feat, tsl, tsl]) => 1=batch 1=chanal 30=width; 30=hight
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        # print('lvl: 2')
        # print(out.size())
        out = self.max_pool1(out)
        # print('lvl: 3')
        # print(out.size())
        out = self.conv_layer3(out)
        # print('lvl: 4')
        # print(out.size())
        out = self.conv_layer4(out)
        # print('lvl: 5')
        # print(out.size())
        out = self.max_pool2(out)
        # print('lvl: 6')
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # print('lvl: 7')
        # print(out.size())
        out = self.fc1(out)
        # print('lvl: 8')
        # print(out.size())
        out = self.relu1(out)
        out = self.fc2(out)
        # print(out)
        return out
