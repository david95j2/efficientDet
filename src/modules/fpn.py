import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(num_channels,num_channels,kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.Relu()
        )
    def forward(self, x):
        return self.layer(x)

class FPN(nn.modules):
    def __init__(self, num_channels):
        super().__init__()

        # Conv layers 1
        self.conv7_1 =  ConvBlock(num_channels)
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)

        # Conv layer 2
        self.conv7_2 =  ConvBlock(num_channels)
        self.conv6_2_up = ConvBlock(num_channels)
        self.conv5_2_up = ConvBlock(num_channels)
        self.conv4_2_up = ConvBlock(num_channels)
        self.conv3_2_up = ConvBlock(num_channels)


        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p6_2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        """
        fpn
            P7_0 ---------> P7_1 ---------> P7_2 -------->
                             |                |
                             ↓                ↓
            P6_0 ---------> P6_1 ---------> P6_2 -------->
                             |                |
                             ↓                ↓
            P5_0 ---------> P5_1 ---------> P5_2 -------->
                             |                |
                             ↓                ↓
            P4_0 ---------> P4_1 ---------> P4_2 -------->
                             |                |
                             ↓                ↓
            P3_0 ---------> P3_1 ---------> P3_2 -------->
        """

        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        
        p7_1 = self.conv7_1(p7_in)
        p6_1 = self.conv6_up(p6_in + self.p6_upsample(p7_in)) 
        p5_1 = self.conv5_up(p5_in + self.p5_upsample(p6_1))          
        p4_1 = self.conv4_up(p4_in + self.p4_upsample(p5_1)) 
        p3_1 = self.conv3_up(p3_in + self.p3_upsample(p4_1))

        p7_out = self.conv7_2(p7_1)
        p6_out = self.conv6_2_up(p6_1+self.conv6_2_up(p7_out))
        p5_out = self.conv5_2_up(p5_1+self.conv5_2_up(p6_out))
        p4_out = self.conv4_2_up(p4_1+self.conv4_2_up(p5_out))
        p3_out = self.conv3_2_up(p3_1+self.conv3_2_up(p4_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out