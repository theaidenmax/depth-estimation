import torch
import torch.nn as nn
import torchvision.models as models 

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class ResNetUNetDepth(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 128x128
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 # 64x64
        self.layer2 = resnet.layer2 # 32x32
        self.layer3 = resnet.layer3 # 16x16
        self.layer4 = resnet.layer4 # 8x8

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoder
        x0 = self.layer0(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # decoder 
        d4 = torch.cat([self.up(x4), x3], dim=1)
        d4 = self.dec4(d4)

        d3 = torch.cat([self.up(d4), x2], dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat([self.up(d3), x1], dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat([self.up(d2), x0], dim=1)
        d1 = self.dec1(d1)

        out = self.up(d1)
        depth = self.final_conv(out)

        return depth

def get_depth_model(pretrained=True):
    return ResNetUNetDepth(pretrained=pretrained)

