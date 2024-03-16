import torch
from torch import nn
from torch.nn import functional as F
from model.utils import Downsample


class Discriminator(nn.Module):
    # ir + vi concat
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(1, 64)
        self.down2 = Downsample(64, 128)
        self.conv1 = nn.Conv2d(128, 256, 3)
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, 3)

    def forward(self, vi):
        # x = torch.cat([ir,vi],axis = 1)
        # x = ir*0.5 + vi*0.5
        x = vi
        x = self.down1(x, is_bn=False)
        x = self.down2(x, is_bn=False)
        x = F.dropout2d(self.bn(F.leaky_relu(self.conv1(x))))
        x = torch.sigmoid(self.last(x))
        return x
