import torch
import torch.nn.functional as F
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient(x):
    # 定义一个函数来计算图像的梯度（或拉普拉斯算子的卷积结果）。
    with torch.no_grad():
        # 使用不追踪梯度的上下文，因为此操作是在推理过程中使用，不需要进行反向传播。
        laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        # 定义拉普拉斯核，这是一个用于图像二阶导数近似的卷积核。
        kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0).to(device)
        # 将拉普拉斯核转换为浮点张量，并添加两个维度以匹配卷积操作的要求，
        # 然后将其移动到指定的设备（如GPU或CPU）。
        return F.conv2d(x, kernel, stride=1, padding=1)
        # 对输入的x（图像或特征图）应用2D卷积，使用定义的拉普拉斯核，
        # 步长设为1，填充设为1（以保持图像尺寸不变）并返回结果。

# 定义下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_bn=True):
        # 生成器的输出层和判别器的输入层不使用BN
        x = self.conv_relu(x)
        if is_bn:
            x = self.bn(x)
        return x
# 定义上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_drop=False):
        # 生成器的输出层和判别器的输入层不使用BN
        x = self.upconv_relu(x)
        x = self.bn(x)
        if is_drop:
            x = F.dropout2d(x)
        return x
# 自定义神经网络层来提取图像的边缘细节并保持形状相同
class EdgeDetectionLayer(nn.Module):
    def __init__(self):
        super(EdgeDetectionLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    def forward(self, x):
        edge_map = F.conv2d(x, weight=torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]*3]).to(x.device), padding=1)
        return x + edge_map


# 自定义神经网络层来提取图像的频谱信息
class FourierTransformLayer(nn.Module):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()
    def forward(self, x):
        # 进行傅里叶变换
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        return x_fft.real


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)
