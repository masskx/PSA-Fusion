import torch
from torch import nn

from model.S import Siamese
from model.utils import Encoder, Decoder
from model.vit import ViT


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Define the encoder
        # self.encoder = Encoder()  # empty encoder
        model = Siamese()  # pre-train encoder
        model.load_state_dict(torch.load("/ML/Mashuai/GAN/AFGAN/Sia_model_checkpoint.pth"), strict=False)
        # for param in model.parameters():  # freeze
        #     param.requires_grad = False
        self.encoder = model.encoder

        # Define the decoder
        self.vit = ViT(
            image_size=256,  # 图像尺寸
            channels=2,
            patch_size=32,  # patch大小
            num_classes=512 * 16 * 16,  # 最终投影类别
            dim=1024,  # 傻傻维度
            depth=6,  # 傻傻深度
            heads=16,  # 多头头数
            mlp_dim=2048,  # 傻傻mlp维度
            dropout=0.1,
            pool='cls',
            emb_dropout=0.1
        )
        self.decoder = Decoder()

    def forward(self, vi, ir):
        # 将红外和可见光塞入vit
        vit_x = self.vit(torch.cat([vi, ir], axis=1))  # 过trans
        x = torch.cat([vi, ir], dim=1)
        # fil = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)  # convert channel
        # fil = fil.to('cuda')
        # print(f'初始图像：{x.shape}')
        x1 = self.encoder(vi)
        x2 = self.encoder(ir)
        # print(f'编码器图像：{x.shape}')
        # x = x + vit_x.view(-1, 512, 16, 16)  # 将编码器和注意力机制的东西融合
        x = self.decoder(x1 + x2 + vit_x.view(-1, 512, 16, 16))
        # print(f'解码器图像：{x.shape}')
        return x

cc = Autoencoder()

ins1 = torch.randn(32, 1, 256, 256)
ins2 = torch.randn(32, 1, 256, 256)
output = cc(ins1,ins2)
print(output.shape)
