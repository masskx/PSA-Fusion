import torch
from torch import nn

from model.S import Siamese
from model.utils import Encoder, Decoder
from model.vit import ViT
from model.vit.cross_vit import CrossViT


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Define the encoder
        self.encoder = Encoder()  # empty encoder
        # model = Siamese()  # pre-train encoder
        # model.load_state_dict(torch.load("/ML/Mashuai/GAN/AFGAN/Sia_model_checkpoint.pth"), strict=False)
        # # for param in model.parameters():  # freeze
        # #     param.requires_grad = False
        # self.encoder = model.encoder

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
        self.cross_vit = CrossViT(
            image_size = 256,
            num_classes = 512 * 4 * 4,
            depth = 4,               # number of multi-scale encoding blocks
            # use same encoder
            sm_dim = 192,            # high res dimension
            sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
            sm_enc_depth = 2,        # high res depth
            sm_enc_heads = 4,        # high res heads
            sm_enc_mlp_dim = 2048,   # high res feedforward dimension
            #########################################
            lg_dim = 192,            # low res dimension
            lg_patch_size = 16,      # low res patch size
            lg_enc_depth = 2,        # low res depth
            lg_enc_heads = 4,        # low res heads
            lg_enc_mlp_dim = 2048,   # low res feedforward dimensions

            cross_attn_depth = 2,    # cross attention rounds
            cross_attn_heads = 4,    # cross attention heads
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.decoder = Decoder()

    def forward(self, vi, ir):
        # 将红外和可见光塞入vit
        # vit_x = self.vit(torch.cat([vi, ir], axis=1))  # 过trans
        vit_x = self.cross_vit(vi,ir) # 将可见光编码
        x = torch.cat([vi, ir], dim=1)
        # print(f'初始图像：{x.shape}')
        x1 = self.encoder(vi)
        x2 = self.encoder(ir)
        # print(f'1编码器图像：{x1.shape}')
        # print(f'2编码器图像：{x2.shape}')
        # 将编码器和注意力机制的东西融合
        x = self.decoder(x1 + x2 + vit_x.view(-1, 512, 4, 4))
        # x = self.decoder(x1 + x2 )
        # print(f'解码器图像：{x.shape}')
        return x

# cc = Autoencoder()
# #
# ins1 = torch.randn(32, 1, 256, 256)
# ins2 = torch.randn(32, 1, 256, 256)
# output = cc(ins1,ins2)
# print(output.shape)
