from torch import nn

from model.AE import Autoencoder
from model.utils import FourierTransformLayer, EdgeDetectionLayer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ae = Autoencoder()
        self.fft = FourierTransformLayer()
        self.edge = EdgeDetectionLayer()

    def forward(self, vi, ir):
        # f_vi = self.fft(vi)  # [b,1,256,256]可见光的频谱信息
        # f_ir = self.fft(ir)  # [b,1,256,256]可见光的频谱信息
        # feature_edge = self.edge(ir) #[b,1,256,256]红外光的边缘信息
        # feature_unet = self.unet(feature_edge,feature_fft) #[b,2,256,256]利用Unet进行融合

        x = self.ae(vi, ir)
        # x = self.ae(x)
        return x
