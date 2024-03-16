# siamese network
from torch import nn, ops
import torch
from model.utils import Encoder
import torch
import torch.functional as F


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: A tuple containing two PyTorch tensors of the same shape.

    Returns:
        A tensor containing the euclidean distance
        (as floating point value) between the vectors.
    """

    x, y = vects
    # Square difference along all dimensions except the first one (batch dimension)
    sum_square = torch.sum((x - y) ** 2, dim=1, keepdim=True)
    # Add epsilon for numerical stability, similar to Keras backend.epsilon()
    return torch.sqrt(torch.max(sum_square, torch.tensor(1e-7)))


# 使用示例：
# 假设你有两个形


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.encoder = Encoder()
        self.flatten = nn.Flatten()
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),

            # nn.ReLU(),
            #
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 64),
            # nn.LayerNorm(64),
            # nn.Sigmoid(),
            # nn.Dropout(p=0.5),
            #
            # nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, vi, ir):
        vi_output = self.encoder(vi)  # 分别经过单通道编码器
        ir_output = self.encoder(ir)
        combine = euclidean_distance((vi_output, ir_output))
        # combine = vi_output * ir_output
        # print(f"combine {combine.shape}")
        # print(f"flatten(combine) {self.flatten(combine).shape}")
        output = self.cls_head(self.flatten(combine))  # mlt
        return output

#
ins = torch.randn(32, 1, 256, 256)
# #
# model = Siamese()
# output = model(ins, ins)
# print(output.shape)
# print(output)

# model = Siamese()  # init
# model.load_state_dict(torch.load("../Sia_model_checkpoint.pth"),strict=False)
# for param in model.parameters():  # freeze
#     param.requires_grad = False
# output = model.encoder(ins)
# print(output.shape)
# # #
