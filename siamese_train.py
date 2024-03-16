import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from dataset.Mydataset import FusionDataset, SiaDataset
from model.S import Siamese

import glob
# 设置一波随机种子
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 获取数据
train_irimgs_path = glob.glob('/ML/Mashuai/GAN/AFGAN/dataset/train/ir/*.png')
train_viimgs_path = glob.glob('/ML/Mashuai/GAN/AFGAN/dataset/train/vi/*.png')

test_irimgs_path = glob.glob('/ML/Mashuai/GAN/AFGAN/dataset/test/MSRS/ir/*.png')
test_viimgs_path = glob.glob('/ML/Mashuai/GAN/AFGAN/dataset/test/MSRS/vi/*.png')

train_ds = SiaDataset(train_irimgs_path[:60], train_viimgs_path[:60])
test_ds = SiaDataset(test_irimgs_path[:10], test_viimgs_path[:10])

BATCHSIZE = 128
LAMDA = 7
epsilon = 5.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 训练集随机打乱
train_dl = data.DataLoader(dataset=train_ds,
                           batch_size=BATCHSIZE,
                           shuffle=True)
test_dl = data.DataLoader(dataset=test_ds,
                          batch_size=BATCHSIZE,
                          shuffle=False)
model = Siamese().to(device)
if torch.cuda.device_count() > 1:  # 多卡训练
    model = nn.DataParallel(model)  # 就在这里wrap一下，模型就会使用所有的GPU
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))


# 定义损失函数
def create_contrastive_loss(margin=1.0):
    def contrastive_loss(y_true, y_pred):
        square_pred = y_pred.pow(2)
        margin_square = torch.clamp(margin - y_pred, min=0).pow(2)
        return torch.mean((1.0 - y_true) * square_pred + y_true * margin_square)

    return contrastive_loss


loss_fn = create_contrastive_loss()
for epoch in range(100):
    model.train()
    train_loss = 0
    test_loss = 0
    train_count = len(train_dl)
    test_count = len(test_dl)
    for step, (x, y) in enumerate(train_dl):
        vi, ir = x
        ir = ir.to(device)
        vi = vi.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(vi, ir)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item()
    with torch.no_grad():
        train_loss /= train_count
        # 训练完一个epoch就打印输出trainLoss
    model.eval()
    for step, (x, y) in enumerate(test_dl):
        vi, ir = x
        ir = ir.to(device)
        vi = vi.to(device)
        y = y.to(device)
        output = model(vi, ir)
        loss = loss_fn(output, y)
        test_loss += loss.item()
    test_loss /= test_count
    print("Epoch:", epoch, end=' ')
    print("train_loss", train_loss, end=' ')
    print("test_loss", test_loss)

torch.save(model.state_dict(), 'Sia_model_checkpoint.pth')
