from __future__ import print_function, division
from PIL import Image

from torchvision import transforms
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel


# 模型存储路径
model_save_path = './checkout/cut/densenet/06-14-11_300-1.000.pth'

# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
# 数据预处理
img_size = 256
transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])

class_names = ['JIC-A','JIC-B','JIC-C1','JIC-C2']  # 这个顺序很重要，要和训练时候的类名顺序一致

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------------ 载入模型并且训练 --------------------------- #


model = torch.load(model_save_path)
model.eval()

#B,C1
image_PIL = Image.open(r'').convert('RGB')  #../dataset/seg_data\HEALTH\jinrilong\5_0.png
# image_PIL = image_PIL.convert('L')

image_tensor = transform_val(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
print(image_tensor.size())
# image_tensor = torch.unsqueeze(image_tensor, 0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)

out = model(image_tensor)
prob = F.softmax(out, dim=1)  # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
print(prob)

pred = torch.argmax(out, 1).detach().cpu().numpy()[0]
print(class_names[pred])
