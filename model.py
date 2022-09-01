import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

# from layers import ConvBlock
# from layers import GlobalAvgPool2d
# from layers import BottleneckBlock
from resnest.torch import resnest50




class MobileNet(nn.Module):
    def __init__(self, num_classes=4):   # num_classes，此处为 二分类值为2
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   # 从预训练模型加载VGG16网络参数
        # net.con
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1280, 1000),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):   # num_classes，此处为 二分类值为2
        super(ResNet18, self).__init__()
        net = models.resnet18(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.fc = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            # nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=4):   # num_classes，此处为 二分类值为2
        super(ResNet50, self).__init__()
        net = models.resnet50(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        net.fc = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50_cam(nn.Module):
    def __init__(self, num_classes=4):   # num_classes，此处为 二分类值为2
        super(ResNet50_cam, self).__init__()
        net = models.resnet50(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        net.avgpool = nn.Sequential()
        net.fc = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNeSt50(nn.Module):
    def __init__(self, num_classes=4):   # num_classes
        super(ResNeSt50, self).__init__()
        # net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        net = resnest50(pretrained = False)

        net.fc = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(2048, 512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Densenet121(nn.Module):
    def __init__(self,num_classes = 4,pretrained=True):
        super(Densenet121,self).__init__()
        net = models.densenet121(pretrained=pretrained)
        net.classifier = nn.Sequential(nn.Linear(1024,out_features=num_classes))
        net.features.norm5 = nn.Sequential()
        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(in_features=1024,out_features=num_classes,bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNext50(nn.Module):
    def __init__(self, num_classes=4):  # num_classes，此处为 二分类值为2
        super(ResNext50, self).__init__()
        net = models.resnext50_32x4d(pretrained=True)  # 从预训练模型加载VGG16网络参数
        # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        net.fc = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(2048, 256),  # 1000不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x











