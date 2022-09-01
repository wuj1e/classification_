# 导入库
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from dataset import dataloader
import torch.nn.parallel
from model.model import *
from torch.utils.data import DataLoader
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from tools.gen_data_list import downsample
from model.res2net_pre import res2net50
# from model.resnetcbam import *
# from model.resnet18 import *
# from resnest import ResNeSt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pred_all = []
    gt_all = []
    gloss = 0
    for batch_idx, (data, label) in enumerate(train_loader):

        data, label = data.to(device), label.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        # exit()
        pred = torch.argmax(output, 1)
        pred = list(pred.detach().cpu().numpy())
        pred_all += pred
        gt_all += list(label.long()[:, 0].cpu().numpy())
        # print(pred_all,gt_all)
        # exit()
        criterion.cuda()
        loss = criterion(output, label.squeeze().long())
        # with torch.autograd.detect_anomaly():
        loss.backward()

        optimizer.step()
        if (batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
        gloss += loss.item()


    print(classification_report(gt_all, pred_all))
    f1 = f1_score(gt_all, pred_all, average='macro')
    acc = accuracy_score(gt_all, pred_all)
    print("f1:{},acc:{}".format(f1,acc))
    return gloss/len(train_loader),f1,acc

# 定义测试过程

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    pred_all = []
    gt_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)

            output = model(data)
            pred = torch.argmax(output, 1)
            pred = list(pred.detach().cpu().numpy())
            pred_all += pred
            gt_all += list(target.long()[:, 0].cpu().numpy())
            test_loss += criterion(output, target.squeeze().long()).item()
    print('val:')
    print(classification_report(gt_all, pred_all))
    f1 = f1_score(gt_all, pred_all,average='macro')
    acc = accuracy_score(gt_all, pred_all)
    print("test_acc:%f"%acc)
    print("gt:{},pre:{}".format(gt_all, pred_all))

    return f1, acc



if __name__ == '__main__':
    train_path = 'list/train.txt'
    val_path = 'list/val1.txt'
    img_size =256     #384
    modellr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = models.resnet50(pretrained=True)
    # fc_inputs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 256),
    #     nn.Dropout(0.5),
    #     nn.Linear(256,4)
    # )
    # model = res2net50(pretrained=True)
    # model = models.resnet50(num_classes = 4)

    model = Densenet121()
    model_dict = model.state_dict()
    pre_dict = torch.load('checkout/cut/densenet/d06-11-18_140-1.000.pth').state_dict()
    #
    # pre_dict = torch.load('resnet50-19c8e357.pth')
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pre_dict)
    # # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)


    model.to(device)

    mon = 'densenet'


    weight = torch.from_numpy(np.array([5.3,5.8,3.7,3.1])).float()   #[7.0,6,2.74,2.93]  [5.5,3.1] [5.3,5.3,3.5,3.1]  5,5b
    criterion = nn.CrossEntropyLoss(weight = weight)
    optimizer = optim.SGD(model.parameters(),lr=modellr,momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=modellr,weight_decay=5e-3)       #,weight_decay=5e-4

    # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # CosLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)
    StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [50,100,160,220,270], gamma=0.50)
    BATCH_SIZE = 16
    EPOCHS = 300
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.CenterCrop(256),
        # transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_data = dataloader(val_path, transform_val)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, drop_last=False)

    loss_list = []
    f1_list = []
    acc_list = []
    # 训练
    best_acc = 0.70
    for epoch in range(1, EPOCHS + 1):
        downpath = 'list/train_down.txt'
        downsample(train_path,downpath,300)
        # 打印当前学习率
        print("lr:%f"%optimizer.state_dict()['param_groups'][0]['lr'])

        train_data = dataloader(train_path, transform_train)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)  # define traindata shuffle参数隐藏说明是index是随机的


        train_loss,train_f1,train_acc = train(model, device, train_loader, optimizer, epoch)

        StepLR.step()
        loss_list.append(train_loss)
        f1_list.append(train_f1)
        acc_list.append(train_acc)

        val_f1,val_acc = val(model,device,val_loader)


        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model, './checkout/cut/{}/best{}_{}-{:.3f}.pth'.format(mon,time.strftime("%m-%d-%H",time.localtime()),epoch,best_acc))

        if val_acc > 0.70:
            if epoch%20 == 0:
                torch.save(model,'./checkout/cut/{}/{}_{}-{:.3f}.pth'.format(mon,time.strftime("%m-%d-%H",time.localtime()),epoch,train_acc))
        # if epoch in range(75,85) :
        #     torch.save(model,'./checkout/cut/ResNet/{}{}_{}-{:.3f}.pth'.format(tag,time.strftime("%m-%d-%H",time.localtime()),epoch,train_acc))

    x1 = range(0, EPOCHS)
    y1 = loss_list
    x2 = range(0, EPOCHS)
    y2 = f1_list
    x3 = range(0,EPOCHS)
    y3 = acc_list
    # 画图train_loss
    # plt.subplot(2, 1, 1)
    plt.plot(x1, y1)
    plt.xlabel('Epoches')
    plt.ylabel('Train loss')
    plt.savefig("Train_loss.jpg")
    plt.show()

    plt.plot(x3,y3)
    # plt.title('Train Acc vs. Epoches')
    plt.xlabel('Epoches')
    plt.ylabel('Train Accuracy')
    plt.ylim(0, 1)
    plt.savefig('Train_f1+acc.png')
    plt.show()