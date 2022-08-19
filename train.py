import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from yolo.yolov1 import yolov1_vgg16, yolov1_vgg16_bn, yolov1_resnet50
from utils.yoloLoss import yoloLoss
from utils.dataset import yoloDataset
from utils.visualize import Visualizer

# --------------------------------  Start  ------------------------------------#
use_gpu = torch.cuda.is_available()
file_root = r"data/VOCdevkit/VOC2007/JPEGImages"
learning_rate = 1e-3
num_epochs = 50
batch_size = 64
use_resnet = False

if use_resnet:
    model = yolov1_resnet50()
else:
    model = yolov1_vgg16_bn()
print(model)

print('loading pretrained model 。。。')
if use_resnet:
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    raw_state_dict = model.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in raw_state_dict.keys() and not k.startswith('fc'):
            print('yes')
            raw_state_dict[k] = new_state_dict[k]
    model.load_state_dict(raw_state_dict)
else:
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    raw_state_dict = model.state_dict()
    for k in list(new_state_dict.keys())[:-4]:
        print(k)
        if k in raw_state_dict.keys() and not k.startswith('fc'):
            print('yes')
            raw_state_dict[k] = new_state_dict[k]
    model.load_state_dict(raw_state_dict)

print('cuda:', torch.cuda.current_device(), torch.cuda.device_count())

#---------------- Loss ------------------------#
# yolov1_vgg: S = 7
# yolov1_resnet: S = 14
criterion = yoloLoss(S=7, B=2, C=20, l_coord=5, l_noobj=0.5)

if use_gpu:
    model.cuda()

model.train()

# different learning rate
params = []
params_dict = dict(model.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate*1}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 训练集
train_dataset = yoloDataset(root=file_root, list_file='data/voc2007train.txt', train=True, grid_num=7, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 测试集
test_dataset = yoloDataset(root=file_root, list_file='data/voc2007test.txt', train=False, grid_num=7, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print('the dataset has {} images'.format(len(train_dataset)))
print('the batch_size is {}'.format(batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='pytorch1.8')
best_test_loss = np.inf

for epoch in range(num_epochs):
    model.train()
    if epoch == 30:
        learning_rate = 1e-4
    if epoch == 40:
        learning_rate = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch {0}/{1}'.format(epoch+1, num_epochs))
    print('Learning Rate for this epoch:{0}'.format(learning_rate))

    total_loss = 0.

    for i, (images, target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        torch.autograd.set_detect_anomaly(True)
        pred = model(images)
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print('Epoch [{0}/{1}, Iter [{2}/{3}] Loss:{4:.4f}, avgerage_loss:{5:.4f}'.format(epoch+1, num_epochs,
                  i+1, len(train_loader), loss.item(), total_loss/(i+1)))
            num_iter += 1
            # vis.plot_train_val(loss_train=total_loss/(i+1))
            # print('loss_train:{}'.format(total_loss/(i+1)))

    # validation
    validation_loss = 0.0
    model.eval()
    for i, (images, target) in enumerate(test_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = model(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    # vis.plot_train_val(loss_val=validation_loss)
    print('loss_val:{}'.format(validation_loss))

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss {:.5f}'.format(best_test_loss))
        torch.save(model.state_dict(), 'weight/best.pt')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()
    torch.save(model.state_dict(), 'weight/yolo.pt')







































