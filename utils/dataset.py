'''
txt描述文件 image_name.jpg x y w h c x y w h c
'''

import os
import os.path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt


class yoloDataset(Dataset):
    image_size = 448
    def __init__(self, root, list_file, train, grid_num, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.grid_num = grid_num
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104) # RGB

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(splited[1 + 5 * i])
                y1 = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c) + 1)  # 0为背景
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes = self.randomScale(img, boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, boxes, labels = self.randomShift(img, boxes, labels)
            img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        # 对bbox做原图的归一化
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        img = self.BGR2RGB(img)
        # img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        target = self.encoder(boxes, labels, grid_nums=self.grid_num)  # 7x7x30
        for t in self.transform:
            img = t(img)
        return img, target


    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels, grid_nums=7):
        """
        :param self: 构造一个与模型输出向量一样的target
        :param boxes: tensor:[[x1,y1,x2,y2],[]]
        :param labels: tensor:[...]
        :param grid_nums: 网格数量
        :return: 7x7x30
        """
        target = torch.zeros((grid_nums, grid_nums, 30))
        cell_size = 1./grid_nums
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
        for i in range(cxcy.size(0)):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i])+9] = 1

            xy = ij * cell_size  # 匹配到的网格左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size   # 网格中的坐标偏移量
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]),  :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5,5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2])/2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shift_img = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shift_img[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width*0.2, width*0.2)
            shift_y = random.uniform(-height*0.2, height*0.2)
            if shift_x >= 0 and shift_y >= 0:
                after_shift_img[int(shift_y):, int(shift_x):, :] = bgr[:height-int(shift_y), :width-int(shift_x), :]
            elif shift_x >= 0 and shift_y < 0:
                after_shift_img[:height+int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width-int(shift_x), :]
            elif shift_x < 0 and shift_y >= 0:
                after_shift_img[int(shift_y):, :width+int(shift_x), :] = bgr[:height-int(shift_y), -int(shift_x):, :]
            elif shift_x < 0 and shift_y < 0:
                after_shift_img[:height+int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]
            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shift_img, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height, width,c = bgr.shape
            bgr = cv2.resize(bgr, (int(width*scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6*height, height)
            w = random.uniform(0.6*width, width)
            x = random.uniform(0, width-w)
            y = random.uniform(0, height-h)
            x, y, w, h = int(x), int(y), int(w), int(h)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h, x:x+w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.unint8)
            return im



def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms


    file_root = r'E:\01_DeepLearning\02_Deep_Learning\01_Pytorch\02_ObjectDetect\01_SSD\ssd-pytorch-master-Bub\VOCdevkit\VOC2007\JPEGImages'
    train_dataset = yoloDataset(root=file_root, list_file='../data/voc2007train.txt', train=True, grid_num=7, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)
    img, target = next(train_iter)
    img1 = img.permute(0, 2, 3, 1)
    img2 = np.squeeze(img1.numpy(), 0)
    plt.imshow(img2, cmap='gray')
    plt.show()
    print(img, target)


if __name__=='__main__':
    main()