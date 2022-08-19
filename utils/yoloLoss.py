import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self, S, B, C, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''compute the intersection over union of two set of boxes, each box:[x1,y1,x2,y2]
        args:
            box1:[N,4]
            box2:[M,4]
        return:
            iou: [N,M]
        '''
        N = box1.size(0)
        M = box2.size(0)

        #------------ intersection --------------#
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
        )

        wh = rb - lt # [N, M, 2]
        wh[wh<0] = 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])   # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])   # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)
        area2 = area2.unsqueeze(0).expand_as(inter)

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        '''
        args:
            pred_tensor: size(batchsize, S, S, B*5+20=30) [x,y,w,h,c]
            target_tensor:size(batchsize, S, S, 30)
        '''
        S, B, C = self.S, self.B, self.C
        N = 5*B + C

        # 批大小
        batch_size = pred_tensor.size(0)
        # 有目标的张量         [bs, 7, 7]
        coord_mask = target_tensor[..., 4] > 0
        # 没有目标的张量       [bs, 7, 7]  一开始的时候有目标的，conf是都给的1，，后来做iou再重新覆盖1和0
        noobj_mask = target_tensor[..., 4] == 0
        # 扩展维度的布尔值相同  [bs, S, S] -> [bs, S, S, N]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        # int8 -> bool 方便后续掩码操作
        coord_mask = coord_mask.bool()
        noobj_mask = noobj_mask.bool()

        # 预测值
        # 提取出预测框里含有目标的张量[n_coord, N]
        coord_pred = pred_tensor[coord_mask].view(-1, N)
        # 提取预测框的bbox和C [n_coord*B, 5]
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # box[x1,y1,w1,h1,c1]、[x2,y2,w2,h2,c2]
        # 预测值的分类信息 [n_coord, C]
        class_pred = coord_pred[:, 5*B:]

        # 标签
        # 提取含有目标的标签张量 [n_coord, N]
        coord_target = target_tensor[coord_mask].view(-1, N)
        # 提取标签的bbox和C  [n_coord*B, 5]
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5*B:]

        #---------------- noobj confidence loss ------------------------#
        # 找到预测值里没有目标的网格张量[n_noobj, N]  n_noobj = S*S-n_coord
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)
        noobj_target = target_tensor[noobj_mask].view(-1, N)
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0)
        for b in range(B):
            # 没有目标置信度置1，noobj_conf_mask[:, 4] = 1, noobj_conf_mask[:, 9] = 1
            noobj_conf_mask[:, 4 + b*5] = 1

        # [n_noobj x 2=len([conf1, conf2])]
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        # [n_noobj x 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        # 计算没有目标的置信度损失
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        #-------------------------- obj  loss ---------------------------#
        # [n_coord x B, 5]
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)
        # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)
        # [n_coord x B, 5], only the last 1=(conf,) is used
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()

        # choose the predicted bbox and having the highest IoU for each target bbox
        # 步长为2是因为一个目标有两个bbox
        for i in range(0, bbox_target.size(0), B):
            # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred = bbox_pred[i:i+B]
            # [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size()))
            # rescale (center_x, center_y) for the image-size to compute IoU correctly
            # w,h是以整张图像大小为基准进行归一化的，而x,y是以网格大小进行归一化的，所以需要将x,y 统一到全图上
            pred_xyxy[:,  :2] = pred[:, :2]/float(S) - 0.5*pred[:, 2:4]  # 左上角坐标
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5*pred[:, 2:4]  # 右下角坐标

            # target bbox at i-th cell
            # 因为每个单元格所包含的目标框在当前实现中是相同的，提取第一个足以  2个bbox的信息完全相同，取一个即可
            # [1, 5=len([x, y, w, h, conf])]
            target = bbox_target[i].view(-1, 5)
            # [1, 5=len([x1, y1, w1, h1, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size()))
            target_xyxy[:,  :2] = target[:, :2]/float(S) - 0.5*target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5*target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            # [4, 5]
            coord_response_mask[i+max_index] = 1
            # [4, 5]
            coord_not_response_mask[i+max_index] = 0
            # [4, 5]
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # 1.BBox location/size and objectness loss for response loss
        # [n_response, 5]
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        # [n_response, 5], only the first 4=(x,y,w,h) are used
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        # [n_response, 5], only the last 1=(conf,) is used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)

        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # 2.class probability loss for the cells which contain objects
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # 3.Total loss
        loss = self.l_coord*(loss_xy + loss_wh) + loss_obj + self.l_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss

