import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],   # 背景
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]

def voc_ap(rec, prec, use_07_metric=False):
    """
        average precision calculations
        [precision integrated to recall]
        :param rec: recall list
        :param prec: precision list
        :param use_07_metric: 2007 metric is 11-recall-point based AP
        :return: average precision
    """

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [1.]))

        for i in range(mpre.size-1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap

def voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False):
    """:cvar
    preds:  {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target: {(image_id,class):[[],]}
    """
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = pred[class_]
        if len(pred) == 0:
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            break
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])

        # 按照置信度进行排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d] # 预测框
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)] # [[],]
                for bbgt in BBGT:
                    # caculate iou
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                            (bbgt[2] - bbgt[0] + 1.) * (bbgt[3] - bbgt[1] + 1.) - inters

                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) # 这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id, class_)]
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))
