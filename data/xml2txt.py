import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
        return objects



'''
yolo训练文件格式转换
'''

txt_file = open('voc2007test.txt', 'w')
test_file = open(r'E:\01_DeepLearning\02_Deep_Learning\01_Pytorch\02_ObjectDetect\01_SSD\ssd-pytorch-master-Bub\VOCdevkit\VOC2007\ImageSets\Main\test.txt',
                 'r')
lines = test_file.readlines()
lines = [x[:-1] for x in lines]

Annotations = r'E:\01_DeepLearning\02_Deep_Learning\01_Pytorch\02_ObjectDetect\01_SSD\ssd-pytorch-master-Bub\VOCdevkit\VOC2007\Annotations'
xml_files = os.listdir(Annotations)

count = 0
for xml_file in xml_files:
    count += 1
    if xml_file.split('.')[0] not in lines:
        continue
    image_path = xml_file.split('.')[0] + '.jpg'
    results = parse_rec(Annotations + '/' + xml_file)
    if len(results) == 0:
        print(xml_file + "contains no objects.")
        continue
    txt_file.write(image_path)
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = VOC_CLASSES.index(class_name)
        txt_file.write(' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' +
                       str(bbox[3]) + ' ' + str(class_name))
    txt_file.write('\n')
txt_file.close()


