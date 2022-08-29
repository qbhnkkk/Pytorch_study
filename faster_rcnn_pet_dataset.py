from xml.dom.minidom import parse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import transforms as T
import os
import cv2 as cv


class PetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = T.Compose([T.ToTensor()])
        self.ann_xmls = list(sorted(os.listdir(os.path.join(root_dir, "annotations/xmls"))))

    def __len__(self):
        return len(self.ann_xmls)

    def num_of_samples(self):
        return len(self.ann_xmls)

    def __getitem__(self, idx):
        # load images and bbox
        bbox_xml_path = os.path.join(self.root_dir, "annotations/xmls", self.ann_xmls[idx])

        # 读取xml
        dom = parse(bbox_xml_path)
        # 获取文档元素对象
        data = dom.documentElement
        # 获取 objects
        objects = data.getElementsByTagName('object')
        node = data.getElementsByTagName('filename')[0]
        file_ame = node.childNodes[0].nodeValue
        image_path = os.path.join(self.root_dir, "images", file_ame)
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # get bounding box coordinates
        boxes = []
        labels = []
        for object_ in objects:
            # 获取标签中内容
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue
            if name == "dog":
                labels.append(np.int(1))
            if name == "cat":
                labels.append(np.int(2))

            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        img, target = self.transforms(img, target)
        return img, target


if __name__ == "__main__":
    ds = PetDataset("D:/project/pytorch_stu/data/pet_data")
    for i in range(len(ds)):
        img, target = ds[i]
        print(i, img.size(), target)
        device = torch.device('cuda:0')
        boxes = target["boxes"]
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        targets = [{k: v.to(device) for k, v in t.items()} for t in [target]]
        if i == 3:
            break
