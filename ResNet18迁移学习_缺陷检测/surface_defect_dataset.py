import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv

# 夹杂 - In - inclusion
# 划痕 - SC - scratch
# 裂纹 - CR- crackle
# 压入氧化皮 - PS - Press in oxide scale
# 麻点 - RS
# 斑点 - PA
defect_labels = ["In", "Sc", "Cr", "PS", "RS", "Pa"]


class SurfaceDefectDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.Resize((200, 200))
                                             ])
        img_files = os.listdir(root_dir)
        self.defect_types = []
        self.images = []
        index = 0
        for file_name in img_files:
            defect_attrs = file_name.split("_")
            d_index = defect_labels.index(defect_attrs[0])
            self.images.append(os.path.join(root_dir, file_name))
            self.defect_types.append(d_index)
            index += 1

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
        img = cv.imread(image_path)  # BGR order
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # h, w, c = img.shape
        # rescale
        # img = cv.resize(img, (200, 200))
        # img = (np.float32(img) /255.0 - 0.5) / 0.5
        # H, W C to C, H, W
        # img = img.transpose((2, 0, 1))
        self.transform(img)
        sample = {'image': self.transform(img), 'defect': self.defect_types[idx]}
        return sample


if __name__ == "__main__":
    ds = SurfaceDefectDataset("D:/project/pytorch_stu/data/enu_surface_defect/train")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['defect'])
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['defect'])
        break
