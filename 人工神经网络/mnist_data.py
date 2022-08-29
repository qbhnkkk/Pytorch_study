import torch
import torchvision as tv
import cv2 as cv
from torch.utils.data import DataLoader

transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,), (0.5,))])

train_ts = tv.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=32, shuffle=True, drop_last=False)
index = 0
for i_batch, sample_batched in enumerate(train_dl):
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    image = sample_batched[0][0].numpy().reshape((28, 28))
    print(image.shape)
    cv.imshow("digit-image", image)
    cv.waitKey(0)
    if index == 4:
        break
    index += 1
