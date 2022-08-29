import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
import cv2 as cv
import numpy as np
from torchinfo import summary

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)),
                                   ])

train_ts = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)


class CNN_Mnist(t.nn.Module):
    def __init__(self):
        super(CNN_Mnist, self).__init__()
        # 卷积层
        self.cnn_layers = t.nn.Sequential(
            # 卷积层
            t.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            # 池化层
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            # 激活函数
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.ReLU()
        )
        # 全连接层
        self.fc_layers = t.nn.Sequential(
            # 全连接层
            t.nn.Linear(7 * 7 * 32, 200),
            t.nn.ReLU(),
            t.nn.Linear(200, 100),
            t.nn.ReLU(),
            t.nn.Linear(100, 10),
            t.nn.LogSoftmax(dim=1)
        )

    # 前向传播
    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7 * 7 * 32)
        out = self.fc_layers(out)
        return out


model = CNN_Mnist().cuda()
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# print(model)
# summary(model,input_size=(32, 1, 28, 28))

def train_mnist():
    loss = t.nn.CrossEntropyLoss(reduction="mean")
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    for s in range(5):
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            y_pred = model.forward(x_train)
            train_loss = loss(y_pred, y_train)
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            # 优化器
            optimizer.zero_grad()
            # 反向传播
            train_loss.backward()
            optimizer.step()

    t.save(model.state_dict(), './cnn_mnist_model.pt')  # 保存模型的参数
    model.eval()

    # 推理
    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        pred_labels = model(test_images.cuda())
        preddicted = t.max(pred_labels, 1)[1]
        correct_count += (preddicted == test_labels.cuda()).sum()
        total += len(test_labels)
    print("total acc : %.2f\n" % (correct_count / total))


# if __name__ == "__main__":
    # 训练
    # train_mnist()
    model.load_state_dict(t.load("./cnn_mnist_model.pt"))  # 加载模型
    model.eval()  # dropout / bn
    # print(model)
    # summary(model, input_size=(32, 1, 28, 28))

    # print(m)
    # image = cv.imread("9.jpg", cv.IMREAD_GRAYSCALE)
    # image = cv.resize(image, (28, 28))
    # cv.imshow("input", image)
    # img_f = np.float32(image) / 255.0 - 0.5
    # img_f = img_f / 0.5
    # img_f = np.reshape(img_f, (1, 1, 28, 28))
    # pred_labels = model(t.from_numpy(img_f))
    # plabels = t.exp(pred_labels)
    # probs = list(plabels.detach().numpy()[0])
    # pred_label = probs.index(max(probs))
    # print("predict digit number: ", pred_label)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
