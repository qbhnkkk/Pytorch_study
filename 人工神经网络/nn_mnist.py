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
model = t.nn.Sequential(
    t.nn.Linear(784, 100),
    t.nn.ReLU(),
    t.nn.Linear(100, 10),
    t.nn.LogSoftmax(dim=1)
)  # 28*28 = 784


def train_mnist():
    loss_fn = t.nn.NLLLoss(reduction="mean")
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    for s in range(10):
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.view(x_train.shape[0], -1)
            y_pred = model(x_train)
            train_loss = loss_fn(y_pred, y_train)
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            model.zero_grad()
            train_loss.backward()
            optimizer.step()

    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        for i in range(len(test_labels)):
            image = test_images[i].view(1, 784)
            with t.no_grad():
                pred_labels = model(image)
            plabels = t.exp(pred_labels)
            probs = list(plabels.numpy()[0])
            pred_label = probs.index(max(probs))
            true_label = test_labels.numpy()[i]
            if pred_label == true_label:
                correct_count += 1
            total += 1
    print("total acc : %.2f\n" % (correct_count / total))
    # 保存模型
    t.save(model, './mnist_model_nn.pt')
    t.save(model.state_dict(), './nn_mnist_model.pt')  # 保存模型的参数


if __name__ == "__main__":
    # train_mnist()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 推理
    model.load_state_dict(t.load("./nn_mnist_model.pt"))  # 加载模型
    model.eval()  # dropout / bn
    print(model)
    # 打印模型结构
    # m = t.load("./mnist_model_nn.pt")
    # print(m)
   #  summary(model, input_size=(32, 3, 784, 784))

    # image = cv.imread("9.jpg", cv.IMREAD_GRAYSCALE)
    # image = cv.resize(image, (28, 28))
    # cv.imshow("input", image)
    # img_f = np.float32(image) / 255.0 - 0.5
    # img_f = img_f / 0.5
    # img_f = np.reshape(img_f, (1, 784))
    # pred_labels = model(t.from_numpy(img_f))
    # plabels = t.exp(pred_labels)
    # probs = list(plabels.detach().numpy()[0])
    # pred_label = probs.index(max(probs))
    # print("predict digit number: ", pred_label)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
