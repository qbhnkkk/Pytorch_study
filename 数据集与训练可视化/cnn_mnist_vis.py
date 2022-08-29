import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)),
                                   ])

train_ts = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)
writer = SummaryWriter('D:\project\pytorch_stu\experiment_01')

# get some random training images
dataiter = iter(train_dl)
images, labels = dataiter.next()

# create grid of images
img_grid = tv.utils.make_grid(images)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)


class CNN_Mnist(t.nn.Module):
    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.cnn_layers = t.nn.Sequential(
            t.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.ReLU()
        )
        self.fc_layers = t.nn.Sequential(
            t.nn.Linear(7 * 7 * 32, 200),
            t.nn.ReLU(),
            t.nn.Linear(200, 100),
            t.nn.ReLU(),
            t.nn.Linear(100, 10),
            t.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7 * 7 * 32)
        out = self.fc_layers(out)
        return out


def train_and_test():
    model = CNN_Mnist().cuda()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    loss = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    writer.add_graph(model, images.cuda())

    for s in range(5):
        m_loss = 0.0
        total1 = 0
        correct_count1 = 0
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            y_pred = model.forward(x_train)
            train_loss = loss(y_pred, y_train)
            m_loss += train_loss.item()
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        for test_images, test_labels in test_dl:
            pred_labels = model(test_images.cuda())
            predicted = t.max(pred_labels, 1)[1]
            correct_count1 += (predicted == test_labels.cuda()).sum()
            total1 += len(test_labels)
            acc = correct_count1 / total1
            if (i + 1) % 100 == 0:
                print(i + 1, acc)

        writer.add_scalar('training loss',
                          m_loss / 1000,
                          s * len(train_dl) + i)

        writer.add_scalar('acc loss',
                          acc,
                          s * len(train_dl) + i)

    t.save(model.state_dict(), './cnn_mnist_model_vis.pt')
    model.eval()
    writer.close()
    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        pred_labels = model(test_images.cuda())
        predicted = t.max(pred_labels, 1)[1]
        correct_count += (predicted == test_labels.cuda()).sum()
        total += len(test_labels)
    print("total acc : %.2f\n" % (correct_count / total))

if __name__ == "__main__":
    train_and_test()
