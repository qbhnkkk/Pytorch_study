import torch
import torchvision
from surface_defect_dataset import SurfaceDefectDataset
from torch.utils.data import DataLoader

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


class SurfaceDefectResNet(torch.nn.Module):

    def __init__(self):
        super(SurfaceDefectResNet, self).__init__()
        self.cnn_layers = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.cnn_layers.fc.in_features
        self.cnn_layers.fc = torch.nn.Linear(num_ftrs, 6)

    def forward(self, x):
        # stack convolution layers
        out = self.cnn_layers(x)
        return out


if __name__ == "__main__":
    # create a complete CNN
    model = SurfaceDefectResNet()
    print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    ds = SurfaceDefectDataset("D:/project/pytorch_stu/data/enu_surface_defect/train")
    num_train_samples = ds.num_of_samples()
    bs = 4
    dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

    # 训练模型的次数
    num_epochs = 15
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    # 损失函数
    cross_loss = torch.nn.CrossEntropyLoss()
    index = 0
    for epoch in  range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, label_batch = \
                sample_batched['image'], sample_batched['defect']
            if train_on_gpu:
                images_batch, label_batch= images_batch.cuda(), label_batch.cuda()
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            m_label_out_ = model(images_batch)
            label_batch = label_batch.long()

            # calculate the batch loss
            loss = cross_loss(m_label_out_, label_batch)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            if index % 100 == 0:
                print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
            index += 1

            # 计算平均损失
        train_loss = train_loss / num_train_samples

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    model.eval()
    torch.save(model.state_dict(), 'surface_defect_model.pt')
