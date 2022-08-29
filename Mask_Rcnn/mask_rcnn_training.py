import torch
import torchvision
from engine import train_one_epoch
from mask_rcnn_custom_dataset import PennFudanDataset
import utils as utils


def main_train():
    # 检查是否可以利用GPU
    # torch.multiprocessing.freeze_support()
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available!')

    # 背景 + 行人
    num_classes = 2
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    device = torch.device('cuda:0')
    model.to(device)

    dataset = PennFudanDataset("D:/project/pytorch_stu/data/PennFudanPed")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,  # num_workers=4,
        collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.1)
    num_epochs = 8
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
    torch.save(model.state_dict(), "mask_rcnn_pedestrian_model.pt")


if __name__ == "__main__":
    main_train()
