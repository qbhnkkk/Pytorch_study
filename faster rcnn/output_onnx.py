import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=3,
                                                             pretrained_backbone=True)
model.load_state_dict(torch.load("./faster_rcnn_pet_model.pt"))
model.eval()
# print(model)
#
dumm_input = [torch.randn(3, 300, 400), torch.randn(3, 500, 400)]
torch.onnx.export(model, dumm_input, "faster_rcnn_pet_model.onnx", verbose=True, opset_version=11)
