import torch
from surface_defect_cnn import SurfaceDefectResNet

model = SurfaceDefectResNet()
model.load_state_dict(torch.load("./surface_defect_model.pt"))
model.eval()
#
dumm_input = torch.randn(1, 3, 200, 200)
torch.onnx.export(model, (dumm_input), "surface_defect_model_resnet18.onnx", verbose=True)