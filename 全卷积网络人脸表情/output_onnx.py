import torch
from emotions_cnn import EmotionsResNet

model = EmotionsResNet()
model.load_state_dict(torch.load("./face_emotions_model2.pt"))
model.eval()
#
dumm_input = torch.randn(1, 3, 64, 64)
torch.onnx.export(model, (dumm_input), "face_emotions_model.onnx", verbose=True)