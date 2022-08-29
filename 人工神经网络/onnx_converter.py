import torch
from cnn_mnist import CNN_Mnist

model = CNN_Mnist()
model.load_state_dict(torch.load("cnn_mnist_model.pt"))
model.eval()

dumm_input1 = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, (dumm_input1), "cnn_mnist.onnx", verbose=True)