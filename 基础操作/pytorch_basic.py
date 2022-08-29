import torch
#
# # 2*2大小的矩阵 默认值为0
# x = torch.empty(2, 2)
# print(x)
# # 随机初始化
# x1 = torch.randn(2, 2)
# print(x1)
# # 初始化为0
# x2 = torch.zeros(3, 3)
# print(x2)
# # 自定义初始化
# x3 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])  # 一维（向量）
# print(x3)
#
# # op操作 （操作数）
# y = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 10])
# z = x3.add(y)
# print(z)
#
# # 维度变换
# x = x.view(-1, 4)  # 变为1*4
# print(x)
# xx2 = x2.view(-1, 9)  # 变为1*9
# print(xx2)
# print(x.size(), xx2.size())
#
# # 由tensor转为numpy
# nx = x.numpy()
# ny = y.numpy()
# print("nx:", nx, "\n ny:", ny)
#
# # 由np转为tensor
# xx = torch.from_numpy(nx.reshape((2, 2)))
# print(xx)
#
# # using CUDA/GPU
# if torch.cuda.is_available():
#     print("GPU Detected")
#     result = x3.cuda() + y.cuda()
#     print(result)

m = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()
# input is of size N x C = 2 X 2
input = torch.randn(2, 2, requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.tensor([0, 1])
output = loss(m(input), target)
print(m(input))
print(target)
print(output)