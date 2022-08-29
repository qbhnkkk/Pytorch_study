import torch

x = torch.randn(1, 5, requires_grad=True)
y = torch.randn(5, 3, requires_grad=True)
z = torch.randn(3, 1, requires_grad=True)

# 自动梯度求导 正向
print("x:\n", x, "y:\n", y, "z:\n", z)
xy = torch.matmul(x, y)
print("xy:\n", xy)
xyz = torch.matmul(xy, z)

# 反向
xyz.backward()
print(x.grad, y.grad, z.grad)

zy = torch.matmul(y, z).view(-1, 5)
print(zy)