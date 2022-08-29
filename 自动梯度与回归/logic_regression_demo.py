import numpy as np
import torch
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 20, dtype=np.float32)
_b = 1 / (1 + np.exp(-x))
y = np.random.normal(_b, 0.005)

# 20x1
x = np.float32(x.reshape(-1, 1))
y = np.float32(y.reshape(-1, 1))


class LogicRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogicRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


input_dim = 1
output_dim = 1
model = LogicRegressionModel(input_dim, output_dim)
criterion = torch.nn.BCELoss()  # 使用二分类交叉熵损失函数
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(300):
    epoch += 1
    # Convert numpy array to torch Variable
    inputs = torch.from_numpy(x).requires_grad_()
    labels = torch.from_numpy(y)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

# Purely inference
predicted_y = model(torch.from_numpy(x).requires_grad_()).data.numpy()
print("标签Y:", y)
print("预测Y:", predicted_y)

# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x, y, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x, predicted_y, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()
