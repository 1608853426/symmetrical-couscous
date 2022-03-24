import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0.0], [0.0], [1.0]])


# 构建模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        r = self.linear(x)
        y_pred = f.sigmoid(r)
        return y_pred


model = LogisticRegressionModel()

# 定义损失函数
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    print("loss: ", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("w", model.linear.weight.item())
print("b", model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test)
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot((0, 10), (0.5, 0.5), c='r')
plt.xlabel('Hour')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
