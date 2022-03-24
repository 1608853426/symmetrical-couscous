# 梯度下降使用的是cost函数计算的梯度,但可能会陷入鞍点
# 随机梯度下降使用lost函数计算的梯度,噪声可能会使我们走出鞍点

import math
import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# forward前馈
def forward(x):
    return x * w


# cost
def loss(x, y):
    return math.pow(forward(x) - y, 2)


# gradient梯度
def gradient(x, y):
    return 2 * x * (forward(x) - y)


loss_list=[]
print("before train: ", 4, forward(4))
for epoch in range(100):
    los = 0
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w -= 0.01 * grad
        l = loss(x_val, y_val)
        los += l
        print("epoch = ", epoch, " w = ", w, ", loss = ", l)
    loss_list.append(los)
print("after train: ", 4, forward(4))
plt.plot(np.arange(0,100,1),loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()