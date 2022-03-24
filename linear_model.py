import math

import numpy as np
import matplotlib.pyplot as plt
#常用visdom做可视化
#meshgrid
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return math.pow((y - y_pred), 2)


w_list = []
mse_list = []

for w in np.arange(0.1, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum/len(x_data))

plt.plot(w_list,mse_list)
plt.xlabel("w")
plt.ylabel("MSE")
plt.show()

