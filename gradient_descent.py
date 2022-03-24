import math

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# forward前馈
def forward(x):
    return x * w


# cost
def cost(xs, ys):
    cost = 0
    for x_val, y_val in zip(xs, ys):
        y_pre = forward(x_val)
        cost += math.pow((y_pre - y_val), 2)
    return cost / len(xs)


# gradient梯度
def gradient(xs, ys):
    gradient = 0
    for x_val, y_val in zip(xs, ys):
        y_pre = forward(x_val)
        gradient += x_val * (y_pre - y_val)
    return 2 * gradient / len(xs)

print("before train: ",4, forward(4))
for epoch in range(100):
    w -= 0.01*gradient(x_data,y_data)
    print("epoch ", epoch, "cost ", cost(x_data,y_data), "w ", w)
print("after train: ", 4, forward(4))