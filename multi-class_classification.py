import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim

# 准备数据
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 均值和方差
])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义模型
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = torch.nn.Linear(784, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, 128)
        self.linear_4 = torch.nn.Linear(128, 64)
        self.linear_5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = self.linear_5(x)
        return x


model = Net()

# 评价和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)


# 开始迭代
def train(epoch):
    total_loss = 0.0
    for i, (x, y) in enumerate(iterable=train_loader, start=0):
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 300 == 299:
            print('[%d, %5d] loss: %.3f' % epoch + 1, i + 1, total_loss / 300)
            total_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(iterable=test_loader, start=0):
            outputs = model(inputs)
            nums, pred = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
