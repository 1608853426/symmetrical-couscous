import torch.nn.functional as F
import torch.optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)
test_dataset = datasets.MNIST(root='./dataset/mnist/',
                              train=False,
                              transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=2)


# 定义model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=10,
                               kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=(5, 5))
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.5)


# 开始迭代
def train(epoch):
    total_loss = 0.0
    for i, (x, y) in enumerate(iterable=train_loader, start=0):
        #x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 300))
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
