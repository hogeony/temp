# pytorch 1.1.0 (gpu, CUDA 10)
import torch, torchvision, time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

batch_size = 100
mnist_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
data_train = torchvision.datasets.MNIST(root='./data', train=True, download= True, transform=mnist_transform)
data_test = torchvision.datasets.MNIST(root='./data', train=False, download= True, transform=mnist_transform)
dataloader_train = DataLoader(dataset=data_train,batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(dataset=data_test,batch_size=batch_size)

torch.cuda.set_device('cuda:0')
model = Model()
model.cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)

time_start = time.time()
for epoch in range(3):
    for idx, (data, target) in enumerate(dataloader_train):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print(epoch)
time_end = time.time()
print(time_end-time_start)
# 30.35199952125549

model.eval()
correct = 0
time_start = time.time()
for data, target in dataloader_test:
    data, target = Variable(data.cuda()), Variable(target.cuda())
    output = model(data)
    prediction = output.data.max(1)[1]   # first column has actual prob.
    correct += prediction.eq(target.data).sum().item()
print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(dataloader_test.dataset)))
time_end = time.time()
print(time_end-time_start)

# Test set: Accuracy: 99.10%
# 1.0610148811340332

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
# 3,274,634
