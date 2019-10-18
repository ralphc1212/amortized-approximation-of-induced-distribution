import torchvision.transforms as transforms

from utils.vgg import *

import torchvision.datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

ood = 'lsun'

trans_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    trans_norm,
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    trans_norm,
])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=2048, shuffle=False, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False, num_workers=8)

oodset = torchvision.datasets.LSUN(root='../../data', classes='test', transform=transform_test)

ood_loader = torch.utils.data.DataLoader(oodset, batch_size=512, shuffle=False, num_workers=4)

print('==> Building model..')
net = VGG('VGG19', dropout=0.1)

net = net.to(device)

net.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])

net.train()

for m in net.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# training
with torch.no_grad():
    all_samples = []
    for i in range(700):
        samples_a_round = []
        for data, target in train_loader:
            data = data.to(device)
            # output = F.softmax(net(data))
            output = net(data)
            samples_a_round.append(output)
        samples_a_round = torch.cat(samples_a_round).cpu()
        all_samples.append(samples_a_round)
    all_samples = torch.stack(all_samples).permute(1,0,2)
    torch.save(all_samples, 'cifar10-vgg19rand-tr-samples.pt')

# testing
with torch.no_grad():
    all_samples = []
    for i in range(700):
        samples_a_round = []
        for data, target in test_loader:
            data = data.to(device)
            # output = F.softmax(net(data))
            output = net(data)
            samples_a_round.append(output)
        samples_a_round = torch.cat(samples_a_round).cpu()
        all_samples.append(samples_a_round)
    all_samples = torch.stack(all_samples).permute(1,0,2)
    torch.save(all_samples, 'cifar10-vgg19rand-te-samples.pt')

# lsun
with torch.no_grad():
    all_samples = []
    for i in range(700):
        samples_a_round = []
        for data, target in ood_loader:
            data = data.to(device)
            # output = F.softmax(net(data))
            output = net(data)
            samples_a_round.append(output)
        samples_a_round = torch.cat(samples_a_round).cpu()
        all_samples.append(samples_a_round)
    all_samples = torch.stack(all_samples).permute(1,0,2)
    torch.save(all_samples, 'cifar10-vgg19rand-lsun-samples.pt')