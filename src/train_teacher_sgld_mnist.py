import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
from torchvision import datasets, transforms
from src.utils.model_zoo import mnist_mlp
import torch.nn.functional as F
from pysgmcmc.optimizers.sgld import SGLD

def train_bayesian(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output,dim=1), target)
        loss.backward()
        optimizer.step()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output,dim=1), target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='run approximation to LeNet on Mnist')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dropout-rate', type=float, default=0.5, metavar='p_drop',
                        help='dropout rate')
    parser.add_argument('--S', type=int, default=500, metavar='N',
                        help='number of posterior samples from the Bayesian model')
    parser.add_argument('--model-path', type=str, default='../saved_models/mnist_sgld/', metavar='N',
                        help='number of posterior samples from the Bayesian model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = mnist_mlp(dropout=False).to(device)
    optimizer = SGLD(model.parameters(), lr=args.lr)

    import copy
    import pickle as pkl

    for epoch in range(1, args.epochs + 1):
        train_bayesian(args, model, device, train_loader, optimizer, epoch)
        print("epoch: {}".format(epoch))
        test(args, model, device, test_loader)

        # save models
        torch.save(model.state_dict(), args.model_path+'sgld-mnist.pt')

    # save samples
    param_samples = []
    while (1):
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = data.view(data.shape[0], -1)
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()
            optimizer.step()
            param_samples.append(copy.deepcopy(model.state_dict()))
            if param_samples.__len__() >= args.S:
                print('1', len(param_samples))
                break
        if param_samples.__len__() >= args.S:
            print('2', len(param_samples))
            break
    with open(args.model_path + "sgld_samples.pkl", "wb") as f:
        print('3', len(param_samples))
        pkl.dump(param_samples, f)

    test(args, model, device, test_loader)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # generate teacher train samples

    with torch.no_grad():
        # obtain ensemble outputs
        all_samples = []
        for i in range(500):
            samples_a_round = []
            model.load_state_dict(param_samples[i])
            for data, target in train_loader:
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                output = F.softmax(model(data))
                samples_a_round.append(output)
            samples_a_round = torch.cat(samples_a_round).cpu()
            all_samples.append(samples_a_round)
        all_samples = torch.stack(all_samples).permute(1,0,2)

        torch.save(all_samples, args.model_path+'mnist-sgld-train-samples.pt')

    # generate teacher test  samples

    with torch.no_grad():
        # obtain ensemble outputs
        all_samples = []
        for i in range(500):
            samples_a_round = []
            model.load_state_dict(param_samples[i])
            for data, target in test_loader:
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                output = F.softmax(model(data))
                samples_a_round.append(output)
            samples_a_round = torch.cat(samples_a_round).cpu()
            all_samples.append(samples_a_round)
        all_samples = torch.stack(all_samples).permute(1,0,2)

        torch.save(all_samples, args.model_path+'mnist-sgld-test-samples.pt')

    # generate teacher omniglot samples

    ood_data = datasets.Omniglot('../../data', download=True, transform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    with torch.no_grad():
        # obtain ensemble outputs
        all_samples = []
        for i in range(500):
            samples_a_round = []
            model.load_state_dict(param_samples[i])
            for data, target in ood_loader:
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                output = F.softmax(model(data))
                samples_a_round.append(output)
            samples_a_round = torch.cat(samples_a_round).cpu()
            all_samples.append(samples_a_round)
        all_samples = torch.stack(all_samples).permute(1,0,2)

        torch.save(all_samples, args.model_path+'mnist-sgld-omniglot-samples.pt')

    # generate teacher SEMEION samples

    ood_data = datasets.SEMEION('../../data', download=True, transform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    with torch.no_grad():
        # obtain ensemble outputs
        all_samples = []
        for i in range(500):
            samples_a_round = []
            model.load_state_dict(param_samples[i])
            for data, target in ood_loader:
                data = data.to(device)
                data = data.view(data.shape[0], -1)
                output = F.softmax(model(data))
                samples_a_round.append(output)
            samples_a_round = torch.cat(samples_a_round).cpu()
            all_samples.append(samples_a_round)
        all_samples = torch.stack(all_samples).permute(1,0,2)

        torch.save(all_samples, args.model_path+'mnist-sgld-SEMEION-samples.pt')

if __name__ == '__main__':
    main()
