import sys
sys.path.append('~/projects/amt_approx_simplex')
import torch.utils.data as tud
from src.utils.model_zoo import mnist_net
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import torch.distributions.dirichlet
from torchvision.datasets import EMNIST

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output,dim=1), target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train_bayesian(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output,dim=1), target)
        loss.backward()
        optimizer.step()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training MCDP Bayes teacher and sampling')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
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
                        help='number of posterior samples')
    parser.add_argument('--model-path', type=str, default='../saved_models/emnist_mcdp/', metavar='N',
                        help='number of posterior samples from the Bayesian model')
    parser.add_argument('--from-model', type=int, default=1, metavar='N',
                        help='if our model is loaded or trained')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    tr_data = EMNIST('../data', split='balanced', train=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    te_data = EMNIST('../data', split='balanced', train=False, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    train_loader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        te_data,
        batch_size=args.batch_size, shuffle=True,  **kwargs)

    model = mnist_net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # --------------- train or load teacher -----------
    if args.from_model == 1:
        print('loading teacher model ...')
        model.load_state_dict(torch.load(args.model_path+'mcdp-emnist.pt'))
    else:
        print('training teacher model ...')
        schedule = [50, 100, 150, 200, 250]
        best = 0
        for epoch in range(1, args.epochs + 1):
            if epoch in schedule:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
            train_bayesian(args, model, device, train_loader, optimizer, epoch)
            print("teacher training epoch: {}".format(epoch))
            test_acc = test(args, model, device, test_loader)
            if test_acc > best:
                torch.save(model.state_dict(), args.model_path+'mcdp-emnist.pt')
                best = test_acc

    train_loader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    print('generating particles for training data ...')
    # for an easier training of amortized approximation,
    # instead of sampling param. during approx,
    # get particles on simplex and store them first.
    with torch.no_grad():
        all_samples = []
        for i in range(500):
            samples_a_round = []
            for data, target in train_loader:
                data = data.to(device)
                output = F.softmax(model(data))
                samples_a_round.append(output)
            samples_a_round = torch.cat(samples_a_round).cpu()
            all_samples.append(samples_a_round)
        all_samples = torch.stack(all_samples).permute(1,0,2)

        torch.save(all_samples, args.model_path+'emnist-mcdp-samples.pt')

if __name__ == '__main__':
    main()
