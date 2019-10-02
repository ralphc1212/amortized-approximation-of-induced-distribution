import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home+'/projects/amt_approx_simplex')
import torch
from src.utils.model_zoo import mnist_mlp, mnist_mlp_h, mnist_mlp_g
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.datasets import MNIST
import numpy as np

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

def train_approx(args, fmodel, gmodel, device, approx_loader, f_optimizer, g_optimizer, output_samples, epoch):
    gmodel.train()
    fmodel.train()
    for batch_idx, (data, target) in enumerate(approx_loader):
        data, target = data.to(device), target.to(device)
        f_optimizer.zero_grad()
        data = data.view(data.shape[0], -1)

        with torch.no_grad():
            g_out = gmodel(data)
            output = output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
                device).clamp(0.00001, 0.99999)

        pi_p_avg = torch.mean(torch.log(output), dim=1)

        f_out = F.softmax(fmodel(data), dim=1)

        # for easier training, the exp() function is changed to softplus,
        # i.e., alpha0 = softplus(g)
        pi_q = f_out.mul(F.softplus(g_out))

        sum_pi_q = pi_q.sum(1)
        lgam_sum_pi_q = torch.lgamma(sum_pi_q)
        loss = -(lgam_sum_pi_q - torch.lgamma(pi_q).sum(1) + (pi_q-1).mul(pi_p_avg).sum(1))

        loss = loss.mean()
        loss.backward()
        f_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

        g_optimizer.zero_grad()

        g_out = gmodel(data)

        with torch.no_grad():
            output = output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
            device).clamp(0.00001, 0.99999)

        pi_p_avg = torch.mean(torch.log(output), dim=1)

        with torch.no_grad():
            f_out = F.softmax(fmodel(data), dim=1)

        pi_q = f_out.mul(F.softplus(g_out))

        sum_pi_q = pi_q.sum(1)
        lgam_sum_pi_q = torch.lgamma(sum_pi_q)
        loss = -(lgam_sum_pi_q - torch.lgamma(pi_q).sum(1) + (pi_q-1).mul(pi_p_avg).sum(1))

        loss = loss.mean()
        loss.backward()
        g_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

def eval_approx(args,  smean, sconc, device, test_loader,
                ood_loader, teacher_test_samples, teacher_ood_samples):
    smean.eval()
    sconc.eval()
    miscls_origin = []
    miscls_approx = []
    entros_origin_1 = []
    fentros_approx_1 = []
    entros_approx_1 = []
    entros_origin_2 = []
    fentros_approx_2 = []
    entros_approx_2 = []
    maxp_origin_1 = []
    maxp_approx_1 = []
    maxp_origin_2 = []
    maxp_approx_2 = []
    gvalue_approx_1 = []
    gvalue_approx_2 = []

    batch_idx = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1)
            pi_q = f_out.mul(g_out)

            samples_p_pi = teacher_test_samples[batch_idx*test_loader.batch_size: (batch_idx+1)*test_loader.batch_size].to(device)
            avg_origin_output = torch.mean(samples_p_pi, dim=1)

            pi_p_avg_batch = avg_origin_output
            origin_result = torch.argmax(pi_p_avg_batch, dim=1)
            approx_result = torch.argmax(pi_q, dim=1)

            miscls_approx.append((1-(approx_result == target).float()).cpu().numpy())
            miscls_origin.append((1-(origin_result == target).float()).cpu().numpy())

            entro_origin = (-torch.bmm(pi_p_avg_batch.view(data.shape[0], 1, -1),
                                      torch.log(pi_p_avg_batch.view(data.shape[0], -1, 1)))).view(-1)

            fentro_approx = (-torch.bmm(f_out.view( data.shape[0], 1, -1),
                                      torch.log(f_out.view(data.shape[0], -1, 1)))).view(-1)

            alpha = pi_q
            alpha0 = alpha.sum(1)

            entro_approx = torch.lgamma(alpha).sum(1) \
                           - torch.lgamma(alpha0) \
                           + (alpha0 - 10).mul(torch.digamma(alpha0)) \
                           - ((alpha - 1 ).mul(torch.digamma(alpha))).sum(1)

            entros_origin_1.append(entro_origin.cpu().numpy())
            fentros_approx_1.append(fentro_approx.cpu().numpy())
            entros_approx_1.append(entro_approx.cpu().numpy())

            maxp_origin = 1./torch.max(pi_p_avg_batch, dim=1)[0]
            maxp_approx = 1./torch.max(f_out, dim=1)[0]

            maxp_origin_1.append(maxp_origin.cpu().numpy())
            maxp_approx_1.append(maxp_approx.cpu().numpy())
            gvalue_approx_1.append(1./g_out.cpu().numpy())
            batch_idx += 1

    miscls_approx = np.concatenate(miscls_approx)
    miscls_origin = np.concatenate(miscls_origin)
    entros_origin_1 = np.concatenate(entros_origin_1)
    fentros_approx_1 = np.concatenate(fentros_approx_1)
    maxp_origin_1 = np.concatenate(maxp_origin_1)
    maxp_approx_1 = np.concatenate(maxp_approx_1)
    gvalue_approx_1 = np.concatenate(gvalue_approx_1)
    correct_approx = np.sum(1-miscls_approx)
    correct_ensemble = np.sum(1-miscls_origin)

    print("AUROC (entros_origin_1): ", roc_auc_score(miscls_origin, entros_origin_1))
    print("AUROC (hentros_approx_1): ", roc_auc_score(miscls_approx, fentros_approx_1))
    print("AUROC (maxp_approx_1):   ", roc_auc_score(miscls_approx, maxp_approx_1))
    print("AUROC (maxp_origin_1):   ", roc_auc_score(miscls_origin, maxp_origin_1))
    print("AUROC (gvalue_approx_1): ", roc_auc_score(miscls_approx, gvalue_approx_1))
    print("AUPR  (entros_origin_1): ", average_precision_score(miscls_origin, entros_origin_1))
    print("AUPR  (hentros_approx_1): ", average_precision_score(miscls_approx, fentros_approx_1))
    print("AUPR  (maxp_approx_1):   ", average_precision_score(miscls_approx, maxp_approx_1))
    print("AUPR  (maxp_origin_1):   ", average_precision_score(miscls_origin, maxp_origin_1))
    print("AUPR  (gvalue_approx_1): ", average_precision_score(miscls_approx, gvalue_approx_1))
    print('approx ACC :', correct_approx / (len(test_loader.dataset)))
    print('ensemble ACC :', correct_ensemble / (len(test_loader.dataset)))


    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(ood_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            g_out = F.softplus(sconc(data))
            f_out = F.softmax(smean(data), dim=1)
            pi_q = f_out.mul(g_out)

            samples_p_pi = teacher_ood_samples[batch_idx*ood_loader.batch_size: (batch_idx+1)*ood_loader.batch_size].to(device)

            avg_origin_output = torch.mean(samples_p_pi, dim=1)

            pi_p_avg_batch = avg_origin_output

            entro_origin = (-torch.bmm(pi_p_avg_batch.view(data.shape[0], 1, -1),
                                      torch.log(pi_p_avg_batch.view(data.shape[0], -1, 1)))).view(-1)

            fentro_approx = (-torch.bmm(f_out.view( data.shape[0], 1, -1),
                                      torch.log(f_out.view(data.shape[0], -1, 1)))).view(-1)

            entros_origin_2.append(entro_origin.cpu().numpy())
            fentros_approx_2.append(fentro_approx.cpu().numpy())

            alpha = pi_q
            alpha0 = alpha.sum(1)

            entro_approx = torch.lgamma(alpha).sum(1) \
                           - torch.lgamma(alpha0) \
                           + (alpha0 - 10).mul(torch.digamma(alpha0)) \
                           - ((alpha - 1 ).mul(torch.digamma(alpha))).sum(1)

            entros_approx_2.append(entro_approx.cpu().numpy())

            maxp_origin = 1./torch.max(pi_p_avg_batch, dim=1)[0]
            maxp_approx = 1./torch.max(f_out, dim=1)[0]

            maxp_origin_2.append(maxp_origin.cpu().numpy())
            maxp_approx_2.append(maxp_approx.cpu().numpy())
            gvalue_approx_2.append(1./g_out.cpu().numpy())
            batch_idx += 1

        entros_origin_2 = np.concatenate(entros_origin_2)
        fentros_approx_2 = np.concatenate(fentros_approx_2)
        maxp_origin_2 = np.concatenate(maxp_origin_2)
        maxp_approx_2 = np.concatenate(maxp_approx_2)
        gvalue_approx_2 = np.concatenate(gvalue_approx_2)

        fentros_approx = np.concatenate([fentros_approx_1, fentros_approx_2])
        entros_origin = np.concatenate([entros_origin_1 , entros_origin_2])
        maxp_approx = np.concatenate([maxp_approx_1 , maxp_approx_2])
        maxp_origin = np.concatenate([maxp_origin_1 , maxp_origin_2])
        gvalue_approx = np.concatenate([gvalue_approx_1 , gvalue_approx_2])
        ood = np.concatenate([np.zeros(test_loader.dataset.__len__()),
                              np.ones(ood_loader.dataset.__len__())])

        print("-----------------------")
        print("AUROC (entros_origin): ", roc_auc_score(ood, entros_origin))
        print("AUROC (hentros_approx): ", roc_auc_score(ood, fentros_approx))
        print("AUROC (maxp_approx):   ", roc_auc_score(ood, maxp_approx))
        print("AUROC (maxp_origin):   ", roc_auc_score(ood, maxp_origin))
        print("AUROC (gvalue_approx): ", roc_auc_score(ood, gvalue_approx))
        print("AUPR  (entros_origin): ", average_precision_score(ood, entros_origin))
        print("AUPR  (hentros_approx): ", average_precision_score(ood, fentros_approx))
        print("AUPR  (maxp_approx):   ", average_precision_score(ood, maxp_approx))
        print("AUPR  (maxp_origin):   ", average_precision_score(ood, maxp_origin))
        print("AUPR  (gvalue_approx): ", average_precision_score(ood, gvalue_approx))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Amortized approximation on MNIST')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--approx-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to approx (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
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
    parser.add_argument('--model-path', type=str, default='../saved_models/mnist_mcdp/', metavar='N',
                        help='path where the model params are saved.')
    parser.add_argument('--from-approx-model', type=int, default=1,
                        help='if our model is loaded or trained')
    parser.add_argument('--test-ood-from-disk', type=int, default=1,
                        help='generate test samples or load from disk')
    parser.add_argument('--ood-name', type=str, default='omniglot',
                        help='name of the used ood dataset')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}

    tr_data = MNIST('../data', train=True, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    te_data = MNIST('../data', train=False, transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]), download=True)

    ood_data = datasets.Omniglot('../data', download=True,  transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))

    if args.ood_name == 'omniglot':
        ood_data = datasets.Omniglot('../data', download=True, transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
    elif args.ood_name == 'SEMEION':
        ood_data = datasets.SEMEION('../data', download=True,  transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))

    train_loader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        te_data,
        batch_size=args.batch_size, shuffle=False,  **kwargs)

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = mnist_mlp().to(device)

    model.load_state_dict(torch.load(args.model_path + 'mcdp-mnist.pt'))

    test(args, model, device, test_loader)

    if args.from_approx_model == 0:
        output_samples = torch.load(args.model_path + 'mnist-mcdp-samples.pt')

    # --------------- training approx ---------

    print('approximating ...')
    fmodel = mnist_mlp_h().to(device)
    gmodel = mnist_mlp_g().to(device)

    if args.from_approx_model == 0:
        f_optimizer = optim.SGD(fmodel.parameters(), lr=args.lr, momentum=args.momentum)
        g_optimizer = optim.SGD(gmodel.parameters(), lr=args.lr, momentum=args.momentum)
        best_acc = 0
        for epoch in range(1, args.approx_epochs + 1):
            train_approx(args, fmodel, gmodel, device, train_loader, f_optimizer, g_optimizer, output_samples, epoch)
            acc = test(args, fmodel, device, test_loader)
            if acc > best_acc:
                torch.save(fmodel.state_dict(), args.model_path + 'mcdp-mnist-mean.pt')
                torch.save(gmodel.state_dict(), args.model_path + 'mcdp-mnist-conc.pt')
                best_acc = acc

    else:
        fmodel.load_state_dict(torch.load(args.model_path + 'mcdp-mnist-mean.pt'))
        gmodel.load_state_dict(torch.load(args.model_path + 'mcdp-mnist-conc.pt'))

    print('generating teacher particles for testing&ood data ...')
    # generate particles for test and ood dataset
    model.train()
    if args.test_ood_from_disk == 1:
        teacher_test_samples = torch.load(args.model_path + 'mnist-mcdp-test-samples.pt')
    else:
        with torch.no_grad():
            # obtain ensemble outputs
            all_samples = []
            for i in range(500):
                samples_a_round = []
                for data, target in test_loader:
                    data = data.to(device)
                    output = F.softmax(model(data))
                    samples_a_round.append(output)
                samples_a_round = torch.cat(samples_a_round).cpu()
                all_samples.append(samples_a_round)
            teacher_test_samples = torch.stack(all_samples).permute(1,0,2)

            torch.save(all_samples, args.model_path + 'mnist-mcdp-test-samples.pt')

    if args.test_ood_from_disk == 1:
        teacher_ood_samples = torch.load(args.model_path + args.ood_name + '-mcdp-ood-samples-trd-mnist.pt')
    else:
        with torch.no_grad():
            # obtain ensemble outputs
            all_samples = []
            for i in range(500):
                samples_a_round = []
                for data, target in ood_loader:
                    data = data.to(device)
                    output = F.softmax(model(data))
                    samples_a_round.append(output)
                samples_a_round = torch.cat(samples_a_round).cpu()
                all_samples.append(samples_a_round)
            teacher_ood_samples = torch.stack(all_samples).permute(1,0,2)

            torch.save(all_samples, args.model_path + args.ood_name + '-mcdp-ood-samples-trd-mnist.pt')

    # fitting individual Dirichlet is not in the sample code as it's time-consuming
    eval_approx(args, fmodel, gmodel, device, test_loader, ood_loader, teacher_test_samples, teacher_ood_samples)

if __name__ == '__main__':
    main()
