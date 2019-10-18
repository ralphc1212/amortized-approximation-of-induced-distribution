import torch
from utils.vgg import *
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.datasets import CIFAR10, LSUN
import numpy as np
from geomloss import SamplesLoss

num_samples = 700

def cost_func(x, y):
    x_norm = x.norm(dim=2)[:, :, None]
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = y.norm(dim=2)[:, None]

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)

    return torch.clamp(dist, 0.0, np.inf)

def test(args, model, device, test_loader, temperature = None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(data.shape[0], -1)
            if temperature:
                output = model(data)/temperature
            else:
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
    LF = SamplesLoss(loss='sinkhorn', p=2, blur=0.05, backend='tensorized', cost=cost_func)

    for batch_idx, (data, target) in enumerate(approx_loader):
        data, target = data.to(device), target.to(device)
        f_optimizer.zero_grad()

        with torch.no_grad():
            # To be consistant with KL, the exp() function is changed to softplus,
            # i.e., alpha0 = softplus(g).
            # Note that, for mmd, the exp() function can be used directly for faster convergence,
            # without tuning hyper-parameters.
            g_out = F.softplus(gmodel(data))
            output = F.softmax((output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
                device)), dim=2).clamp(0.0001, 0.9999)

        f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)

        s1 = torch.distributions.Dirichlet(pi).rsample((num_samples,)).permute(1,0,2)

        loss = LF(output, s1).mean()

        loss.backward()
        f_optimizer.step()

        if batch_idx == 0:
            print('Train Epoch: {}, Loss: {:.6f}'.format(
                epoch, loss.item()))

        g_optimizer.zero_grad()

        g_out = F.softplus(gmodel(data))

        with torch.no_grad():
            output = F.softmax((output_samples[batch_idx * approx_loader.batch_size:(batch_idx + 1) * approx_loader.batch_size].to(
                    device)), dim=2).clamp(0.0001, 0.9999)

        with torch.no_grad():
            f_out = F.softmax(fmodel(data), dim=1)

        pi = f_out.mul(g_out)
        s1 = torch.distributions.Dirichlet(pi).rsample((num_samples,)).permute(1,0,2)

        loss = LF(output, s1).mean()

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
            # data = data.view(data.shape[0], -1)

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
            # data = data.view(data.shape[0], -1)

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
    parser = argparse.ArgumentParser(description='Amortized approximation on Cifar10')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--approx-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to approx (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dropout-rate', type=float, default=0.5, metavar='p_drop',
                        help='dropout rate')
    parser.add_argument('--model-path', type=str, default='./checkpoint/', metavar='N',
                        help='path where the model params are saved.')
    parser.add_argument('--from-approx-model', type=int, default=1,
                        help='if our model is loaded or trained')
    parser.add_argument('--test-ood-from-disk', type=int, default=1,
                        help='generate test samples or load from disk')
    parser.add_argument('--ood-name', type=str, default='lsun',
                        help='name of the used ood dataset')

    ood = 'lsun'

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': False} if use_cuda else {}

    trans_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        trans_norm,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        trans_norm,
    ])

    tr_data = CIFAR10(root='../../data', train=True, download=True, transform=transform_train)

    te_data = CIFAR10(root='../../data', train=False, download=True, transform=transform_test)

    ood_data = LSUN(root='../../data', classes='test', transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        tr_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        te_data,
        batch_size=args.batch_size, shuffle=False,  **kwargs)

    ood_loader = torch.utils.data.DataLoader(
        ood_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = VGG('VGG19').to(device)

    model.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])

    test(args, model, device, test_loader)

    if args.from_approx_model == 0:
        output_samples = torch.load('./cifar10-vgg19rand-tr-samples.pt')

    # --------------- training approx ---------

    print('approximating ...')
    fmodel = VGG('VGG19').to(device)
    gmodel = VGG('VGG19', concentration=True).to(device)

    if args.from_approx_model == 0:
        f_optimizer = optim.SGD(fmodel.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        g_optimizer = optim.SGD(gmodel.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        best_acc = 0
        for epoch in range(1, args.approx_epochs + 1):
            train_approx(args, fmodel, gmodel, device, train_loader, f_optimizer, g_optimizer, output_samples, epoch)
            acc = test(args, fmodel, device, test_loader)

            if acc > best_acc:
                torch.save(fmodel.state_dict(), args.model_path + 'cifar10rand-mean-emd.pt')
                torch.save(gmodel.state_dict(), args.model_path + 'cifar10rand-conc-emd.pt')
                best_acc = acc

    else:
        fmodel.load_state_dict(torch.load(args.model_path + 'cifar10rand-mean-emd.pt'))
        gmodel.load_state_dict(torch.load(args.model_path + 'cifar10rand-conc-emd.pt'))
    
    teacher_test_samples = torch.load('./cifar10-vgg19rand-te-samples.pt')

    teacher_ood_samples  = torch.load('./cifar10-vgg19rand-lsun-samples.pt')

    # fitting individual Dirichlet is not in the sample code as it's time-consuming
    eval_approx(args, fmodel, gmodel, device, test_loader, ood_loader, teacher_test_samples, teacher_ood_samples)

if __name__ == '__main__':
    main()
