#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F


# --------------- for Bayesian LR ---------------

class meanNet_fc(nn.Module):
    def __init__(self, input_dim):
        super(meanNet_fc, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim*2))
        self.fc2 = nn.Linear(int(input_dim*2), int(input_dim/2))
        # self.fc3 = nn.Linear(input_dim, int(input_dim*2))
        self.fc4 = nn.Linear(int(input_dim/2), 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        # return F.relu(x)
        return x

class gNet_fc(nn.Module):
    def __init__(self, input_dim):
        super(gNet_fc, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim*2))
        self.fc2 = nn.Linear(int(input_dim*2), int(input_dim/2))
        # self.fc3 = nn.Linear(input_dim, int(input_dim*2))
        self.fc4 = nn.Linear(int(input_dim/2), 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        # return F.relu(x)
        return x

class disc_simple_gundam(nn.Module):

    def __init__(self, input_dim, out_dim=2):
        super(disc_simple_gundam, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.x_transform1 = nn.Linear(input_dim, int(input_dim*2))
        self.x_transform2 = nn.Linear(int(input_dim*2), int(input_dim*2))
        self.trans_layer1_weight = nn.Linear(int(input_dim*2), out_dim*out_dim*10)
        self.trans_layer1_bias = nn.Linear(int(input_dim*2), out_dim*10)
        self.trans_layer2_weight = nn.Linear(int(input_dim*2), out_dim*10*out_dim*2)
        self.trans_layer2_bias = nn.Linear(int(input_dim*2), out_dim*2)
        self.trans_layer3_weight = nn.Linear(int(input_dim*2), out_dim*2)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.x_transform1(x))
        x = F.relu(self.x_transform2(x))
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, self.out_dim*10)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, self.out_dim*10)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], self.out_dim*10, self.out_dim*2)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, self.out_dim*2)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], self.out_dim*2, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

# --------------- for Bayesian LeNet ------------

## the pytorch example NN for MNIST

class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout2d(x, p=0.5)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(x, p=0.5)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout2d(x, p=0.5)
        x = self.fc2(x)
        return x

class mnist_net_f(nn.Module):
    def __init__(self):
        super(mnist_net_f, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class mnist_net_g(nn.Module):
    def __init__(self):
        super(mnist_net_g, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --------------- for Mnist example_-------------

class mnist_exp(nn.Module):
    def __init__(self, dropout =False, out_dim = 10):
        super(mnist_exp, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, 0.4)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = F.dropout(x, 0.3)
        x = self.fc3(x)
        return x

    # def __init__(self, dropout =False):
    #     super(mnist_exp, self).__init__()
    #     self.layer = nn.Sequential(nn.Linear(784, 400), nn.Linear(400, 400), nn.Linear(400, 3))
    #
    # def forward(self, x):
    #     x = F.softplus(self.layer(x))
    #     return x

class disc_on_input(nn.Module):
    
    def __init__(self, out_dim = 10):
        super(disc_on_input, self).__init__()
        self.x_transform = nn.Linear(784, 400)
        self.pi_tranform = nn.Linear(out_dim, 400)
        self.fc_joint1 = nn.Linear(800, 400)
        self.fc_joint2 = nn.Linear(400, 400)
        self.fc_joint3 = nn.Linear(400, 1)

    def forward(self, x, pi):
        x = F.relu(self.x_transform(x))
        pi = F.relu(self.pi_tranform(pi))
        j = torch.cat((x.view(x.shape[0], -1),
                      pi.view(pi.shape[0], -1)), dim=1)
        j = F.relu(self.fc_joint1(j))
        j = F.relu(self.fc_joint2(j))
        j = self.fc_joint3(j)
        return j

# class disc_on_input_10cls(nn.Module):
#
#     def __init__(self, out_dim=10):
#         super(disc_on_input_10cls, self).__init__()
#         self.x_transform = nn.Linear(784, 1024)
#         self.pi_tranform = nn.Linear(out_dim, 1024)
#         self.fc_joint1 = nn.Linear(2048, 512)
#         # self.fc_joint2 = nn.Linear(1024, 256)
#         self.fc_joint3 = nn.Linear(512, 1)
#
#     def forward(self, x, pi):
#         x = F.relu(self.x_transform(x))
#         pi = F.relu(self.pi_tranform(pi))
#         j = torch.cat((x.view(x.shape[0], -1),
#                        pi.view(pi.shape[0], -1)), dim=1)
#         j = F.relu(self.fc_joint1(j))
#         # j = F.relu(self.fc_joint2(j))
#         j = self.fc_joint3(j)
#         return j

class disc_on_input_10cls(nn.Module):

    def __init__(self, out_dim=47):
        super(disc_on_input_10cls, self).__init__()
        self.x_transform = nn.Linear(784, 256)
        self.pi_tranform = nn.Linear(out_dim, 512)
        self.fc_joint1 = nn.Linear(768, 256)
        # self.fc_joint2 = nn.Linear(1024, 256)
        self.fc_joint3 = nn.Linear(256, 1)

    def forward(self, x, pi):
        x = F.relu(self.x_transform(x))
        pi = F.relu(self.pi_tranform(pi))
        # print(x.shape, pi.shape)
        j = torch.cat((x.view(x.shape[0], -1),
                       pi.view(pi.shape[0], -1)), dim=1)
        j = F.relu(self.fc_joint1(j))
        # j = F.relu(self.fc_joint2(j))
        j = self.fc_joint3(j)
        return j

class disc_on_input_fix_feat(nn.Module):

    def __init__(self, out_dim=47):
        super(disc_on_input_fix_feat, self).__init__()
        # self.x_transform = nn.Linear(784, 256)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.pi_tranform = nn.Linear(out_dim, 512)
        self.fc_joint1 = nn.Linear(1312, 256)
        # self.fc_joint2 = nn.Linear(1024, 256)
        self.fc_joint3 = nn.Linear(256, 1)

    def forward(self, x, pi):
        x = x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        pi = F.relu(self.pi_tranform(pi))
        # print(x.shape, pi.shape)
        j = torch.cat((x.view(x.shape[0], -1),
                       pi.view(pi.shape[0], -1)), dim=1)
        j = F.relu(self.fc_joint1(j))
        # j = F.relu(self.fc_joint2(j))
        j = self.fc_joint3(j)
        return j

class disc_genby_input_conv_small(nn.Module):

    def __init__(self, out_dim=10):
        super(disc_genby_input_conv_small, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 1024)
        self.trans_layer1_weight = nn.Linear(1024, out_dim*128)
        self.trans_layer1_bias = nn.Linear(1024, 128)
        self.trans_layer2_weight = nn.Linear(1024, 128*32)
        self.trans_layer2_bias = nn.Linear(1024, 32)
        self.trans_layer3_weight = nn.Linear(1024, 32)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = x.expand((pi.shape[0],)+x.shape)
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, 128)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, 128)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], 128, 32)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, 32)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], 32, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

class disc_genby_input_conv(nn.Module):

    def __init__(self, out_dim=10):
        super(disc_genby_input_conv, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 1024)
        self.trans_layer1_weight = nn.Linear(1024, out_dim*400)
        self.trans_layer1_bias = nn.Linear(1024, 400)
        self.trans_layer2_weight = nn.Linear(1024, 400*128)
        self.trans_layer2_bias = nn.Linear(1024, 128)
        self.trans_layer3_weight = nn.Linear(1024, 128)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = x.expand((pi.shape[0],)+x.shape)
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, 400)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, 400)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], 400, 128)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, 128)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], 128, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

class disc_genby_input(nn.Module):

    def __init__(self, out_dim=10):
        super(disc_genby_input, self).__init__()
        self.out_dim = out_dim
        self.x_transform1 = nn.Linear(784, 400)
        self.x_transform2 = nn.Linear(400, 1024)
        self.trans_layer1_weight = nn.Linear(1024, out_dim*400)
        self.trans_layer1_bias = nn.Linear(1024, 400)
        self.trans_layer2_weight = nn.Linear(1024, 400*128)
        self.trans_layer2_bias = nn.Linear(1024, 128)
        self.trans_layer3_weight = nn.Linear(1024, 128)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.x_transform1(x))
        x = F.relu(self.x_transform2(x))
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, 400)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, 400)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], 400, 128)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, 128)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], 128, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

class disc_genby_input_10cls(nn.Module):

    def __init__(self, out_dim=10):
        super(disc_genby_input_10cls, self).__init__()
        self.out_dim = out_dim
        self.x_transform1 = nn.Linear(784, 1024)
        self.x_transform2 = nn.Linear(1024, 1024)
        self.trans_layer1_weight = nn.Linear(1024, out_dim*512)
        self.trans_layer1_bias = nn.Linear(1024, 512)
        self.trans_layer2_weight = nn.Linear(1024, 512*256)
        self.trans_layer2_bias = nn.Linear(1024, 256)
        self.trans_layer3_weight = nn.Linear(1024, 256)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.x_transform1(x))
        x = F.relu(self.x_transform2(x))
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, 512)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, 512)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], 512, 256)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, 256)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], 256, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

class disc_genby_input_modi(nn.Module):

    def __init__(self, out_dim=10):
        super(disc_genby_input_modi, self).__init__()
        self.out_dim = out_dim
        self.x_transform1 = nn.Linear(784, 1024)
        self.x_transform2 = nn.Linear(1024, 2048)
        self.trans_layer1_weight = nn.Linear(2048, out_dim*128)
        self.trans_layer1_bias = nn.Linear(2048, 128)
        self.trans_layer2_weight = nn.Linear(2048, 400*64)
        self.trans_layer2_bias = nn.Linear(2048, 64)
        self.trans_layer3_weight = nn.Linear(2048, 64)

    # out_dim * 400 + 400 + 400 * 128 + 128 + 128
    def forward(self, x, pi):
        x = F.relu(self.x_transform1(x))
        x = F.relu(self.x_transform2(x))
        weight1 = self.trans_layer1_weight(x).view(x.shape[0], self.out_dim, 128)
        bias1 = self.trans_layer1_bias(x).view(x.shape[0], 1, 128)
        weight2 = self.trans_layer2_weight(x).view(x.shape[0], 128, 64)
        bias2 = self.trans_layer2_bias(x).view(x.shape[0], 1, 64)
        weight3 = self.trans_layer3_weight(x).view(x.shape[0], 64, 1)

        pi = torch.bmm(pi.view(pi.shape[0], 1, pi.shape[1]), weight1)
        pi = F.relu(pi + bias1)
        pi = torch.bmm(pi, weight2)
        pi = F.relu(pi + bias2)
        pi = torch.bmm(pi, weight3).view(pi.shape[0], 1)
        return pi

class mnist_exp_g(nn.Module):
    def __init__(self, dropout =False):
        super(mnist_exp_g, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = F.dropout(x, 0.25)
        x = self.fc3(x)
        return x

class mnist_expG(nn.Module):
    def __init__(self, dropout =False):
        super(mnist_expG, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 1)
        self.dropout = dropout

    def forward(self, x):

        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, 0.25)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = F.dropout(x, 0.25)
        x = self.fc3(x)
        return x

class simple_disc(nn.Module):
    def __init__(self):
        super(simple_disc, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class mnist_wsamples(nn.Module):
    def __init__(self, out_dim = 10):
        super(mnist_wsamples, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.fc1 = nn.Linear(64*7*7, 256)
        # self.fc2 = nn.Linear(256, 10)

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, out_dim)
    def forward(self, x, samples):
        x = F.relu(self.fc1(x))
        x = x.mul(samples[0])
        x = F.relu(self.fc2(x))
        x = x.mul(samples[1])
        x = self.fc3(x)
        return x

class mnist_alphaNet(nn.Module):
    def __init__(self):
        super(mnist_alphaNet, self).__init__()

        self.gconv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.gconv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.gfc1 = nn.Linear(320, 50)
        self.gfc2 = nn.Linear(50, 10)

        self.fconv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fconv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.ffc1 = nn.Linear(320, 50)
        self.ffc2 = nn.Linear(50, 10)

    def forward(self, x):
        gx = F.relu(self.gconv1(x))
        gx = F.dropout2d(gx, p=0.5, training=self.training)
        gx = F.max_pool2d(gx, 2)
        gx = F.relu(self.gconv2(gx))
        gx = F.dropout2d(gx, p=0.5, training=self.training)
        gx = F.max_pool2d(gx, 2)
        gx = gx.view(-1, 320)
        gx = F.relu(self.gfc1(gx))
        gx = F.dropout(gx, p=0.5, training=self.training)
        gx = self.gfc2(gx)

        fx = F.relu(self.fconv1(x))
        fx = F.dropout2d(fx, p=0.5, training=self.training)
        fx = F.max_pool2d(fx, 2)
        fx = F.relu(self.fconv2(fx))
        fx = F.dropout2d(fx, p=0.5, training=self.training)
        fx = F.max_pool2d(fx, 2)
        fx = fx.view(-1, 320)
        fx = F.relu(self.ffc1(fx))
        fx = F.dropout(fx, p=0.5, training=self.training)
        fx = F.softmax(self.ffc2(fx), dim=1)

        x = fx * torch.exp(gx)

        return x

class mnist_alphaNet_G(nn.Module):
    def __init__(self):
        super(mnist_alphaNet_G, self).__init__()

        self.gconv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.gconv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.gfc1 = nn.Linear(320, 50)
        self.gfc2 = nn.Linear(50, 1)

        # self.gconv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        # self.gconv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        # self.gfc1 = nn.Linear(64*7*7, 256)
        # self.gfc2 = nn.Linear(256, 1)

    #     self.fc1 = nn.Linear(784, 400)
    #     self.fc2 = nn.Linear(400, 400)
    #     self.fc3 = nn.Linear(400, 1)
    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = F.relu(self.fc2(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.fc3(x)
    #     return x

    def forward(self, x):
        gx = F.relu(self.gconv1(x))
        gx = F.dropout2d(gx, p=0.5, training=self.training)
        gx = F.max_pool2d(gx, 2)
        gx = F.relu(self.gconv2(gx))
        gx = F.dropout2d(gx, p=0.5, training=self.training)
        gx = F.max_pool2d(gx, 2)
        gx = gx.view(-1, 320)
        # gx = gx.view(-1, 64*7*7)
        gx = F.relu(self.gfc1(gx))
        gx = F.dropout(gx, p=0.5, training=self.training)
        gx = self.gfc2(gx)

        # gx = F.softplus(self.gconv1(x))
        # gx = F.dropout2d(gx, p=0.5, training=self.training)
        # gx = F.max_pool2d(gx, 2)
        # gx = F.softplus(self.gconv2(gx))
        # gx = F.dropout2d(gx, p=0.5, training=self.training)
        # gx = F.max_pool2d(gx, 2)
        # gx = gx.view(-1, 320)
        # # gx = gx.view(-1, 64*7*7)
        # gx = F.softplus(self.gfc1(gx))
        # gx = F.dropout(gx, p=0.5, training=self.training)
        # gx = self.gfc2(gx)

        return gx

class mnist_alphaNet_F(nn.Module):
    def __init__(self):
        super(mnist_alphaNet_F, self).__init__()

        # self.fconv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.fconv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.ffc1 = nn.Linear(320, 50)
        # self.ffc2 = nn.Linear(50, 10)

        self.fconv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.fconv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.ffc1 = nn.Linear(64*7*7, 256)
        self.ffc2 = nn.Linear(256, 10)

    def forward(self, x):
        fx = F.relu(self.fconv1(x))
        fx = F.dropout2d(fx, p=0.5, training=self.training)
        fx = F.max_pool2d(fx, 2)
        fx = F.relu(self.fconv2(fx))
        fx = F.dropout2d(fx, p=0.5, training=self.training)
        fx = F.max_pool2d(fx, 2)
        fx = fx.view(-1, 320)
        # fx = fx.view(-1, 64*7*7)
        fx = F.relu(self.ffc1(fx))
        fx = F.dropout(fx, p=0.5, training=self.training)
        fx = F.softmax(self.ffc2(fx), dim=1)

        return fx

class mnist_gNet(nn.Module):
    def __init__(self):
        super(mnist_gNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

# class mnist_gNet_fc(nn.Module):
#     def __init__(self):
#         super(mnist_gNet_fc, self).__init__()
#         self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(3)
#         self.fc1 = nn.Linear(432, 1)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.bn1(x)
#         # x = F.dropout2d(x, p=0.5, training=self.training)
#         x = F.max_pool2d(x, 2)
#
#         x = x.view(-1, 432)
#         return self.fc1(x)


# --------------- for sgld example_--------------

class mnist_sgld(nn.Module):
    def __init__(self, n_filters1=64,
                 n_filters2=64,
                 n_fc=256,
                 dropout=False):
        super(mnist_sgld, self).__init__()

        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.n_fc = n_fc
        self.dropout = dropout

        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, self.n_filters1, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(self.n_filters1, self.n_filters2, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(self.n_filters2 * 7 * 7, self.n_fc)
        self.fc2 = nn.Linear(self.n_fc, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.n_filters2 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        if self.dropout: x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x)


### for cifar10 vgg ###

vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.features = self._make_layers(vgg16_cfg)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           nn.Dropout2d(p= 0.3),
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class conv_mnf(nn.Module):
    def __init__(self, in_channel, num_filter, kernel_size,
                  mu_w, mu_bias, clipped_var_w, clipped_var_bias):
        super(conv_mnf, self).__init__()
        self.num_filter = num_filter
        self.kernel_size = kernel_size
        # self.padding = padding
        # self.subsample = subsample
        # self.thres_var = thres_var
        self.conv_mu  = nn.Conv2d(in_channel, num_filter, 5, 1)
        self.conv_var = nn.Conv2d(in_channel, num_filter, 5, 1)
        self.conv_mu.weight = mu_w
        self.conv_mu.bias = mu_bias
        self.conv_var.weight = clipped_var_w
        self.conv_var.bias = clipped_var_bias

    def forward(self, h, sample_z):
        # sample_z = sample_z.view((-1,1,1))
        mu_out  = self.conv_mu(h)
        var_out = self.conv_var(torch.pow(h,2))
        mean_gout = mu_out * sample_z
        var_gout = torch.sqrt(var_out) * torch.randn(mean_gout.size()).cuda()
        out = mean_gout + var_gout
        return out

class fc_mnf(nn.Module):
    def __init__(self,  mu_w, mu_bias, clipped_var_w, clipped_var_bias):
        super(fc_mnf, self).__init__()
        # self.num_filter = num_filter
        # self.num_row = num_row
        # self.num_col = num_col
        # self.padding = padding
        # self.input_shape = input_shape,
        # self.subsample = subsample
        self.clipped_var_w = clipped_var_w.cuda()
        self.mu_w = mu_w.cuda()
        self.mu_bias = mu_bias.view(1,-1).cuda()
        self.clipped_var_bias = clipped_var_bias.cuda()

    def forward(self, h, sample_z):
        xt = h * sample_z
        mu_out = torch.mm(xt, self.mu_w) + self.mu_bias
        varin = torch.mm(torch.pow(h, 2), self.clipped_var_w) + self.clipped_var_bias
        xin = torch.sqrt(varin)
        sigma_out = xin * torch.randn(mu_out.size()).cuda()
        output = mu_out + sigma_out
        return output

class lenet_mnf(nn.Module):
    def __init__(self,
                 params,
                 in_channels = (1,20),
                 num_filters = (20,50),
                 kernel_sizes = (5,5),
                 ):
        super(lenet_mnf, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = conv_mnf(in_channels[0],num_filters[0],kernel_sizes[0],
                              params[0][0],params[0][1],params[0][2],params[0][3],)
        self.conv2 = conv_mnf(in_channels[1],num_filters[1],kernel_sizes[1],
                              params[1][0],params[1][1],params[1][2],params[1][3],)
        self.fc1 = fc_mnf(params[2][0],params[2][1],params[2][2],params[2][3],)
        self.fc2 = fc_mnf(params[3][0],params[3][1],params[3][2],params[3][3],)


    def forward(self, x, sample_z):
        x = F.relu(self.max_pool(self.conv1(x, sample_z[0].view(-1,1,1))))
        x = F.relu(self.max_pool(self.conv2(x, sample_z[1].view(-1,1,1))))
        x = x.view((x.shape[0],-1))
        x = F.relu(self.fc1(x, sample_z[2]))
        x = self.fc2(x, sample_z[3])
        return x