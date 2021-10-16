"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.in_to_out = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out_sum = self.in_to_out(x)
        output = F.log_softmax(out_sum, dim = 1)
        return output

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        num_nodes = 120
        self.in_to_hid = nn.Linear(28 * 28, num_nodes)
        self.hid_to_out = nn.Linear(num_nodes, 10)


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        hid_sum = self.in_to_hid(x)
        hid = torch.tanh(hid_sum)
        out_sum = self.hid_to_out(hid)
        output = F.log_softmax(out_sum, dim = 1)
        return output

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        channels1 = 8
        kernal1 = 5
        stride1 = 1
        pad1 =  0 #int(kernal1 / 2)
        pool1 = 2

        width1 = int((1 + (28 + 2 * pad1 - kernal1) / stride1) / pool1)

        channels2 = 16
        kernal2 = 3
        stride2 = 1
        pad2 =  0 #int(kernal2 / 2)  2
        pool2 = 2
        
        width2 = int((1 + (width1 + 2 * pad2 - kernal2) / stride2) / pool2)
        
        size = channels2 * (width2 ** 2)

        full = 150

        self.conv1 = nn.Conv2d(1, channels1, kernal1,
                               stride = stride1, padding = pad1)
        self.pool1 = nn.MaxPool2d(pool1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernal2,
                               stride = stride2, padding = pad2)
        self.pool2 = nn.MaxPool2d(pool2)
        self.full = nn.Linear(size, full)
        self.out = nn.Linear(full, 10)

    def forward(self, x):
        conv1_sum = self.conv1(x)
        relu1 = nn.ReLU()
        conv1 = relu1(conv1_sum)
        pool1 = self.pool1(conv1)
        conv2_sum = self.conv2(pool1)
        relu2 = nn.ReLU()
        conv2 = relu2(conv2_sum)
        pool2 = self.pool2(conv2)
        pool2 = torch.flatten(pool2, start_dim = 1)
        full_sum = self.full(pool2)
        relu3 = nn.ReLU()
        full = relu3(full_sum)
        out_sum = self.out(full)
        output = F.log_softmax(out_sum, dim = 1)
        return output
