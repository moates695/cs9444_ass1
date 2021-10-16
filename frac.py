"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.in_to_hid1 = nn.Linear(2, hid)
        self.hid1_to_hid2 = nn.Linear(hid, hid)
        self.hid2_to_out = nn.Linear(hid, 1)

    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(hid1_sum)
        
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        
        out_sum = self.hid2_to_out(self.hid2)
        output = torch.sigmoid(out_sum)
        return output

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.in_to_hid1 = nn.Linear(2, hid)
        self.hid1_to_hid2 = nn.Linear(hid, hid)
        self.hid2_to_hid3 = nn.Linear(hid, hid)
        self.hid3_to_out = nn.Linear(hid, 1)

    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(hid1_sum)
        
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.tanh(hid2_sum)
        
        hid3_sum = self.hid2_to_hid3(self.hid2)
        self.hid3 = torch.tanh(hid3_sum)
        
        out_sum = self.hid3_to_out(self.hid3)
        output = torch.sigmoid(out_sum)
        return output

class DenseNet(torch.nn.Module):
    def __init__(self, hid):
        super(DenseNet, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(2, hid)
        self.in_to_hid2 = torch.nn.Linear(2, hid)
        self.in_to_out = torch.nn.Linear(2, 1)
        self.hid1_to_hid2 = torch.nn.Linear(hid, hid)
        self.hid1_to_out = torch.nn.Linear(hid, 1)
        self.hid2_to_out = torch.nn.Linear(hid, 1)       
    
    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.tanh(hid1_sum)
        
        in_to_hid2 = self.in_to_hid2(input)
        hid1_to_hid2 = self.hid1_to_hid2(self.hid1)
        hid2_sum = in_to_hid2 + hid1_to_hid2
        self.hid2 = torch.tanh(hid2_sum)
        
        in_to_out = self.in_to_out(input)
        hid1_to_out = self.hid1_to_out(self.hid1)
        hid2_to_out = self.hid2_to_out(self.hid2)
        out_sum = in_to_out + hid1_to_out + hid2_to_out
        output = torch.sigmoid(out_sum)
        return output