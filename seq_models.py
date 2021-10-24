"""
   seq_models.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import math


class SRN_model(nn.Module):
    def __init__(self, num_input, num_hid, num_out, batch_size=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.H0= nn.Parameter(torch.Tensor(num_hid))
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))

    def init_hidden(self):
        H0 = torch.tanh(self.H0)
        return(H0.unsqueeze(0).expand(self.batch_size,-1))
 
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t = self.init_hidden().to(x.device)
        else:
            h_t = init_states
            
        for t in range(seq_size):
            x_t = x[:, t, :]
            c_t = x_t @ self.W + h_t @ self.U + self.hid_bias
            h_t = torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output

class LSTM_model(nn.Module):
    def __init__(self,num_input,num_hid,num_out,batch_size=1,num_layers=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        c_seq = []
        f_seq = []
        i_seq = []
        g_seq = []
        o_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid).to(x.device), 
                        torch.zeros(batch_size,self.num_hid).to(x.device))
        else:
            h_t, c_t = init_states
         
        NH = self.num_hid
        for t in range(seq_size):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :NH]),     # input gate
                torch.sigmoid(gates[:, NH:NH*2]), # forget gate
                torch.tanh(gates[:, NH*2:NH*3]),  # new values
                torch.sigmoid(gates[:, NH*3:]),   # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            c_seq.append(c_t.unsqueeze(0))
            f_seq.append(f_t.unsqueeze(0))
            i_seq.append(f_t.unsqueeze(0))
            g_seq.append(g_t.unsqueeze(0))
            o_seq.append(o_t.unsqueeze(0))
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        c_seq = torch.cat(c_seq, dim=0)
        c_seq = c_seq.transpose(0,1).contiguous()
        f_seq = torch.cat(f_seq, dim=0)
        f_seq = f_seq.transpose(0,1).contiguous()
        i_seq = torch.cat(i_seq, dim=0)
        i_seq = i_seq.transpose(0,1).contiguous()
        g_seq = torch.cat(g_seq, dim=0)
        g_seq = g_seq.transpose(0,1).contiguous()
        o_seq = torch.cat(o_seq, dim=0)
        o_seq = o_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output, c_seq, f_seq, i_seq, g_seq, o_seq
