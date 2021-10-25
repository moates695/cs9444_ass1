"""
   seq_plot.py
   COMP9444, CSE, UNSW
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from seq_models import SRN_model, LSTM_model
from reber import lang_reber
from anbn import lang_anbn
from sklearn.linear_model import Perceptron

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='reber', help='reber, anbn or anbncn')
parser.add_argument('--embed', type=bool, default=False, help='embedded or not (reber)')
parser.add_argument('--length', type=int, default=0, help='min (reber) or max (anbn)')
# network options
parser.add_argument('--model', type=str, default='srn', help='srn or lstm')
parser.add_argument('--hid', type=int, default=0, help='number of hidden units')
# visualization options
parser.add_argument('--out_path', type=str, default='net', help='outputs path')
parser.add_argument('--epoch', type=int, default=100, help='epoch to load from')
parser.add_argument('--num_plot', type=int, default=10, help='number of plots')
args = parser.parse_args()

if args.lang == 'reber':
    num_class = 7
    hid_default = 2
    lang = lang_reber(args.embed,args.length)
    if args.embed:
        max_state = 18
    else:
        max_state =  6
elif args.lang == 'anbn':
    num_class = 2
    hid_default = 2
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)
    max_state = args.length
elif args.lang == 'anbncn':
    num_class = 3
    hid_default = 3
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)
    max_state = args.length

if args.hid == 0:
    args.hid = hid_default
    
if args.model == 'srn':
    net = SRN_model(num_class,args.hid,num_class)
elif args.model == 'lstm':
    net = LSTM_model(num_class,args.hid,num_class)

path = args.out_path+'/'
net.load_state_dict(torch.load(path+'%s_%s%d_%d.pth'
                    %(args.lang,args.model,args.hid,args.epoch)))

np.set_printoptions(suppress=True,precision=2)

for weight in net.parameters():
    print(weight.data.numpy())

if args.model == 'srn':
    if args.hid == 2:
        plt.plot(net.H0.data[0],net.H0.data[1],'bx') 
    elif args.hid == 3:    
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(net.H0.data[0],net.H0.data[1],net.H0.data[2],'bx') 
else:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(0,0,0,'bx')
    
context_act = []
context_comp = {}

with torch.no_grad():
    net.eval()

    for epoch in range(args.num_plot):

        input, seq, target, state = lang.get_sequence()
        label = seq[1:]

        net.init_hidden()
        if args.model == 'lstm':
            hidden_seq, output, c_seq, f_seq, i_seq, g_seq, o_seq = net(input)

            hidden = hidden_seq.squeeze()
            context = c_seq.squeeze()
            context = context.detach().numpy()
            if state[2] == 2:
                for k in range(1, len(state)-2):
                    context_act.append(np.concatenate([context[k], np.array([0])]))
            else:
                for k in range(2, len(state)-2):
                    context_act.append(np.concatenate([context[k], np.array([1])]))

            for k in range(1, len(state)-2):
                if state[k+1] not in context_comp:
                    context_comp[state[k+1]] = []
                context_comp[state[k+1]].append(context[k])

            lang.print_outputs(epoch, seq, state, hidden, target, output, 
                            c_seq, f_seq, i_seq, g_seq, o_seq)
            sys.stdout.flush()
        else:
            hidden_seq, output =  net(input)

            hidden = hidden_seq.squeeze()

            lang.print_outputs(epoch, seq, state, hidden, target, output)
            sys.stdout.flush()

        if args.hid == 2:
            plt.scatter(hidden[:,0],hidden[:,1], c=state[1:],
                        cmap='jet', vmin=0, vmax=max_state)
        else:
            ax.scatter(hidden[:,0],hidden[:,1],hidden[:,2],
                       c=state[1:], cmap='jet',
                       vmin=0, vmax=max_state)

    plt.show()

if args.model == 'lstm':
    #context_act = context_act.detach().numpy()
    context_act = np.asarray(context_act)
    perceptron = Perceptron(max_iter=10000).fit(context_act[:,:-1], context_act[:,-1])
    score = perceptron.score(context_act[:,:-1], context_act[:, -1])
    if score == 1:
        print('linearly seperable')
    else:
        print('NOT linearly seperable')
    print(f'score: {score}')

    for key in context_comp.keys():
        context_comp[key] = sum(context_comp[key]) / len(context_comp[key])

    for i in range(2, 10):
        try:
            dist = np.sum(np.square(context_comp[i] - context_comp[i + 8]))
            print(f'dist {i}->{i+8}: {dist}')
        except KeyError:
            pass