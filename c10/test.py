import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import torchvision
from torchvision import transforms

from spikingjelly.clock_driven.functional import reset_net, set_monitor

import numpy as np
import os
import sys
import time
import argparse
#from tqdm import tqdm

from model import Cifar10Net

sys.path.append('..')

from gradrewire import GradRewiring
from deeprewire import DeepRewiring
from ccep import CCEP

############## Reproducibility ##############
_seed_ = 2020
np.random.seed(_seed_)
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-penalty', type=float, default=1e-3)
parser.add_argument('-s', '--sparsity', type=float)
parser.add_argument('-gpu', type=str)
parser.add_argument('--dataset-dir', type=str)
parser.add_argument('--dump-dir', type=str)
parser.add_argument('-T', type=int, default=8)
parser.add_argument('-N', '--epoch', type=int, default=2048)
parser.add_argument('-soft', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument(
    '-m', '--mode', choices=['deep', 'grad', 'no_prune'], default='no_prune')

# Epoch interval when recording data (firing rate, acc. on test set, etc.) on TEST set
parser.add_argument('-i1', '--interval-test', type=int, default=128)

# Step interval when recording data (loss, acc. on train set) on TRAIN set
parser.add_argument('-i2', '--interval-train', type=int, default=1024)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = args.batch_size
learning_rate = args.learning_rate
dataset_dir = args.dataset_dir
dump_dir = args.dump_dir
T = args.T
penalty = args.penalty
s = args.sparsity
soft = args.soft
test = args.test
no_prune = (args.mode == 'no_prune')
i1 = args.interval_test
i2 = args.interval_train
N = args.epoch



if __name__ == "__main__":
    if os.path.exists(os.path.join(model_dir, 'net.pkl')):
        module = torch.load(os.path.join(model_dir, 'net.pkl'), map_location='cuda')
        print(f'Load existing model, Train steps: {net.train_times}, Epochs: {net.epochs}')
    else:
        module = Cifar10Net(T=T).cuda()
        print(f'Create new model')
    module = model.conv1
    print(list(module.named_parameters()))
    print(list(module.named_buffers()))
