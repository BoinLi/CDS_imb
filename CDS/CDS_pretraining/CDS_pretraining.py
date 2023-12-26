from __future__ import print_function
import argparse
import torch.optim as optim
import torchvision.models as models

import os
from lib.LinearAverage_2 import LinearAverage
# from test import NN, kNN_DA, recompute_memory
from return_dataset import *
import logging
from utils_dh import setup_logging
import sys

from custom_function.selfsup import *

torch.backends.cudnn.benchmark=True


# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--eta_b', type=float, default=1.0, metavar='ETAB',
                    help='eta b')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./output',
                    help='dir to save checkpoint') 
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                    help='which network ')
parser.add_argument('--split', type=int, default=0, metavar='N',
                    help='which split to use')
parser.add_argument('--epoch', type=int, default=0, metavar='N',
                    help='how many labeled examples to use for target domain')
parser.add_argument('--pretrained_batch', type=int, default=32, metavar='N',
                    help='how many labeled examples to use for target domain')
parser.add_argument('--low_dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='M', help='batch_size')
parser.add_argument('--nce-k', default=0, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

parser.add_argument('--n_neighbor', default=700, type=int,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--temp2', default=1.0, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--lambda_value', default=1.0, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='validation phase')
parser.add_argument('--training_da', action='store_false', default=True,
                    help='validation phase')

parser.add_argument('--imagenet', action='store_true', default=False,
                    help='validation phase')

parser.add_argument('--DC', action='store_true', default=False,
                    help='validation phase')
parser.add_argument('--instance', action='store_true', default=True,
                    help='validation phase')
parser.add_argument('--t2s', action='store_true', default=True,
                    help='validation phase')
parser.add_argument('--s2t', action='store_true', default=True,
                    help='validation phase')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

parser.add_argument('--data-A', metavar='DIR Domain A', help='path to domain A dataset')
parser.add_argument('--data-B', metavar='DIR Domain B', help='path to domain B dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--clean-model', default='', type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='the directory of the experiment')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

use_gpu = torch.cuda.is_available()

# source_loader, target_loader_unl, val_loader, _, class_list = return_dataset_selfsup(args, batch_size=args.batch_size)

dirA = args.data_A   
dirB = args.data_B

train_datasetA = TrainDataset(dirA, 'RS')
eval_datasetA = EvalDataset(dirA, 'RS')

train_datasetB = TrainDataset(dirB, 'UT')
eval_datasetB = EvalDataset(dirB, 'UT')

train_loaderA = torch.utils.data.DataLoader(
    train_datasetA, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

eval_loaderA = torch.utils.data.DataLoader(
    eval_datasetA, batch_size=args.batch_size * 2, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None)

train_loaderB = torch.utils.data.DataLoader(
    train_datasetB, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

eval_loaderB = torch.utils.data.DataLoader(
    eval_datasetB, batch_size=args.batch_size * 2, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None)


# torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
params = []
device = 'cuda:0'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True

class  Model(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.net = models.__dict__['resnet50'](num_classes=args.low_dim)

        dim_mlp = self.net.fc.weight.shape[1]
        self.net.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.net.fc)

    def forward(self, x):
        return self.net(x)

        
model = Model(args)
model = model.cuda()
setup_seed(args.seed)


if args.clean_model:
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            clean_checkpoint = torch.load(args.clean_model)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'net' in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint['state_dict']['module.encoder_q.'+k_parts]
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

lemniscate_s = LinearAverage(args.low_dim, train_loaderA.dataset.__len__() , args.nce_t, args.nce_m)
lemniscate_t = LinearAverage(args.low_dim, train_loaderB.dataset.__len__(), args.nce_t, args.nce_m)
ndata = train_loaderA.dataset.__len__() + train_loaderB.dataset.__len__()
lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)


lemniscate_s.cuda()
lemniscate_t.cuda()
lemniscate.cuda()
######


logging.info(args.lambda_value)
model.train()

optimizer = optim.SGD(model.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

## CDS


# if os.path.exists(args.checkpath)==False:
#     os.mkdir(args.checkpath)

if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

info_save = open(os.path.join(args.exp_dir, 'imbUCIR.txt'), 'w')


Best_mAP_A = 0.
Best_mAP_B = 0.

for epoch in range(0,50):

    logging.info('epoch: start:{}'.format(epoch))
    res_A, res_B = train_selfsup_only(epoch, args, model.net,  lemniscate_s, lemniscate_t, optimizer, info_save)

    print("res_A: {}; res_B:{}".format(res_A, res_B))
    info_save.write("res_A: {}; res_B:{} \n".format(res_A, res_B))
    info_save.flush()

    if (Best_mAP_A + Best_mAP_B) / 2 < (res_A + res_B) /2:
            Best_mAP_A = res_A
            Best_mAP_B = res_B

    print("Best_mAP_A: {}; Best_mAP_B:{}".format(Best_mAP_A, Best_mAP_B))
    info_save.write("Best_mAP_A: {}; Best_mAP_B:{}".format(Best_mAP_A, Best_mAP_B))
    info_save.flush()


