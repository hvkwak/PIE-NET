#
# Some of the codes are from https://github.com/charlesq34/pointnet2/blob/master/train_multi_gpu.py
#
import argparse
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
import fnmatch
from trainer import NetworkTrainer
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,5'

from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
import tf_util
import scipy.io as sio
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='How many gpus to use [default: 1]')
#parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')    # change gpu device number
parser.add_argument('--model', default='model_all', help='Model name [default: model]')
parser.add_argument('--stage_1_log_dir', default='stage_1_log', help='Log dir [default: log]')
parser.add_argument('--stage_2_log_dir', default='stage_2_log', help='Log dir [default: log]')
parser.add_argument('--stage_3_log_dir', default='stage_3_log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8096, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--stage',type=int,default=1,help='network stage')
parser.add_argument('--resume',type=int,default=0,help='resume training')
FLAGS = parser.parse_args()



if __name__ == "__main__":
    Trainer = NetworkTrainer(FLAGS, BASE_DIR, ROOT_DIR)
    Trainer.log_string('pid: %s'%(str(os.getpid())))
    Trainer.build_graph_31()
    if Trainer.STAGE == 1:
        Trainer.train_graph_31()
    elif Trainer.STAGE == 2:
        Trainer.build_graph_32()
        Trainer.train_graph_32()
    elif Trainer.STAGE == 3:
        Trainer.build_graph_32()
        Trainer.eval_graph_32()
    Trainer.LOG_FOUT.close()