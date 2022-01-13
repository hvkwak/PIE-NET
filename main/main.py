#
# Some of the codes are from https://github.com/charlesq34/pointnet2/blob/master/train_multi_gpu.py
#
import argparse
import math
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
parser.add_argument('--num_point', type=int, default=8096, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--stage',type=int,default=1,help='network stage')
parser.add_argument('--resume',type=int,default=0,help='resume training')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE // NUM_GPUS

NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
#GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
# STAGE = 1: Training, Section 3.1.
# STAGE = 2: Training, Section 3.2.
STAGE = FLAGS.stage
RESUME = FLAGS.resume

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
if STAGE == 1:
    LOG_DIR = FLAGS.stage_1_log_dir
else:
    LOG_DIR = FLAGS.stage_2_log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp main.py %s' % (ROOT_DIR+"/"+LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def average_gradients(tower_grads):
    #
    # code from https://github.com/charlesq34/pointnet2/blob/master/train_multi_gpu.py
    #
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        #for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            print("g:", g)
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(input_tensor=grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_learning_rate_stage_2(batch,base_learning_rate):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def corner_pair_label_generator(sample_256_64_idx, \
                                sample_pair_idx, \
                                sample_pair_valid_mask, \
                                sample_corner_pairs_available, \
                                points_cloud, \
                                batch_open_gt_pair_idx, \
                                batch_open_gt_256_64_idx):
                                # add more gt_*

    batch_num = len(sample_corner_pairs_available)    
    sample_valid_mask_256_64_labels_for_loss = np.zeros((batch_num, 256, 64), dtype = np.int32) # output should be (batch_num, 256, 64, 2)
    sample_valid_mask_pair_labels_for_loss = np.zeros((batch_num, 256, 1), dtype = np.int16)
    points_cloud_np = points_cloud.numpy()
    dist_threshold = 0.01

    # sample_valid_mask_256_64    
    
    for i in range(batch_num):
        # per batch
        if sample_corner_pairs_available[i]:
            sample_valid_mask_pair_numpy = sample_pair_valid_mask[i].numpy()
            k = 0
            found_in_gt_open_pair = False
            while sample_valid_mask_pair_numpy[k][0] == 1:

                # per curve pair k in one batch
                if sample_pair_idx[i][k].numpy() in batch_open_gt_pair_idx[i, :, :]:
                    found_in_gt_open_pair = True
                    # indices match exactly
                    gt_idx = np.where(sample_pair_idx[i][k].numpy() in batch_open_gt_pair_idx[i, :, :])[0]
                    # gt_idx = np.where(batch_open_gt_pair_idx[i][k].numpy() in my_mat['open_gt_pair_idx'][0, 0])[0][0]
                    # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                    mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                    sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                    sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                elif np.flip(sample_pair_idx[i][k].numpy()) in batch_open_gt_pair_idx[i, :, :]:
                    found_in_gt_open_pair = True
                    gt_idx = np.where(np.flip(sample_pair_idx[i][k].numpy()) in batch_open_gt_pair_idx[i, :, :])[0]
                    # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                    mask = np.in1d(sample_pair_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                    # update here labels
                    sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                    sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1

                if not found_in_gt_open_pair:
                # not exact match, but see if there is one nearby.
                    # calculate distances NN.
                    distance = np.sqrt(np.sum((points_cloud_np[i][sample_pair_idx[i][k].numpy(), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        found_in_gt_open_pair = True
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                
                if not found_in_gt_open_pair:
                    distance = np.sqrt(np.sum((points_cloud_np[i][np.flip(sample_pair_idx[i][k].numpy()), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        found_in_gt_open_pair = True
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                k = k+1

    return sample_valid_mask_256_64_labels_for_loss, sample_valid_mask_pair_labels_for_loss

def corner_pair_neighbor_search(points_cloud, pred_corner):
    """ builds a sphere between two predicted corner points, sample 64 points within the spehere.

    Args:
        points_cloud ([tf.float32], batch_size, 8096, 3): original point cloud
        pred_corner ([tf.float32], batch_size, 8096, 2): predicted corner points

    Returns:
        sampled points, its indices and valid masks
    """    
    
    corner_points = tf.where(pred_corner[..., 1] > 0.999)
    corner_pair_available = [False]*DEVICE_BATCH_SIZE
    corner_valid_mask_pair = []

    # organize corner_pairs per batch
    corner_pair_idx = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        idx = tf.boolean_mask(corner_points, corner_points[:,0] == per_batch)[:,1]
        if idx.shape[0] > 1:
            corner_pair_available[per_batch] = True
            idx_r = tf.repeat(idx, idx.shape[0])
            idx_b = tf.tile(idx, [idx.shape[0]])
            two_col = tf.stack([idx_r, idx_b], 1)
            corner_pair_idx.append(two_col[two_col[:,0] < two_col[:, 1]])
        else:
            corner_pair_idx.append([])
            

    # per batch sample the points
    corner_pair_256_64_idx = []
    corner_pair_sample_points = [] # (8, 256, 64, 3)
    corner_valid_mask_256_64 = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        rest_num = 256
        if corner_pair_available[per_batch]:

            # first increase the precision to float64, 
            # otherwise it may go wrong when it comes to finding points within radius, 
            # where it may also include the two corner points at the end.
            points_cloud = tf.cast(points_cloud, dtype = tf.float64) 

            # find neighbors
            xyz1 = tf.gather(points_cloud[per_batch], indices=corner_pair_idx[per_batch][:, 0], axis=0)
            xyz2 = tf.gather(points_cloud[per_batch], indices=corner_pair_idx[per_batch][:, 1], axis=0)
            ball_center = tf.reduce_mean(tf.stack([xyz1, xyz2], axis = 0), axis = 0)
            distance_from_ball_center = tf.sqrt(tf.reduce_sum(tf.square(tf.math.subtract(tf.expand_dims(ball_center,axis=1), tf.expand_dims(points_cloud[per_batch],axis=0))), axis = 2))
            r = tf.sqrt(tf.reduce_sum(tf.square(xyz1 - xyz2), axis = 1)) / 2.0 # radius
            within_range = tf.math.less(distance_from_ball_center, tf.multiply(tf.ones_like(distance_from_ball_center), tf.expand_dims(r, axis = 1)))

            # per corner pair within this batch, subsample the indicies
            idx_256_64 = []
            valid_mask_256_64 = []
            corner_pair_num = within_range.shape[0]
            
            # if there are more than 256 pairs, just take first 256.
            if corner_pair_num > 256 : corner_pair_num = 256
            rest_num = 256 - corner_pair_num

            for per_corner in tf.range(corner_pair_num):
                # make sure that corner points(end points) are not within the range.
                assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
                assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
                candidnate_num = tf.where(within_range[per_corner, :]).shape[0]
                #
                # raise error or debug when within_range.shape[0] = 0
                # or if within_range.shape[0] = 3?
                if 64 <= candidnate_num:
                    idx_nums = tf.concat([tf.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), tf.squeeze(tf.random.shuffle(tf.where(within_range[per_corner, :]))[:62]), tf.expand_dims(corner_pair_idx[per_batch][per_corner][-1], axis = 0)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))
                    valid_mask_256_64.append(tf.expand_dims(tf.ones_like(idx_nums), axis = 0))

                elif 0 < candidnate_num < 64:
                    n = candidnate_num
                    dummy_num = 64 - 1 - n
                    middle_indicies = tf.squeeze(tf.where(within_range[per_corner, :]))
                    if candidnate_num == 1: 
                        middle_indicies = tf.expand_dims(middle_indicies, axis = 0)
                    idx_nums = tf.concat([tf.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), middle_indicies, tf.repeat(corner_pair_idx[per_batch][per_corner][-1], dummy_num)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))
                    valid_mask_256_64.append(tf.expand_dims(tf.concat([tf.ones((64 - (dummy_num - 1)), dtype = tf.int64), tf.zeros((dummy_num - 1), dtype = tf.int64)], axis = 0), axis = 0))

            if rest_num > 0: 
                idx_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
                valid_mask_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_pair_256_64_idx.append(tf.concat(idx_256_64, axis = 0))
            corner_valid_mask_256_64.append(tf.concat(valid_mask_256_64, axis = 0))
            corner_pair_sample_points.append(tf.gather(points_cloud[per_batch], indices=corner_pair_256_64_idx[per_batch], axis=0))
        else:
            corner_pair_256_64_idx.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_valid_mask_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_pair_sample_points.append(tf.zeros((rest_num, 64, 3), dtype = tf.float32))

        valid_mask = tf.expand_dims(tf.cast(tf.sequence_mask(256 - rest_num, 256), dtype=tf.uint8), axis = 1)
        corner_valid_mask_pair.append(valid_mask)

    return corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_idx, corner_valid_mask_pair, corner_valid_mask_256_64, corner_pair_available

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            if STAGE==1 or STAGE==2:
                pointclouds_pl, labels_edge_p, labels_corner_p, reg_edge_p, reg_corner_p = MODEL.placeholder_inputs_31(BATCH_SIZE,NUM_POINT)
                open_gt_corner_pair_sample_points_pl, open_gt_corner_valid_mask_256_64, open_gt_labels_256_64, open_gt_labels_pair = MODEL.placeholder_inputs_32(BATCH_SIZE)
                #open_gt_corner_pair_sample_points_pl, open_gt_256_64_idx, open_gt_mask, open_gt_type, open_gt_res, \
                #open_gt_sample_points, open_gt_valid_mask, open_gt_pair_idx = MODEL.placeholder_inputs_32(BATCH_SIZE)

                is_training_31 = tf.compat.v1.placeholder(tf.bool, shape=())
                is_training_32 = tf.compat.v1.placeholder(tf.bool, shape=())
                
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                #batch_stage_1 = tf.Variable(0,name='stage1/batch')
                batch = tf.compat.v1.get_variable('batch', [],initializer=tf.compat.v1.constant_initializer(0), trainable=False)
                bn_decay = get_bn_decay(batch)
                tf.compat.v1.summary.scalar('bn_decay', bn_decay)

                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.compat.v1.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                
                print("--- Get model and loss")
                # -------------------------------------------
                # Get model and loss on multiple GPU devices
                # -------------------------------------------
                # Allocating variables on CPU first will greatly accelerate multi-gpu training.
                # Ref: https://github.com/kuza55/keras-extras/issues/21

                # check if this works
                MODEL.get_model_31(pointclouds_pl, is_training_31, STAGE, bn_decay=bn_decay)
                MODEL.get_model_32(open_gt_corner_pair_sample_points_pl, is_training_32, bn_decay=bn_decay)
                #pred_labels_edge_p, pred_labels_corner_p, pred_reg_edge_p, pred_reg_corner_p = MODEL.get_model_31(pointclouds_pl, is_training_31, STAGE, bn_decay=bn_decay)
                #corner_pair_sample_points_pl, corner_pair_256_64_idx, corner_pair_valid_mask, corner_pair_available = corner_pair_neighbor_search(pointclouds_pl, pred_labels_corner_p)
                #corner_pair_sample_points_cloud_pl = tf.concat([corner_pair_sample_points_pl[i] for i in range(len(corner_pair_sample_points_pl))], axis = 0) # this will be [2048, 64, 3]
                #MODEL.get_model_32(corner_pair_sample_points_cloud_pl, is_training_32, bn_decay=bn_decay)

                tower_grads_stage1 = []
                tower_grads_stage2 = []

                # Sec. 3.1
                pred_labels_edge_p_gpu = []
                pred_labels_corner_p_gpu = []
                pred_reg_edge_p_gpu = []
                pred_reg_corner_p_gpu = []

                # Sec. 3.2
                pred_seg_p_gpu = []
                pred_cls_p_gpu = []
                pred_reg_p_gpu = []

                total_loss_gpu = []

                for i in range(NUM_GPUS):
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.compat.v1.name_scope('gpu_%d'%(i)) as scope:
                            # Evenly split input data to each GPU
                            ## check if dimension numbers are correct:
                            batch_pc = tf.slice(pointclouds_pl, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                            batch_labels_edge_p = tf.slice(labels_edge_p,[i*DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])
                            batch_labels_corner_p = tf.slice(labels_corner_p,[i*DEVICE_BATCH_SIZE, 0], [DEVICE_BATCH_SIZE, -1])
                            batch_reg_edge_p = tf.slice(reg_edge_p, [i*DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                            batch_reg_corner_p = tf.slice(reg_corner_p, [i*DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])

                            pred_labels_edge_p, pred_labels_corner_p, pred_reg_edge_p, pred_reg_corner_p = MODEL.get_model_31(batch_pc, is_training_31, STAGE, bn_decay=bn_decay)
                            # LOSS
                            if STAGE == 1 or STAGE == 2:
                                edge_3_1_loss,   edge_3_1_recall,   edge_3_1_acc,\
                                corner_3_1_loss, corner_3_1_recall, corner_3_1_acc,\
                                reg_edge_3_1_loss, reg_corner_3_1_loss, loss_31 = MODEL.get_stage_1_loss(pred_labels_edge_p, \
                                                                                                               pred_labels_corner_p, \
                                                                                                               batch_labels_edge_p, \
                                                                                                               batch_labels_corner_p, \
                                                                                                               pred_reg_edge_p, \
                                                                                                               pred_reg_corner_p, \
                                                                                                               batch_reg_edge_p, \
                                                                                                               batch_reg_corner_p)
                            #elif STAGE == 2: # Section 3.2.
                                # GT
                                device_batch_size_3_2 = BATCH_SIZE*256//2
                                batch_corner_pair_sample_points_pl = tf.slice(open_gt_corner_pair_sample_points_pl, [i*device_batch_size_3_2,0,0], [device_batch_size_3_2,-1,-1])
                                batch_open_gt_256_64_labels = tf.slice(open_gt_labels_256_64, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                                batch_open_gt_256_64_valid_mask = tf.slice(open_gt_corner_valid_mask_256_64, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                                batch_open_gt_pair_valid_mask = tf.slice(open_gt_labels_pair, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])

                                # maybe we don't need these:
                                #batch_open_gt_256_64_idx = tf.slice(open_gt_256_64_idx, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_res = tf.slice(open_gt_res, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_sample_points = tf.slice(open_gt_sample_points, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_mask = tf.slice(open_gt_mask, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_valid_mask = tf.slice(open_gt_valid_mask, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_pair_idx = tf.slice(open_gt_pair_idx, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #batch_open_gt_type = tf.slice(open_gt_type, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                                #
                                
                                # Note:
                                # add the end_points term in loss!
                                #
                                pred_open_curve_seg, end_points = MODEL.get_model_32(batch_corner_pair_sample_points_pl, is_training_32, bn_decay=bn_decay)
                                
                                loss_32 = MODEL.get_stage_2_loss(pred_open_curve_seg, \
                                                            batch_open_gt_256_64_labels, \
                                                            batch_open_gt_256_64_valid_mask, \
                                                            batch_open_gt_pair_valid_mask, \
                                                            #pred_open_curve_reg, \
                                                            #batch_open_gt_res, \
                                                            #batch_open_gt_sample_points, \
                                                            #batch_open_gt_256_64_idx, \
                                                            #batch_open_gt_mask, \
                                                            #batch_open_gt_valid_mask, \
                                                            #batch_open_gt_pair_idx, \
                                                            #batch_open_gt_type,\
                                                            end_points, \
                                                            0.0001)
                            if STAGE == 1 or STAGE == 2:
                                tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_loss' % (i), edge_3_1_loss)
                                tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_recall' % (i), edge_3_1_recall)
                                tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_acc' % (i), edge_3_1_acc)                
                                tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_loss' % (i), corner_3_1_loss)
                                tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_recall' % (i), corner_3_1_recall)
                                tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_acc' % (i), corner_3_1_acc)
                                tf.compat.v1.summary.scalar('%d_GPU_reg_edge_3_1_loss' % (i), reg_edge_3_1_loss)
                                tf.compat.v1.summary.scalar('%d_GPU_reg_corner_3_1_loss' % (i), reg_corner_3_1_loss)
                                tf.compat.v1.summary.scalar('%d_GPU_loss'% (i), loss_31)
                                grads = optimizer.compute_gradients(loss_31) # here's where the loss and gradients are covered.
                                tower_grads_stage1.append(grads[0:108])

                                ## check this: 
                                # losses = tf.compat.v1.get_collection('losses', scope)
                                # total_loss = tf.add_n(losses, name='total_loss')
                                pred_labels_edge_p_gpu.append(pred_labels_edge_p)
                                pred_labels_corner_p_gpu.append(pred_labels_corner_p)
                                pred_reg_edge_p_gpu.append(pred_reg_edge_p)
                                pred_reg_corner_p_gpu.append(pred_reg_corner_p)
                                total_loss_gpu.append(loss_31)
                            
                            #elif STAGE == 2:
                                #tf.compat.v1.summary.scalar('%d_GPU_seg_3_2_loss' % (i), seg_3_2_loss)
                                #tf.compat.v1.summary.scalar('%d_GPU_seg_3_2_acc' % (i), seg_3_2_acc)
                                tf.compat.v1.summary.scalar('%d_GPU_(mat_diff included) loss'% (i), loss_32)
                                grads = optimizer.compute_gradients(loss_32) # here's where the loss and gradients are covered.
                                tower_grads_stage2.append(grads[108:])


                                pred_seg_p_gpu.append(pred_open_curve_seg)
                                #pred_cls_p_gpu.append(pred_open_curve_cls)
                                #pred_reg_p_gpu.append(pred_open_curve_reg)
                                total_loss_gpu.append(loss_32)


                                ## check this: 
                                # losses = tf.compat.v1.get_collection('losses', scope)
                                # total_loss = tf.add_n(losses, name='total_loss')
                            
                                '''
                                open_pre_mask, open_pre_class_logits, open_pre_res, \
                                open_pre_sample_points, open_ball_radius, open_ball_center, \
                                open_b_spline_curve_pre, open_line_curve_pre, \
                                = MODEL.get_model_32(pc_batch, is_training_pl, STAGE,bn_decay=bn_decay)

                                MODEL.get_stage_1_loss(pred_labels_edge_p, pred_labels_corner_p, labels_edge_p_batch, labels_corner_p_batch, \
                                                        pred_reg_edge_p, pred_reg_corner_p, reg_edge_p_batch, reg_corner_p_batch)
                                
                                losses = tf.compat.v1.get_collection('losses', scope)
                                total_loss = tf.add_n(losses, name='total_loss')
                                for l in losses + [total_loss]:
                                    tf.compat.v1.summary.scalar(l.op.name, l)
                                '''
                
                ## Merge pred and losses from multiple GPUs
                if STAGE == 1:
                    pred_labels_edge_p = tf.concat(pred_labels_edge_p_gpu, 0)
                    pred_labels_corner_p = tf.concat(pred_labels_corner_p_gpu, 0)
                    pred_reg_edge_p = tf.concat(pred_reg_edge_p_gpu, 0)
                    pred_reg_corner_p = tf.concat(pred_reg_corner_p_gpu, 0)
                    total_loss = tf.reduce_mean(input_tensor = total_loss_gpu)
                
                    # Get training operator
                    grads = average_gradients(tower_grads_stage1)
                    train_op = optimizer.apply_gradients(grads, global_step=batch)
                    # train_op = optimizer.minimize(loss, global_step=batch_stage_1)

                    # Add ops to save and restore all the variables.
                    saver = tf.compat.v1.train.Saver(max_to_keep=10)
                
                elif STAGE == 2:
                    pred_open_curve_seg = tf.concat(pred_seg_p_gpu, 0)
                    total_loss = tf.reduce_mean(input_tensor = total_loss_gpu)

                    # Get training operator
                    grads = average_gradients(tower_grads_stage2)
                    train_op = optimizer.apply_gradients(grads, global_step=batch)

                    # Add ops to save and restore all the variables.
                    saver = tf.compat.v1.train.Saver(max_to_keep=10)
            '''
            elif STAGE==2:
                print('stage_2')
                pointclouds_pl,proposal_nx_pl,dof_mask_pl,dof_score_pl= MODEL.placeholder_inputs_stage_2(BATCH_SIZE,NUM_POINT)
                is_training_feature= tf.constant(False, dtype=tf.bool)
                is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch_stage_2 = tf.Variable(0,name='stage2/batch_2')
                bn_decay = get_bn_decay(batch_stage_2)
                tf.compat.v1.summary.scalar('bn_decay', bn_decay)
                print("--- Get model and loss")
                # Get model and loss 
                end_points,dof_feat,simmat_feat = MODEL.get_feature(pointclouds_pl, is_training_feature,STAGE,bn_decay=bn_decay)
                pred_dof_score,all_feat = MODEL.get_stage_2(dof_feat,simmat_feat,dof_mask_pl,proposal_nx_pl,is_training_pl,bn_decay=bn_decay)
                loss = MODEL.get_stage_2_loss(pred_dof_score,dof_score_pl,dof_mask_pl)
                tf.compat.v1.summary.scalar('loss', loss)
                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch_stage_2)
                tf.compat.v1.summary.scalar('learning_rate', learning_rate)
                variables = tf.contrib.framework.get_variables_to_restore()
                variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='pointnet']
                variables_to_train = [v for v in variables if v.name.split('/')[0]=='stage2']
                if OPTIMIZER == 'momentum':
                    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch_stage_2,var_list = variables_to_train)
                # Add ops to save and restore all the variables.
                saver = tf.compat.v1.train.Saver(var_list = variables_to_resotre)
                saver2 = tf.compat.v1.train.Saver(max_to_keep=100)
            '''
        
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)
        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        
        
        # Init variables
        if RESUME == 0:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        elif RESUME == 1:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, BASE_DIR+'/stage_1_log/model100.ckpt')
        
        if STAGE==1 or STAGE == 2:
            ops = {'pointclouds_pl': pointclouds_pl,
               'labels_edge_p': labels_edge_p,
               'labels_corner_p': labels_corner_p,
               #'labels_direction': labels_direction,
               'reg_edge_p': reg_edge_p,
               'reg_corner_p': reg_corner_p,
               'open_gt_corner_pair_sample_points_pl': open_gt_corner_pair_sample_points_pl,
               'open_gt_corner_valid_mask_256_64': open_gt_corner_valid_mask_256_64,
               'open_gt_labels_256_64': open_gt_labels_256_64,
               'open_gt_labels_pair': open_gt_labels_pair,
               'pred_open_curve_seg': pred_open_curve_seg,
               #'labels_type': labels_type,
               #'simmat_pl': simmat_pl,
               #'neg_simmat_pl': neg_simmat_pl,
               'is_training_31': is_training_31,
               'is_training_32': is_training_32,
               'pred_labels_edge_p': pred_labels_edge_p,                   #  'pred_labels_edge_points'
               'pred_labels_corner_p': pred_labels_corner_p, 
               #'pred_labels_direction': pred_labels_direction,
               'pred_reg_edge_p': pred_reg_edge_p,   
               'pred_reg_corner_p': pred_reg_corner_p,
               #'pred_labels_type': pred_labels_type,
               #'pred_simmat': pred_simmat,
               #'pred_conf': pred_conf_logits,
               'edge_3_1_loss': edge_3_1_loss,
               'edge_3_1_recall':edge_3_1_recall,
               'edge_3_1_acc': edge_3_1_acc,               
               'corner_3_1_loss': corner_3_1_loss,
               'corner_3_1_recall':corner_3_1_recall,
               'corner_3_1_acc': corner_3_1_acc, 
               'reg_edge_3_1_loss': reg_edge_3_1_loss,
               'reg_corner_3_1_loss': reg_corner_3_1_loss,
               'seg_3_2_loss': loss_32,
               #'seg_3_2_acc': seg_3_2_acc,
               #'task_2_2_loss': task_2_2_loss,
               #'task_3_loss': task_3_loss,
               #'task_4_loss': task_4_loss,
               #'task_4_acc': task_4_acc,
               #'task_5_loss': task_5_loss,
               #'task_6_loss': task_6_loss,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
               #'end_points': end_points
            for epoch in range(MAX_EPOCH):
                log_string('**** TRAIN EPOCH %03d ****' % (epoch))
                train_one_epoch(sess,ops,train_writer)
                sys.stdout.flush()
                #log_string('**** TEST EPOCH %03d ****' % (epoch))
                #eval_one_epoch(sess, ops, test_writer)
                #sys.stdout.flush()
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model"+str(epoch)+".ckpt"
                    save_path = saver.save(sess, os.path.join(LOG_DIR, model_ccc_path))
                    log_string("Model saved in file: %s" % save_path)
        '''
        elif STAGE==2:
            ops = {'pointclouds_pl': pointclouds_pl,
               'proposal_nx_pl': proposal_nx_pl,
               'dof_mask_pl': dof_mask_pl,
               'dof_score_pl': dof_score_pl,
               'pred_dof_score': pred_dof_score,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch_stage_2,
               'all_feat':all_feat,
               'end_points': end_points}
            for epoch in range(MAX_EPOCH):
                log_string('**** TRAIN EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                train_one_epoch_stage_2(sess,ops,train_writer)
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model"+str(epoch)+".ckpt"
                    save_path = saver2.save(sess, os.path.join(LOG_DIR, model_ccc_path))
                    log_string("Model saved in file: %s" % save_path)
        '''

def train_one_epoch(sess, ops, train_writer):
    is_training_31 = STAGE == 1 # train until Sec. 3.1.
    is_training_32 = STAGE == 2 # train until Sec. 3.2.
    train_matrices_names_list = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/train_data/new_train/'), '*.mat')
    matrix_num = len(train_matrices_names_list)
    permutation = np.random.permutation(matrix_num)
    for i in range(len(permutation)//4):
        load_data_start_time = time.time()
        loadpath = BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4]]
        train_data = sio.loadmat(loadpath)['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        for j in range(3):
            temp_load_data_start_time = time.time()
            temp_loadpath = BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4+j+1]]
            temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
            temp_load_data_duration = time.time() - temp_load_data_start_time
            log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
            train_data = np.concatenate((train_data,temp_train_data),axis = 0)
            print(train_data.shape)

        #push_eval(train_data, ops, sess, train_writer, is_training)
        # num_data = 64*4 = 256
        # num_batch = 256 // 32 = 8
        num_data = train_data.shape[0]  # = 256
        num_batch = num_data // BATCH_SIZE   # 256 // 32 = 8
        total_loss = 0.0
        total_edge_3_1_loss = 0.0
        total_edge_3_1_recall = 0.0
        total_edge_3_1_acc = 0.0
        total_corner_3_1_loss = 0.0
        total_corner_3_1_recall = 0.0
        total_corner_3_1_acc = 0.0
        total_reg_edge_3_1_loss = 0.0
        total_reg_corner_3_1_loss = 0.0
        total_seg_3_2_loss = 0.0
        total_cls_3_2_loss = 0.0
        total_reg_3_2_loss = 0.0
#        total_task_2_2_loss = 0.0
#        total_task_3_loss = 0.0
#        total_task_4_loss = 0.0
#        total_task_4_acc = 0.0
#        total_task_5_loss = 0.0
#        total_task_6_loss = 0.0
        process_start_time = time.time()
        pred_labels_edge_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
        pred_labels_corner_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
        pred_reg_edge_p_val = np.zeros((num_data, NUM_POINT, 3), np.float32)
        pred_reg_corner_p_val = np.zeros((num_data, NUM_POINT, 3), np.float32)
        pred_open_curve_seg = np.zeros((num_data*256, NUM_POINT, 3), np.float32)

        np.random.shuffle(train_data)
        for j in range(num_batch):
            # remember that num_batch will be 8
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]

            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
            batch_labels_edge_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)  # edge point label 0/1
            batch_labels_corner_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)  # edge point label 0/1
            batch_regression_edge = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)  # each point normal estimation
            batch_regression_corner = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)

            if STAGE == 2:
                batch_open_gt_256_64_idx = np.zeros((BATCH_SIZE, 256, 64), np.int32)
                batch_open_gt_mask = np.zeros((BATCH_SIZE, 256, 64), np.int32)
                batch_open_gt_type = np.zeros((BATCH_SIZE, 256, 1), np.int32)
                batch_open_gt_res = np.zeros((BATCH_SIZE, 256, 6), np.float32)
                batch_open_gt_sample_points = np.zeros((BATCH_SIZE,256, 64, 3), np.float32)
                batch_open_gt_valid_mask = np.zeros((BATCH_SIZE,256, 1), np.int32)
                batch_open_gt_pair_idx = np.zeros((BATCH_SIZE,256, 2), np.int32)

            #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            for cnt in range(BATCH_SIZE):
                # cnt: 0 ... 31
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
                batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
                batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
                #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
                batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']

                ## check if these dimensions are correct
                if STAGE == 2:
                    batch_open_gt_256_64_idx[cnt, ...] = tmp_data[0, 0]['open_gt_256_64_idx']
                    batch_open_gt_sample_points[cnt, ...] = tmp_data[0, 0]['open_gt_sample_points']
                    batch_open_gt_mask[cnt, ...] = tmp_data[0, 0]['open_gt_mask']
                    batch_open_gt_type[cnt, ...] = tmp_data[0, 0]['open_gt_type']
                    batch_open_gt_res[cnt, ...] = tmp_data[0, 0]['open_gt_res']
                    batch_open_gt_valid_mask[cnt, ...] = tmp_data[0, 0]['open_gt_valid_mask']
                    batch_open_gt_pair_idx[cnt, ...] = tmp_data[0, 0]['open_gt_pair_idx']

                #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                #tmp_simmat = tmp_data['similar_matrix'][0,0]
                #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                #tmp_neg_simmat = 1 - tmp_simmat
                #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['labels_edge_p']: batch_labels_edge_p,
                         ops['labels_corner_p']: batch_labels_corner_p,
                         #ops['labels_direction']: batch_labels_direction,
                         ops['reg_edge_p']: batch_regression_edge,
                         ops['reg_corner_p']: batch_regression_corner,
                         #ops['labels_type']: batch_labels_type,
                         #ops['simmat_pl']: batch_simmat_pl,
                         #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                         #ops['is_training_32']: is_training_32}
                         ops['is_training_31']: is_training_31}
                         
                 
                    
#            summary, step, _, task_1_loss_val,task_1_recall_val,task_1_acc_val,task_2_1_loss_val,task_2_1_acc_val,task_2_2_loss_val, \
#                                 task_3_loss_val,task_4_loss_val,task_4_acc_val,task_5_loss_val, \
#                                 task_6_loss_val, loss_val = sess.run([ops['merged'], ops['step'], \
#                                 ops['train_op'], ops['task_1_loss'], ops['task_1_recall'],ops['task_1_acc'],ops['task_2_1_loss'], \
#                                 ops['task_2_1_acc'],ops['task_2_2_loss'],ops['task_3_loss'],ops['task_4_loss'], \
#                                 ops['task_4_acc'],ops['task_5_loss'],ops['task_6_loss'],ops['loss']],feed_dict=feed_dict)            
            
            summary, step, _, \
            edge_3_1_loss_val, edge_3_1_recall_val, edge_3_1_acc_val, \
            corner_3_1_loss_val, corner_3_1_recall_val, corner_3_1_acc_val, \
            reg_edge_3_1_loss_val, reg_corner_3_1_loss_val, loss_val, \
            pred_labels_edge_p_val[begin_idx:end_idx,:,:], pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
            pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:] = \
                sess.run([ops['merged'], ops['step'], ops['train_op'], \
                    ops['edge_3_1_loss'], ops['edge_3_1_recall'], ops['edge_3_1_acc'],\
                    ops['corner_3_1_loss'], ops['corner_3_1_recall'], ops['corner_3_1_acc'],\
                    ops['reg_edge_3_1_loss'], ops['reg_corner_3_1_loss'], ops['loss'], \
                    ops['pred_labels_edge_p'], ops['pred_labels_corner_p'], \
                    ops['pred_reg_edge_p'], ops['pred_reg_corner_p']],feed_dict=feed_dict)

            if STAGE == 2:
                # end_idx - begin_idx = 32
                corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_idx, corner_valid_mask_pair, corner_valid_mask_256_64, corner_pair_available = corner_pair_neighbor_search(batch_inputs, pred_labels_corner_p_val[begin_idx:end_idx,:,:])
                open_gt_labels_256_64, open_gt_labels_pair = corner_pair_label_generator(corner_pair_256_64_idx, \
                                                                              corner_pair_idx, \
                                                                              corner_valid_mask_pair, \
                                                                              corner_pair_available, \
                                                                              batch_inputs, \
                                                                              batch_open_gt_pair_idx, \
                                                                              batch_open_gt_256_64_idx)
                # Loss computation
                # 1. Compute CrossEntropy predicted labels <-> gt_labels_256_64
                # 2. corner_valid_mask_256_64 will take elements that were valid only
                # 3. gt_labels_pair will finally will let us take valid proposals only
                # 4. (optional) corner_valid_mask_256_64 will balance loss the per curve.

                corner_pair_sample_points = tf.concat([corner_pair_sample_points[i] for i in range(len(corner_pair_sample_points))], axis = 0) # this will be [N, 64, 3]
                #corner_pair_sample_points_label = tf.concat([corner_pair_sample_points_label[i] for i in range(len(corner_pair_sample_points_label))], axis = 0)
                corner_valid_mask_256_64 = tf.concat([corner_valid_mask_256_64[i] for i in range(len(corner_valid_mask_256_64))], axis = 0) 
                #corner_valid_mask_pair = tf.concat([corner_valid_mask_pair[i] for i in range(len(corner_valid_mask_pair))], axis = 0) 
                #corner_pair_available = tf.concat([corner_pair_available[i] for i in range(len(corner_pair_available))], axis = 0) 

                feed_dict = {ops['open_gt_corner_pair_sample_points_pl']: corner_pair_sample_points,\
                             #ops['corner_pair_sample_points_label']: corner_pair_sample_points_label,\
                             #ops['corner_valid_mask_pair']: corner_valid_mask_pair, \
                             ops['open_gt_corner_valid_mask_256_64']: corner_valid_mask_256_64, \
                             ops['open_gt_labels_256_64']: open_gt_labels_256_64, \
                             ops['open_gt_labels_pair']: open_gt_labels_pair, \
                             ops['is_training_32']: is_training_32}

                summary, step, _, \
                seg_3_2_loss_val, pred_open_curve_seg[begin_idx*256:end_idx*256, ...] = \
                    sess.run([ops['merged'], ops['step'], ops['train_op'], \
                        ops[''], ops['pred_open_curve_seg']],feed_dict=feed_dict)                    


            train_writer.add_summary(summary, step)
            total_loss += loss_val
            total_edge_3_1_loss += edge_3_1_loss_val
            total_edge_3_1_acc += edge_3_1_acc_val
            total_edge_3_1_recall += edge_3_1_recall_val
            total_corner_3_1_loss += corner_3_1_loss_val
            total_corner_3_1_acc += corner_3_1_acc_val
            total_corner_3_1_recall += corner_3_1_recall_val
            total_reg_edge_3_1_loss += reg_edge_3_1_loss_val
            total_reg_corner_3_1_loss += reg_corner_3_1_loss_val
            if STAGE == 2:
                total_seg_3_2_loss += seg_3_2_loss_val
#            total_task_2_1_loss += task_2_1_loss_val
#            total_task_2_1_acc += task_2_1_acc_val
#            total_task_2_2_loss += task_2_2_loss_val
#            total_task_3_loss += task_3_loss_val
#            total_task_4_loss += task_4_loss_val
#            total_task_4_acc += task_4_acc_val
#            total_task_5_loss += task_5_loss_val
#            total_task_6_loss += task_6_loss_val
            #print('loss: %f' % loss_val)
        total_loss = total_loss * 1.0 / num_batch
        total_edge_3_1_loss = total_edge_3_1_loss * 1.0 / num_batch
        total_edge_3_1_acc = total_edge_3_1_acc * 1.0 / num_batch
        total_edge_3_1_recall = total_edge_3_1_recall * 1.0 / num_batch
        total_corner_3_1_loss = total_corner_3_1_loss * 1.0 / num_batch
        total_corner_3_1_acc = total_corner_3_1_acc * 1.0 / num_batch
        total_corner_3_1_recall = total_corner_3_1_recall * 1.0 / num_batch
        total_reg_edge_3_1_loss = total_reg_edge_3_1_loss * 1.0 / num_batch
        total_reg_corner_3_1_loss = total_reg_corner_3_1_loss * 1.0 / num_batch
        if STAGE == 2:
            total_seg_3_2_loss = total_seg_3_2_loss * 1.0 / num_batch
#        total_task_2_1_loss = total_task_2_1_loss * 1.0 / num_batch
#        total_task_2_1_acc = total_task_2_1_acc * 1.0 / num_batch
#        total_task_2_2_loss = total_task_2_2_loss * 1.0 / num_batch
#        total_task_3_loss = total_task_3_loss * 1.0 / num_batch
#        total_task_4_loss = total_task_4_loss * 1.0 / num_batch
#        total_task_4_acc = total_task_4_acc * 1.0 / num_batch
#        total_task_5_loss = total_task_5_loss * 1.0 / num_batch
#        total_task_6_loss = total_task_6_loss * 1.0 / num_batch
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
        log_string('\t\tTraining Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
        log_string('\t\tTraining Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
        log_string('\t\tTraining Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
        log_string('\t\tTraining Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
        log_string('\t\tTraining Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
        log_string('\t\tTraining Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
        log_string('\t\tTraining Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
        log_string('\t\tTraining Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)
        if STAGE == 2:
            log_string('\t\tTraining total_Seg_3_2 Mean_Loss: %f' % total_seg_3_2_loss)
#        log_string('\t\tTraining TASK 2_1 Mean_loss: %f' % total_task_2_1_loss)
#        log_string('\t\tTraining TASK 2_1 Accuracy: %f' % total_task_2_1_acc)
#        log_string('\t\tTraining TASK 2_2 Mean_loss: %f' % total_task_2_2_loss)
#        log_string('\t\tTraining TASK 3 Mean_loss: %f' % total_task_3_loss)
#        log_string('\t\tTraining TASK 4 Mean_loss: %f' % total_task_4_loss)
#        log_string('\t\tTraining TASK 4 Accuracy: %f' % total_task_4_acc)
#        log_string('\t\tTraining TASK 5 Mean_loss: %f' % total_task_5_loss)
#        log_string('\t\tTraining TASK 6 Mean_loss: %f' % total_task_6_loss)
        

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    train_or_test = "EVAL"
    is_training_31 = False
    is_training_32 = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    # just use one matrix.
    test_matrices_name = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/test_data/new_test/'), '40.mat')
    loadpath = BASE_DIR + '/test_data/new_test/'+test_matrices_name[0]
    test_data = sio.loadmat(loadpath)['Training_data']

    num_data = test_data.shape[0]
    num_batch = num_data // BATCH_SIZE
    total_loss = 0.0
    total_edge_3_1_loss = 0.0
    total_edge_3_1_recall = 0.0
    total_edge_3_1_acc = 0.0
    total_corner_3_1_loss = 0.0
    total_corner_3_1_recall = 0.0
    total_corner_3_1_acc = 0.0
    total_reg_edge_3_1_loss = 0.0
    total_reg_corner_3_1_loss = 0.0
    #        total_task_2_2_loss = 0.0
    #        total_task_3_loss = 0.0
    #        total_task_4_loss = 0.0
    #        total_task_4_acc = 0.0
    #        total_task_5_loss = 0.0
    #        total_task_6_loss = 0.0
    process_start_time = time.time()
    pred_labels_edge_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
    pred_labels_corner_p_val = np.zeros((num_data, NUM_POINT, 2), np.float32)
    pred_reg_edge_p_val = np.zeros((num_data, NUM_POINT, 3), np.float32)
    pred_reg_corner_p_val = np.zeros((num_data, NUM_POINT, 3), np.float32)
    input_labels_edge_p = np.zeros((num_data,NUM_POINT),np.int32)
    input_labels_corner_p = np.zeros((num_data,NUM_POINT),np.int32)
    np.random.shuffle(test_data)
    for j in range(num_batch):
        begin_idx = j*BATCH_SIZE
        end_idx = (j+1)*BATCH_SIZE
        data_cells = test_data[begin_idx: end_idx,0]
        batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
        batch_labels_edge_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)  # edge point label 0/1
        batch_labels_corner_p = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)  # edge point label 0/1
        #batch_labels_direction = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
        batch_regression_edge = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)  # each point normal estimation
        batch_regression_corner = np.zeros((BATCH_SIZE,NUM_POINT,3),np.float32)
        #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
        #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
        #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
        for cnt in range(BATCH_SIZE):
            tmp_data = data_cells[cnt]
            batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
            batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
            input_labels_edge_p[begin_idx+cnt, :] = np.squeeze(tmp_data[0,0]['edge_points_label'])
            batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
            input_labels_corner_p[begin_idx+cnt, :] = np.squeeze(tmp_data[0,0]['corner_points_label'])
            #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
            batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
            batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']
            #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
            #tmp_simmat = tmp_data['similar_matrix'][0,0]
            #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
            #tmp_neg_simmat = 1 - tmp_simmat
            #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
            #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
        feed_dict = {ops['pointclouds_pl']: batch_inputs,
                    ops['labels_edge_p']: batch_labels_edge_p,
                    ops['labels_corner_p']: batch_labels_corner_p,
                    #ops['labels_direction']: batch_labels_direction,
                    ops['reg_edge_p']: batch_regression_edge,
                    ops['reg_corner_p']: batch_regression_corner,
                    #ops['labels_type']: batch_labels_type,
                    #ops['simmat_pl']: batch_simmat_pl,
                    #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                    ops['is_training_31']: is_training_31,
                    ops['is_training_32']: is_training_32}
                    
                        
    #            summary, step, _, task_1_loss_val,task_1_recall_val,task_1_acc_val,task_2_1_loss_val,task_2_1_acc_val,task_2_2_loss_val, \
    #                                 task_3_loss_val,task_4_loss_val,task_4_acc_val,task_5_loss_val, \
    #                                 task_6_loss_val, loss_val = sess.run([ops['merged'], ops['step'], \
    #                                 ops['train_op'], ops['task_1_loss'], ops['task_1_recall'],ops['task_1_acc'],ops['task_2_1_loss'], \
    #                                 ops['task_2_1_acc'],ops['task_2_2_loss'],ops['task_3_loss'],ops['task_4_loss'], \
    #                                 ops['task_4_acc'],ops['task_5_loss'],ops['task_6_loss'],ops['loss']],feed_dict=feed_dict)
        summary, step, _, \
        edge_3_1_loss_val, edge_3_1_recall_val, edge_3_1_acc_val, \
        corner_3_1_loss_val, corner_3_1_recall_val, corner_3_1_acc_val, \
        reg_edge_3_1_loss_val, reg_corner_3_1_loss_val, loss_val, \
        pred_labels_edge_p_val[begin_idx:end_idx,:,:], pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
        pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:] = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], \
                ops['edge_3_1_loss'], ops['edge_3_1_recall'], ops['edge_3_1_acc'],\
                ops['corner_3_1_loss'], ops['corner_3_1_recall'], ops['corner_3_1_acc'],\
                ops['reg_edge_3_1_loss'], ops['reg_corner_3_1_loss'], ops['loss'], \
                ops['pred_labels_edge_p'], ops['pred_labels_corner_p'], \
                ops['pred_reg_edge_p'], ops['pred_reg_corner_p']],feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        total_loss += loss_val
        total_edge_3_1_loss += edge_3_1_loss_val
        total_edge_3_1_acc += edge_3_1_acc_val
        total_edge_3_1_recall += edge_3_1_recall_val
        total_corner_3_1_loss += corner_3_1_loss_val
        total_corner_3_1_acc += corner_3_1_acc_val
        total_corner_3_1_recall += corner_3_1_recall_val
        total_reg_edge_3_1_loss += reg_edge_3_1_loss_val
        total_reg_corner_3_1_loss += reg_corner_3_1_loss_val
    #            total_task_2_1_loss += task_2_1_loss_val
    #            total_task_2_1_acc += task_2_1_acc_val
    #            total_task_2_2_loss += task_2_2_loss_val
    #            total_task_3_loss += task_3_loss_val
    #            total_task_4_loss += task_4_loss_val
    #            total_task_4_acc += task_4_acc_val
    #            total_task_5_loss += task_5_loss_val
    #            total_task_6_loss += task_6_loss_val
                #print('loss: %f' % loss_val)
    total_loss = total_loss * 1.0 / num_batch
    total_edge_3_1_loss = total_edge_3_1_loss * 1.0 / num_batch
    total_edge_3_1_acc = total_edge_3_1_acc * 1.0 / num_batch
    total_edge_3_1_recall = total_edge_3_1_recall * 1.0 / num_batch
    total_corner_3_1_loss = total_corner_3_1_loss * 1.0 / num_batch
    total_corner_3_1_acc = total_corner_3_1_acc * 1.0 / num_batch
    total_corner_3_1_recall = total_corner_3_1_recall * 1.0 / num_batch
    total_reg_edge_3_1_loss = total_reg_edge_3_1_loss * 1.0 / num_batch
    total_reg_corner_3_1_loss = total_reg_corner_3_1_loss * 1.0 / num_batch
    #        total_task_2_1_loss = total_task_2_1_loss * 1.0 / num_batch
    #        total_task_2_1_acc = total_task_2_1_acc * 1.0 / num_batch
    #        total_task_2_2_loss = total_task_2_2_loss * 1.0 / num_batch
    #        total_task_3_loss = total_task_3_loss * 1.0 / num_batch
    #        total_task_4_loss = total_task_4_loss * 1.0 / num_batch
    #        total_task_4_acc = total_task_4_acc * 1.0 / num_batch
    #        total_task_5_loss = total_task_5_loss * 1.0 / num_batch
    #        total_task_6_loss = total_task_6_loss * 1.0 / num_batch
    process_duration = time.time() - process_start_time
    examples_per_sec = num_data/process_duration
    sec_per_batch = process_duration/num_batch
    log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
    % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
    log_string('\t\t%s Edge_3_1 Mean_Loss: %f' % ("EVAL", total_edge_3_1_loss))
    log_string('\t\t%s Edge_3_1 Mean_Accuracy: %f' % ("EVAL", total_edge_3_1_acc))
    log_string('\t\t%s Edge_3_1 Mean_Recall: %f' % ("EVAL", total_edge_3_1_recall))
    log_string('\t\t%s Corner_3_1 Mean_Loss: %f' % ("EVAL", total_corner_3_1_loss))
    log_string('\t\t%s Corner_3_1 Mean_Accuracy: %f' % ("EVAL", total_corner_3_1_acc))
    log_string('\t\t%s Corner_3_1 Mean_Recall: %f' % ("EVAL", total_corner_3_1_recall))
    log_string('\t\t%s Reg_Edge_3_1 Mean_Loss: %f' % ("EVAL", total_reg_edge_3_1_loss))
    log_string('\t\t%s Reg_Corner_3_1 Mean_Loss: %f' % ("EVAL", total_reg_corner_3_1_loss))

    #        log_string('\t\tTraining TASK 2_1 Mean_loss: %f' % total_task_2_1_loss)
    #        log_string('\t\tTraining TASK 2_1 Accuracy: %f' % total_task_2_1_acc)
    #        log_string('\t\tTraining TASK 2_2 Mean_loss: %f' % total_task_2_2_loss)
    #        log_string('\t\tTraining TASK 3 Mean_loss: %f' % total_task_3_loss)
    #        log_string('\t\tTraining TASK 4 Mean_loss: %f' % total_task_4_loss)
    #        log_string('\t\tTraining TASK 4 Accuracy: %f' % total_task_4_acc)
    #        log_string('\t\tTraining TASK 5 Mean_loss: %f' % total_task_5_loss)
    #        log_string('\t\tTraining TASK 6 Mean_loss: %f' % total_task_6_loss)
    sio.savemat('./test_result/test_pred_'+test_matrices_name[0], {'input_point_cloud': test_data, \
                                                'labels_edge_p': input_labels_edge_p, \
                                                'labels_corner_p': input_labels_corner_p, \
                                                'pred_labels_edge_p': pred_labels_edge_p_val, \
                                                'pred_labels_corner_p': pred_labels_corner_p_val, \
                                                'pred_reg_edge_p': pred_reg_edge_p_val, \
                                                'pred_reg_corner_p': pred_reg_corner_p_val})







'''
def train_one_epoch_stage_2(sess, ops, train_writer):
    is_training = True
    permutation = np.random.permutation(328)
    for i in range(len(permutation)/4):
        load_data_start_time = time.time()
        loadpath = BASE_DIR + '/train_data_stage_2/train_stage_2_data_'+str(permutation[i*4]+1)+'.mat'
        train_data = sio.loadmat(loadpath)['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        for j in range(3):
            temp_load_data_start_time = time.time()
            temp_loadpath = BASE_DIR + '/train_data_stage_2/train_stage_2_data_'+str(permutation[i*4+j+1]+1)+'.mat'
            temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
            temp_load_data_duration = time.time() - temp_load_data_start_time
            log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
            train_data = np.concatenate((train_data,temp_train_data),axis = 0)
            print(train_data.shape)
        
        num_data = train_data.shape[0]
        num_batch = num_data // BATCH_SIZE
        total_loss = 0.0
        process_start_time = time.time()
        np.random.shuffle(train_data)
        
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
            batch_inputs = np.zeros((BATCH_SIZE,NUM_POINT,6),np.float32)
            batch_dof_mask = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_proposal_nx = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            batch_dof_score = np.zeros((BATCH_SIZE,NUM_POINT),np.float32)
            for cnt in range(BATCH_SIZE):
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data['inputs_all'][0, 0]
                batch_dof_mask[cnt,:] = np.squeeze(tmp_data['dof_mask'][0,0])
                batch_proposal_nx[cnt,:] = np.squeeze(tmp_data['proposal_nx'][0,0])
                batch_dof_score[cnt,:] = np.squeeze(tmp_data['dof_score'][0,0])
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['proposal_nx_pl']: batch_proposal_nx,
                         ops['dof_mask_pl']: batch_dof_mask,
                         ops['dof_score_pl']: batch_dof_score,
                         ops['is_training_pl']: is_training}
                    
            summary, step, _, loss_val = sess.run([ops['merged'], ops['step'], \
                                 ops['train_op'],ops['loss']],feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            total_loss += loss_val
            #print('loss: %f' % loss_val)
        total_loss = total_loss * 1.0 / num_batch
        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
           % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
'''

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
