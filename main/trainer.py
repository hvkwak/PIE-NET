#
# Some of the codes are from https://github.com/charlesq34/pointnet2/blob/master/train_multi_gpu.py
#
import sys
import os
import socket
import importlib
import random
import tensorflow as tf
import numpy as np
import fnmatch
import time
from datetime import datetime
import scipy.io as sio
import scipy.special as ssp
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graphier_FPS import graphier_FPS
from multiprocess import parallel_apply_along_axis
#from func_311_312 import idx_pair_generator, corner_pair_label_generator
SN = 128
def idx_pair_generator(cloud_corner_reshaped):
    # generates idx_pairs, suitable for multiprocessing.
    # cloud_corner_reshaped: (self.BATCH_SIZE, 8096, 5)
    
    
    # Input:
    # cloud_corner_reshaped[..., 0:3]: (np.float32) (self.BATCH_SIZE, 8096, 3) contains point_cloud
    # cloud_corner_reshaped[..., 3:5]: (np.float32) (self.BATCH_SIZE, 8096, 2) contains pred_corner(softmax outputs of corner points)

    # returns
    # idx_pairs and corresponding pointclouds, labels and valid masks

    cloud_corner_concat = cloud_corner_reshaped.reshape(-1, 5)
    points_cloud = cloud_corner_concat[..., 0:3]
    pred_corner = cloud_corner_concat[..., 3:5]

    threshold = 0.99
    # idx_pair with -1: invalid
    idx_pair = np.zeros((256, 2), dtype = np.int32)
    idx = np.where(pred_corner[..., 1] > threshold)[0]

    # check if we have enough corner points from softmax
    while idx.shape[0] < 23:
        threshold = threshold - 0.001
        idx = np.where(pred_corner[..., 1] > threshold)[0]
    point_cloud_corners = points_cloud[idx, ...]
    
    # K-means clustering: possibly there may be 23 corner points!
    # + non maximum supression?
    # init
    best_distance_std = np.Inf
    best_k = -1
    early_stopped = False

    # search for best_k of clusters k = 8, ... 23
    for k in range(4, 24):
        if not early_stopped:
            best_distance_mean = np.Inf
            _, centroids_idx = graphier_FPS(point_cloud_corners, k, np.arange(point_cloud_corners.shape[0]))
            centroids_xyz = point_cloud_corners[centroids_idx, ...]
            # distances to centroids
            distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
            mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
            while mean_mindistance_to_centroids < best_distance_mean:
                best_distance_mean = mean_mindistance_to_centroids
                classes = np.argmin(distances, axis = 0) # each one belongs to the nearest cluster
                centroids_idx = np.stack([np.where(classes == i)[0][np.argmin(np.sum((point_cloud_corners[np.where(classes == i)[0], ...] - np.mean(point_cloud_corners[np.where(classes == i)[0], ...]))**2))] for i in range(k)])
                centroids_xyz = point_cloud_corners[centroids_idx, ...]
                #centroids_xyz = np.stack([np.mean(point_cloud_corners[np.where(classes == i)[0], ...], axis = 0) for i in range(k)], axis = 0)
                distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
                mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
            distance_std = np.std(np.min(distances, 0))
            # early stop
            if distance_std < 0.001 and best_distance_mean < 0.001:
                #best_k = k
                early_stopped = True
                break
                # check for early stopping
                #print("best_distance_std: ", best_distance_std)
                #print("best_distance_mean: ", best_distance_mean)
                #print("best_k: ", best_k)                
                #print(best_distance_mean)
                #print(k)
                #if best_distance_std < 0.0003 and best_distance_mean < 0.0001: # Early stopping
                #    #print(best_distance)
                #    early_stopped = True
                #    break
        
    
    #if not early_stopped:
    #    best_k = 23
    class_centroids_idx = idx[centroids_idx]
    # best_k found. non maximum supression!
    '''
    clusters_ready = False # we need at least one centroid per corner.
    while not clusters_ready:
        best_distance_mean = np.Inf
        _, centroids_idx = graphier_FPS(point_cloud_corners, best_k, np.arange(point_cloud_corners.shape[0]))
        centroids_xyz = point_cloud_corners[centroids_idx, ...]
        # distances to centroids
        distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
        mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
        while mean_mindistance_to_centroids < best_distance_mean:
            best_distance_mean = mean_mindistance_to_centroids
            classes = np.argmin(distances, axis = 0) # each one belongs to the nearest cluster
            centroids_idx = np.stack([np.where(classes == i)[0][np.argmin(np.sum((point_cloud_corners[np.where(classes == i)[0], ...] - np.mean(point_cloud_corners[np.where(classes == i)[0], ...]))**2))] for i in range(k)])
            centroids_xyz = point_cloud_corners[centroids_idx, ...]
            #centroids_xyz = np.stack([np.mean(point_cloud_corners[np.where(classes == i)[0], ...], axis = 0) for i in range(best_k)], axis = 0)
            distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
            mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
        if np.unique(classes).shape[0] == best_k:
            clusters_ready = True
        else:
            # update this until they are all ok
            random_num = random.randint(-1, 1) # increase, decrease, or it stays the same.
            if 4 < best_k + random_num < 24: best_k = best_k + random_num
    '''
            

    # non maximum suppression(nms) in every cluster: choose the best corner points per cluster.
    '''
    idx_in_class = idx[np.where(classes == k)[0]] # per class k = 0, 1, 2, ...
    
    np.argmin(np.sum((centroids_xyz[k, ...] - point_cloud_corners[np.where(classes == k)[0]])**2, axis = 1), axis = 0)
    argmax_idx = np.argmax(pred_corner[idx_in_class, 1])
    idx_in_class[argmax_idx]
    '''

    # take best_k centroids
    # class_centroids_idx = np.array([idx[np.where(classes == k)[0]][np.argmin(np.sum((centroids_xyz[k, ...] - point_cloud_corners[np.where(classes == k)[0]])**2, axis = 1), axis = 0)] for k in range(best_k)])
    

    if class_centroids_idx.shape[0] > 1:
        idx = np.sort(class_centroids_idx)
        idx_r = np.repeat(idx, idx.shape[0])
        idx_b = np.tile(idx, [idx.shape[0]])
        two_col = np.stack([idx_r, idx_b], 1)
        idx_pair[0:(idx.shape[0]*(idx.shape[0]-1)//2), ...] = two_col[two_col[:,0] < two_col[:, 1]]
        #idx_pair.append()

    # idx_pair ready!
    rest_num = 256

    # first increase the precision to float64, 
    # otherwise it may go wrong when it comes to finding points within radius, 
    # where it may also include the two corner points at the end.
    points_cloud = np.float64(points_cloud)

    # find neighbors
    xyz1 = points_cloud[idx_pair[:, 0], :]
    xyz2 = points_cloud[idx_pair[:, 1], :]
    ball_center = np.mean(np.stack([xyz1, xyz2], axis = 0), axis = 0)

    distance_from_ball_center = np.sqrt(np.sum(((np.expand_dims(ball_center, 1) - np.expand_dims(points_cloud,axis=0))**2), axis = 2))
    # for safety: - 0.00001
    r = np.sqrt(np.sum((xyz1 - xyz2)**2, axis = 1)) / 2.0  - 0.00001
    within_range = distance_from_ball_center < np.multiply(np.ones_like(distance_from_ball_center), np.expand_dims(r, axis = 1))
    corner_pair_num = class_centroids_idx.shape[0]*(class_centroids_idx.shape[0] - 1) // 2

    # memory arrays
    points_256_sn_3 = np.zeros((256, SN, 3), dtype = np.float32)
    idx_256_sn = np.zeros((256, SN), dtype = np.int32)
    valid_mask_256_sn = np.zeros((256, SN), dtype = np.int32)
        
    # if there are more than 256 pairs, just take first 256.
    if corner_pair_num > 256 : corner_pair_num = 256
    rest_num = 256 - corner_pair_num

    if corner_pair_num > 0:
        for per_corner in range(corner_pair_num):
            # make sure that corner points(end points) are not within the range.
            assert within_range[per_corner, ...][idx_pair[per_corner][0]] == False
            assert within_range[per_corner, ...][idx_pair[per_corner][1]] == False
            #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
            #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
            candidnate_num = np.where(within_range[per_corner, :])[0].shape[0]
            #
            # raise error or debug when within_range.shape[0] = 0
            # or if within_range.shape[0] = 3?
            if SN-2 <= candidnate_num:
                middle_indicies = np.where(within_range[per_corner, :])[0]
                np.random.shuffle(middle_indicies)
                idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), np.squeeze(middle_indicies[:SN-2]), np.expand_dims(idx_pair[per_corner][-1], axis = 0)], axis = 0)
                idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                #idx_256_sn.append(np.expand_dims(idx_nums, axis = 0))
                valid_mask_256_sn[per_corner, :] = np.expand_dims(np.ones_like(idx_nums), axis = 0)

            elif 0 < candidnate_num < SN-2:
                n = candidnate_num
                dummy_num = SN - 1 - n
                middle_indicies = np.where(within_range[per_corner, :])[0]
                #if candidnate_num == 1: 
                    #middle_indicies = np.expand_dims(middle_indicies, axis = 0)
                idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), middle_indicies, np.repeat(idx_pair[per_corner][-1], dummy_num)], axis = 0)
                idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                valid_mask_256_sn[per_corner, :] = np.expand_dims(np.concatenate([np.ones((SN - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0)
            elif candidnate_num == 0:
                dummy_num = SN-1
                idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), np.repeat(idx_pair[per_corner][-1], dummy_num)], axis = 0)
                idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                valid_mask_256_sn[per_corner, :] = np.expand_dims(np.concatenate([np.ones((SN - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0)

        if rest_num > 0: 
            idx_256_sn[corner_pair_num:, :] = np.zeros((rest_num, SN), dtype = np.int64)
            valid_mask_256_sn[corner_pair_num:, :] = np.zeros((rest_num, SN), dtype = np.int64)
    
        # return idx_256_sn
        # return valid_mask_256_sn
        #idx_B_256_sn.append(np.concatenate(idx_256_sn, axis = 0))
        #valid_mask_256_sn_per_batch.append(np.concatenate(valid_mask_256_sn_per_corner, axis = 0))
        points_256_sn_3 = np.stack([points_cloud[idx_256_sn[k], ...] for k in range(256)], axis = 0)
        
    valid_mask = np.ones(256, dtype = np.int8)
    valid_mask[-rest_num:] = 0
    valid_mask = np.expand_dims(valid_mask, axis = 1)
    tp = np.dtype([
        ('points_256_sn_3', 'O'),
        ('idx_256_sn', 'O'),
        ('idx_pair', 'O'),
        ('valid_mask', 'O'),
        ('valid_mask_256_sn', 'O')
    ])
    corner_neighbors = np.zeros((1, 1), dtype = tp)
    for tp_name in tp.names:
        save_this_piece = locals()[tp_name]
        corner_neighbors[tp_name][0, 0] = save_this_piece
    return corner_neighbors

def corner_pair_label_generator(corner_pair_label_generator_inputs):

    # from corner_pair_neighbor_search
    #points_256_sn_3 = corner_pair_label_generator_inputs[0, ...]['points_256_sn_3'][()]
    idx_256_sn = corner_pair_label_generator_inputs[0, ...]['idx_256_sn'][()]
    idx_pair = corner_pair_label_generator_inputs[0, ...]['idx_pair'][()]
    mask_256_1_if_proposed = corner_pair_label_generator_inputs[0, ...]['mask_256_1_if_proposed'][()]
    #mask_256_sn_if_proposed = corner_pair_label_generator_inputs[0, ...]['mask_256_sn_if_proposed'][()]

    # from batch
    batch_inputs = corner_pair_label_generator_inputs[0, ...]['batch_inputs'][()]
    batch_open_gt_pair_idx = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_pair_idx'][()]
    batch_open_gt_256_sn_idx = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_256_sn_idx'][()]
    batch_open_gt_type = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_type'][()]
    batch_open_gt_type_one_hot = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_type_one_hot'][()]
    
    batch_num = 1
    mask_256_sn_if_edge = np.zeros((256, SN), dtype = np.int32) # output should be (batch_num, 256, SN, 2)
    mask_256_1_if_edge = np.zeros((256, 1), dtype = np.int16)
    sample_corner_type = np.zeros((256), dtype = np.int16)
    sample_corner_type_one_hot = np.zeros((256, 4), dtype = np.int16)
    gt_idx_array = np.zeros((256), dtype = np.int16)
    dist_threshold = 0.007
    n_values = 4

    k = 0
    while k < 256 and mask_256_1_if_proposed[k][0] == 1:

        # per curve pair k in one batch
        if (idx_pair[k] == batch_open_gt_pair_idx).all(axis = 1).any():
            # indices match exactly
            gt_idx = np.where((idx_pair[k] == batch_open_gt_pair_idx).all(axis = 1))[0]
            # gt_idx = np.where(batch_open_gt_pair_idx[i][k].numpy() in my_mat['open_gt_pair_idx'][0, 0])[0][0]
            # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
            gt_idx_array[k] = gt_idx
            sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
            # one-hot encoding
            sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
            mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
            mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
            mask_256_1_if_edge[k, 0] = 1
            k = k+1
            continue
        elif (np.flip(idx_pair[k]) == batch_open_gt_pair_idx).all(axis = 1).any():
            gt_idx = np.where((np.flip(idx_pair[k]) == batch_open_gt_pair_idx).all(axis = 1))[0]
            sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
            gt_idx_array[k] = gt_idx
            sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
            # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
            mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
            # update here labels
            mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
            mask_256_1_if_edge[k, 0] = 1
            k = k+1
            continue

        # not exact match, but see if there is one nearby.
        # calculate distances NN.
        distance = np.sqrt(np.sum((batch_inputs[idx_pair[k], :] - batch_inputs[batch_open_gt_pair_idx, :])**2, axis = 2))
        #print("distance.mean(axis = 1).min(): ", np.min(distance.mean(axis = 1)))
        if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
            gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
            gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
            #print("distance[gt_idx]: ", distance[gt_idx, ...])
            sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
            gt_idx_array[k] = gt_idx
            sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
            mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[:, :][gt_idx])
            mask[0], mask[-1] = True, True
            mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
            mask_256_1_if_edge[k, 0] = 1
            k = k+1
            continue

        distance = np.sqrt(np.sum((batch_inputs[np.flip(idx_pair[k]), :] - batch_inputs[batch_open_gt_pair_idx, :])**2, axis = 2))
        #print("distance.mean(axis = 1).min(): ", np.min(distance.mean(axis = 1)))
        if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
            gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
            gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
            #print("distance[gt_idx]: ", distance[gt_idx, ...])
            sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
            gt_idx_array[k] = gt_idx
            sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
            mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
            mask[0], mask[-1] = True, True
            mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
            mask_256_1_if_edge[k, 0] = 1
            k = k+1
            continue
        
        # Null class
        #assert batch_open_gt_type[gt_idx][0] == 0
        sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[0]
        k = k+1

    mask_256_sn_if_edge = mask_256_sn_if_edge.reshape((-1, SN))
    mask_256_1_if_edge = mask_256_1_if_edge.reshape(-1, 1)
    labels_type = sample_corner_type.reshape(-1)
    labels_type_one_hot = sample_corner_type_one_hot.reshape(-1, 4)
    gt_idx_array = gt_idx_array
    tp = np.dtype([
        ('mask_256_sn_if_edge', 'O'),
        ('mask_256_1_if_edge', 'O'),
        ('labels_type', 'O'),
        ('labels_type_one_hot', 'O'),
        ('gt_idx_array', 'O')
    ])
    corner_pair_label_generator_outputs = np.zeros((1, 1), dtype = tp)
    for tp_name in tp.names:
        save_this_piece = locals()[tp_name]
        corner_pair_label_generator_outputs[0, 0][tp_name] = save_this_piece
    
    return corner_pair_label_generator_outputs
    #return mask_256_sn_if_edge, mask_256_1_if_edge, labels_type, labels_type_one_hot, gt_idx_array



class NetworkTrainer:
    
    def __init__(self, FLAGS, BASE_DIR, ROOT_DIR):

        self.BASE_DIR = BASE_DIR
        self.ROOT_DIR = ROOT_DIR
        sys.path.append(self.BASE_DIR)
        sys.path.append(os.path.join(self.BASE_DIR, 'models'))
        sys.path.append(os.path.join(self.ROOT_DIR, 'utils'))
        
        self.EPOCH_CNT = 0
        self.NUM_GPUS = FLAGS.num_gpus
        self.BATCH_SIZE = FLAGS.batch_size
        assert(self.BATCH_SIZE % self.NUM_GPUS == 0)
        self.DEVICE_BATCH_SIZE = self.BATCH_SIZE // self.NUM_GPUS

        self.NUM_POINT = FLAGS.num_point
        self.MAX_EPOCH = FLAGS.max_epoch
        self.BASE_LEARNING_RATE = FLAGS.learning_rate
        
        self.MOMENTUM = FLAGS.momentum
        self.OPTIMIZER = FLAGS.optimizer
        self.DECAY_STEP = FLAGS.decay_step
        self.DECAY_RATE = FLAGS.decay_rate
        # STAGE = 1: Section 3.1.
        # STAGE = 2: Section 3.1. + Section 3.2.
        self.STAGE = FLAGS.stage
        self.RESUME = FLAGS.resume

        self.MODEL = importlib.import_module(FLAGS.model) # import network module
        self.MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
        if self.STAGE == 1:
            self.LOG_DIR = os.path.join(self.BASE_DIR, FLAGS.stage_1_log_dir)
        elif self.STAGE == 2:
            self.LOG_DIR = os.path.join(self.BASE_DIR, FLAGS.stage_2_log_dir)
        elif self.STAGE == 3:
            self.LOG_DIR = os.path.join(self.BASE_DIR, FLAGS.stage_3_log_dir)

        if not os.path.exists(self.LOG_DIR): 
            os.mkdir(self.LOG_DIR)
            os.system('cp %s %s' % (self.MODEL_FILE, self.LOG_DIR)) # bkp of model def
            os.system('cp main_oop.py %s' % (ROOT_DIR+"/"+self.LOG_DIR)) # bkp of train procedure
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log.txt'), 'w')
        self.LOG_FOUT.write(str(FLAGS)+'\n')

        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
        self.BN_DECAY_CLIP = 0.99
        self.HOSTNAME = socket.gethostname()

        self.graph_31 = tf.Graph()
        self.graph_32 = tf.Graph()

    def build_graph_32(self):
        with self.graph_32.as_default():
            with tf.device('/cpu:0'):
                self.points_256_sn_3, \
                self.mask_256_sn_if_edge, \
                self.labels_type, \
                self.labels_type_one_hot, \
                self.mask_256_sn_if_proposed, \
                self.mask_256_1_if_edge,\
                self.mask_256_1_if_proposed = self.MODEL.placeholder_inputs_32(self.BATCH_SIZE)
                self.is_training_32 = tf.compat.v1.placeholder(tf.bool, shape=())
                
                self.batch_32 = tf.compat.v1.get_variable('batch', [],initializer=tf.compat.v1.constant_initializer(0), trainable=False)
                self.bn_decay_32 = self.get_bn_decay(self.batch_32)
                tf.compat.v1.summary.scalar('bn_decay', self.bn_decay_32)

                print("--- Get training operator")
                # Get training operator
                self.learning_rate_32 = self.get_learning_rate(self.batch_32)
                tf.compat.v1.summary.scalar('learning_rate', self.learning_rate_32)
                if self.OPTIMIZER == 'momentum':
                    self.optimizer_32 = tf.compat.v1.train.MomentumOptimizer(self.learning_rate_32, momentum=self.MOMENTUM)
                elif self.OPTIMIZER == 'adam':
                    self.optimizer_32 = tf.compat.v1.train.AdamOptimizer(self.learning_rate_32)

                self.MODEL.get_model_32(self.points_256_sn_3, self.is_training_32, bn_decay=self.bn_decay_32)
                #self.MODEL.get_model_32(self.points_256_sn_3, self.is_training_32, self.STAGE, bn_decay=self.bn_decay_32)

                tower_grads_stage2 = []
                pred_seg_p_gpu = []
                pred_cls_p_gpu = []
                pred_reg_Line_p_gpu = []
                pred_reg_BSpline_p_gpu = []
                seg_3_2_loss_p_gpu = []
                cls_3_2_loss_p_gpu = []
                #reg_3_2_loss_p_gpu = []
                #mat_diff_3_2_loss_p_gpu = []
                total_loss_gpu = []
                #end_points_p_gpu = []

                for i in range(self.NUM_GPUS):
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.compat.v1.name_scope('gpu_%d'%(i)) as scope:
                            device_batch_size_3_2 = self.BATCH_SIZE*256//self.NUM_GPUS
                            batch_points_256_sn_3 = tf.slice(self.points_256_sn_3, [i*device_batch_size_3_2,0,0], [device_batch_size_3_2,-1,-1])
                            batch_mask_256_sn_if_edge = tf.slice(self.mask_256_sn_if_edge, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_mask_256_sn_if_proposed = tf.slice(self.mask_256_sn_if_proposed, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_mask_256_1_if_edge = tf.slice(self.mask_256_1_if_edge, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_mask_256_1_if_proposed = tf.slice(self.mask_256_1_if_proposed, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_labels_type = tf.slice(self.labels_type, [i*device_batch_size_3_2], [device_batch_size_3_2])
                            batch_labels_type_one_hot = tf.slice(self.labels_type_one_hot, [i*device_batch_size_3_2, 0], [device_batch_size_3_2, -1])
                            #pred_open_curve_seg, pred_open_curve_cls, pred_open_curve_reg_BSpline, pred_open_curve_reg_Line, end_points = self.MODEL.get_model_32(batch_points_256_sn_3, self.is_training_32, bn_decay=self.bn_decay_32)
                            #pred_open_curve_seg, pred_open_curve_cls, end_points = self.MODEL.get_model_32(batch_points_256_sn_3, self.is_training_32, bn_decay=self.bn_decay_32)
                            pred_open_curve_seg, pred_open_curve_cls = self.MODEL.get_model_32(batch_points_256_sn_3, self.is_training_32, bn_decay=self.bn_decay_32)
                            #pred_open_curve_seg, pred_open_curve_cls = self.MODEL.get_model_32(batch_points_256_sn_3, self.is_training_32, self.STAGE, bn_decay=self.bn_decay_32)
                            
                            #loss_32, seg_3_2_loss, cls_3_2_loss, reg_3_2_loss, mat_diff_loss = self.MODEL.get_stage_2_loss(pred_open_curve_seg, \
                            loss_32, seg_3_2_loss, cls_3_2_loss = self.MODEL.get_stage_2_loss(pred_open_curve_seg, \
                                                        pred_open_curve_cls, \
                                                        #pred_open_curve_reg_Line, \
                                                        #pred_open_curve_reg_BSpline, \
                                                        batch_mask_256_sn_if_edge, \
                                                        batch_mask_256_sn_if_proposed, \
                                                        batch_mask_256_1_if_edge, \
                                                        batch_mask_256_1_if_proposed,\
                                                        batch_labels_type, \
                                                        batch_labels_type_one_hot, \
                                                        batch_points_256_sn_3, \
                                                        #batch_open_gt_res, \
                                                        #batch_open_gt_sample_points, \
                                                        #batch_open_gt_256_sn_idx, \
                                                        #batch_open_gt_mask, \
                                                        #batch_open_gt_valid_mask, \
                                                        #batch_open_gt_pair_idx, \
                                                        #batch_open_gt_type,\
                                                        #end_points, \
                                                        0.0001)

                            tf.compat.v1.summary.scalar('%d_GPU_(mat_diff included) loss'% (i), loss_32)
                            grads = self.optimizer_32.compute_gradients(loss_32) # here's where the loss and gradients are covered.
                            tower_grads_stage2.append(grads)
                            pred_seg_p_gpu.append(pred_open_curve_seg)
                            pred_cls_p_gpu.append(pred_open_curve_cls)
                            #pred_reg_Line_p_gpu.append(pred_open_curve_reg_Line)
                            #pred_reg_BSpline_p_gpu.append(pred_open_curve_reg_BSpline)
                            #end_points_p_gpu.append(end_points)
                            
                            #pred_reg_p_gpu.append(pred_open_curve_reg)
                            total_loss_gpu.append(loss_32)
                            seg_3_2_loss_p_gpu.append(seg_3_2_loss)
                            cls_3_2_loss_p_gpu.append(cls_3_2_loss)
                            #reg_3_2_loss_p_gpu.append(reg_3_2_loss)
                            #mat_diff_3_2_loss_p_gpu.append(mat_diff_loss)

                # average or concat
                self.pred_open_curve_seg = tf.concat(pred_seg_p_gpu, 0)
                self.pred_open_curve_cls = tf.concat(pred_cls_p_gpu, 0)
                #self.pred_open_curve_reg_Line = tf.concat(pred_reg_Line_p_gpu, 0)
                #self.pred_open_curve_reg_BSpline = tf.concat(pred_reg_BSpline_p_gpu, 0)
                #self.end_points = end_points_p_gpu
                self.total_loss_32 = tf.reduce_mean(input_tensor = total_loss_gpu)
                self.seg_3_2_loss = tf.reduce_mean(input_tensor = seg_3_2_loss_p_gpu)
                self.cls_3_2_loss = tf.reduce_mean(input_tensor = cls_3_2_loss_p_gpu)
                #self.reg_3_2_loss = tf.reduce_mean(input_tensor = reg_3_2_loss_p_gpu)
                #self.mat_diff_loss = tf.reduce_mean(input_tensor = mat_diff_loss)

                # Get training operator
                grads = self.average_gradients(tower_grads_stage2)
                self.train_optimizer_32 = self.optimizer_32.apply_gradients(grads, global_step=self.batch_32)
                # Get training operator
                self.saver_32 = tf.compat.v1.train.Saver(max_to_keep=10)       

    def build_graph_31(self):
        with self.graph_31.as_default():
            with tf.device('/cpu:0'):
                self.pointclouds_pl, self.labels_edge_p, self.labels_corner_p, self.reg_edge_p, self.reg_corner_p = self.MODEL.placeholder_inputs_31(self.BATCH_SIZE, self.NUM_POINT)
                self.is_training_31 = tf.compat.v1.placeholder(tf.bool, shape=())
                self.batch_31 = tf.compat.v1.get_variable('batch', [],initializer=tf.compat.v1.constant_initializer(0), trainable=False)
                self.bn_decay_31 = self.get_bn_decay(self.batch_31)
                tf.compat.v1.summary.scalar('bn_decay', self.bn_decay_31)

                print("--- Get training operator")
                # Get training operator
                self.learning_rate_31 = self.get_learning_rate(self.batch_31)
                tf.compat.v1.summary.scalar('learning_rate', self.learning_rate_31)
                if self.OPTIMIZER == 'momentum':
                    self.optimizer_31 = tf.compat.v1.train.MomentumOptimizer(self.learning_rate_31, momentum=self.MOMENTUM)
                elif self.OPTIMIZER == 'adam':
                    self.optimizer_31 = tf.compat.v1.train.AdamOptimizer(self.learning_rate_31)
                
                print("--- Get model and loss")
                # -------------------------------------------
                # Get model and loss on multiple GPU devices
                # -------------------------------------------
                # Allocating variables on CPU first will greatly accelerate multi-gpu training.
                # Ref: https://github.com/kuza55/keras-extras/issues/21
                self.MODEL.get_model_31(self.pointclouds_pl, self.is_training_31, self.STAGE, bn_decay=self.bn_decay_31)
                
                # Sec. 3.1
                tower_grads_stage1 = []
                pred_labels_edge_p_gpu = []
                pred_labels_corner_p_gpu = []
                pred_reg_edge_p_gpu = []
                pred_reg_corner_p_gpu = []
                total_loss_gpu = []
                edge_3_1_loss_p_gpu = []
                edge_3_1_recall_p_gpu = []
                edge_3_1_acc_p_gpu = []
                corner_3_1_loss_p_gpu = []
                corner_3_1_recall_p_gpu = []
                corner_3_1_acc_p_gpu = []
                reg_edge_3_1_loss_p_gpu = []
                reg_corner_3_1_loss_p_gpu = []

                for i in range(self.NUM_GPUS):
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.compat.v1.name_scope('gpu_%d'%(i)) as scope:
                            # Evenly split input data to each GPU
                            ## check if dimension numbers are correct:
                            batch_pc = tf.slice(self.pointclouds_pl, [i*self.DEVICE_BATCH_SIZE,0,0], [self.DEVICE_BATCH_SIZE,-1,-1])
                            batch_labels_edge_p = tf.slice(self.labels_edge_p,[i*self.DEVICE_BATCH_SIZE, 0], [self.DEVICE_BATCH_SIZE, -1])
                            batch_labels_corner_p = tf.slice(self.labels_corner_p,[i*self.DEVICE_BATCH_SIZE, 0], [self.DEVICE_BATCH_SIZE, -1])
                            batch_reg_edge_p = tf.slice(self.reg_edge_p, [i*self.DEVICE_BATCH_SIZE, 0, 0], [self.DEVICE_BATCH_SIZE, -1, -1])
                            batch_reg_corner_p = tf.slice(self.reg_corner_p, [i*self.DEVICE_BATCH_SIZE, 0, 0], [self.DEVICE_BATCH_SIZE, -1, -1])
                            
                            
                            pred_labels_edge_p, pred_labels_corner_p, pred_reg_edge_p, pred_reg_corner_p = self.MODEL.get_model_31(batch_pc, self.is_training_31, self.STAGE, bn_decay=self.bn_decay_31)
                            # LOSS
                            edge_3_1_loss,   edge_3_1_recall,   edge_3_1_acc,\
                            corner_3_1_loss, corner_3_1_recall, corner_3_1_acc,\
                            reg_edge_3_1_loss, reg_corner_3_1_loss, loss_31 = self.MODEL.get_stage_1_loss(pred_labels_edge_p, \
                                                                                                            pred_labels_corner_p, \
                                                                                                            batch_labels_edge_p, \
                                                                                                            batch_labels_corner_p, \
                                                                                                            pred_reg_edge_p, \
                                                                                                            pred_reg_corner_p, \
                                                                                                            batch_reg_edge_p, \
                                                                                                            batch_reg_corner_p)
                            tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_loss' % (i), edge_3_1_loss)
                            tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_recall' % (i), edge_3_1_recall)
                            tf.compat.v1.summary.scalar('%d_GPU_edge_3_1_acc' % (i), edge_3_1_acc)                
                            tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_loss' % (i), corner_3_1_loss)
                            tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_recall' % (i), corner_3_1_recall)
                            tf.compat.v1.summary.scalar('%d_GPU_corner_3_1_acc' % (i), corner_3_1_acc)
                            tf.compat.v1.summary.scalar('%d_GPU_reg_edge_3_1_loss' % (i), reg_edge_3_1_loss)
                            tf.compat.v1.summary.scalar('%d_GPU_reg_corner_3_1_loss' % (i), reg_corner_3_1_loss)
                            tf.compat.v1.summary.scalar('%d_GPU_loss'% (i), loss_31)
                            grads = self.optimizer_31.compute_gradients(loss_31) # here's where the loss and gradients are covered.
                            tower_grads_stage1.append(grads)

                            ## check this: 
                            # losses = tf.compat.v1.get_collection('losses', scope)
                            # total_loss = tf.add_n(losses, name='total_loss')
                            pred_labels_edge_p_gpu.append(pred_labels_edge_p)
                            pred_labels_corner_p_gpu.append(pred_labels_corner_p)
                            pred_reg_edge_p_gpu.append(pred_reg_edge_p)
                            pred_reg_corner_p_gpu.append(pred_reg_corner_p)
                            total_loss_gpu.append(loss_31)
                            edge_3_1_loss_p_gpu.append(edge_3_1_loss)
                            edge_3_1_recall_p_gpu.append(edge_3_1_recall)
                            edge_3_1_acc_p_gpu.append(edge_3_1_acc)
                            corner_3_1_loss_p_gpu.append(corner_3_1_loss)
                            corner_3_1_recall_p_gpu.append(corner_3_1_recall)
                            corner_3_1_acc_p_gpu.append(corner_3_1_acc)
                            reg_edge_3_1_loss_p_gpu.append(reg_edge_3_1_loss)
                            reg_corner_3_1_loss_p_gpu.append(reg_corner_3_1_loss)

                ## Merge pred and losses from multiple GPUs
                self.pred_labels_edge_p = tf.concat(pred_labels_edge_p_gpu, 0)
                self.pred_labels_corner_p = tf.concat(pred_labels_corner_p_gpu, 0)
                self.pred_reg_edge_p = tf.concat(pred_reg_edge_p_gpu, 0)
                self.pred_reg_corner_p = tf.concat(pred_reg_corner_p_gpu, 0)
                self.total_loss_31 = tf.reduce_mean(input_tensor = total_loss_gpu)
                self.edge_3_1_loss = tf.reduce_mean(edge_3_1_loss_p_gpu)
                self.edge_3_1_recall = tf.reduce_mean(edge_3_1_recall_p_gpu)
                self.edge_3_1_acc = tf.reduce_mean(edge_3_1_acc_p_gpu)
                self.corner_3_1_loss = tf.reduce_mean(corner_3_1_loss_p_gpu)
                self.corner_3_1_recall = tf.reduce_mean(corner_3_1_recall_p_gpu)
                self.corner_3_1_acc = tf.reduce_mean(corner_3_1_acc_p_gpu)
                self.reg_edge_3_1_loss = tf.reduce_mean(reg_edge_3_1_loss_p_gpu)
                self.reg_corner_3_1_loss = tf.reduce_mean(reg_corner_3_1_loss_p_gpu)

                # Get training operator
                grads = self.average_gradients(tower_grads_stage1)
                self.train_optimizer_31 = self.optimizer_31.apply_gradients(grads, global_step=self.batch_31)
                # train_op = optimizer.minimize(loss, global_step=batch_stage_1)
                # Add ops to save and restore all the variables.
                self.saver_31 = tf.compat.v1.train.Saver(max_to_keep=10)

    def eval_graph_32(self):
        # init graph_31
        with self.graph_31.as_default():
            # Create a session
            config_31 = tf.compat.v1.ConfigProto()
            config_31.gpu_options.allow_growth = True
            config_31.allow_soft_placement = True
            config_31.log_device_placement = False
            self.sess_31 = tf.compat.v1.Session(graph = self.graph_31, config=config_31)
            self.merged_31 = tf.compat.v1.summary.merge_all()
            self.test_writer_31 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'), self.sess_31.graph)                            
            # load graph_31
            init_31 = tf.compat.v1.global_variables_initializer()
            self.sess_31.run(init_31)
            self.saver_31.restore(self.sess_31, self.BASE_DIR+'/stage_1_log/model_31_498.ckpt')
            # build train ops
            self.train_ops_31 = {'pointclouds_pl': self.pointclouds_pl,
                                'labels_edge_p': self.labels_edge_p,
                                'labels_corner_p': self.labels_corner_p,
                                #'labels_direction': labels_direction,
                                'reg_edge_p': self.reg_edge_p,
                                'reg_corner_p': self.reg_corner_p,
                                #'labels_type': labels_type,
                                #'simmat_pl': simmat_pl,
                                #'neg_simmat_pl': neg_simmat_pl,
                                'is_training_31': self.is_training_31,
                                'pred_labels_edge_p': self.pred_labels_edge_p,                   #  'pred_labels_edge_points'
                                'pred_labels_corner_p': self.pred_labels_corner_p,
                                #'pred_labels_direction': pred_labels_direction,
                                'pred_reg_edge_p': self.pred_reg_edge_p,
                                'pred_reg_corner_p': self.pred_reg_corner_p,
                                #'pred_labels_type': pred_labels_type,
                                #'pred_simmat': pred_simmat,
                                #'pred_conf': pred_conf_logits,
                                'edge_3_1_loss': self.edge_3_1_loss,
                                'edge_3_1_recall':self.edge_3_1_recall,
                                'edge_3_1_acc': self.edge_3_1_acc,
                                'corner_3_1_loss': self.corner_3_1_loss,
                                'corner_3_1_recall': self.corner_3_1_recall,
                                'corner_3_1_acc': self.corner_3_1_acc,
                                'reg_edge_3_1_loss': self.reg_edge_3_1_loss,
                                'reg_corner_3_1_loss': self.reg_corner_3_1_loss,
                                #'seg_3_2_acc': seg_3_2_acc,
                                #'task_2_2_loss': task_2_2_loss,
                                #'task_3_loss': task_3_loss,
                                #'task_4_loss': task_4_loss,
                                #'task_4_acc': task_4_acc,
                                #'task_5_loss': task_5_loss,
                                #'task_6_loss': task_6_loss,
                                'loss': self.total_loss_31,
                                'train_op': self.train_optimizer_31,
                                'merged': self.merged_31,
                                'step': self.batch_31}
        tf.compat.v1.reset_default_graph()
        # init graph_32
        with self.graph_32.as_default():
            # Create a session
            config_32 = tf.compat.v1.ConfigProto()
            config_32.gpu_options.allow_growth = True
            config_32.allow_soft_placement = True
            config_32.log_device_placement = False
            self.sess_32 = tf.compat.v1.Session(graph = self.graph_32, config=config_32)
            self.merged_32 = tf.compat.v1.summary.merge_all()
            self.test_writer_32 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'), self.sess_32.graph)
            # load graph_31
            init_32 = tf.compat.v1.global_variables_initializer()
            self.sess_32.run(init_32)
            self.saver_32.restore(self.sess_32, self.BASE_DIR+'/stage_2_log/model_32_498.ckpt')
            
            self.train_ops_32 = {
               'points_256_sn_3': self.points_256_sn_3,
               'mask_256_sn_if_proposed': self.mask_256_sn_if_proposed,
               'mask_256_1_if_edge': self.mask_256_1_if_edge,
               'mask_256_1_if_proposed': self.mask_256_1_if_proposed,
               'mask_256_sn_if_edge': self.mask_256_sn_if_edge,
               'labels_type': self.labels_type,
               'labels_type_one_hot': self.labels_type_one_hot,
               'pred_open_curve_seg': self.pred_open_curve_seg,
               'pred_open_curve_cls': self.pred_open_curve_cls,
               #'pred_open_curve_reg_Line': self.pred_open_curve_reg_Line,
               #'pred_open_curve_reg_BSpline': self.pred_open_curve_reg_BSpline,
               #'end_points': self.end_points,
               'is_training_32': self.is_training_32,
               'total_seg_loss_32': self.seg_3_2_loss,
               'total_cls_loss_32': self.cls_3_2_loss,
               #'total_reg_loss_32': self.reg_3_2_loss,
               #'total_mat_diff_loss_32': self.mat_diff_loss,
               'total_loss_32': self.total_loss_32,
               'train_op': self.train_optimizer_32,
               'merged': self.merged_32,
               'step': self.batch_32}
        for epoch in range(1):
            self.log_string('**** TEST EPOCH %03d ****' % (epoch))
            self.eval_one_epoch_32(epoch)
            sys.stdout.flush()
            #eval_one_epoch(sess, ops, test_writer)
            #sys.stdout.flush()
            # Save the variables to disk.

    def eval_one_epoch_32(self, epoch):
        save_eval_subsection31 = True
        save_eval_subsection311 = True
        save_eval_subsection312 = True
        save_eval_subsection32 = True
        # Just one test_data matrix.
        is_training_31 = True
        is_training_32 = True
        #train_matrices_names_list = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/test_data/new_test/'), '70.mat')        
        load_data_start_time = time.time()
        # change this any time!
        loadpath = self.BASE_DIR + '/test_data/new_test/' + '70.mat'
        loadpath_1 = self.BASE_DIR + '/test_data/new_test/' + 'test_0.mat'
        train_data = sio.loadmat(loadpath)['Training_data']
        train_data_1 = sio.loadmat(loadpath_1)['Training_data']
        train_data = np.concatenate([train_data, train_data_1], axis = 0)
        #
        load_data_duration = time.time() - load_data_start_time
        self.log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        self.log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath_1,load_data_duration))

        num_data = train_data.shape[0]  # = 256
        num_batch = num_data // self.BATCH_SIZE   # 256 // 32 = 8
        total_loss_3_1 = 0.0
        total_edge_3_1_loss = 0.0
        total_edge_3_1_recall = 0.0
        total_edge_3_1_acc = 0.0
        total_corner_3_1_loss = 0.0
        total_corner_3_1_recall = 0.0
        total_corner_3_1_acc = 0.0
        total_reg_edge_3_1_loss = 0.0
        total_reg_corner_3_1_loss = 0.0
        
        total_loss_3_2 = 0.0
        total_seg_3_2_loss = 0.0
        total_cls_3_2_loss = 0.0
        #total_reg_3_2_loss = 0.0
        #total_mat_diff_3_2_loss = 0.0
        process_start_time = time.time()
        pred_labels_edge_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
        pred_labels_corner_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
        pred_reg_edge_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
        pred_reg_corner_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
        pred_open_curve_seg = np.zeros((num_data*256, SN, 2), np.float32)
        pred_open_curve_cls = np.zeros((num_data*256, 4), np.float32)
        
        #pred_open_curve_reg_Line = np.zeros((num_data*256, 6), np.float32)
        #pred_open_curve_reg_BSpline = np.zeros((num_data*256, 21), np.float32)

        #proposed_points_256_sn_3 = []
        #proposed_sample_points_idx = []
        #proposed_mask_256_1_if_proposed = []
        #edge_mask_256_1_if_edge = []
        #gt_idx_arrays = []
        for j in range(num_batch):
            # remember that num_batch will be 8
            begin_idx = j*self.BATCH_SIZE
            end_idx = (j+1)*self.BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]

            batch_inputs = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
            batch_labels_edge_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
            batch_labels_corner_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
            batch_regression_edge = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # each point normal estimation
            batch_regression_corner = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)

            batch_open_gt_256_sn_idx = np.zeros((self.BATCH_SIZE, 256, SN), np.int32)
            #batch_open_gt_mask = np.zeros((self.BATCH_SIZE, 256, SN), np.int32)
            batch_open_gt_type = np.zeros((self.BATCH_SIZE, 256, 1), np.int32)
            batch_open_gt_type_one_hot = np.zeros((self.BATCH_SIZE, 256, 4), np.int32)
            #batch_open_gt_res = np.zeros((self.BATCH_SIZE, 256, 6), np.float32)
            #batch_open_gt_sample_points = np.zeros((self.BATCH_SIZE,256, SN, 3), np.float32)
            #batch_open_gt_valid_mask = np.zeros((self.BATCH_SIZE,256, 1), np.int32)
            batch_open_gt_pair_idx = np.zeros((self.BATCH_SIZE,256, 2), np.int32)

            #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            for cnt in range(self.BATCH_SIZE):
                # cnt: 0 ... 31
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
                batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
                batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
                #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
                batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']

                ## check if these dimensions are correct
                batch_open_gt_256_sn_idx[cnt, ...] = tmp_data[0, 0]['open_gt_256_sn_idx']
                #batch_open_gt_sample_points[cnt, ...] = tmp_data[0, 0]['open_gt_sample_points']
                #batch_open_gt_mask[cnt, ...] = tmp_data[0, 0]['open_gt_mask']
                batch_open_gt_type[cnt, ...] = tmp_data[0, 0]['open_gt_type'][:, 0][:, np.newaxis]
                #batch_open_gt_res[cnt, ...] = tmp_data[0, 0]['open_gt_res']
                #batch_open_gt_valid_mask[cnt, ...] = tmp_data[0, 0]['open_gt_valid_mask']
                batch_open_gt_pair_idx[cnt, ...] = tmp_data[0, 0]['open_gt_pair_idx']
                batch_open_gt_type_one_hot[cnt, ...] = tmp_data[0, 0]['open_type_onehot']

                #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                #tmp_simmat = tmp_data['similar_matrix'][0,0]
                #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                #tmp_neg_simmat = 1 - tmp_simmat
                #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
            
            # run section 3.1.
            print("run graph3.1.")
            with self.graph_31.as_default():
                # batch waits.. feed this.
                feed_dict = {self.train_ops_31['pointclouds_pl']: batch_inputs,
                            self.train_ops_31['labels_edge_p']: batch_labels_edge_p,
                            self.train_ops_31['labels_corner_p']: batch_labels_corner_p,
                            #ops['labels_direction']: batch_labels_direction,
                            self.train_ops_31['reg_edge_p']: batch_regression_edge,
                            self.train_ops_31['reg_corner_p']: batch_regression_corner,
                            #ops['labels_type']: batch_labels_type,
                            #ops['simmat_pl']: batch_simmat_pl,
                            #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                            #ops['is_training_32']: is_training_32}
                            self.train_ops_31['is_training_31']: is_training_31}

                summary, step, _, \
                edge_3_1_loss_val, edge_3_1_recall_val, edge_3_1_acc_val, \
                corner_3_1_loss_val, corner_3_1_recall_val, corner_3_1_acc_val, \
                reg_edge_3_1_loss_val, reg_corner_3_1_loss_val, loss_3_1_val, \
                pred_labels_edge_p_val[begin_idx:end_idx,:,:], pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:] = \
                    self.sess_31.run([self.train_ops_31['merged'], self.train_ops_31['step'], self.train_ops_31['train_op'], \
                        self.train_ops_31['edge_3_1_loss'], self.train_ops_31['edge_3_1_recall'], self.train_ops_31['edge_3_1_acc'],\
                        self.train_ops_31['corner_3_1_loss'], self.train_ops_31['corner_3_1_recall'], self.train_ops_31['corner_3_1_acc'],\
                        self.train_ops_31['reg_edge_3_1_loss'], self.train_ops_31['reg_corner_3_1_loss'], self.train_ops_31['loss'], \
                        self.train_ops_31['pred_labels_edge_p'], self.train_ops_31['pred_labels_corner_p'], \
                        self.train_ops_31['pred_reg_edge_p'], self.train_ops_31['pred_reg_corner_p']],feed_dict=feed_dict)

                self.test_writer_31.add_summary(summary, step)
                total_loss_3_1 += loss_3_1_val
                total_edge_3_1_loss += edge_3_1_loss_val
                total_edge_3_1_acc += edge_3_1_acc_val
                total_edge_3_1_recall += edge_3_1_recall_val
                total_corner_3_1_loss += corner_3_1_loss_val
                total_corner_3_1_acc += corner_3_1_acc_val
                total_corner_3_1_recall += corner_3_1_recall_val
                total_reg_edge_3_1_loss += reg_edge_3_1_loss_val
                total_reg_corner_3_1_loss += reg_corner_3_1_loss_val

            # Save results for visualization: Sec.3.1.
            # make sure that this includes 'stage', 'batch_num', 'subsection'

            self.save_pred_results(is_save = save_eval_subsection31, \
                                    epoch = epoch, \
                                    batch_inputs = batch_inputs, \
                                    pred_labels_edge_p_val = pred_labels_edge_p_val[begin_idx:end_idx,:,:], \
                                    pred_reg_edge_p_val = pred_reg_edge_p_val[begin_idx:end_idx,:,:], \
                                    pred_labels_corner_p_val = pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                                    pred_reg_corner_p_val = pred_reg_corner_p_val[begin_idx:end_idx,:,:], \
                                    batch_num = j, \
                                    batch_labels_edge_p = batch_labels_edge_p, \
                                    batch_labels_corner_p = batch_labels_corner_p, \
                                    batch_regression_edge = batch_regression_edge, \
                                    batch_regression_corner = batch_regression_corner,\
                                    stage = self.STAGE, \
                                    subsection = 31,\
                                    train_or_eval = "eval")

            # here takes the post processing place.
            pred_labels_corner_p_val_softmax = ssp.softmax(pred_labels_corner_p_val[begin_idx:end_idx,:,:], axis = 2)
            pred_labels_edge_p_val_softmax = ssp.softmax(pred_labels_edge_p_val[begin_idx:end_idx,:,:], axis = 2)

            # apply correction vectors
            batch_inputs = self.apply_residual_vectors(batch_inputs, pred_labels_edge_p_val_softmax, pred_labels_corner_p_val_softmax, pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:])

            
            points_256_sn_3, idx_256_sn, idx_pair, mask_256_1_if_proposed, mask_256_sn_if_proposed = self.corner_pair_neighbor_search(batch_inputs, pred_labels_corner_p_val_softmax)
            
            # Save results for visualization: corner_pair_neighbor_search(), subsection = 311
            self.save_pred_results(is_save = save_eval_subsection311, \
                                    epoch = epoch, \
                                    batch_inputs = batch_inputs, \
                                    batch_num = j, \
                                    pred_labels_corner_p_val_softmax = pred_labels_corner_p_val_softmax, \
                                    points_256_sn_3 = points_256_sn_3, \
                                    idx_256_sn = idx_256_sn, \
                                    idx_pair = idx_pair, \
                                    mask_256_1_if_proposed = mask_256_1_if_proposed, \
                                    mask_256_sn_if_proposed = mask_256_sn_if_proposed, \
                                    stage = self.STAGE, \
                                    subsection = 311, \
                                    train_or_eval = "eval")


            # Note: Masks for loss!
            # cls - mask_256_1_if_proposed will let you take only the proposals.
            # seg - mask_256_sn_if_edge                    will create loss in the first place, 
            #     - mask_256_sn_if_proposed           will take valid candidates only and
            #     - mask_256_1_if_edge                 will take suitable corners only.
            #proposed_points_256_sn_3.append(points_256_sn_3)
            #proposed_sample_points_idx.append(idx_B_256_sn)
            #proposed_mask_256_1_if_proposed.append(mask_256_1_if_proposed)

            # combine with GT
            tp = np.dtype([
                #('points_256_sn_3', 'O'),
                ('idx_256_sn', 'O'),
                ('idx_pair', 'O'),
                ('mask_256_1_if_proposed', 'O'),
                #('mask_256_sn_if_proposed', 'O'),
                ('batch_inputs', 'O'),
                ('batch_open_gt_pair_idx', 'O'),
                ('batch_open_gt_256_sn_idx', 'O'),
                ('batch_open_gt_type', 'O'),
                ('batch_open_gt_type_one_hot', 'O'),
            ])
            corner_pair_label_generator_inputs = np.zeros((self.BATCH_SIZE, 1), dtype = tp)
            for tp_name in tp.names:
                for k in range(self.BATCH_SIZE):
                    save_this_piece = locals()[tp_name]
                    corner_pair_label_generator_inputs[k, 0][tp_name] = save_this_piece[k, ...]

            # this continues with corner_pair_label_generator
            mask_256_sn_if_edge, mask_256_1_if_edge, labels_type, labels_type_one_hot = self.corner_pair_label_generator_parallel(corner_pair_label_generator_inputs)            
            #corner_pair_label_generator_outputs = self.corner_pair_label_generator_parallel(corner_pair_label_generator_inputs)
            #mask_256_sn_if_edge = corner_pair_label_generator_outputs[:, ...]['mask_256_sn_if_edge'][()]
            #mask_256_1_if_edge = corner_pair_label_generator_outputs[:, ...]['mask_256_1_if_edge'][()]
            #labels_type = corner_pair_label_generator_outputs[:, ...]['labels_type'][()]
            #labels_type_one_hot = corner_pair_label_generator_outputs[:, ...]['labels_type_one_hot'][()]
            #gt_idx_array = corner_pair_label_generator_outputs[:, ...]['gt_idx_array'][()]

            # make sure that this includes 'stage', 'batch_num', 'subsection'

            self.save_pred_results(is_save = save_eval_subsection312, \
                                   epoch = epoch, \
                                   batch_inputs = batch_inputs, \
                                   points_256_sn_3 = points_256_sn_3.reshape(-1, 256, SN, 3), \
                                   idx_256_sn = idx_256_sn, \
                                   idx_pair = idx_pair, \
                                   mask_256_sn_if_proposed = mask_256_sn_if_proposed.reshape(-1, 256, SN), \
                                   mask_256_1_if_proposed = mask_256_1_if_proposed.reshape(-1, 256, 1), \
                                   batch_open_gt_pair_idx = batch_open_gt_pair_idx, \
                                   batch_open_gt_256_sn_idx = batch_open_gt_256_sn_idx, \
                                   batch_open_gt_type = batch_open_gt_type, \
                                   batch_open_gt_type_one_hot = batch_open_gt_type_one_hot, \
                                   batch_num = j, \
                                   mask_256_sn_if_edge = mask_256_sn_if_edge.reshape(-1, 256, SN), \
                                   mask_256_1_if_edge = mask_256_1_if_edge.reshape(-1, 256, 1) ,\
                                   labels_type = labels_type, \
                                   labels_type_one_hot = labels_type_one_hot, \
                                   stage = self.STAGE, \
                                   subsection = 312,\
                                   train_or_eval = "eval")


            # note that these per batch.  e.g. points_256_sn_3: (self.BATCH_SIZE, 256, SN, 3)
            points_256_sn_3 = points_256_sn_3.reshape(-1, SN, 3).astype(np.float32)
            mask_256_sn_if_edge = mask_256_sn_if_edge.reshape(-1, SN).astype(np.int32)
            mask_256_1_if_edge = mask_256_1_if_edge.reshape(-1, 1).astype(np.int32)
            mask_256_sn_if_proposed = mask_256_sn_if_proposed.reshape(-1, SN).astype(np.int32)
            mask_256_1_if_proposed = mask_256_1_if_proposed.reshape(-1, 1).astype(np.int32)
            labels_type = labels_type.reshape(-1).astype(np.int32)
            labels_type_one_hot = labels_type_one_hot.reshape(-1, 4).astype(np.int32)
            #edge_mask_256_1_if_edge.append(mask_256_1_if_edge)
            #gt_idx_arrays.append(gt_idx_array)
            
            print("run graph3.2.")
            with self.graph_32.as_default():
                feed_dict = {self.train_ops_32['points_256_sn_3']: points_256_sn_3,\
                                self.train_ops_32['labels_type']: labels_type, \
                                self.train_ops_32['labels_type_one_hot']: labels_type_one_hot, \
                                self.train_ops_32['mask_256_sn_if_edge']: mask_256_sn_if_edge, \
                                self.train_ops_32['mask_256_sn_if_proposed']: mask_256_sn_if_proposed, \
                                self.train_ops_32['mask_256_1_if_edge']: mask_256_1_if_edge, \
                                self.train_ops_32['mask_256_1_if_proposed']: mask_256_1_if_proposed, \
                                self.train_ops_32['is_training_32']: is_training_32}
                '''
                # session run
                summary, step, _, \
                seg_3_2_loss_val, \
                cls_3_2_loss_val, \
                reg_3_2_loss_val, \
                mat_diff_3_2_loss_val, \
                loss_3_2_val, \
                pred_open_curve_seg[begin_idx*256:end_idx*256, ...], \
                pred_open_curve_cls[begin_idx*256:end_idx*256, ...], \
                pred_open_curve_reg_Line[begin_idx*256:end_idx*256, ...], \
                pred_open_curve_reg_BSpline[begin_idx*256:end_idx*256, ...], \
                = self.sess_32.run([self.train_ops_32['merged'], \
                                    self.train_ops_32['step'], \
                                    self.train_ops_32['train_op'], \
                                    self.train_ops_32['total_seg_loss_32'], \
                                    self.train_ops_32['total_cls_loss_32'], \
                                    self.train_ops_32['total_reg_loss_32'], \
                                    self.train_ops_32['total_mat_diff_loss_32'], \
                                    self.train_ops_32['total_loss_32'], \
                                    self.train_ops_32['pred_open_curve_seg'], \
                                    self.train_ops_32['pred_open_curve_cls'], \
                                    self.train_ops_32['pred_open_curve_reg_Line'], \
                                    self.train_ops_32['pred_open_curve_reg_BSpline'], \
                                    ],feed_dict=feed_dict)
                '''

                # session run
                summary, step, _, \
                seg_3_2_loss_val, \
                cls_3_2_loss_val, \
                loss_3_2_val, \
                pred_open_curve_seg[begin_idx*256:end_idx*256, ...], \
                pred_open_curve_cls[begin_idx*256:end_idx*256, ...], \
                = self.sess_32.run([self.train_ops_32['merged'], \
                                    self.train_ops_32['step'], \
                                    self.train_ops_32['train_op'], \
                                    self.train_ops_32['total_seg_loss_32'], \
                                    self.train_ops_32['total_cls_loss_32'], \
                                    self.train_ops_32['total_loss_32'], \
                                    self.train_ops_32['pred_open_curve_seg'], \
                                    self.train_ops_32['pred_open_curve_cls'], \
                                    ],feed_dict=feed_dict)
                # loss
                total_seg_3_2_loss += seg_3_2_loss_val
                total_cls_3_2_loss += cls_3_2_loss_val
                #total_reg_3_2_loss += reg_3_2_loss_val
                #total_mat_diff_3_2_loss += mat_diff_3_2_loss_val
                total_loss_3_2 += loss_3_2_val
            # make sure that this includes 'stage', 'batch_num', 'subsection'

            self.save_pred_results(is_save = save_eval_subsection32, \
                        epoch = epoch, \
                        batch_inputs = batch_inputs,\
                        points_256_sn_3 = points_256_sn_3.reshape(-1, 256, SN, 3), \
                        labels_type = labels_type.reshape(-1, 256, 1), \
                        batch_num = j, \
                        pred_open_curve_seg = pred_open_curve_seg[begin_idx*256:end_idx*256, ...].reshape(-1, 256, SN, 2), \
                        pred_open_curve_cls = pred_open_curve_cls[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 4), \
                        #pred_open_curve_reg_Line = pred_open_curve_reg_Line[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 6), \
                        #pred_open_curve_reg_BSpline = pred_open_curve_reg_BSpline[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 21), \
                        stage = self.STAGE, \
                        subsection = 32,\
                        train_or_eval = "eval")
            tf.compat.v1.reset_default_graph()

        # total loss
        total_loss_3_1 = total_loss_3_1 * 1.0 / num_batch
        total_edge_3_1_loss = total_edge_3_1_loss * 1.0 / num_batch
        total_edge_3_1_acc = total_edge_3_1_acc * 1.0 / num_batch
        total_edge_3_1_recall = total_edge_3_1_recall * 1.0 / num_batch
        total_corner_3_1_loss = total_corner_3_1_loss * 1.0 / num_batch
        total_corner_3_1_acc = total_corner_3_1_acc * 1.0 / num_batch
        total_corner_3_1_recall = total_corner_3_1_recall * 1.0 / num_batch
        total_reg_edge_3_1_loss = total_reg_edge_3_1_loss * 1.0 / num_batch
        total_reg_corner_3_1_loss = total_reg_corner_3_1_loss * 1.0 / num_batch
        
        total_loss_3_2 = total_loss_3_2 * 1.0 / num_batch
        total_cls_3_2_loss = total_cls_3_2_loss * 1.0 / num_batch
        total_seg_3_2_loss = total_seg_3_2_loss * 1.0 / num_batch
        #total_reg_3_2_loss = total_reg_3_2_loss * 1.0 / num_batch
        #total_mat_diff_3_2_loss = total_mat_diff_3_2_loss * 1.0 / num_batch

        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        self.log_string('\t%s: step: %f total_loss_3_1: %f total_loss_3_2: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
        % (datetime.now(),step, total_loss_3_1, total_loss_3_2 ,process_duration,examples_per_sec,sec_per_batch))
        self.log_string('\t\tEvaluation Total_3_1 Mean_Loss: %f' % total_loss_3_1)
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
        self.log_string('\t\tEvaluation Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
        self.log_string('\t\tEvaluation Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)
        self.log_string('\t\tEvaluation Total_3_2 Mean_Loss: %f' % total_loss_3_2)
        self.log_string('\t\tEvaluation Seg_3_2 Mean_Loss: %f' % total_seg_3_2_loss)
        self.log_string('\t\tEvaluation Cls_3_2 Mean_Loss: %f' % total_cls_3_2_loss)
        #self.log_string('\t\tEvaluation Reg_3_2 Mean_Loss: %f' % total_reg_3_2_loss)
        #self.log_string('\t\tEvaluation Mat_Diff_3_2 Mean_Loss: %f' % total_mat_diff_3_2_loss)
        
    def train_graph_32(self):
        # init graph_31
        with self.graph_31.as_default():
            # Create a session
            config_31 = tf.compat.v1.ConfigProto()
            config_31.gpu_options.allow_growth = True
            config_31.allow_soft_placement = True
            config_31.log_device_placement = False
            self.sess_31 = tf.compat.v1.Session(graph = self.graph_31, config=config_31)
            self.merged_31 = tf.compat.v1.summary.merge_all()
            self.test_writer_31 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'), self.sess_31.graph)                            
            # load graph_31
            init_31 = tf.compat.v1.global_variables_initializer()
            self.sess_31.run(init_31)
            self.saver_31.restore(self.sess_31, self.BASE_DIR+'/stage_1_log/model_31_98.ckpt')
            # build train ops
            self.train_ops_31 = {'pointclouds_pl': self.pointclouds_pl,
                                'labels_edge_p': self.labels_edge_p,
                                'labels_corner_p': self.labels_corner_p,
                                #'labels_direction': labels_direction,
                                'reg_edge_p': self.reg_edge_p,
                                'reg_corner_p': self.reg_corner_p,
                                #'labels_type': labels_type,
                                #'simmat_pl': simmat_pl,
                                #'neg_simmat_pl': neg_simmat_pl,
                                'is_training_31': self.is_training_31,
                                'pred_labels_edge_p': self.pred_labels_edge_p,                   #  'pred_labels_edge_points'
                                'pred_labels_corner_p': self.pred_labels_corner_p,
                                #'pred_labels_direction': pred_labels_direction,
                                'pred_reg_edge_p': self.pred_reg_edge_p,
                                'pred_reg_corner_p': self.pred_reg_corner_p,
                                #'pred_labels_type': pred_labels_type,
                                #'pred_simmat': pred_simmat,
                                #'pred_conf': pred_conf_logits,
                                'edge_3_1_loss': self.edge_3_1_loss,
                                'edge_3_1_recall':self.edge_3_1_recall,
                                'edge_3_1_acc': self.edge_3_1_acc,
                                'corner_3_1_loss': self.corner_3_1_loss,
                                'corner_3_1_recall': self.corner_3_1_recall,
                                'corner_3_1_acc': self.corner_3_1_acc,
                                'reg_edge_3_1_loss': self.reg_edge_3_1_loss,
                                'reg_corner_3_1_loss': self.reg_corner_3_1_loss,
                                #'seg_3_2_acc': seg_3_2_acc,
                                #'task_2_2_loss': task_2_2_loss,
                                #'task_3_loss': task_3_loss,
                                #'task_4_loss': task_4_loss,
                                #'task_4_acc': task_4_acc,
                                #'task_5_loss': task_5_loss,
                                #'task_6_loss': task_6_loss,
                                'loss': self.total_loss_31,
                                'train_op': self.train_optimizer_31,
                                'merged': self.merged_31,
                                'step': self.batch_31}
        tf.compat.v1.reset_default_graph()
        # init graph_32
        with self.graph_32.as_default():
            # Create a session
            config_32 = tf.compat.v1.ConfigProto()
            config_32.gpu_options.allow_growth = True
            config_32.allow_soft_placement = True
            config_32.log_device_placement = False
            self.sess_32 = tf.compat.v1.Session(graph = self.graph_32, config=config_32)
            self.merged_32 = tf.compat.v1.summary.merge_all()
            self.test_writer_32 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'), self.sess_32.graph)
            # Init variables
            if self.RESUME == 0:
                init = tf.compat.v1.global_variables_initializer()
                self.sess_32.run(init)
            elif self.RESUME == 1:
                init = tf.compat.v1.global_variables_initializer()
                self.sess_32.run(init)
                self.saver_32.restore(self.sess_32, self.BASE_DIR+'/stage_2_log/model_32_.ckpt')
            
            self.train_ops_32 = {
               'points_256_sn_3': self.points_256_sn_3,
               'mask_256_sn_if_proposed': self.mask_256_sn_if_proposed,
               'mask_256_1_if_edge': self.mask_256_1_if_edge,
               'mask_256_1_if_proposed': self.mask_256_1_if_proposed,
               'mask_256_sn_if_edge': self.mask_256_sn_if_edge,
               'labels_type': self.labels_type,
               'labels_type_one_hot': self.labels_type_one_hot,
               'pred_open_curve_seg': self.pred_open_curve_seg,
               'pred_open_curve_cls': self.pred_open_curve_cls,
               #'pred_open_curve_reg_Line': self.pred_open_curve_reg_Line,
               #'pred_open_curve_reg_BSpline': self.pred_open_curve_reg_BSpline,
               #'end_points': self.end_points,
               'is_training_32': self.is_training_32,
               'total_seg_loss_32': self.seg_3_2_loss,
               'total_cls_loss_32': self.cls_3_2_loss,
               #'total_reg_loss_32': self.reg_3_2_loss,
               #'total_mat_diff_loss_32': self.mat_diff_loss,
               'total_loss_32': self.total_loss_32,
               'train_op': self.train_optimizer_32,
               'merged': self.merged_32,
               'step': self.batch_32}        
        tf.compat.v1.reset_default_graph()
        for epoch in range(self.MAX_EPOCH):
            self.log_string('**** TRAIN EPOCH %03d ****' % (epoch))
            self.train_one_epoch_32(epoch)
            sys.stdout.flush()
            self.log_string('**** TEST EPOCH %03d ****' % (epoch))
            self.eval_one_epoch_32(epoch)
            sys.stdout.flush()
            # Save the variables to disk.
            if epoch % 2 == 0:
                with self.graph_32.as_default():
                    model_ccc_path = "model_32_"+str(epoch)+".ckpt"
                    save_path = self.saver_32.save(self.sess_32, os.path.join(self.LOG_DIR, model_ccc_path))
                    self.log_string("Model saved in file: %s" % save_path)
                tf.compat.v1.reset_default_graph()
            
    def train_one_epoch_32(self, epoch):
        save_train_subsection31 = False
        save_train_subsection311 = False
        save_train_subsection312 = False
        save_train_subsection32 = False
        is_training_31 = True
        is_training_32 = True
        train_matrices_names_list = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/train_data/new_train/'), '*.mat')
        matrix_num = len(train_matrices_names_list)
        permutation = np.random.permutation(matrix_num)
        for i in range(len(permutation)//4):
            load_data_start_time = time.time()
            loadpath = self.BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4]]
            train_data = sio.loadmat(loadpath)['Training_data']
            load_data_duration = time.time() - load_data_start_time
            self.log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
            for j in range(3):
                temp_load_data_start_time = time.time()
                temp_loadpath = self.BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4+j+1]]
                temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
                temp_load_data_duration = time.time() - temp_load_data_start_time
                self.log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
                train_data = np.concatenate((train_data,temp_train_data),axis = 0)
                #print(train_data.shape)

            #push_eval(train_data, ops, sess, train_writer, is_training)
            # num_data = SN*4 = 256
            # num_batch = 256 // 32 = 8
            num_data = train_data.shape[0]  # = 256
            num_batch = num_data // self.BATCH_SIZE   # 256 // 32 = 8
            total_loss_3_1 = 0.0
            total_edge_3_1_loss = 0.0
            total_edge_3_1_recall = 0.0
            total_edge_3_1_acc = 0.0
            total_corner_3_1_loss = 0.0
            total_corner_3_1_recall = 0.0
            total_corner_3_1_acc = 0.0
            total_reg_edge_3_1_loss = 0.0
            total_reg_corner_3_1_loss = 0.0
            
            total_loss_3_2 = 0.0
            total_seg_3_2_loss = 0.0
            total_cls_3_2_loss = 0.0
            total_reg_3_2_loss = 0.0
            #total_mat_diff_3_2_loss = 0.0
    #        total_task_2_2_loss = 0.0
    #        total_task_3_loss = 0.0
    #        total_task_4_loss = 0.0
    #        total_task_4_acc = 0.0
    #        total_task_5_loss = 0.0
    #        total_task_6_loss = 0.0
            process_start_time = time.time()
            pred_labels_edge_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_labels_corner_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_reg_edge_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            pred_reg_corner_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            pred_open_curve_seg = np.zeros((num_data*256, SN, 2), np.float32)
            pred_open_curve_cls = np.zeros((num_data*256, 4), np.float32)

            #pred_open_curve_reg_Line = np.zeros((num_data*256, 6), np.float32)
            #pred_open_curve_reg_BSpline = np.zeros((num_data*256, 21), np.float32)

            #np.random.shuffle(train_data)
            for j in range(num_batch):
                # remember that num_batch will be 8
                begin_idx = j*self.BATCH_SIZE
                end_idx = (j+1)*self.BATCH_SIZE
                data_cells = train_data[begin_idx: end_idx,0]

                batch_inputs = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
                batch_labels_edge_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
                batch_labels_corner_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
                batch_regression_edge = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # each point normal estimation
                batch_regression_corner = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)

                batch_open_gt_256_sn_idx = np.zeros((self.BATCH_SIZE, 256, SN), np.int32)
                #batch_open_gt_mask = np.zeros((self.BATCH_SIZE, 256, SN), np.int32)
                batch_open_gt_type = np.zeros((self.BATCH_SIZE, 256, 1), np.int32)
                batch_open_gt_type_one_hot = np.zeros((self.BATCH_SIZE, 256, 4), np.int32)
                #batch_open_gt_res = np.zeros((self.BATCH_SIZE, 256, 6), np.float32)
                #batch_open_gt_sample_points = np.zeros((self.BATCH_SIZE,256, SN, 3), np.float32)
                #batch_open_gt_valid_mask = np.zeros((self.BATCH_SIZE,256, 1), np.int32)
                batch_open_gt_pair_idx = np.zeros((self.BATCH_SIZE,256, 2), np.int32)

                #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
                #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
                #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
                for cnt in range(self.BATCH_SIZE):
                    # cnt: 0 ... 31
                    tmp_data = data_cells[cnt]
                    batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
                    batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
                    batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
                    #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                    batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
                    batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']

                    ## check if these dimensions are correct
                    ## Debug !
                    batch_open_gt_256_sn_idx[cnt, ...] = tmp_data[0, 0]['open_gt_256_sn_idx']
                    #batch_open_gt_sample_points[cnt, ...] = tmp_data[0, 0]['open_gt_sample_points']
                    #batch_open_gt_mask[cnt, ...] = tmp_data[0, 0]['open_gt_mask']
                    batch_open_gt_type[cnt, ...] = tmp_data[0, 0]['open_gt_type'][:, 0][:, np.newaxis]
                    #batch_open_gt_res[cnt, ...] = tmp_data[0, 0]['open_gt_res']
                    #batch_open_gt_valid_mask[cnt, ...] = tmp_data[0, 0]['open_gt_valid_mask']
                    batch_open_gt_pair_idx[cnt, ...] = tmp_data[0, 0]['open_gt_pair_idx']
                    batch_open_gt_type_one_hot[cnt, ...] = tmp_data[0, 0]['open_type_onehot']

                    #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                    #tmp_simmat = tmp_data['similar_matrix'][0,0]
                    #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                    #tmp_neg_simmat = 1 - tmp_simmat
                    #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                    #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
                print("run graph3.1.")
                # run section 3.1.
                with self.graph_31.as_default():
                    # batch waits.. feed this.
                    feed_dict = {self.train_ops_31['pointclouds_pl']: batch_inputs,
                                self.train_ops_31['labels_edge_p']: batch_labels_edge_p,
                                self.train_ops_31['labels_corner_p']: batch_labels_corner_p,
                                #ops['labels_direction']: batch_labels_direction,
                                self.train_ops_31['reg_edge_p']: batch_regression_edge,
                                self.train_ops_31['reg_corner_p']: batch_regression_corner,
                                #ops['labels_type']: batch_labels_type,
                                #ops['simmat_pl']: batch_simmat_pl,
                                #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                                #ops['is_training_32']: is_training_32}
                                self.train_ops_31['is_training_31']: is_training_31}

                    summary, step, _, \
                    edge_3_1_loss_val, edge_3_1_recall_val, edge_3_1_acc_val, \
                    corner_3_1_loss_val, corner_3_1_recall_val, corner_3_1_acc_val, \
                    reg_edge_3_1_loss_val, reg_corner_3_1_loss_val, loss_3_1_val, \
                    pred_labels_edge_p_val[begin_idx:end_idx,:,:], pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                    pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:] = \
                        self.sess_31.run([self.train_ops_31['merged'], self.train_ops_31['step'], self.train_ops_31['train_op'], \
                            self.train_ops_31['edge_3_1_loss'], self.train_ops_31['edge_3_1_recall'], self.train_ops_31['edge_3_1_acc'],\
                            self.train_ops_31['corner_3_1_loss'], self.train_ops_31['corner_3_1_recall'], self.train_ops_31['corner_3_1_acc'],\
                            self.train_ops_31['reg_edge_3_1_loss'], self.train_ops_31['reg_corner_3_1_loss'], self.train_ops_31['loss'], \
                            self.train_ops_31['pred_labels_edge_p'], self.train_ops_31['pred_labels_corner_p'], \
                            self.train_ops_31['pred_reg_edge_p'], self.train_ops_31['pred_reg_corner_p']],feed_dict=feed_dict)

                    self.save_pred_results(is_save = save_train_subsection31, \
                                            epoch = epoch, \
                                            batch_inputs = batch_inputs, \
                                            pred_labels_edge_p_val = pred_labels_edge_p_val[begin_idx:end_idx,:,:], \
                                            pred_reg_edge_p_val = pred_reg_edge_p_val[begin_idx:end_idx,:,:], \
                                            pred_labels_corner_p_val = pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                                            pred_reg_corner_p_val = pred_reg_corner_p_val[begin_idx:end_idx,:,:], \
                                            batch_num = j, \
                                            batch_labels_edge_p = batch_labels_edge_p, \
                                            batch_labels_corner_p = batch_labels_corner_p, \
                                            batch_regression_edge = batch_regression_edge, \
                                            batch_regression_corner = batch_regression_corner,\
                                            stage = self.STAGE, \
                                            subsection = 31,\
                                            train_or_eval = "train")
                                                        
                    self.test_writer_31.add_summary(summary, step)
                    total_loss_3_1 += loss_3_1_val
                    total_edge_3_1_loss += edge_3_1_loss_val
                    total_edge_3_1_acc += edge_3_1_acc_val
                    total_edge_3_1_recall += edge_3_1_recall_val
                    total_corner_3_1_loss += corner_3_1_loss_val
                    total_corner_3_1_acc += corner_3_1_acc_val
                    total_corner_3_1_recall += corner_3_1_recall_val
                    total_reg_edge_3_1_loss += reg_edge_3_1_loss_val
                    total_reg_corner_3_1_loss += reg_corner_3_1_loss_val
                tf.compat.v1.reset_default_graph()
                # section 3.1.
                #np.save(os.path.join(self.BASE_DIR, 's31_batch_inputs.npy'), batch_inputs)
                #np.save(os.path.join(self.BASE_DIR, 's31_pred_labels_edge_p_val.npy'), pred_labels_edge_p_val[begin_idx:end_idx,:,:])
                #np.save(os.path.join(self.BASE_DIR, 's31_pred_labels_corner_p_val.npy'), pred_labels_corner_p_val[begin_idx:end_idx,:,:])
                #np.save(os.path.join(self.BASE_DIR, 's31_pred_reg_edge_p_val.npy'), pred_reg_edge_p_val[begin_idx:end_idx,:,:])
                #np.save(os.path.join(self.BASE_DIR, 's31_pred_reg_corner_p_val.npy'), pred_reg_corner_p_val[begin_idx:end_idx,:,:])

                # here takes the post processing place: corner_pair_neighbor_search(0)
                pred_labels_corner_p_val_softmax = ssp.softmax(pred_labels_corner_p_val[begin_idx:end_idx,:,:], axis = 2)
                pred_labels_edge_p_val_softmax = ssp.softmax(pred_labels_edge_p_val[begin_idx:end_idx,:,:], axis = 2)

                # apply correction vectors
                batch_inputs = self.apply_residual_vectors(batch_inputs, pred_labels_edge_p_val_softmax, pred_labels_corner_p_val_softmax, pred_reg_edge_p_val[begin_idx:end_idx,:,:], pred_reg_corner_p_val[begin_idx:end_idx,:,:])

                points_256_sn_3, idx_256_sn, idx_pair, mask_256_1_if_proposed, mask_256_sn_if_proposed = self.corner_pair_neighbor_search(batch_inputs, pred_labels_corner_p_val_softmax)
                # make sure that this includes 'stage', 'batch_num', 'subsection'
                
                self.save_pred_results(is_save = save_train_subsection311, \
                                        epoch = epoch, \
                                        batch_inputs = batch_inputs, \
                                        batch_num = j, \
                                        pred_labels_corner_p_val_softmax = pred_labels_corner_p_val_softmax, \
                                        points_256_sn_3 = points_256_sn_3, \
                                        idx_256_sn = idx_256_sn, \
                                        idx_pair = idx_pair, \
                                        mask_256_1_if_proposed = mask_256_1_if_proposed, \
                                        mask_256_sn_if_proposed = mask_256_sn_if_proposed, \
                                        stage = self.STAGE, \
                                        subsection = 311,\
                                        train_or_eval = "train")
                

                # combine with GT
                tp = np.dtype([
                    #('points_256_sn_3', 'O'),
                    ('idx_256_sn', 'O'),
                    ('idx_pair', 'O'),
                    ('mask_256_1_if_proposed', 'O'),
                    #('mask_256_sn_if_proposed', 'O'),
                    ('batch_inputs', 'O'),
                    ('batch_open_gt_pair_idx', 'O'),
                    ('batch_open_gt_256_sn_idx', 'O'),
                    ('batch_open_gt_type', 'O'),
                    ('batch_open_gt_type_one_hot', 'O'),
                ])
                corner_pair_label_generator_inputs = np.zeros((self.BATCH_SIZE, 1), dtype = tp)
                for tp_name in tp.names:
                    for k in range(self.BATCH_SIZE):
                        save_this_piece = locals()[tp_name]
                        corner_pair_label_generator_inputs[k, 0][tp_name] = save_this_piece[k, ...]
                
                # this continues with corner_pair_label_generator
                #print("run corner_pair_label_generator_parallel()")
                mask_256_sn_if_edge, mask_256_1_if_edge, labels_type, labels_type_one_hot = self.corner_pair_label_generator_parallel(corner_pair_label_generator_inputs)

                #mask_256_sn_if_edge = corner_pair_label_generator_outputs[:, ...]['mask_256_sn_if_edge'][()]
                #mask_256_1_if_edge = corner_pair_label_generator_outputs[:, ...]['mask_256_1_if_edge'][()]
                #labels_type = corner_pair_label_generator_outputs[:, ...]['labels_type'][()]
                #labels_type_one_hot = corner_pair_label_generator_outputs[:, ...]['labels_type_one_hot'][()]
                #gt_idx_array = corner_pair_label_generator_outputs[:, ...]['gt_idx_array'][()]

                
                # Note: Masks at loss!
                # cls - mask_256_1_if_proposed will let you take only the proposals.
                # seg - mask_256_sn_if_edge                    will create loss in the first place, 
                #     - mask_256_sn_if_proposed           will take valid candidates only and
                #     - mask_256_1_if_edge                 will take suitable corners only.
                
                # note that these per batch.  e.g. points_256_sn_3: (self.BATCH_SIZE, 256, SN, 3)
                points_256_sn_3 = points_256_sn_3.reshape(-1, SN, 3).astype(np.float32)
                mask_256_sn_if_edge = mask_256_sn_if_edge.reshape(-1, SN).astype(np.int32)
                mask_256_1_if_edge = mask_256_1_if_edge.reshape(-1, 1).astype(np.int32)
                mask_256_sn_if_proposed = mask_256_sn_if_proposed.reshape(-1, SN).astype(np.int32)
                mask_256_1_if_proposed = mask_256_1_if_proposed.reshape(-1, 1).astype(np.int32)
                labels_type = labels_type.reshape(-1).astype(np.int32)
                labels_type_one_hot = labels_type_one_hot.reshape(-1, 4).astype(np.int32)

                # make sure that this includes 'stage', 'batch_num', 'subsection'
                
                self.save_pred_results(is_save = save_train_subsection312, \
                                    epoch = epoch, \
                                    batch_inputs = batch_inputs, \
                                    points_256_sn_3 = points_256_sn_3, \
                                    idx_256_sn = idx_256_sn, \
                                    idx_pair = idx_pair, \
                                    mask_256_sn_if_proposed = mask_256_sn_if_proposed, \
                                    mask_256_1_if_proposed = mask_256_1_if_proposed, \
                                    batch_open_gt_pair_idx = batch_open_gt_pair_idx, \
                                    batch_open_gt_256_sn_idx = batch_open_gt_256_sn_idx, \
                                    batch_open_gt_type = batch_open_gt_type, \
                                    batch_open_gt_type_one_hot = batch_open_gt_type_one_hot, \
                                    batch_num = j, \
                                    mask_256_sn_if_edge = mask_256_sn_if_edge, \
                                    mask_256_1_if_edge = mask_256_1_if_edge ,\
                                    labels_type = labels_type, \
                                    labels_type_one_hot = labels_type_one_hot, \
                                    stage = self.STAGE, \
                                    subsection = 312,\
                                    train_or_eval = "train")
                

                print("run graph3.2.")
                with self.graph_32.as_default():
                    feed_dict = {self.train_ops_32['points_256_sn_3']: points_256_sn_3,\
                                 self.train_ops_32['labels_type']: labels_type, \
                                 self.train_ops_32['labels_type_one_hot']: labels_type_one_hot, \
                                 self.train_ops_32['mask_256_sn_if_edge']: mask_256_sn_if_edge, \
                                 self.train_ops_32['mask_256_sn_if_proposed']: mask_256_sn_if_proposed, \
                                 self.train_ops_32['mask_256_1_if_edge']: mask_256_1_if_edge, \
                                 self.train_ops_32['mask_256_1_if_proposed']: mask_256_1_if_proposed, \
                                 self.train_ops_32['is_training_32']: is_training_32}
                    # session run
                    summary, step, _, \
                    seg_3_2_loss_val, \
                    cls_3_2_loss_val, \
                    loss_3_2_val, \
                    pred_open_curve_seg[begin_idx*256:end_idx*256, ...], \
                    pred_open_curve_cls[begin_idx*256:end_idx*256, ...], \
                    = self.sess_32.run([self.train_ops_32['merged'], \
                                        self.train_ops_32['step'], \
                                        self.train_ops_32['train_op'], \
                                        self.train_ops_32['total_seg_loss_32'], \
                                        self.train_ops_32['total_cls_loss_32'], \
                                        self.train_ops_32['total_loss_32'], \
                                        self.train_ops_32['pred_open_curve_seg'], \
                                        self.train_ops_32['pred_open_curve_cls'], \
                                        ],feed_dict=feed_dict)                                 
                    '''
                    # session run
                    summary, step, _, \
                    seg_3_2_loss_val, \
                    cls_3_2_loss_val, \
                    reg_3_2_loss_val, \
                    mat_diff_3_2_loss_val, \
                    loss_3_2_val, \
                    pred_open_curve_seg[begin_idx*256:end_idx*256, ...], \
                    pred_open_curve_cls[begin_idx*256:end_idx*256, ...], \
                    pred_open_curve_reg_Line[begin_idx*256:end_idx*256, ...], \
                    pred_open_curve_reg_BSpline[begin_idx*256:end_idx*256, ...], \
                    = self.sess_32.run([self.train_ops_32['merged'], \
                                        self.train_ops_32['step'], \
                                        self.train_ops_32['train_op'], \
                                        self.train_ops_32['total_seg_loss_32'], \
                                        self.train_ops_32['total_cls_loss_32'], \
                                        self.train_ops_32['total_reg_loss_32'], \
                                        self.train_ops_32['total_mat_diff_loss_32'], \
                                        self.train_ops_32['total_loss_32'], \
                                        self.train_ops_32['pred_open_curve_seg'], \
                                        self.train_ops_32['pred_open_curve_cls'], \
                                        self.train_ops_32['pred_open_curve_reg_Line'], \
                                        self.train_ops_32['pred_open_curve_reg_BSpline'], \
                                        ],feed_dict=feed_dict)
                    '''
                    # loss
                    total_seg_3_2_loss += seg_3_2_loss_val
                    total_cls_3_2_loss += cls_3_2_loss_val
                    #total_reg_3_2_loss += reg_3_2_loss_val
                    #total_mat_diff_3_2_loss += mat_diff_3_2_loss_val
                    total_loss_3_2 += loss_3_2_val
                    
                    self.save_pred_results(is_save = save_train_subsection32, \
                                epoch = epoch, \
                                batch_inputs = batch_inputs,\
                                points_256_sn_3 = points_256_sn_3.reshape(-1, 256, SN, 3), \
                                labels_type = labels_type.reshape(-1, 256, 1), \
                                batch_num = j, \
                                pred_open_curve_seg = pred_open_curve_seg[begin_idx*256:end_idx*256, ...].reshape(-1, 256, SN, 2), \
                                pred_open_curve_cls = pred_open_curve_cls[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 4), \
                                #pred_open_curve_reg_Line = pred_open_curve_reg_Line[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 6), \
                                #pred_open_curve_reg_BSpline = pred_open_curve_reg_BSpline[begin_idx*256:end_idx*256, ...].reshape(-1, 256, 21), \
                                stage = self.STAGE, \
                                subsection = 32,\
                                train_or_eval = "train")
                    
                tf.compat.v1.reset_default_graph()
            # total loss
            total_loss_3_1 = total_loss_3_1 * 1.0 / num_batch
            total_edge_3_1_loss = total_edge_3_1_loss * 1.0 / num_batch
            total_edge_3_1_acc = total_edge_3_1_acc * 1.0 / num_batch
            total_edge_3_1_recall = total_edge_3_1_recall * 1.0 / num_batch
            total_corner_3_1_loss = total_corner_3_1_loss * 1.0 / num_batch
            total_corner_3_1_acc = total_corner_3_1_acc * 1.0 / num_batch
            total_corner_3_1_recall = total_corner_3_1_recall * 1.0 / num_batch
            total_reg_edge_3_1_loss = total_reg_edge_3_1_loss * 1.0 / num_batch
            total_reg_corner_3_1_loss = total_reg_corner_3_1_loss * 1.0 / num_batch
            
            total_loss_3_2 = total_loss_3_2 * 1.0 / num_batch
            total_cls_3_2_loss = total_cls_3_2_loss * 1.0 / num_batch
            total_seg_3_2_loss = total_seg_3_2_loss * 1.0 / num_batch
            #total_reg_3_2_loss = total_reg_3_2_loss * 1.0 / num_batch
            #total_mat_diff_3_2_loss = total_mat_diff_3_2_loss * 1.0 / num_batch

            
                
            process_duration = time.time() - process_start_time
            examples_per_sec = num_data/process_duration
            sec_per_batch = process_duration/num_batch
            self.log_string('\t%s: step: %f total_loss_3_1: %f total_loss_3_2: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
            % (datetime.now(),step, total_loss_3_1, total_loss_3_2 ,process_duration,examples_per_sec,sec_per_batch))
            self.log_string('\t\tTraining Total_3_1 Mean_Loss: %f' % total_loss_3_1)
            self.log_string('\t\tTraining Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
            self.log_string('\t\tTraining Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
            self.log_string('\t\tTraining Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
            self.log_string('\t\tTraining Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
            self.log_string('\t\tTraining Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
            self.log_string('\t\tTraining Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
            self.log_string('\t\tTraining Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
            self.log_string('\t\tTraining Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)            
            self.log_string('\t\tTraining Total_3_2 Mean_Loss: %f' % total_loss_3_2)
            self.log_string('\t\tTraining Seg_3_2 Mean_Loss: %f' % total_seg_3_2_loss)
            self.log_string('\t\tTraining Cls_3_2 Mean_Loss: %f' % total_cls_3_2_loss)
            #self.log_string('\t\tTraining Reg_3_2 Mean_Loss: %f' % total_reg_3_2_loss)
            #self.log_string('\t\tTraining Mat_Diff_3_2 Mean_Loss: %f' % total_mat_diff_3_2_loss)
                        
    def train_graph_31(self):
        
        with self.graph_31.as_default():
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            self.sess_31 = tf.compat.v1.Session(graph = self.graph_31, config=config)
            # Add summary writers
            self.merged = tf.compat.v1.summary.merge_all()
            self.train_writer_31 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'train'), self.sess_31.graph)
            self.test_writer_31 = tf.compat.v1.summary.FileWriter(os.path.join(self.LOG_DIR, 'test'), self.sess_31.graph)        

            # Init variables
            if self.RESUME == 0:
                init = tf.compat.v1.global_variables_initializer()
                self.sess_31.run(init)
            elif self.RESUME == 1:
                init = tf.compat.v1.global_variables_initializer()
                self.sess_31.run(init)
                self.saver_31.restore(self.sess_31, self.BASE_DIR+'/stage_1_log/model_31_98.ckpt')

            self.train_ops_31 = {'pointclouds_pl': self.pointclouds_pl,
                'labels_edge_p': self.labels_edge_p,
                'labels_corner_p': self.labels_corner_p,
                #'labels_direction': labels_direction,
                'reg_edge_p': self.reg_edge_p,
                'reg_corner_p': self.reg_corner_p,
                #'labels_type': labels_type,
                #'simmat_pl': simmat_pl,
                #'neg_simmat_pl': neg_simmat_pl,
                'is_training_31': self.is_training_31,
                'pred_labels_edge_p': self.pred_labels_edge_p,                   #  'pred_labels_edge_points'
                'pred_labels_corner_p': self.pred_labels_corner_p, 
                #'pred_labels_direction': pred_labels_direction,
                'pred_reg_edge_p': self.pred_reg_edge_p,   
                'pred_reg_corner_p': self.pred_reg_corner_p,
                #'pred_labels_type': pred_labels_type,
                #'pred_simmat': pred_simmat,
                #'pred_conf': pred_conf_logits,
                'edge_3_1_loss': self.edge_3_1_loss,
                'edge_3_1_recall':self.edge_3_1_recall,
                'edge_3_1_acc': self.edge_3_1_acc,               
                'corner_3_1_loss': self.corner_3_1_loss,
                'corner_3_1_recall': self.corner_3_1_recall,
                'corner_3_1_acc': self.corner_3_1_acc, 
                'reg_edge_3_1_loss': self.reg_edge_3_1_loss,
                'reg_corner_3_1_loss': self.reg_corner_3_1_loss,
                #'seg_3_2_acc': seg_3_2_acc,
                #'task_2_2_loss': task_2_2_loss,
                #'task_3_loss': task_3_loss,
                #'task_4_loss': task_4_loss,
                #'task_4_acc': task_4_acc,
                #'task_5_loss': task_5_loss,
                #'task_6_loss': task_6_loss,
                'loss': self.total_loss_31,
                'train_op': self.train_optimizer_31,
                'merged': self.merged,
                'step': self.batch_31}

            for epoch in range(self.MAX_EPOCH):                
                self.log_string('**** TRAIN EPOCH %03d ****' % (epoch))
                self.train_one_epoch_31()
                sys.stdout.flush()
                self.log_string('**** TEST EPOCH %03d ****' % (epoch))
                self.eval_one_epoch_31(epoch)
                sys.stdout.flush()
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model_31_"+str(epoch)+".ckpt"
                    save_path = self.saver_31.save(self.sess_31, os.path.join(self.LOG_DIR, model_ccc_path))
                    self.log_string("Model saved in file: %s" % save_path)

    def train_one_epoch_31(self):
        save_train_subsection31 = False
        is_training_31 = True
        train_matrices_names_list = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/train_data/new_train/'), '*.mat')
        matrix_num = len(train_matrices_names_list)
        permutation = np.random.permutation(matrix_num)
        #permutation = np.array([0, 2, 3, 1])
        for i in range(len(permutation)//4):
            load_data_start_time = time.time()
            loadpath = self.BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4]]
            train_data = sio.loadmat(loadpath)['Training_data']
            load_data_duration = time.time() - load_data_start_time
            self.log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
            for j in range(3):
                temp_load_data_start_time = time.time()
                temp_loadpath = self.BASE_DIR + '/train_data/new_train/'+train_matrices_names_list[permutation[i*4+j+1]]
                temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
                temp_load_data_duration = time.time() - temp_load_data_start_time
                self.log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
                train_data = np.concatenate((train_data,temp_train_data),axis = 0)
                #print(train_data.shape)

            #push_eval(train_data, ops, sess, train_writer, is_training)
            # num_data = SN*4 = 256
            # num_batch = 256 // 32 = 8
            num_data = train_data.shape[0]  # = 256
            num_batch = num_data // self.BATCH_SIZE   # 256 // 32 = 8
            total_loss = 0.0
            total_edge_3_1_loss = 0.0
            total_edge_3_1_recall = 0.0
            total_edge_3_1_acc = 0.0
            total_corner_3_1_loss = 0.0
            total_corner_3_1_recall = 0.0
            total_corner_3_1_acc = 0.0
            total_reg_edge_3_1_loss = 0.0
            total_reg_corner_3_1_loss = 0.0
            #total_seg_3_2_loss = 0.0
            #total_cls_3_2_loss = 0.0
            #total_reg_3_2_loss = 0.0
    #        total_task_2_2_loss = 0.0
    #        total_task_3_loss = 0.0
    #        total_task_4_loss = 0.0
    #        total_task_4_acc = 0.0
    #        total_task_5_loss = 0.0
    #        total_task_6_loss = 0.0
            process_start_time = time.time()
            pred_labels_edge_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_labels_corner_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_reg_edge_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            pred_reg_corner_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            #np.random.shuffle(train_data)
            for j in range(num_batch):
                # remember that num_batch will be 8
                begin_idx = j*self.BATCH_SIZE
                end_idx = (j+1)*self.BATCH_SIZE
                data_cells = train_data[begin_idx: end_idx,0]

                batch_inputs = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
                batch_labels_edge_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
                batch_labels_corner_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
                batch_regression_edge = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # each point normal estimation
                batch_regression_corner = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)

                #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
                #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
                #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
                for cnt in range(self.BATCH_SIZE):
                    # cnt: 0 ... 31
                    tmp_data = data_cells[cnt]
                    batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
                    batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
                    batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
                    #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                    batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
                    batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']

                    #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                    #tmp_simmat = tmp_data['similar_matrix'][0,0]
                    #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                    #tmp_neg_simmat = 1 - tmp_simmat
                    #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                    #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
                feed_dict = {self.train_ops_31['pointclouds_pl']: batch_inputs,
                            self.train_ops_31['labels_edge_p']: batch_labels_edge_p,
                            self.train_ops_31['labels_corner_p']: batch_labels_corner_p,
                            #ops['labels_direction']: batch_labels_direction,
                            self.train_ops_31['reg_edge_p']: batch_regression_edge,
                            self.train_ops_31['reg_corner_p']: batch_regression_corner,
                            #ops['labels_type']: batch_labels_type,
                            #ops['simmat_pl']: batch_simmat_pl,
                            #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                            #ops['is_training_32']: is_training_32}
                            self.train_ops_31['is_training_31']: is_training_31}
                            
                    
                        
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
                    self.sess_31.run([self.train_ops_31['merged'], self.train_ops_31['step'], self.train_ops_31['train_op'], \
                        self.train_ops_31['edge_3_1_loss'], self.train_ops_31['edge_3_1_recall'], self.train_ops_31['edge_3_1_acc'],\
                        self.train_ops_31['corner_3_1_loss'], self.train_ops_31['corner_3_1_recall'], self.train_ops_31['corner_3_1_acc'],\
                        self.train_ops_31['reg_edge_3_1_loss'], self.train_ops_31['reg_corner_3_1_loss'], self.train_ops_31['loss'], \
                        self.train_ops_31['pred_labels_edge_p'], self.train_ops_31['pred_labels_corner_p'], \
                        self.train_ops_31['pred_reg_edge_p'], self.train_ops_31['pred_reg_corner_p']],feed_dict=feed_dict)                 

                self.train_writer_31.add_summary(summary, step)
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
            self.log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
            % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
            self.log_string('\t\tTraining Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
            self.log_string('\t\tTraining Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
            self.log_string('\t\tTraining Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
            self.log_string('\t\tTraining Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
            self.log_string('\t\tTraining Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
            self.log_string('\t\tTraining Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
            self.log_string('\t\tTraining Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
            self.log_string('\t\tTraining Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)      

    def eval_one_epoch_31(self, epoch):
        save_eval_subsection31 = True
        is_training_31 = True
        # make sure that there are 4*n matrices
        #train_matrices_names_list = fnmatch.filter(os.listdir('/raid/home/hyovin.kwak/PIE-NET/main/test_data/new_test/'), '*.mat')
        #matrix_num = len(train_matrices_names_list)
        #permutation = np.random.permutation(matrix_num)
        load_data_start_time = time.time()
        loadpath = self.BASE_DIR + '/test_data/new_test/' + '70.mat'
        loadpath_1 = self.BASE_DIR + '/test_data/new_test/' + 'test_0.mat'
        train_data = sio.loadmat(loadpath)['Training_data']
        train_data_1 = sio.loadmat(loadpath_1)['Training_data']
        train_data = np.concatenate([train_data, train_data_1], axis = 0)
        #
        load_data_duration = time.time() - load_data_start_time
        self.log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))

        #push_eval(train_data, ops, sess, train_writer, is_training)
        # num_data = SN*4 = 256
        # num_batch = 256 // 32 = 8
        num_data = train_data.shape[0]  # = 256
        num_batch = num_data // self.BATCH_SIZE   # 256 // 32 = 8
        total_loss = 0.0
        total_edge_3_1_loss = 0.0
        total_edge_3_1_recall = 0.0
        total_edge_3_1_acc = 0.0
        total_corner_3_1_loss = 0.0
        total_corner_3_1_recall = 0.0
        total_corner_3_1_acc = 0.0
        total_reg_edge_3_1_loss = 0.0
        total_reg_corner_3_1_loss = 0.0
        #total_seg_3_2_loss = 0.0
        #total_cls_3_2_loss = 0.0
        #total_reg_3_2_loss = 0.0
#        total_task_2_2_loss = 0.0
#        total_task_3_loss = 0.0
#        total_task_4_loss = 0.0
#        total_task_4_acc = 0.0
#        total_task_5_loss = 0.0
#        total_task_6_loss = 0.0
        process_start_time = time.time()
        pred_labels_edge_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
        pred_labels_corner_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
        pred_reg_edge_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
        pred_reg_corner_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
        #np.random.shuffle(train_data)
        for j in range(num_batch):
            # remember that num_batch will be 8
            begin_idx = j*self.BATCH_SIZE
            end_idx = (j+1)*self.BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]

            batch_inputs = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # input point clouds  # original code  =6
            batch_labels_edge_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
            batch_labels_corner_p = np.zeros((self.BATCH_SIZE,self.NUM_POINT),np.int32)  # edge point label 0/1
            batch_regression_edge = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)  # each point normal estimation
            batch_regression_corner = np.zeros((self.BATCH_SIZE,self.NUM_POINT,3),np.float32)

            #batch_labels_type = np.zeros((BATCH_SIZE,NUM_POINT),np.int32)
            #batch_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            #batch_neg_simmat_pl = np.zeros((BATCH_SIZE, NUM_POINT, NUM_POINT), np.float32)
            for cnt in range(self.BATCH_SIZE):
                # cnt: 0 ... 31
                tmp_data = data_cells[cnt]
                batch_inputs[cnt,:,:] = tmp_data[0,0]['down_sample_point']
                batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data[0,0]['edge_points_label'])
                batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data[0,0]['corner_points_label'])
                #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                batch_regression_edge[cnt,:,:] = tmp_data[0,0]['edge_points_residual_vector']
                batch_regression_corner[cnt,:,:] = tmp_data[0,0]['corner_points_residual_vector']

                #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                #tmp_simmat = tmp_data['similar_matrix'][0,0]
                #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                #tmp_neg_simmat = 1 - tmp_simmat
                #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
            feed_dict = {self.train_ops_31['pointclouds_pl']: batch_inputs,
                        self.train_ops_31['labels_edge_p']: batch_labels_edge_p,
                        self.train_ops_31['labels_corner_p']: batch_labels_corner_p,
                        #ops['labels_direction']: batch_labels_direction,
                        self.train_ops_31['reg_edge_p']: batch_regression_edge,
                        self.train_ops_31['reg_corner_p']: batch_regression_corner,
                        #ops['labels_type']: batch_labels_type,
                        #ops['simmat_pl']: batch_simmat_pl,
                        #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                        #ops['is_training_32']: is_training_32}
                        self.train_ops_31['is_training_31']: is_training_31}
                        
                    
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
                self.sess_31.run([self.train_ops_31['merged'], self.train_ops_31['step'], self.train_ops_31['train_op'], \
                    self.train_ops_31['edge_3_1_loss'], self.train_ops_31['edge_3_1_recall'], self.train_ops_31['edge_3_1_acc'],\
                    self.train_ops_31['corner_3_1_loss'], self.train_ops_31['corner_3_1_recall'], self.train_ops_31['corner_3_1_acc'],\
                    self.train_ops_31['reg_edge_3_1_loss'], self.train_ops_31['reg_corner_3_1_loss'], self.train_ops_31['loss'], \
                    self.train_ops_31['pred_labels_edge_p'], self.train_ops_31['pred_labels_corner_p'], \
                    self.train_ops_31['pred_reg_edge_p'], self.train_ops_31['pred_reg_corner_p']],feed_dict=feed_dict)                 
            
            # make sure that this includes 'stage', 'batch_num', 'subsection'
            
            self.save_pred_results(is_save = save_eval_subsection31, \
                                    epoch = epoch, \
                                    batch_inputs = batch_inputs, \
                                    pred_labels_edge_p_val = pred_labels_edge_p_val[begin_idx:end_idx,:,:], \
                                    pred_reg_edge_p_val = pred_reg_edge_p_val[begin_idx:end_idx,:,:], \
                                    pred_labels_corner_p_val = pred_labels_corner_p_val[begin_idx:end_idx,:,:], \
                                    pred_reg_corner_p_val = pred_reg_corner_p_val[begin_idx:end_idx,:,:], \
                                    batch_num = j, \
                                    batch_labels_edge_p = batch_labels_edge_p, \
                                    batch_labels_corner_p = batch_labels_corner_p, \
                                    batch_regression_edge = batch_regression_edge, \
                                    batch_regression_corner = batch_regression_corner,\
                                    stage = self.STAGE, \
                                    subsection = 31,\
                                    train_or_eval = "eval")
            
            
            self.train_writer_31.add_summary(summary, step)
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

        process_duration = time.time() - process_start_time
        examples_per_sec = num_data/process_duration
        sec_per_batch = process_duration/num_batch
        self.log_string('\t%s: step: %f loss: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
        % (datetime.now(),step,total_loss,process_duration,examples_per_sec,sec_per_batch))
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
        self.log_string('\t\tEvaluation Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
        self.log_string('\t\tEvaluation Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
        self.log_string('\t\tEvaluation Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
        self.log_string('\t\tEvaluation Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)
        
    def save_pred_results(self, is_save, epoch, **kwargs):
        # Note:
        # make sure that **kwargs contains the following names:
        # 'data_mat_name', 'stage', 'batch_num', 'subsection'
        if is_save:
            dir = os.path.join(self.BASE_DIR, "test_results")
            if not os.path.exists(dir): 
                os.mkdir(dir)
            epoch = 'epoch_' +str(epoch) + '_'
            stage = 'stage_' + str(kwargs['stage'])+'_'
            batch = 'batch_' + str(kwargs['batch_num'])+'_'
            train_or_eval = str(kwargs['train_or_eval'])+'_'
            subsection = 'subsection_' + str(kwargs['subsection'])
            filename = epoch+stage+batch+train_or_eval+subsection
            np.save(os.path.join(dir, filename + '.npy'), kwargs)

    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def average_gradients(self, tower_grads):
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

    def get_learning_rate(self, batch):
        learning_rate = tf.compat.v1.train.exponential_decay(
                            self.BASE_LEARNING_RATE,  # Base learning rate.
                            batch * self.BATCH_SIZE,  # Current index into the dataset.
                            self.DECAY_STEP,          # Decay step.
                            self.DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate        

    def get_learning_rate_stage_2(self, batch,base_learning_rate):
        learning_rate = tf.compat.v1.train.exponential_decay(
                            base_learning_rate,  # Base learning rate.
                            batch * self.BATCH_SIZE,  # Current index into the dataset.
                            self.DECAY_STEP,          # Decay step.
                            self.DECAY_RATE,          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.compat.v1.train.exponential_decay(
                        self.BN_INIT_DECAY,
                        batch*self.BATCH_SIZE,
                        self.BN_DECAY_DECAY_STEP,
                        self.BN_DECAY_DECAY_RATE,
                        staircase=True)
        bn_decay = tf.minimum(self.BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def apply_residual_vectors_per_batch(self, inputs_concat_apply_res_vectors_reshaped):

        edge_threshold = 0.85
        corner_threshold = 0.99
        inputs_concat_apply_res_vectors = inputs_concat_apply_res_vectors_reshaped.reshape(-1, 13)
        batch_inputs = inputs_concat_apply_res_vectors[..., 0:3]
        edge_softmax = inputs_concat_apply_res_vectors[..., 3:5]
        corner_softmax = inputs_concat_apply_res_vectors[..., 5:7]
        pred_res_vec_edge = inputs_concat_apply_res_vectors[..., 7:10]
        pred_res_vec_corner = inputs_concat_apply_res_vectors[..., 10:13]
        
        # get the complement idx
        edge_idx = np.where(edge_softmax[..., 1] > edge_threshold)[0]
        corner_idx = np.where(corner_softmax[..., 1] > corner_threshold)[0]
        #edge_not_corner_idx = edge_idx[np.where(~np.in1d(edge_idx, corner_idx))[0]]
        edge_not_corner_idx = edge_idx[~np.in1d(edge_idx, corner_idx)]

        batch_inputs[edge_not_corner_idx, ...] = batch_inputs[edge_not_corner_idx, ...] + pred_res_vec_edge[edge_not_corner_idx, ...]
        batch_inputs[corner_idx, ...] = batch_inputs[corner_idx, ...] + pred_res_vec_corner[corner_idx, ...]
        tp = np.dtype([
            ('batch_inputs', 'O'),
        ])
        batch_inputs_memory = np.zeros((1, 1), dtype = tp)
        for tp_name in tp.names:
            save_this_piece = locals()[tp_name]
            batch_inputs_memory[tp_name][0, 0] = save_this_piece

        return batch_inputs

    def apply_residual_vectors(self, batch_inputs, edge_softmax, corner_softmax, pred_res_vec_edge, pred_res_vec_corner):
        ''' Note:
        batch_inputs: (np.float32) (self.BATCH_SIZE, 8096, 3)
        edge_softmax: (np.float32) (self.BATCH_SIZE, 8096, 2)
        corner_softmax: (np.float32) (self.BATCH_SIZE, 8096, 2)
        pred_res_vec_edge: (np.float32) (self.BATCH_SIZE, 8096, 3)
        pred_res_vec_corner: (np.float32) (self.BATCH_SIZE, 8096, 3)
        '''
        batch_inputs = np.apply_along_axis(self.apply_residual_vectors_per_batch, 1, np.concatenate([batch_inputs, edge_softmax, corner_softmax, pred_res_vec_edge, pred_res_vec_corner], axis = 2).reshape(self.BATCH_SIZE, -1))
        return batch_inputs
    '''
    def idx_pair_generator(self, cloud_corner_reshaped):
        # generates idx_pairs, suitable for multiprocessing.
        # cloud_corner_reshaped: (self.BATCH_SIZE, 8096, 5)
        
        
        # Input:
        # cloud_corner_reshaped[..., 0:3]: (np.float32) (self.BATCH_SIZE, 8096, 3) contains point_cloud
        # cloud_corner_reshaped[..., 3:5]: (np.float32) (self.BATCH_SIZE, 8096, 2) contains pred_corner(softmax outputs of corner points)

        # returns
        # idx_pairs and corresponding pointclouds, labels and valid masks

        cloud_corner_concat = cloud_corner_reshaped.reshape(-1, 5)
        points_cloud = cloud_corner_concat[..., 0:3]
        pred_corner = cloud_corner_concat[..., 3:5]

        threshold = 0.95
        # idx_pair with -1: invalid
        idx_pair = np.zeros((256, 2), dtype = np.int32) -1
        idx = np.where(pred_corner[..., 1] > threshold)[0]

        # check if we have enough corner points from softmax - THIS WAS A BAD IDEA
        while idx.shape[0] < 2:
            threshold = threshold - 0.001
            idx = np.where(pred_corner[..., 1] > threshold)[0]
        point_cloud_corners = points_cloud[idx, ...]
        
        # K-means clustering: possibly there may be 23 corner points!
        # + non maximum supression?
        # init
        best_distance_std = np.Inf
        best_k = -1
        early_stopped = False

        # search for best_k of clusters k = 2, 3, 4, ... 23
        for k in range(4, 24):
            if not early_stopped:
                best_distance_mean = np.Inf
                _, centroids_idx = graphier_FPS(point_cloud_corners, k, np.arange(point_cloud_corners.shape[0]))
                centroids_xyz = point_cloud_corners[centroids_idx, ...]
                # distances to centroids
                distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
                mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
                while mean_mindistance_to_centroids < best_distance_mean:
                    best_distance_mean = mean_mindistance_to_centroids
                    classes = np.argmin(distances, axis = 0) # each one belongs to the nearest cluster
                    centroids_xyz = np.stack([np.mean(point_cloud_corners[np.where(classes == i)[0], ...], axis = 0) for i in range(k)], axis = 0)
                    distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
                    mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
                distance_std = np.std(np.min(distances, 0))
                if distance_std < best_distance_std:
                    best_distance_std = distance_std
                    best_k = k
                    #print(best_distance_mean)
                    #print(k)
                    if best_distance_std < 0.0003 and best_distance_mean < 0.000001: # Early stopping
                        print("best_distance_std: ", best_distance_std)
                        print("best_distance_mean: ", best_distance_mean)
                        print("best_k: ", best_k)
                        #print(best_distance)
                        early_stopped = True
                        break
            
            # if distance absolutely small: break! This has to be debugged.


        # best_k found. non maximum supression!
        best_distance_mean = np.Inf
        _, centroids_idx = graphier_FPS(point_cloud_corners, best_k, np.arange(point_cloud_corners.shape[0]))
        centroids_xyz = point_cloud_corners[centroids_idx, ...]
        # distances to centroids
        distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
        mean_mindistance_to_centroids = np.mean(np.min(distances, 0))
        while mean_mindistance_to_centroids < best_distance_mean:
            best_distance_mean = mean_mindistance_to_centroids
            classes = np.argmin(distances, axis = 0) # each one belongs to the nearest cluster
            centroids_xyz = np.stack([np.mean(point_cloud_corners[np.where(classes == i)[0], ...], axis = 0) for i in range(best_k)], axis = 0)
            distances = np.sum((np.expand_dims(centroids_xyz, axis = 1) - point_cloud_corners)**2, axis = 2)
            mean_mindistance_to_centroids = np.mean(np.min(distances, 0))

        # non maximum suppression(nms) in every cluster: choose the best corner points per cluster.
        
        idx_in_class = idx[np.where(classes == k)[0]] # per class k = 0, 1, 2, ...
        
        np.argmin(np.sum((centroids_xyz[k, ...] - point_cloud_corners[np.where(classes == k)[0]])**2, axis = 1), axis = 0)
        argmax_idx = np.argmax(pred_corner[idx_in_class, 1])
        idx_in_class[argmax_idx]
        
        class_centroids_idx = np.array([idx[np.where(classes == k)[0]][np.argmin(np.sum((centroids_xyz[k, ...] - point_cloud_corners[np.where(classes == k)[0]])**2, axis = 1), axis = 0)] for k in range(best_k)])
        

        if class_centroids_idx.shape[0] > 1:
            idx = np.sort(class_centroids_idx)
            idx_r = np.repeat(idx, idx.shape[0])
            idx_b = np.tile(idx, [idx.shape[0]])
            two_col = np.stack([idx_r, idx_b], 1)
            idx_pair[0:(idx.shape[0]*(idx.shape[0]-1)//2), ...] = two_col[two_col[:,0] < two_col[:, 1]]
            #idx_pair.append()

        # idx_pair ready!
        rest_num = 256

        # first increase the precision to float64, 
        # otherwise it may go wrong when it comes to finding points within radius, 
        # where it may also include the two corner points at the end.
        points_cloud = np.float64(points_cloud)

        # find neighbors
        xyz1 = points_cloud[idx_pair[:, 0], :]
        xyz2 = points_cloud[idx_pair[:, 1], :]
        ball_center = np.mean(np.stack([xyz1, xyz2], axis = 0), axis = 0)

        distance_from_ball_center = np.sqrt(np.sum(((np.expand_dims(ball_center, 1) - np.expand_dims(points_cloud,axis=0))**2), axis = 2))
        # for safety: - 0.00001
        r = np.sqrt(np.sum((xyz1 - xyz2)**2, axis = 1)) / 2.0  - 0.00001
        within_range = distance_from_ball_center < np.multiply(np.ones_like(distance_from_ball_center), np.expand_dims(r, axis = 1))
        corner_pair_num = class_centroids_idx.shape[0]*(class_centroids_idx.shape[0] - 1) // 2

        # memory arrays
        points_256_sn_3 = np.zeros((256, 64, 3), dtype = np.float32)
        idx_256_sn = np.zeros((256, 64), dtype = np.int32) -1
        valid_mask_256_sn = np.zeros((256, 64), dtype = np.int32) -1
            
        # if there are more than 256 pairs, just take first 256.
        if corner_pair_num > 256 : corner_pair_num = 256
        rest_num = 256 - corner_pair_num

        if corner_pair_num > 0:
            for per_corner in range(corner_pair_num):
                # make sure that corner points(end points) are not within the range.
                assert within_range[per_corner, ...][idx_pair[per_corner][0]] == False
                assert within_range[per_corner, ...][idx_pair[per_corner][1]] == False
                #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
                #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
                candidnate_num = np.where(within_range[per_corner, :])[0].shape[0]
                #
                # raise error or debug when within_range.shape[0] = 0
                # or if within_range.shape[0] = 3?
                if 63 <= candidnate_num:
                    middle_indicies = np.where(within_range[per_corner, :])[0]
                    np.random.shuffle(middle_indicies)
                    idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), np.squeeze(middle_indicies[:62]), np.expand_dims(idx_pair[per_corner][-1], axis = 0)], axis = 0)
                    idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                    #idx_256_sn.append(np.expand_dims(idx_nums, axis = 0))
                    valid_mask_256_sn[per_corner, :] = np.expand_dims(np.ones_like(idx_nums), axis = 0)

                elif 0 < candidnate_num < 63:
                    n = candidnate_num
                    dummy_num = 64 - 1 - n
                    middle_indicies = np.where(within_range[per_corner, :])[0]
                    #if candidnate_num == 1: 
                        #middle_indicies = np.expand_dims(middle_indicies, axis = 0)
                    idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), middle_indicies, np.repeat(idx_pair[per_corner][-1], dummy_num)], axis = 0)
                    idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                    valid_mask_256_sn[per_corner, :] = np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0)
                elif candidnate_num == 0:
                    dummy_num = 63
                    idx_nums = np.concatenate([np.expand_dims(idx_pair[per_corner][0], axis = 0), np.repeat(idx_pair[per_corner][-1], dummy_num)], axis = 0)
                    idx_256_sn[per_corner, :] = np.expand_dims(idx_nums, axis = 0)
                    valid_mask_256_sn[per_corner, :] = np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0)

            if rest_num > 0: 
                idx_256_sn[corner_pair_num:, :] = np.zeros((rest_num, 64), dtype = np.int64)
                valid_mask_256_sn[corner_pair_num:, :] = np.zeros((rest_num, 64), dtype = np.int64)
        
            # return idx_256_sn
            # return valid_mask_256_sn
            #idx_B_256_sn.append(np.concatenate(idx_256_sn, axis = 0))
            #valid_mask_256_sn_per_batch.append(np.concatenate(valid_mask_256_sn_per_corner, axis = 0))
            points_256_sn_3 = np.stack([points_cloud[idx_256_sn[k], ...] for k in range(256)], axis = 0)
            
        valid_mask = np.ones(256, dtype = np.int8)
        valid_mask[-rest_num:] = 0
        valid_mask = np.expand_dims(valid_mask, axis = 1)
        tp = np.dtype([
            ('points_256_sn_3', 'O'),
            ('idx_256_sn', 'O'),
            ('idx_pair', 'O'),
            ('valid_mask', 'O'),
            ('valid_mask_256_sn', 'O')
        ])
        corner_neighbors = np.zeros((1, 1), dtype = tp)
        for tp_name in tp.names:
            save_this_piece = locals()[tp_name]
            corner_neighbors[tp_name][0, 0] = save_this_piece
        return corner_neighbors
    '''

    def corner_pair_neighbor_search(self, points_cloud, pred_corner):
        """ builds a sphere between two predicted corner points, sample 62 points within the spehere.
        Note: We name this 62 points 'neighbors' + 2 points at the both ends = 64 points

        Args:
            points_cloud ([tf.float32], batch_size, 8096, 3): original point cloud
            pred_corner ([tf.float32], batch_size, 8096, 2): predicted corner points

        Returns:
            points_B_256_sn_3: float32, sampled points in the neighborhood, 64 points per corner pair.
            idx_B_256_sn: their indices in points_cloud
            idx_pair: and their pair indices
            mask_if_proposed (list, (N, 256, 1)): valid mask based on proposed corners. 
                                                    (Note: elements of valid_mask_all_pair are 1 when they are just "proposed" anyway, not subject to if it is real in gt.)
            valid_mask_256_sn_per_batch (list, (N, 256, 64)): valid mask based on the candidates num.
                                                            (Note: elements of valid_mask_256_sn_per_batch are 1 when they are just "proposed".)
            pairs_available: list of booleans, whethere at least one pair is available per batch
        """    
        
        # we do this in multiprocessing.
        '''
        start = time.process_time()
        idx_pair_generator_results_B_5 = np.apply_along_axis(self.idx_pair_generator, 1, np.concatenate([points_cloud, pred_corner], axis = 2).reshape(self.BATCH_SIZE, -1)) # starts with != -1 if there exists.
        print("1 CPU:", time.process_time() - start)
        start = time.process_time()
        idx_pair_generator_results_B_5 = parallel_apply_along_axis(self.idx_pair_generator, 1, np.concatenate([points_cloud, pred_corner], axis = 2).reshape(self.BATCH_SIZE, -1)) # starts with != -1 if there exists.
        print("4 CPUs: ", time.process_time() - start)
        '''
        idx_pair_generator_results_B_5 = parallel_apply_along_axis(idx_pair_generator, 1, np.concatenate([points_cloud, pred_corner], axis = 2).reshape(self.BATCH_SIZE, -1)) # starts with != -1 if there exists.
        #idx_pair_generator_results_B_5 = np.apply_along_axis(idx_pair_generator, 1, np.concatenate([points_cloud, pred_corner], axis = 2).reshape(self.BATCH_SIZE, -1)) # starts with != -1 if there exists.
        points_256_sn_3 = np.stack([idx_pair_generator_results_B_5['points_256_sn_3'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        idx_256_sn = np.stack([idx_pair_generator_results_B_5['idx_256_sn'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        idx_pair = np.stack([idx_pair_generator_results_B_5['idx_pair'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        mask_256_1_if_proposed = np.stack([idx_pair_generator_results_B_5['valid_mask'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        mask_256_sn_if_proposed = np.stack([idx_pair_generator_results_B_5['valid_mask_256_sn'][i, 0, 0] for i in range(self.BATCH_SIZE)])

        return points_256_sn_3, idx_256_sn, idx_pair, mask_256_1_if_proposed, mask_256_sn_if_proposed
        
        # per batch sample the points
        idx_B_256_sn = []
        points_B_256_sn_3 = [] # (BATCH_SIZE, 256, 64, 3)
        valid_mask_256_sn_per_batch = []


        for per_batch in range(self.BATCH_SIZE):
            rest_num = 256

            if pairs_available[per_batch]:

                # first increase the precision to float64, 
                # otherwise it may go wrong when it comes to finding points within radius, 
                # where it may also include the two corner points at the end.
                points_cloud = np.float64(points_cloud)

                # find neighbors
                xyz1 = points_cloud[per_batch, ...][idx_pair[per_batch][:, 0], :]
                xyz2 = points_cloud[per_batch, ...][idx_pair[per_batch][:, 1], :]
                
                ball_center = np.mean(np.stack([xyz1, xyz2], axis = 0), axis = 0)
                distance_from_ball_center = np.sqrt(np.sum(((np.expand_dims(ball_center, 1) - np.expand_dims(points_cloud[per_batch],axis=0))**2), axis = 2))
                r = np.sqrt(np.sum((xyz1 - xyz2)**2, axis = 1)) / 2.0 # radius
                within_range = distance_from_ball_center < np.multiply(np.ones_like(distance_from_ball_center), np.expand_dims(r, axis = 1))

                # per corner pair within this batch, subsample the indicies
                idx_256_sn = []
                valid_mask_256_sn_per_corner = []
                corner_pair_num = within_range.shape[0]
                
                # if there are more than 256 pairs, just take first 256.
                if corner_pair_num > 256 : corner_pair_num = 256
                rest_num = 256 - corner_pair_num

                for per_corner in range(corner_pair_num):
                    # make sure that corner points(end points) are not within the range.
                    assert within_range[per_corner, ...][idx_pair[per_batch][per_corner]][0] == False
                    assert within_range[per_corner, ...][idx_pair[per_batch][per_corner]][1] == False
                    #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
                    #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
                    candidnate_num = np.where(within_range[per_corner, :])[0].shape[0]
                    #
                    # raise error or debug when within_range.shape[0] = 0
                    # or if within_range.shape[0] = 3?
                    if 63 <= candidnate_num:
                        middle_indicies = np.where(within_range[per_corner, :])[0]
                        np.random.shuffle(middle_indicies)
                        idx_nums = np.concatenate([np.expand_dims(idx_pair[per_batch][per_corner][0], axis = 0), np.squeeze(middle_indicies[:62]), np.expand_dims(idx_pair[per_batch][per_corner][-1], axis = 0)], axis = 0)
                        idx_256_sn.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_sn_per_corner.append(np.expand_dims(np.ones_like(idx_nums), axis = 0))

                    elif 0 < candidnate_num < 63:
                        n = candidnate_num
                        dummy_num = 64 - 1 - n
                        middle_indicies = np.where(within_range[per_corner, :])[0]
                        #if candidnate_num == 1: 
                            #middle_indicies = np.expand_dims(middle_indicies, axis = 0)
                        idx_nums = np.concatenate([np.expand_dims(idx_pair[per_batch][per_corner][0], axis = 0), middle_indicies, np.repeat(idx_pair[per_batch][per_corner][-1], dummy_num)], axis = 0)
                        idx_256_sn.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_sn_per_corner.append(np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0))
                    
                    elif candidnate_num == 0:
                        dummy_num = 63
                        idx_nums = np.concatenate([np.expand_dims(idx_pair[per_batch][per_corner][0], axis = 0), np.repeat(idx_pair[per_batch][per_corner][-1], dummy_num)], axis = 0)
                        idx_256_sn.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_sn_per_corner.append(np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0))
                if rest_num > 0: 
                    idx_256_sn.append(np.zeros((rest_num, 64), dtype = np.int64))
                    valid_mask_256_sn_per_corner.append(np.zeros((rest_num, 64), dtype = np.int64))
                idx_B_256_sn.append(np.concatenate(idx_256_sn, axis = 0))
                valid_mask_256_sn_per_batch.append(np.concatenate(valid_mask_256_sn_per_corner, axis = 0))
                points_B_256_sn_3.append(points_cloud[per_batch][idx_B_256_sn[per_batch], ...])
            else:
                idx_B_256_sn.append(np.zeros((rest_num, 64), dtype = np.int64))
                valid_mask_256_sn_per_batch.append(np.zeros((rest_num, 64), dtype = np.int64))
                points_B_256_sn_3.append(np.zeros((rest_num, 64, 3), dtype = np.float32))
            valid_mask = np.ones(256, dtype = np.int8)
            valid_mask[-rest_num:] = 0
            valid_mask = np.expand_dims(valid_mask, axis = 1)
            valid_mask_all_pair.append(valid_mask)

        return points_B_256_sn_3, idx_B_256_sn, idx_pair, valid_mask_all_pair, valid_mask_256_sn_per_batch

    def corner_pair_label_generator(self, corner_pair_label_generator_inputs):

        # from corner_pair_neighbor_search
        #points_256_sn_3 = corner_pair_label_generator_inputs[0, ...]['points_256_sn_3'][()]
        idx_256_sn = corner_pair_label_generator_inputs[0, ...]['idx_256_sn'][()]
        idx_pair = corner_pair_label_generator_inputs[0, ...]['idx_pair'][()]
        mask_256_1_if_proposed = corner_pair_label_generator_inputs[0, ...]['mask_256_1_if_proposed'][()]
        #mask_256_sn_if_proposed = corner_pair_label_generator_inputs[0, ...]['mask_256_sn_if_proposed'][()]

        # from batch
        batch_inputs = corner_pair_label_generator_inputs[0, ...]['batch_inputs'][()]
        batch_open_gt_pair_idx = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_pair_idx'][()]
        batch_open_gt_256_sn_idx = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_256_sn_idx'][()]
        batch_open_gt_type = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_type'][()]
        batch_open_gt_type_one_hot = corner_pair_label_generator_inputs[0, ...]['batch_open_gt_type_one_hot'][()]
        
        batch_num = 1
        mask_256_sn_if_edge = np.zeros((256, SN), dtype = np.int32) # output should be (batch_num, 256, SN, 2)
        mask_256_1_if_edge = np.zeros((256, 1), dtype = np.int16)
        sample_corner_type = np.zeros((256), dtype = np.int16)
        sample_corner_type_one_hot = np.zeros((256, 4), dtype = np.int16)
        gt_idx_array = np.zeros((256), dtype = np.int16)
        dist_threshold = 0.001
        n_values = 4

        k = 0
        while k < 256 and mask_256_1_if_proposed[k][0] == 1:

            # per curve pair k in one batch
            if (idx_pair[k] == batch_open_gt_pair_idx).all(axis = 1).any():
                # indices match exactly
                gt_idx = np.where((idx_pair[k] == batch_open_gt_pair_idx).all(axis = 1))[0]
                # gt_idx = np.where(batch_open_gt_pair_idx[i][k].numpy() in my_mat['open_gt_pair_idx'][0, 0])[0][0]
                # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
                gt_idx_array[k] = gt_idx
                sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
                # one-hot encoding
                sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
                mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
                mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
                mask_256_1_if_edge[k, 0] = 1
                k = k+1
                continue
            elif (np.flip(idx_pair[k]) == batch_open_gt_pair_idx).all(axis = 1).any():
                gt_idx = np.where((np.flip(idx_pair[k]) == batch_open_gt_pair_idx).all(axis = 1))[0]
                sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
                gt_idx_array[k] = gt_idx
                sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
                # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
                mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
                # update here labels
                mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
                mask_256_1_if_edge[k, 0] = 1
                k = k+1
                continue

            # not exact match, but see if there is one nearby.
            # calculate distances NN.
            distance = np.sqrt(np.sum((batch_inputs[idx_pair[k], :] - batch_inputs[batch_open_gt_pair_idx, :])**2, axis = 2))
            if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
                gt_idx_array[k] = gt_idx
                sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
                mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[:, :][gt_idx])
                mask[0], mask[-1] = True, True
                mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
                mask_256_1_if_edge[k, 0] = 1
                k = k+1
                continue

            distance = np.sqrt(np.sum((batch_inputs[np.flip(idx_pair[k]), :] - batch_inputs[batch_open_gt_pair_idx, :])**2, axis = 2))
            if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                sample_corner_type[k] = batch_open_gt_type[gt_idx][0]
                gt_idx_array[k] = gt_idx
                sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[gt_idx][0]]
                mask = np.in1d(idx_256_sn[k], batch_open_gt_256_sn_idx[gt_idx])
                mask[0], mask[-1] = True, True
                mask_256_sn_if_edge[k, :] = mask.astype(np.int32)
                mask_256_1_if_edge[k, 0] = 1
                k = k+1
                continue
            
            # Null class
            #assert batch_open_gt_type[gt_idx][0] == 0
            sample_corner_type_one_hot[k, ...] = np.eye(n_values, dtype = np.int16)[0]
            k = k+1

        mask_256_sn_if_edge = mask_256_sn_if_edge.reshape((-1, SN))
        mask_256_1_if_edge = mask_256_1_if_edge.reshape(-1, 1)
        labels_type = sample_corner_type.reshape(-1)
        labels_type_one_hot = sample_corner_type_one_hot.reshape(-1, 4)
        gt_idx_array = gt_idx_array
        tp = np.dtype([
            ('mask_256_sn_if_edge', 'O'),
            ('mask_256_1_if_edge', 'O'),
            ('labels_type', 'O'),
            ('labels_type_one_hot', 'O'),
            ('gt_idx_array', 'O')
        ])
        corner_pair_label_generator_outputs = np.zeros((1, 1), dtype = tp)
        for tp_name in tp.names:
            save_this_piece = locals()[tp_name]
            corner_pair_label_generator_outputs[0, 0][tp_name] = save_this_piece
        
        return corner_pair_label_generator_outputs
        #return mask_256_sn_if_edge, mask_256_1_if_edge, labels_type, labels_type_one_hot, gt_idx_array

    def corner_pair_label_generator_parallel(self, corner_pair_label_generator_inputs):
        #start = time.process_time()
        corner_pair_label_generator_outputs = parallel_apply_along_axis(corner_pair_label_generator, 1, corner_pair_label_generator_inputs) # starts with != -1 if there exists.
        #corner_pair_label_generator_outputs = np.apply_along_axis(corner_pair_label_generator, 1, corner_pair_label_generator_inputs) # starts with != -1 if there exists.

        #idx_pair_generator_results_B_5 = parallel_apply_along_axis(self.idx_pair_generator, 1, np.concatenate([points_cloud, pred_corner], axis = 2).reshape(self.BATCH_SIZE, -1)) # starts with != -1 if there exists.
        mask_256_sn_if_edge = np.stack([corner_pair_label_generator_outputs['mask_256_sn_if_edge'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        mask_256_1_if_edge = np.stack([corner_pair_label_generator_outputs['mask_256_1_if_edge'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        labels_type = np.stack([corner_pair_label_generator_outputs['labels_type'][i, 0, 0] for i in range(self.BATCH_SIZE)])
        labels_type_one_hot = np.stack([corner_pair_label_generator_outputs['labels_type_one_hot'][i, 0, 0] for i in range(self.BATCH_SIZE)])        

        return mask_256_sn_if_edge, mask_256_1_if_edge, labels_type, labels_type_one_hot
        #print("4 CPUs: ", time.process_time() - start)
        #start = time.process_time()
        #corner_pair_label_generator_outputs = np.apply_along_axis(corner_pair_label_generator, 1, corner_pair_label_generator_inputs) # starts with != -1 if there exists.
        #print("4 CPUs: ", time.process_time() - start)
        return corner_pair_label_generator_outputs

    '''
    def corner_pair_label_generator(self, \
                                    sample_256_sn_idx, \
                                    sample_pair_idx, \
                                    sample_pair_valid_mask, \
                                    sample_corner_pairs_available, \
                                    points_cloud, \
                                    batch_open_gt_pair_idx, \
                                    batch_open_gt_256_sn_idx, \
                                    batch_open_gt_type, \
                                    batch_open_gt_type_one_hot):
                                # add more gt_*

        batch_num = len(sample_corner_pairs_available)
        sample_valid_mask_256_sn_labels_for_loss = np.zeros((batch_num, 256, SN), dtype = np.int32) # output should be (batch_num, 256, SN, 2)
        sample_valid_mask_pair_labels_for_loss = np.zeros((batch_num, 256, 1), dtype = np.int16)
        sample_corner_type = np.zeros((batch_num, 256), dtype = np.int16)
        sample_corner_type_one_hot = np.zeros((batch_num, 256, 4), dtype = np.int16)
        gt_idx_array = np.zeros((batch_num, 256), dtype = np.int16)
        points_cloud_np = points_cloud
        dist_threshold = 1.5
        n_values = 4

        # sample_valid_mask_256_sn
        for i in range(batch_num):

            # per batch
            if sample_corner_pairs_available[i]:
                sample_valid_mask_pair_i_copy = sample_pair_valid_mask[i].copy()
                k = 0
                
                while k < 256 and sample_valid_mask_pair_i_copy[k][0] == 1:

                    # per curve pair k in one batch
                    if (sample_pair_idx[i][k] == batch_open_gt_pair_idx[i, :, :]).all(axis = 1).any():
                        # indices match exactly
                        gt_idx = np.where((sample_pair_idx[i][k] == batch_open_gt_pair_idx[i, :, :]).all(axis = 1))[0]
                        # gt_idx = np.where(batch_open_gt_pair_idx[i][k].numpy() in my_mat['open_gt_pair_idx'][0, 0])[0][0]
                        # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
                        gt_idx_array[i, k] = gt_idx
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        # one-hot encoding
                        sample_corner_type_one_hot[i, k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[i, gt_idx][0]]
                        

                        mask = np.in1d(sample_256_sn_idx[i][k], batch_open_gt_256_sn_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_sn_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue
                    elif (np.flip(sample_pair_idx[i][k]) == batch_open_gt_pair_idx[i, :, :]).all(axis = 1).any():
                        gt_idx = np.where((np.flip(sample_pair_idx[i][k]) == batch_open_gt_pair_idx[i, :, :]).all(axis = 1))[0]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        gt_idx_array[i, k] = gt_idx
                        sample_corner_type_one_hot[i, k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[i, gt_idx][0]]
                        # my_mat[0, 0]['open_gt_256_sn_idx'][gt_idx, :]
                        mask = np.in1d(sample_256_sn_idx[i][k], batch_open_gt_256_sn_idx[i, :, :][gt_idx])
                        # update here labels
                        sample_valid_mask_256_sn_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue

                    # not exact match, but see if there is one nearby.
                    # calculate distances NN.
                    distance = np.sqrt(np.sum((points_cloud_np[i][sample_pair_idx[i][k], :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        gt_idx_array[i, k] = gt_idx
                        sample_corner_type_one_hot[i, k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[i, gt_idx][0]]
                        mask = np.in1d(sample_256_sn_idx[i][k], batch_open_gt_256_sn_idx[i, :, :][gt_idx])
                        mask[0], mask[-1] = True, True
                        sample_valid_mask_256_sn_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue

                    distance = np.sqrt(np.sum((points_cloud_np[i][np.flip(sample_pair_idx[i][k]), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        gt_idx_array[i, k] = gt_idx
                        sample_corner_type_one_hot[i, k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[i, gt_idx][0]]
                        mask = np.in1d(sample_256_sn_idx[i][k], batch_open_gt_256_sn_idx[i, :, :][gt_idx])
                        mask[0], mask[-1] = True, True
                        sample_valid_mask_256_sn_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue
                    
                    # Null class
                    assert batch_open_gt_type[i, gt_idx][0] == 0
                    sample_corner_type_one_hot[i, k, ...] = np.eye(n_values, dtype = np.int16)[batch_open_gt_type[i, gt_idx][0]]
                    k = k+1
        
        return sample_valid_mask_256_sn_labels_for_loss.reshape((-1, SN)), sample_valid_mask_pair_labels_for_loss.reshape(-1, 1), sample_corner_type.reshape(-1), sample_corner_type_one_hot.reshape(-1, 4), gt_idx_array
        '''