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
parser.add_argument('--max_epoch', type=int, default=101, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--stage',type=int,default=2,help='network stage')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
NUM_GPUS = FLAGS.num_gpus
BATCH_SIZE = FLAGS.batch_size
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE // NUM_GPUS

NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
STAGE = FLAGS.stage

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
if STAGE == 1:
    LOG_DIR = FLAGS.stage_1_log_dir
else:
    LOG_DIR = FLAGS.stage_2_log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
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
            if STAGE==1:

                print('stage_1')
                # remember that reg_{edge, corner}_p is label.
                pointclouds_pl, labels_edge_p, labels_corner_p, reg_edge_p, reg_corner_p = MODEL.placeholder_inputs(BATCH_SIZE,NUM_POINT)
                is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
                
                # Note the global_step=batch parameter to minimize. 
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch_stage_1 = tf.Variable(0,name='stage1/batch')
                bn_decay = get_bn_decay(batch_stage_1)
                tf.compat.v1.summary.scalar('bn_decay', bn_decay)

                print("--- Get training operator")
                # Get training operator
                learning_rate = get_learning_rate(batch_stage_1)
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
                tower_grads = []
                pred_labels_edge_p_gpu = []
                pred_labels_corner_p_gpu = []
                pred_reg_edge_p_gpu = []
                pred_reg_corner_p_gpu = []

                total_loss_gpu = []
                for i in range(NUM_GPUS):
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.compat.v1.name_scope('gpu_%d'%(i)) as scope:
                            # Evenly split input data to each GPU
                            pc_batch = tf.slice(pointclouds_pl, [i*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])

                            ## check if dimension numbers are correct:
                            labels_edge_p_batch = tf.slice(labels_edge_p,[i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])
                            labels_corner_p_batch = tf.slice(labels_corner_p,[i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])
                            reg_edge_p_batch = tf.slice(reg_edge_p, [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])
                            reg_corner_p_batch = tf.slice(reg_corner_p, [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])

                            end_points,dof_feat,simmat_feat = MODEL.get_feature(pc_batch, is_training_pl,STAGE,bn_decay=bn_decay)
                            pred_labels_edge_p, pred_labels_corner_p, pred_reg_edge_p, pred_reg_corner_p  = MODEL.get_stage_1(dof_feat,simmat_feat, is_training_pl,bn_decay=bn_decay)

                            edge_3_1_loss,   edge_3_1_recall,   edge_3_1_acc,\
                            corner_3_1_loss, corner_3_1_recall, corner_3_1_acc,\
                            reg_edge_3_1_loss, reg_corner_3_1_loss, loss = MODEL.get_stage_1_loss(pred_labels_edge_p, \
                                                                                                pred_labels_corner_p, \
                                                                                                labels_edge_p_batch, \
                                                                                                labels_corner_p_batch, \
                                                                                                pred_reg_edge_p, \
                                                                                                pred_reg_corner_p, \
                                                                                                reg_edge_p_batch, \
                                                                                                reg_corner_p_batch)

                            tf.compat.v1.summary.scalar('edge_3_1_loss', edge_3_1_loss)
                            tf.compat.v1.summary.scalar('edge_3_1_recall', edge_3_1_recall)
                            tf.compat.v1.summary.scalar('edge_3_1_acc', edge_3_1_acc)                
                            tf.compat.v1.summary.scalar('corner_3_1_loss', corner_3_1_loss)
                            tf.compat.v1.summary.scalar('corner_3_1_recall', corner_3_1_recall)
                            tf.compat.v1.summary.scalar('corner_3_1_acc', corner_3_1_acc)
                            tf.compat.v1.summary.scalar('reg_edge_3_1_loss', reg_edge_3_1_loss)
                            tf.compat.v1.summary.scalar('reg_corner_3_1_loss', reg_corner_3_1_loss)
                            #tf.summary.scalar('labels_type_loss', task_4_loss)
                            #tf.summary.scalar('labels_type_acc', task_4_acc)
                            tf.compat.v1.summary.scalar('loss', loss)

                            grads = optimizer.compute_gradients(loss)
                            tower_grads.append(grads)

                            ## check this: 
                            # losses = tf.compat.v1.get_collection('losses', scope)
                            # total_loss = tf.add_n(losses, name='total_loss')
                            pred_labels_edge_p_gpu.append(pred_labels_edge_p)
                            pred_labels_corner_p_gpu.append(pred_labels_corner_p)
                            pred_reg_edge_p_gpu.append(pred_reg_edge_p)
                            pred_reg_corner_p_gpu.append(pred_reg_corner_p)
                            total_loss_gpu.append(loss)
                
                # Merge pred and losses from multiple GPUs
                pred_labels_edge_p = tf.concat(pred_labels_edge_p_gpu, 0)
                pred_labels_corner_p = tf.concat(pred_labels_corner_p_gpu, 0)
                pred_reg_edge_p = tf.concat(pred_reg_edge_p_gpu, 0)
                pred_reg_corner_p = tf.concat(pred_reg_corner_p_gpu, 0)
                total_loss = tf.reduce_mean(input_tensor = total_loss_gpu)
                
                # Get training operator 
                grads = average_gradients(tower_grads)
                train_op = optimizer.apply_gradients(grads, global_step=batch_stage_1)
            

                train_op = optimizer.minimize(loss, global_step=batch_stage_1)
                # Add ops to save and restore all the variables.
                saver = tf.compat.v1.train.Saver(max_to_keep=10)


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
        if STAGE == 1:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
        else:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            saver.restore(sess, BASE_DIR+'/stage_1_log/model100.ckpt')
        
        if STAGE==1:
            ops = {'pointclouds_pl': pointclouds_pl,
               'labels_edge_p': labels_edge_p,
               'labels_corner_p': labels_corner_p,
               #'labels_direction': labels_direction,
               'reg_edge_p': reg_edge_p,
               'reg_corner_p': reg_corner_p,
               #'labels_type': labels_type,
               #'simmat_pl': simmat_pl,
               #'neg_simmat_pl': neg_simmat_pl,
               'is_training_pl': is_training_pl,
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
               'reg_edge_3_1_loss': reg_corner_3_1_loss,
               #'task_2_2_loss': task_2_2_loss,
               #'task_3_loss': task_3_loss,
               #'task_4_loss': task_4_loss,
               #'task_4_acc': task_4_acc,
               #'task_5_loss': task_5_loss,
               #'task_6_loss': task_6_loss,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch_stage_1,
               'end_points': end_points}
            for epoch in range(MAX_EPOCH):
                log_string('**** TRAIN EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                train_one_epoch_stage_1(sess,ops,train_writer)
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model"+str(epoch)+".ckpt"
                    save_path = saver.save(sess, os.path.join(LOG_DIR, model_ccc_path))
                    log_string("Model saved in file: %s" % save_path)
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


def train_one_epoch_stage_1(sess, ops, train_writer):
    is_training = True
    permutation = np.random.permutation(32)
    for i in range(len(permutation)//4):
        load_data_start_time = time.time()
        loadpath = BASE_DIR + '/train_data/new_train/'+str(permutation[i*4]+1)+'.mat'   # change training data path
        train_data = sio.loadmat(loadpath)['Training_data']
        load_data_duration = time.time() - load_data_start_time
        log_string('\t%s: %s load time: %f' % (datetime.now(),loadpath,load_data_duration))
        for j in range(3):
            temp_load_data_start_time = time.time()
            temp_loadpath = BASE_DIR + '/train_data/'+str(permutation[i*4+j+1]+1)+'.mat'      # change training data path
            temp_train_data = sio.loadmat(temp_loadpath)['Training_data']
            temp_load_data_duration = time.time() - temp_load_data_start_time
            log_string('\t%s: %s load time: %f' % (datetime.now(),temp_loadpath,temp_load_data_duration))
            train_data = np.concatenate((train_data,temp_train_data),axis = 0)
            print(train_data.shape)

        num_data = train_data.shape[0]
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
        np.random.shuffle(train_data)
        for j in range(num_batch):
            begin_idx = j*BATCH_SIZE
            end_idx = (j+1)*BATCH_SIZE
            data_cells = train_data[begin_idx: end_idx,0]
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
                batch_inputs[cnt,:,:] = tmp_data['down_sample_point'][0, 0]
                batch_labels_edge_p[cnt,:] = np.squeeze(tmp_data['edge_points_label'][0,0])
                batch_labels_corner_p[cnt,:] = np.squeeze(tmp_data['corner_points_label'][0,0])
                #batch_labels_direction[cnt,:] = np.squeeze(tmp_data['motion_direction_class'][0,0])
                batch_regression_edge[cnt,:,:] = tmp_data['edge_points_residual_vector'][0,0]
                batch_regression_corner[cnt,:,:] = tmp_data['corner_points_residual_vector'][0,0]
                #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                #tmp_simmat = tmp_data['similar_matrix'][0,0]
                #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                #tmp_neg_simmat = 1 - tmp_simmat
                #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
            feed_dict = {ops['pointclouds_pl']: batch_inputs,
                         ops['labels_key_p']: batch_labels_edge_p,
                         ops['labels_corner_p']: batch_labels_corner_p,
                         #ops['labels_direction']: batch_labels_direction,
                         ops['reg_edge_p']: batch_regression_edge,
                         ops['reg_corner_p']: batch_regression_corner,
                         #ops['labels_type']: batch_labels_type,
                         #ops['simmat_pl']: batch_simmat_pl,
                         #ops['neg_simmat_pl']: batch_neg_simmat_pl,
                         ops['is_training_pl']: is_training}
                 
                    
#            summary, step, _, task_1_loss_val,task_1_recall_val,task_1_acc_val,task_2_1_loss_val,task_2_1_acc_val,task_2_2_loss_val, \
#                                 task_3_loss_val,task_4_loss_val,task_4_acc_val,task_5_loss_val, \
#                                 task_6_loss_val, loss_val = sess.run([ops['merged'], ops['step'], \
#                                 ops['train_op'], ops['task_1_loss'], ops['task_1_recall'],ops['task_1_acc'],ops['task_2_1_loss'], \
#                                 ops['task_2_1_acc'],ops['task_2_2_loss'],ops['task_3_loss'],ops['task_4_loss'], \
#                                 ops['task_4_acc'],ops['task_5_loss'],ops['task_6_loss'],ops['loss']],feed_dict=feed_dict)
            summary, step, _, edge_3_1_loss_val, edge_3_1_recall_val, edge_3_1_acc_val, \
            corner_3_1_loss_val, corner_3_1_recall_val, corner_3_1_acc_val, \
                reg_edge_3_1_loss_val, reg_corner_3_1_loss_val, loss_val = \
                sess.run([ops['merged'], ops['step'], ops['train_op'], \
                    ops['edge_3_1_loss'], ops['edge_3_1_recall'], ops['edge_3_1_acc'],\
                    ops['corner_3_1_loss'], ops['corner_3_1_recall'], ops['corner_3_1_acc'],\
                    ops['reg_edge_3_1_loss'], ops['reg_corner_3_1_loss'], ops['loss']],feed_dict=feed_dict)
            
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
        log_string('\t\tTraining Edge_3_1 Mean_loss: %f' % total_edge_3_1_loss)
        log_string('\t\tTraining Edge_3_1 Accuracy: %f' % total_edge_3_1_acc)
        log_string('\t\tTraining Edge_3_1 Recall: %f' % total_edge_3_1_recall)
        log_string('\t\tTraining Corner_3_1 Mean_loss: %f' % total_corner_3_1_loss)
        log_string('\t\tTraining Corner_3_1 1 Accuracy: %f' % total_corner_3_1_acc)
        log_string('\t\tTraining Corner_3_1 1 Recall: %f' % total_corner_3_1_recall)
        log_string('\t\tTraining Reg_Edge_3_1 Mean_loss: %f' % total_reg_edge_3_1_loss)
        log_string('\t\tTraining Reg_Edge_3_1 Recall: %f' % total_reg_edge_3_1_loss)
#        log_string('\t\tTraining TASK 2_1 Mean_loss: %f' % total_task_2_1_loss)
#        log_string('\t\tTraining TASK 2_1 Accuracy: %f' % total_task_2_1_acc)
#        log_string('\t\tTraining TASK 2_2 Mean_loss: %f' % total_task_2_2_loss)
#        log_string('\t\tTraining TASK 3 Mean_loss: %f' % total_task_3_loss)
#        log_string('\t\tTraining TASK 4 Mean_loss: %f' % total_task_4_loss)
#        log_string('\t\tTraining TASK 4 Accuracy: %f' % total_task_4_acc)
#        log_string('\t\tTraining TASK 5 Mean_loss: %f' % total_task_5_loss)
#        log_string('\t\tTraining TASK 6 Mean_loss: %f' % total_task_6_loss)

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


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
