#
# Some of the codes are from https://github.com/charlesq34/pointnet2/blob/master/train_multi_gpu.py
#
import sys
import os
import socket
import importlib
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
            self.LOG_DIR = FLAGS.stage_1_log_dir
        elif self.STAGE == 2:
            self.LOG_DIR = FLAGS.stage_2_log_dir

        if not os.path.exists(self.LOG_DIR): 
            os.mkdir(self.LOG_DIR)
            os.system('cp %s %s' % (self.MODEL_FILE, self.LOG_DIR)) # bkp of model def
            os.system('cp main_oop.py %s' % (ROOT_DIR+"/"+self.LOG_DIR)) # bkp of train procedure
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log_train.txt'), 'w')
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
                self.open_gt_corner_pair_sample_points_pl, \
                self.open_gt_corner_valid_mask_256_64, \
                self.open_gt_labels_256_64, \
                self.open_gt_labels_pair,\
                self.open_gt_type_label = self.MODEL.placeholder_inputs_32(self.BATCH_SIZE)

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

                self.MODEL.get_model_32(self.open_gt_corner_pair_sample_points_pl, self.is_training_32, bn_decay=self.bn_decay_32)

                tower_grads_stage2 = []
                pred_seg_p_gpu = []
                pred_cls_p_gpu = []
                seg_3_2_loss_p_gpu = []
                cls_3_2_loss_p_gpu = []
                total_loss_gpu = []
                end_points_p_gpu = []

                for i in range(self.NUM_GPUS):
                    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=True):
                        with tf.device('/gpu:%d'%(i)), tf.compat.v1.name_scope('gpu_%d'%(i)) as scope:
                            device_batch_size_3_2 = self.BATCH_SIZE*256//2
                            batch_corner_pair_sample_points_pl = tf.slice(self.open_gt_corner_pair_sample_points_pl, [i*device_batch_size_3_2,0,0], [device_batch_size_3_2,-1,-1])
                            batch_open_gt_256_64_labels = tf.slice(self.open_gt_labels_256_64, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_open_gt_256_64_valid_mask = tf.slice(self.open_gt_corner_valid_mask_256_64, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_open_gt_pair_valid_mask = tf.slice(self.open_gt_labels_pair, [i*device_batch_size_3_2,0], [device_batch_size_3_2,-1])
                            batch_open_gt_type_label = tf.slice(self.open_gt_type_label, [i*device_batch_size_3_2], [device_batch_size_3_2])
                            pred_open_curve_seg, pred_open_curve_cls, end_points = self.MODEL.get_model_32(batch_corner_pair_sample_points_pl, self.is_training_32, bn_decay=self.bn_decay_32)
                            
                            loss_32, seg_3_2_loss, cls_3_2_loss = self.MODEL.get_stage_2_loss(pred_open_curve_seg, \
                                                        pred_open_curve_cls, \
                                                        batch_open_gt_256_64_labels, \
                                                        batch_open_gt_256_64_valid_mask, \
                                                        batch_open_gt_pair_valid_mask, \
                                                        batch_open_gt_type_label,\
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

                            tf.compat.v1.summary.scalar('%d_GPU_(mat_diff included) loss'% (i), loss_32)
                            grads = self.optimizer_32.compute_gradients(loss_32) # here's where the loss and gradients are covered.
                            tower_grads_stage2.append(grads)
                            pred_seg_p_gpu.append(pred_open_curve_seg)
                            pred_cls_p_gpu.append(pred_open_curve_cls)
                            end_points_p_gpu.append(end_points)
                            
                            #pred_reg_p_gpu.append(pred_open_curve_reg)
                            total_loss_gpu.append(loss_32)
                            seg_3_2_loss_p_gpu.append(seg_3_2_loss)
                            cls_3_2_loss_p_gpu.append(cls_3_2_loss)

                # average or concat
                self.pred_open_curve_seg = tf.concat(pred_seg_p_gpu, 0)
                self.pred_open_curve_cls = tf.concat(pred_cls_p_gpu, 0)
                self.end_points = end_points_p_gpu
                self.total_loss_32 = tf.reduce_mean(input_tensor = total_loss_gpu)
                self.seg_3_2_loss = tf.reduce_mean(input_tensor = seg_3_2_loss_p_gpu)
                self.cls_3_2_loss = tf.reduce_mean(input_tensor = cls_3_2_loss_p_gpu)

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
            self.saver_31.restore(self.sess_31, self.BASE_DIR+'/stage_1_log/model_31_2.ckpt')
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
                'open_gt_corner_pair_sample_points_pl': self.open_gt_corner_pair_sample_points_pl,
               'open_gt_corner_valid_mask_256_64': self.open_gt_corner_valid_mask_256_64,
               'open_gt_labels_256_64': self.open_gt_labels_256_64,
               'open_gt_labels_pair': self.open_gt_labels_pair,
               'open_gt_type_label': self.open_gt_type_label,
               'pred_open_curve_seg': self.pred_open_curve_seg,
               'pred_open_curve_cls': self.pred_open_curve_cls,
               'end_points': self.end_points,
               'is_training_32': self.is_training_32,
               'total_seg_loss_32': self.seg_3_2_loss,
               'total_cls_loss_32': self.cls_3_2_loss,
               'total_loss_32': self.total_loss_32,
               'train_op': self.train_optimizer_32,
               'merged': self.merged_32,
               'step': self.batch_32}        

        for epoch in range(self.MAX_EPOCH):
            self.log_string('**** TRAIN EPOCH %03d ****' % (epoch))
            self.train_one_epoch_32()
            sys.stdout.flush()
            #log_string('**** TEST EPOCH %03d ****' % (epoch))
            #eval_one_epoch(sess, ops, test_writer)
            #sys.stdout.flush()
            # Save the variables to disk.
            '''
            if epoch % 2 == 0:
                model_ccc_path = "model"+str(epoch)+".ckpt"
                save_path = self.saver_32.save(self.sess_32, os.path.join(self.LOG_DIR, model_ccc_path))
                self.log_string("Model saved in file: %s" % save_path)
            '''

    def train_one_epoch_32(self):
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
                print(train_data.shape)

            #push_eval(train_data, ops, sess, train_writer, is_training)
            # num_data = 64*4 = 256
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
            pred_open_curve_seg = np.zeros((num_data*256, 64, 2), np.float32)
            pred_open_curve_cls = np.zeros((num_data*256, 4), np.float32)
            np.random.shuffle(train_data)
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

                batch_open_gt_256_64_idx = np.zeros((self.BATCH_SIZE, 256, 64), np.int32)
                batch_open_gt_mask = np.zeros((self.BATCH_SIZE, 256, 64), np.int32)
                batch_open_gt_type = np.zeros((self.BATCH_SIZE, 256, 1), np.int32)
                batch_open_gt_res = np.zeros((self.BATCH_SIZE, 256, 6), np.float32)
                batch_open_gt_sample_points = np.zeros((self.BATCH_SIZE,256, 64, 3), np.float32)
                batch_open_gt_valid_mask = np.zeros((self.BATCH_SIZE,256, 1), np.int32)
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
                    batch_open_gt_256_64_idx[cnt, ...] = tmp_data[0, 0]['open_gt_256_64_idx']
                    #batch_open_gt_sample_points[cnt, ...] = tmp_data[0, 0]['open_gt_sample_points']
                    #batch_open_gt_mask[cnt, ...] = tmp_data[0, 0]['open_gt_mask']
                    batch_open_gt_type[cnt, ...] = tmp_data[0, 0]['open_gt_type']
                    #batch_open_gt_res[cnt, ...] = tmp_data[0, 0]['open_gt_res']
                    #batch_open_gt_valid_mask[cnt, ...] = tmp_data[0, 0]['open_gt_valid_mask']
                    batch_open_gt_pair_idx[cnt, ...] = tmp_data[0, 0]['open_gt_pair_idx']

                    #batch_labels_type[cnt,:] = np.squeeze(tmp_data['motion_dof_type'][0,0])
                    #tmp_simmat = tmp_data['similar_matrix'][0,0]
                    #batch_simmat_pl[cnt,:,:] = tmp_simmat + tmp_simmat.T
                    #tmp_neg_simmat = 1 - tmp_simmat
                    #tmp_neg_simmat = tmp_neg_simmat - np.eye(NUM_POINT) 
                    #batch_neg_simmat_pl[cnt,:,:] = tmp_neg_simmat
                
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

                # here takes the post processing place.
                
                pred_labels_corner_p_val_softmax = ssp.softmax(pred_labels_corner_p_val[begin_idx:end_idx,:,:], axis = 2)
                corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_idx, corner_valid_mask_pair, corner_valid_mask_256_64, corner_pair_available = self.corner_pair_neighbor_search(batch_inputs, pred_labels_corner_p_val_softmax)
                open_gt_labels_256_64, open_gt_labels_pair, open_gt_type = self.corner_pair_label_generator(corner_pair_256_64_idx, \
                                                                            corner_pair_idx, \
                                                                            corner_valid_mask_pair, \
                                                                            corner_pair_available, \
                                                                            batch_inputs, \
                                                                            batch_open_gt_pair_idx, \
                                                                            batch_open_gt_256_64_idx, \
                                                                            batch_open_gt_type)
                corner_pair_sample_points = np.concatenate([corner_pair_sample_points[i] for i in range(len(corner_pair_sample_points))], axis = 0) # this will be [N, 64, 3]
                #corner_pair_sample_points_label = tf.concat([corner_pair_sample_points_label[i] for i in range(len(corner_pair_sample_points_label))], axis = 0)
                corner_valid_mask_256_64 = np.concatenate([corner_valid_mask_256_64[i] for i in range(len(corner_valid_mask_256_64))], axis = 0) 
                corner_pair_sample_points = corner_pair_sample_points.astype(np.float32)
                corner_valid_mask_256_64 = corner_valid_mask_256_64.astype(np.int32)
                open_gt_labels_pair = open_gt_labels_pair.astype(np.int32)
                open_gt_labels_256_64 = open_gt_labels_256_64.astype(np.int32)
                open_gt_type = open_gt_type.astype(np.int32)
                #

                with self.graph_32.as_default():
                    feed_dict = {self.train_ops_32['open_gt_corner_pair_sample_points_pl']: corner_pair_sample_points,\
                                #ops['corner_pair_sample_points_label']: corner_pair_sample_points_label,\
                                self.train_ops_32['open_gt_corner_valid_mask_256_64']: corner_valid_mask_256_64, \
                                self.train_ops_32['open_gt_labels_256_64']: open_gt_labels_256_64, \
                                self.train_ops_32['open_gt_labels_pair']: open_gt_labels_pair, \
                                self.train_ops_32['open_gt_type_label']: open_gt_type, \
                                self.train_ops_32['is_training_32']: is_training_32}
                    # session run
                    summary, step, _, \
                    seg_3_2_loss_val, \
                    cls_3_2_loss_val, \
                    loss_3_2_val, \
                    pred_open_curve_seg[begin_idx*256:end_idx*256, ...], \
                    pred_open_curve_cls[begin_idx*256:end_idx*256, ...], \
                    end_points = \
                        self.sess_32.run([self.train_ops_32['merged'], \
                                          self.train_ops_32['step'], \
                                          self.train_ops_32['train_op'], \
                                          self.train_ops_32['total_seg_loss_32'], \
                                          self.train_ops_32['total_cls_loss_32'], \
                                          self.train_ops_32['total_loss_32'], \
                                          self.train_ops_32['pred_open_curve_seg'], \
                                          self.train_ops_32['pred_open_curve_cls'], \
                                          self.train_ops_32['end_points']
                                         ],feed_dict=feed_dict)
                    # loss
                    total_seg_3_2_loss += seg_3_2_loss_val
                    total_cls_3_2_loss += cls_3_2_loss_val
                    total_loss_3_2 += loss_3_2_val
                

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
            
                
            process_duration = time.time() - process_start_time
            examples_per_sec = num_data/process_duration
            sec_per_batch = process_duration/num_batch
            self.log_string('\t%s: step: %f total_loss_3_1: %f total_loss_3_2: %f duration time %.3f (%.1f examples/sec; %.3f sec/batch)' \
            % (datetime.now(),step, total_loss_3_1, total_loss_3_2 ,process_duration,examples_per_sec,sec_per_batch))
            self.log_string('\t\tTest Total_3_1 Mean_Loss: %f' % total_loss_3_1)
            self.log_string('\t\tTest Edge_3_1 Mean_Loss: %f' % total_edge_3_1_loss)
            self.log_string('\t\tTest Edge_3_1 Mean_Accuracy: %f' % total_edge_3_1_acc)
            self.log_string('\t\tTest Edge_3_1 Mean_Recall: %f' % total_edge_3_1_recall)
            self.log_string('\t\tTest Corner_3_1 Mean_Loss: %f' % total_corner_3_1_loss)
            self.log_string('\t\tTest Corner_3_1 Mean_Accuracy: %f' % total_corner_3_1_acc)
            self.log_string('\t\tTest Corner_3_1 Mean_Recall: %f' % total_corner_3_1_recall)
            self.log_string('\t\tTest Reg_Edge_3_1 Mean_Loss: %f' % total_reg_edge_3_1_loss)
            self.log_string('\t\tTest Reg_Corner_3_1 Mean_Loss: %f' % total_reg_corner_3_1_loss)
            
            self.log_string('\t\tTraining Total_3_2 Mean_Loss: %f' % total_loss_3_2)
            self.log_string('\t\tTraining Seg_3_2 Mean_Loss: %f' % total_seg_3_2_loss)
            self.log_string('\t\tTraining Cls_3_2 Mean_Loss: %f' % total_cls_3_2_loss)
            

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
                self.saver_31.restore(self.sess_31, self.BASE_DIR+'/stage_1_log/model_31_2.ckpt')

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
                #log_string('**** TEST EPOCH %03d ****' % (epoch))
                #eval_one_epoch(sess, ops, test_writer)
                #sys.stdout.flush()
                # Save the variables to disk.
                if epoch % 2 == 0:
                    model_ccc_path = "model_31_"+str(epoch)+".ckpt"
                    save_path = self.saver_31.save(self.sess_31, os.path.join(self.LOG_DIR, model_ccc_path))
                    self.log_string("Model saved in file: %s" % save_path)

    def train_one_epoch_31(self):
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
                print(train_data.shape)

            #push_eval(train_data, ops, sess, train_writer, is_training)
            # num_data = 64*4 = 256
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
            pred_labels_edge_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_labels_corner_p_val = np.zeros((num_data, self.NUM_POINT, 2), np.float32)
            pred_reg_edge_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            pred_reg_corner_p_val = np.zeros((num_data, self.NUM_POINT, 3), np.float32)
            np.random.shuffle(train_data)
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

    def corner_pair_neighbor_search(self, points_cloud, pred_corner):
        """ builds a sphere between two predicted corner points, sample 64 points within the spehere.

        Args:
            points_cloud ([tf.float32], batch_size, 8096, 3): original point cloud
            pred_corner ([tf.float32], batch_size, 8096, 2): predicted corner points

        Returns:
            sampled points, its indices and valid masks
        """    
        
        
        corner_pair_available = [False]*self.BATCH_SIZE
        corner_valid_mask_pair = []
        corner_pair_idx = []
        
        for per_batch in range(self.BATCH_SIZE):
            threshold = 0.90
            idx = np.where(pred_corner[per_batch, ...][..., 1] > threshold)[0]

            if idx.shape[0] > 23:
                _, idx = graphier_FPS(points_cloud[per_batch, idx, :], 23, idx)

            if idx.shape[0] > 1:
                idx = np.sort(idx)
                corner_pair_available[per_batch] = True
                idx_r = np.repeat(idx, idx.shape[0])
                idx_b = np.tile(idx, [idx.shape[0]])
                two_col = np.stack([idx_r, idx_b], 1)
                corner_pair_idx.append(two_col[two_col[:,0] < two_col[:, 1]])
            else:
                corner_pair_idx.append([])

        # per batch sample the points
        corner_pair_256_64_idx = []
        corner_pair_sample_points = [] # (8, 256, 64, 3)
        corner_valid_mask_256_64 = []
        for per_batch in range(self.BATCH_SIZE):
            rest_num = 256
            if corner_pair_available[per_batch]:

                # first increase the precision to float64, 
                # otherwise it may go wrong when it comes to finding points within radius, 
                # where it may also include the two corner points at the end.
                points_cloud = np.float64(points_cloud)

                # find neighbors
                xyz1 = points_cloud[per_batch, ...][corner_pair_idx[per_batch][:, 0], :]
                xyz2 = points_cloud[per_batch, ...][corner_pair_idx[per_batch][:, 1], :]
                
                ball_center = np.mean(np.stack([xyz1, xyz2], axis = 0), axis = 0)
                distance_from_ball_center = np.sqrt(np.sum(((np.expand_dims(ball_center, 1) - np.expand_dims(points_cloud[per_batch],axis=0))**2), axis = 2))
                r = np.sqrt(np.sum((xyz1 - xyz2)**2, axis = 1)) / 2.0 # radius
                within_range = distance_from_ball_center < np.multiply(np.ones_like(distance_from_ball_center), np.expand_dims(r, axis = 1))

                # per corner pair within this batch, subsample the indicies
                idx_256_64 = []
                valid_mask_256_64 = []
                corner_pair_num = within_range.shape[0]
                
                # if there are more than 256 pairs, just take first 256.
                if corner_pair_num > 256 : corner_pair_num = 256
                rest_num = 256 - corner_pair_num

                for per_corner in range(corner_pair_num):
                    # make sure that corner points(end points) are not within the range.
                    assert within_range[per_corner, ...][corner_pair_idx[per_batch][per_corner]][0] == False
                    assert within_range[per_corner, ...][corner_pair_idx[per_batch][per_corner]][1] == False
                    #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
                    #assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
                    candidnate_num = np.where(within_range[per_corner, :])[0].shape[0]
                    #
                    # raise error or debug when within_range.shape[0] = 0
                    # or if within_range.shape[0] = 3?
                    if 63 <= candidnate_num:
                        middle_indicies = np.where(within_range[per_corner, :])[0]
                        np.random.shuffle(middle_indicies)
                        idx_nums = np.concatenate([np.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), np.squeeze(middle_indicies[:62]), np.expand_dims(corner_pair_idx[per_batch][per_corner][-1], axis = 0)], axis = 0)
                        idx_256_64.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_64.append(np.expand_dims(np.ones_like(idx_nums), axis = 0))

                    elif 0 < candidnate_num < 63:
                        n = candidnate_num
                        dummy_num = 64 - 1 - n
                        middle_indicies = np.where(within_range[per_corner, :])[0]
                        #if candidnate_num == 1: 
                            #middle_indicies = np.expand_dims(middle_indicies, axis = 0)
                        idx_nums = np.concatenate([np.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), middle_indicies, np.repeat(corner_pair_idx[per_batch][per_corner][-1], dummy_num)], axis = 0)
                        idx_256_64.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_64.append(np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0))
                    
                    elif candidnate_num == 0:
                        dummy_num = 63
                        idx_nums = np.concatenate([np.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), np.repeat(corner_pair_idx[per_batch][per_corner][-1], dummy_num)], axis = 0)
                        idx_256_64.append(np.expand_dims(idx_nums, axis = 0))
                        valid_mask_256_64.append(np.expand_dims(np.concatenate([np.ones((64 - (dummy_num - 1)), dtype = np.int64), np.zeros((dummy_num - 1), dtype = np.int64)], axis = 0), axis = 0))
                if rest_num > 0: 
                    idx_256_64.append(np.zeros((rest_num, 64), dtype = np.int64))
                    valid_mask_256_64.append(np.zeros((rest_num, 64), dtype = np.int64))
                corner_pair_256_64_idx.append(np.concatenate(idx_256_64, axis = 0))
                corner_valid_mask_256_64.append(np.concatenate(valid_mask_256_64, axis = 0))
                corner_pair_sample_points.append(points_cloud[per_batch][corner_pair_256_64_idx[per_batch], ...])
            else:
                corner_pair_256_64_idx.append(np.zeros((rest_num, 64), dtype = np.int64))
                corner_valid_mask_256_64.append(np.zeros((rest_num, 64), dtype = np.int64))
                corner_pair_sample_points.append(np.zeros((rest_num, 64, 3), dtype = np.float32))
            valid_mask = np.ones(256, dtype = np.int8)
            valid_mask[-rest_num:] = 0
            valid_mask = np.expand_dims(valid_mask, axis = 1)
            corner_valid_mask_pair.append(valid_mask)

        return corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_idx, corner_valid_mask_pair, corner_valid_mask_256_64, corner_pair_available

    def corner_pair_label_generator(self, \
                                    sample_256_64_idx, \
                                    sample_pair_idx, \
                                    sample_pair_valid_mask, \
                                    sample_corner_pairs_available, \
                                    points_cloud, \
                                    batch_open_gt_pair_idx, \
                                    batch_open_gt_256_64_idx, \
                                    batch_open_gt_type):
                                # add more gt_*

        batch_num = len(sample_corner_pairs_available)    
        sample_valid_mask_256_64_labels_for_loss = np.zeros((batch_num, 256, 64), dtype = np.int32) # output should be (batch_num, 256, 64, 2)
        sample_valid_mask_pair_labels_for_loss = np.zeros((batch_num, 256, 1), dtype = np.int16)
        sample_corner_type = np.zeros((batch_num, 256), dtype = np.int16)
        points_cloud_np = points_cloud
        dist_threshold = 0.5

        # sample_valid_mask_256_64
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
                        # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        

                        mask = np.in1d(sample_256_64_idx[i][k], batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue
                    elif (np.flip(sample_pair_idx[i][k]) == batch_open_gt_pair_idx[i, :, :]).all(axis = 1).any():
                        gt_idx = np.where((np.flip(sample_pair_idx[i][k]) == batch_open_gt_pair_idx[i, :, :]).all(axis = 1))[0]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                        mask = np.in1d(sample_256_64_idx[i][k], batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        # update here labels
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
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
                        mask = np.in1d(sample_256_64_idx[i][k], batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        mask[0], mask[-1] = True, True
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue
                    

                    distance = np.sqrt(np.sum((points_cloud_np[i][np.flip(sample_pair_idx[i][k]), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        sample_corner_type[i, k] = batch_open_gt_type[i, gt_idx][0]
                        mask = np.in1d(sample_256_64_idx[i][k], batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        mask[0], mask[-1] = True, True
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                        k = k+1
                        continue
                    k = k+1
        
        return sample_valid_mask_256_64_labels_for_loss.reshape((-1, 64)), sample_valid_mask_pair_labels_for_loss.reshape(-1, 1), sample_corner_type.reshape(-1)