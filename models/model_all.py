import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from transform_nets import input_transform_net, feature_transform_net

def smooth_l1_dist(deltas, sigma2=2.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

def placeholder_inputs_32(batch_size):
    # placeholders for Section 3.2., batch_size(default) = 32
    open_gt_corner_pair_sample_points_pl = tf.compat.v1.placeholder(tf.float32, shape = (batch_size*256, 64, 3))
    open_gt_corner_valid_mask_256_64 = tf.compat.v1.placeholder(tf.int32, shape = (batch_size*256, 64))
    open_gt_labels_256_64 = tf.compat.v1.placeholder(tf.int32, shape = (batch_size*256, 64))
    open_gt_labels_pair = tf.compat.v1.placeholder(tf.int32, shape = (batch_size*256, 1))
    open_gt_type_cls_label = tf.compat.v1.placeholder(tf.int32, shape = (batch_size*256))

    #open_gt_256_64_idx_pl = tf.compat.v1.placeholder(tf.int32, shape = (batch_size,256,64))
    #open_gt_mask_pl = tf.compat.v1.placeholder(tf.int32, shape = (batch_size,256,64))
    #open_gt_type_pl = tf.compat.v1.placeholder(tf.int32, shape = (batch_size,256,1))
    #open_gt_res_pl = tf.compat.v1.placeholder(tf.float32, shape = (batch_size,256,6))
    #open_gt_sample_points_pl = tf.compat.v1.placeholder(tf.float32, shape = (batch_size,256,64,3))
    #open_gt_valid_mask_pl = tf.compat.v1.placeholder(tf.int32, shape = (batch_size,256,1))
    #open_gt_pair_idx_pl = tf.compat.v1.placeholder(tf.int32, shape = (batch_size,256,2))
    return open_gt_corner_pair_sample_points_pl, open_gt_corner_valid_mask_256_64, open_gt_labels_256_64, open_gt_labels_pair, open_gt_type_cls_label

def placeholder_inputs_31(batch_size,num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,num_point,3))  # input
    labels_edge_p = tf.compat.v1.placeholder(tf.int32,shape=(batch_size,num_point))  # edge points label 0/1
    labels_corner_p = tf.compat.v1.placeholder(tf.int32,shape=(batch_size,num_point)) 
    #labels_direction = tf.placeholder(tf.int32,shape=(batch_size,num_point))
    reg_edge_p = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,num_point,3))
    reg_corner_p = tf.compat.v1.placeholder(tf.float32,shape=(batch_size,num_point,3))
#    labels_type = tf.placeholder(tf.int32,shape=(batch_size,num_point))
#    simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    neg_simmat_pl = tf.placeholder(tf.float32,shape=(batch_size,num_point,num_point))
#    return pointclouds_pl,labels_key_p,labels_direction,regression_direction,regression_position,labels_type,simmat_pl,neg_simmat_pl
    return pointclouds_pl, labels_edge_p, labels_corner_p, reg_edge_p, reg_corner_p


def get_model_32(point_cloud, is_training, bn_decay=None):
    """ Classification/Segmentation/Regression PointNet """
    # Code from PointNet in "https://github.com/charlesq34/pointnet"
    # Implementation of Section 3.2.

    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    # T-Net
    with tf.compat.v1.variable_scope('stage2/transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)
    # Shared MLPs
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/conv2', bn_decay=bn_decay)
    # T-Net
    with tf.compat.v1.variable_scope('stage2/transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    # Shared MLPs again.
    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    # this goes to classification.
    global_feat_reshape = tf.reshape(global_feat, [batch_size, -1])

    # head 2: Classification, this determines wheter we need to generate a line, circle or BSpline.
    class_head = tf_util.fully_connected(global_feat_reshape, 512, bn=True, is_training=is_training,
                                  scope='stage2/cls/fc1', bn_decay=bn_decay)
    class_head = tf_util.dropout(class_head, keep_prob=0.7, is_training=is_training,
                          scope='stage2/cls/dp1')
    class_head = tf_util.fully_connected(class_head, 256, bn=True, is_training=is_training,
                                  scope='stage2/cls/fc2', bn_decay=bn_decay)
    class_head = tf_util.dropout(class_head, keep_prob=0.7, is_training=is_training,
                          scope='stage2/cls/dp2')
    pred_open_curve_cls = tf_util.fully_connected(class_head, 4, activation_fn=None, scope='stage2/cls/fc3')


    # head 1: Segmentation, this determines wheter a particular point belongs to the candidate curve.
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)
    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/seg/conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/seg/conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/seg/conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='stage2/seg/conv9', bn_decay=bn_decay)
    
    seg_head = tf_util.conv2d(net, 2, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='stage2/seg/conv10')
    pred_open_curve_seg = tf.squeeze(seg_head, [2]) # BxNxC (e.g. 8x64x2)

    '''
    # head 3: Regression, this identifies the parameters of the proposed curve.    
    regression_head = tf_util.fully_connected(global_feat_reshape, 512, bn=True, is_training=is_training,
                                  scope='stage2/reg/fc1', bn_decay=bn_decay)
    regression_head = tf_util.dropout(regression_head, keep_prob=0.7, is_training=is_training,
                          scope='stage2/reg/dp1')
    regression_head = tf_util.fully_connected(regression_head, 256, bn=True, is_training=is_training,
                                  scope='stage2/reg/fc2', bn_decay=bn_decay)
    regression_head = tf_util.dropout(regression_head, keep_prob=0.7, is_training=is_training,
                          scope='stage2/reg/dp2')
    pred_open_curve_reg = tf_util.fully_connected(regression_head, 9, activation_fn=None, scope='stage2/reg/fc3') # 3 coordinates
    '''
    
    return pred_open_curve_seg, pred_open_curve_cls, end_points

def get_model_31(point_cloud, is_training, STAGE, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,0])

#    # Set Abstraction layers#
#    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer1')
#    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer2')
#    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, #is_training=is_training, bn_decay=bn_decay, scope='pointnet/layer3')

#    # Feature Propagation layers
#    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='pointnet/fa_layer1')
#    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='pointnet/fa_layer2')
#    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='pointnet/fa_layer3')

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=0.05, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='stage1/layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius=0.1, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='stage1/layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=1024, radius=0.2, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='stage1/layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=512, radius=0.4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='stage1/layer4')

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='stage1/fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='stage1/fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='stage1/fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='stage1/fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='stage1/pointnet/fc1', bn_decay=bn_decay)

    end_points['feats'] = net
    if STAGE == 1:
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='stage1/pointnet/dp1')
    #if is_training:
    #    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='pointnet/dp1')        
    
    # dof_feature
    dof_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='stage1/pointnet/fc_dof', bn_decay=bn_decay)
    # simmat_feature
    #simmat_feat = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet/fc_simmat', bn_decay=bn_decay)

    #return end_points,dof_feat,simmat_feat
    #batch_size = dof_feat.get_shape()[0]

    # Section3.1 Point classification
    # Edge Points Classification
    feat3_1_1 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/feat3_1_1/fc1', bn_decay=bn_decay)
    pred_labels_edge_p = tf_util.conv1d(feat3_1_1, 2, 1, padding='VALID', activation_fn=None, scope='stage1/feat3_1_1/fc2', bn_decay=bn_decay)
    
    # Corner Points Classification
    feat3_1_2 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/feat3_1_2/fc1', bn_decay=bn_decay)
    pred_labels_corner_p = tf_util.conv1d(feat3_1_2, 2, 1, padding='VALID', activation_fn=None, scope='stage1/feat3_1_2/fc2', bn_decay=bn_decay)

    # Edge Points Regression
    feat3_1_3 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/feat3_1_3/fc1', bn_decay=bn_decay)
    pred_reg_edge_p = tf_util.conv1d(feat3_1_3, 3, 1, padding='VALID', activation_fn=None, scope='stage1/feat3_1_3/fc2', bn_decay=bn_decay)

    # Corner Points Regression
    feat3_1_4 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/feat3_1_4/fc1', bn_decay=bn_decay)
    pred_reg_corner_p = tf_util.conv1d(feat3_1_4, 3, 1, padding='VALID', activation_fn=None, scope='stage1/feat3_1_4/fc2', bn_decay=bn_decay)
#
#    #task_4: dof_type
#    feat4 = tf_util.conv1d(dof_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task4/fc1', bn_decay=bn_decay)
#    pred_labels_type = tf_util.conv1d(feat4, 4, 1, padding='VALID', activation_fn=None, scope='stage1/task4/fc2', bn_decay=bn_decay)
#
#    #task_5: similar matrix
#    feat5 = tf_util.conv1d(simmat_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task_5/fc1', bn_decay=bn_decay)
#    r = tf.reduce_sum(feat5*feat5,2)
#    r = tf.reshape(r, [batch_size, -1, 1])
#    D = r-2*tf.matmul(feat5,tf.transpose(feat5,perm=[0,2,1]))+tf.transpose(r, perm=[0,2,1])
#    pred_simmat = tf.maximum(10*D,0.)
#
#    #task_6: confidence map
#    feat6 = tf_util.conv1d(simmat_feat,128,1,padding='VALID',activation_fn = None,scope = 'stage1/task6/fc1', bn_decay=bn_decay)
#    conf_logits = tf_util.conv1d(feat6,1,1,padding='VALID',activation_fn = None,scope = 'stage1/task_6/fc2', bn_decay=bn_decay)
#    pred_conf_logits = tf.nn.sigmoid(conf_logits, name='stage1/task_6/confidence')

#    return pred_labels_key_p,pred_labels_direction,pred_regression_direction,pred_regression_position, \
#                                             pred_labels_type,pred_simmat,pred_conf_logits
    return pred_labels_edge_p, pred_labels_corner_p, pred_reg_edge_p, pred_reg_corner_p


def get_stage_2_loss(pred_open_curve_seg, \
                     pred_open_curve_cls,\
                     gt_256_64_labels, \
                     gt_256_64_valid_mask, \
                     gt_pair_valid_mask, \
                     gt_type_label,
                     #pred_open_curve_reg, \
                     #end_points, \
                     #batch_open_gt_res, \
                     #batch_open_gt_sample_points, \
                     #batch_open_gt_256_64_idx, \
                     #batch_open_gt_mask, \
                     #batch_open_gt_valid_mask, \
                     #batch_open_gt_pair_idx, \
                     #batch_open_gt_type,\
                     end_points, \
                     reg_weight = 0.001):
    
    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1]
    mat_diff = tf.matmul(transform, tf.transpose(a=transform, perm=[0,2,1])) # use perm [0, 2, 1] to keep batch
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    #print_op = tf.compat.v1.print("Debug output:", mat_diff_loss)
    #with tf.control_dependencies([print_op]):
    #    mat_diff_loss = tf.compat.v1.identity(mat_diff_loss)
        
    #tf.print(mat_diff_loss)
    #tf.compat.v1.summary.scalar('mat loss', mat_diff_loss)

    # Loss computation
    # 1. Compute CrossEntropy predicted labels <-> gt_labels_256_64
    # 2. gt_256_64_valid_mask will take elements that were valid only
    # 3. gt_pair_valid_mask will finally will let us take valid proposals only
    # 4. (optional) corner_valid_mask_256_64 will balance loss the per curve.
    num_corner_pairs = gt_256_64_labels.get_shape()[0]

    # Change this accordingly. make sure that loss balancing takes place.
    mask_256_64 = tf.cast(gt_256_64_valid_mask, tf.float32)
    neg_mask_256_64 = tf.ones_like(mask_256_64) - mask_256_64
    Np_256_64 = tf.expand_dims(tf.reduce_sum(mask_256_64, axis=1),1) + 0.001
    Nn_256_64 = tf.expand_dims(tf.reduce_sum(neg_mask_256_64, axis=1),1)

    #print_op2 = tf.compat.v1.print("Debug output mask_256_64:", mask_256_64, mask_256_64.shape)
    #with tf.control_dependencies([print_op2]):
    #    mask_256_64 = tf.compat.v1.identity(mask_256_64)

    #print_op3 = tf.compat.v1.print("Debug output Np_256_64:", Np_256_64, Np_256_64.shape)
    #with tf.control_dependencies([print_op3]):
    #    Np_256_64 = tf.compat.v1.identity(Np_256_64)        

    mask_pair = tf.cast(gt_pair_valid_mask, tf.float32)
    neg_mask_pair = tf.ones_like(mask_pair) - mask_pair
    Np_pair = tf.reduce_sum(mask_pair) + 0.001
    Nn_pair = tf.reduce_sum(neg_mask_pair)

    seg_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_open_curve_seg, labels = gt_256_64_labels)

    #print_op1 = tf.compat.v1.print("Debug output seg_entropy_loss:", seg_entropy_loss, seg_entropy_loss.shape)
    #with tf.control_dependencies([print_op1]):
    #    seg_entropy_loss = tf.compat.v1.identity(seg_entropy_loss)
    
    seg_entropy_loss_valid_256_64 = seg_entropy_loss*(mask_256_64*(Nn_256_64/Np_256_64)+1)
    seg_3_2_loss = tf.reduce_mean(seg_entropy_loss_valid_256_64*(mask_pair*(Nn_pair/Np_pair)+1))
    
    #tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_open_curve_seg, axis = 2, output_type=tf.int32), gt_256_64_labels), tf.float32)*mask_256_64, axis = 1)
    #seg_3_2_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_open_curve_seg, axis = 2, output_type=tf.int32), gt_256_64_labels), tf.float32)*mask, axis = 1) / tf.reduce_sum(mask, axis = 1))
    #seg_3_2_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_open_curve_seg, axis = 2, output_type=tf.int32), gt_256_64_labels), tf.float32)*mask, axis = 1) / tf.reduce_sum(mask, axis = 1))
    
    cls_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_open_curve_cls, labels=gt_type_label)
    cls_3_2_loss = tf.reduce_mean(input_tensor=cls_entropy_loss*(mask_pair*(Nn_pair/Np_pair)+1))
    tf.compat.v1.summary.scalar('cls_entropy_loss', cls_3_2_loss)








    loss = seg_3_2_loss + mat_diff_loss*reg_weight + cls_3_2_loss
    



    # Section 3.2 Open Curve Proposal Loss
    # num_curves = tf.reduce_sum(tf.where(batch_open_gt_valid_mask == 1))

    ## Concat all the GT
    #batch_open_gt_256_64_idx = tf.concat([batch_open_gt_256_64_idx], axis = 0)
    #batch_open_gt_sample_points = tf.concat([batch_open_gt_sample_points], axis = 0)
    #batch_open_gt_valid_mask = tf.concat([batch_open_gt_valid_mask], axis = 0)
    #batch_open_gt_mask = tf.concat([batch_open_gt_mask], axis = 0)
    #batch_open_gt_pair_idx = tf.concat([batch_open_gt_pair_idx], axis = 0)

    # change int to float
    #batch_open_gt_valid_mask = tf.cast(batch_open_gt_valid_mask, tf.float32) # (256, 1)
    #batch_open_gt_mask = tf.cast(batch_open_gt_mask, tf.float32) # (256, 64, 0/1)

    #neg_batch_open_gt_valid_mask = tf.ones_like(batch_open_gt_valid_mask)-batch_open_gt_valid_mask
    #neg_batch_open_gt_mask = tf.ones_like(batch_open_gt_mask)-batch_open_gt_mask



    '''
    #loss: Section 3.1. edge
    mask = tf.cast(labels_edge_p, tf.float32) # change int to float
    neg_mask = tf.ones_like(mask)-mask
    Np = tf.expand_dims(tf.reduce_sum(mask,axis=1),1)     
    Ng = tf.expand_dims(tf.reduce_sum(neg_mask,axis=1),1)  
    edge_3_1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_edge_p,labels = labels_edge_p)*(mask*(Ng/Np)+1))
    edge_3_1_recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_edge_p,axis=2,output_type = tf.int32),\
                        labels_edge_p),tf.float32)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
    edge_3_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_edge_p,axis=2,output_type = tf.int32),\
                        labels_edge_p),tf.float32),axis = 1)/num_point)
    '''

    return loss, seg_3_2_loss, cls_3_2_loss
    #return seg_3_2_loss, seg_3_2_acc, loss + mat_diff_loss*reg_weight

def get_stage_1_loss(pred_labels_edge_p, \
                     pred_labels_corner_p, \
                     labels_edge_p, \
                     labels_corner_p, \
                     pred_reg_edge_p, \
                     pred_reg_corner_p, \
                     reg_edge_p, \
                     reg_corner_p):
    # returns losses from Section 3.1.
    # batch_size = pred_labels_edge_p.get_shape()[0]
    num_point = pred_labels_edge_p.get_shape()[1]

    #loss: Section 3.1. edge
    mask = tf.cast(labels_edge_p, tf.float32) # change int to float, labels_edpge_p: # batch_size,num_point
    neg_mask = tf.ones_like(mask)-mask
    Np = tf.expand_dims(tf.reduce_sum(mask,axis=1),1)     
    Ng = tf.expand_dims(tf.reduce_sum(neg_mask,axis=1),1)  
    edge_3_1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_edge_p,labels = labels_edge_p)*(mask*(Ng/Np)+1))
    edge_3_1_recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_edge_p,axis=2,output_type = tf.int32),\
                        labels_edge_p),tf.float32)*mask,axis = 1)/tf.reduce_sum(mask,axis=1))
    edge_3_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_edge_p,axis=2,output_type = tf.int32),\
                        labels_edge_p),tf.float32),axis = 1)/num_point)
    
    #loss:task1_1: corner
    mask_1_1 = tf.cast(labels_corner_p,tf.float32)
    neg_mask_1_1 = tf.ones_like(mask_1_1)-mask_1_1
    Np_1_1 = tf.expand_dims(tf.reduce_sum(mask_1_1,axis=1),1)     
    Ng_1_1 = tf.expand_dims(tf.reduce_sum(neg_mask_1_1,axis=1),1) 
    corner_3_1_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_labels_corner_p,labels = labels_corner_p)*(mask_1_1*(Ng_1_1/Np_1_1)+1))
    corner_3_1_recall = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_corner_p,axis=2,output_type = tf.int32),\
                        labels_corner_p),tf.float32)*mask_1_1,axis = 1)/tf.reduce_sum(mask_1_1,axis=1))
    corner_3_1_acc = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_labels_corner_p,axis=2,output_type = tf.int32),\
                        labels_corner_p),tf.float32),axis = 1)/num_point)

    # regression edge & corner loss
    reg_edge_3_1_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_reg_edge_p-reg_edge_p),axis=2)*mask, \
                            axis = 1)/tf.reduce_sum(mask,axis=1))
    reg_corner_3_1_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(smooth_l1_dist(pred_reg_corner_p-reg_corner_p),axis=2)*mask_1_1, \
                            axis = 1)/tf.reduce_sum(mask_1_1,axis=1))


    # check for optimal lambda_edge to balance.
    lambda_edge, lambda_corner  = 100.0, 100.0

    # check if these are ok:
    L_edge = edge_3_1_loss+lambda_edge*reg_edge_3_1_loss
    L_corner = corner_3_1_loss+lambda_corner*reg_corner_3_1_loss
    #L_edge = tf.add(edge_3_1_loss, lambda_edge*reg_edge_3_1_loss)
    #L_corner = tf.add(corner_3_1_loss, lambda_corner*reg_corner_3_1_loss)
    loss_31 = L_edge + L_corner

    tf.summary.scalar('all_loss', loss_31)
    tf.compat.v1.add_to_collection('losses', loss_31)
    '''
    tf.compat.v1.add_to_collection('losses', edge_3_1_loss)
    tf.compat.v1.add_to_collection('losses', corner_3_1_loss)
    tf.compat.v1.add_to_collection('losses', lambda_edge*reg_edge_3_1_loss)
    tf.compat.v1.add_to_collection('losses', lambda_corner*reg_corner_3_1_loss)
    '''

    #return task_1_loss,task_1_recall,task_1_acc,task_2_1_loss,task_2_1_acc,task_2_2_loss,task_3_loss,task_4_loss,task_4_acc,task_5_loss,task_6_loss,loss
    return edge_3_1_loss, edge_3_1_recall, edge_3_1_acc, corner_3_1_loss, corner_3_1_recall, corner_3_1_acc, reg_edge_3_1_loss, reg_corner_3_1_loss, loss_31


''' stage3?
def get_stage_2(dof_feat,simmat_feat,dof_mask_pl,proposal_nx_pl,is_training,bn_decay=None):

    dof_feat = tf_util.conv1d(dof_feat,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/smat_fc1')
    simmat_feat = tf_util.conv1d(simmat_feat,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/dof_fc1')
    proposal_nx_pl = tf.expand_dims(proposal_nx_pl,axis = -1)
    proposal_nx_pl = tf.cast(tf.tile(proposal_nx_pl,[1,1,512]),tf.float32)
    simmat_feat_mul = simmat_feat * proposal_nx_pl
    simmat_feat_reduce = tf.reduce_max(simmat_feat_mul,axis=1)
    simmat_feat_expand = tf.tile(tf.expand_dims(simmat_feat_reduce,axis=1),[1,4096,1])
    simmat_feat_all = tf.reduce_max(simmat_feat,axis=1)
    simmat_feat_all = tf.tile(tf.expand_dims(simmat_feat_all,axis=1),[1,4096,1])
    all_feat = tf.concat([dof_feat,simmat_feat_expand],axis = 1)
    dof_mask_pl = tf.expand_dims(dof_mask_pl,axis =-1)
    dof_mask_pl = tf.cast(tf.tile(dof_mask_pl,[1,1,1024]),tf.float32)
    all_feat = all_feat * dof_mask_pl
    feat1 = tf_util.conv1d(all_feat,1024,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc1')
    feat2 = tf_util.conv1d(feat1,512,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc2')
    feat3 = tf_util.conv1d(feat2,256,1,padding='VALID',activation_fn = None,scope = 'stage2/task1/fc3')
    pred_dof_score = tf_util.conv1d(feat3, 1,1, padding='VALID', activation_fn=None, scope='stage2/task1/fc4')
    pred_dof_score = tf.nn.sigmoid(pred_dof_score, name='stage2/task_1/score')
    pred_dof_score = tf.squeeze(pred_dof_score,axis = -1)
    return pred_dof_score, all_feat
'''