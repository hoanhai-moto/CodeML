# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def swish(features):
    return features * tf.nn.sigmoid(features)
# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                     tower_conv2_2, tower_pool], 3)
    return net


def upsample(inputs, out_shape):
    inputs = tf.image.resize_nearest_neighbor(inputs, out_shape)
    return inputs


def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
                                   reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='SAME'):
                print(inputs)


                # 100 x 24 x 32
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                tf.summary.image("CD1", tf.reshape(tf.transpose(net, [0, 3, 1, 2]),
                                                   (-1, net.get_shape()[1], net.get_shape()[2], 1)), 32)
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                # net = slim.conv2d(net, 32, 3, padding='VALID',
                #                   scope='Conv2d_2a_3x3')
                #
                # print(net)
                # end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')

                print(net)
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')

                print(net)
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                # net = slim.conv2d(net, 80, 1, padding='VALID',
                #                   scope='Conv2d_3b_1x1')

                net = slim.conv2d(net, 64, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')

                print(net)
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                # net = slim.conv2d(net, 192, 3, padding='VALID',
                #                   scope='Conv2d_4a_3x3')
                net = slim.conv2d(net, 128, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')

                print(net)
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=1, padding='VALID',
                                  scope='Conv2d_4b_3x3')

                print(net)
                end_points['Conv2d_4b_3x3'] = net
                ##
                # # --Bao edited--#
                # net_1 = slim.conv2d(net, 16, 3, stride=2, scope='conv1')
                # print(net_1)
                # net_1 = slim.conv2d(net_1, 32, 3, stride=2, scope='conv2')
                # print(net_1)
                #
                # net_1_1 = slim.conv2d(net_1, 64, 3, stride=2, scope='conv3')
                # net_1_2 = slim.conv2d(net_1, 64, 3, stride=2, scope='conv4')
                #
                # net_1_1 = slim.conv2d(net_1_1, 1, 1, stride=1, scope='conv5')
                # net_1_2 = slim.conv2d(net_1_2, 1, 1, stride=1, scope='conv6')
                #
                # print(net_1_1)
                #
                # save_net_1 = net_1_1
                # net_1_1 = slim.flatten(net_1_1)
                # net_1_1 = tf.nn.softmax(net_1_1)
                #
                # net_1_2 = slim.flatten(net_1_2)
                # net_1_2 = tf.nn.softmax(net_1_2)
                #
                # net_1_1 = tf.add(net_1_1, net_1_2)
                # print(net_1_1)
                #
                # net_1_1 = tf.reshape(net_1_1, (
                #     -1, save_net_1.get_shape()[1], save_net_1.get_shape()[2], save_net_1.get_shape()[3]))
                #
                # print(net_1_1)
                # net_1_1 = upsample(net_1_1, inputs.get_shape()[1:3])
                # print(net_1_1)
                #
                # net = tf.multiply(inputs, net_1_1)

                # --------------#
                ##
                # 5 x Inception-resnet-A
                # net = slim.repeat(net, 5, block35, scale=0.17)
                net = slim.repeat(net, 2, block35, scale=0.17)
                print(net)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                    print(net)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                # net = slim.repeat(net, 10, block17, scale=0.10)
                net = slim.repeat(net, 2, block17, scale=0.10)
                print(net)
                end_points['Mixed_6b'] = net

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                    print(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                # net = slim.repeat(net, 5, block8, scale=0.20)
                net = slim.repeat(net, 2, block8, scale=0.20)
                print(net)
                end_points['Mixed_8a'] = net

                net = block8(net, activation_fn=None)
                print(net)
                end_points['Mixed_8b'] = net

                ##--Bao edited--##

                net = slim.dropout(net, dropout_keep_prob)
                net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv2d_bottleneck')
                print(net)
                print(net.get_shape()[1:3])
                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool2D')
                print(net)
                net = tf.squeeze(net, [1, 2], name='squeeze')
                print(net)

                ##-------------##

                # with tf.variable_scope('Logits'):
                #     end_points['PrePool'] = net
                #     #pylint: disable=no-member
                #
                #     net = slim.dropout(net, dropout_keep_prob)
                #     net = slim.conv2d(net, 1024, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv2d_bottleneck')
                #     net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool_1a_8x8')
                #     print (net)
                #     net = tf.squeeze(net, [1, 2], name='squeeze')
                #     print(net)

                #     net = slim.flatten(net)
                #
                #     net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                        scope='Dropout')
                #
                #     end_points['PreLogitsFlatten'] = net
                #
                # print (net)
                # net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                #         scope='Bottleneck', reuse=False)

    return net, end_points


def inception_resnet_v1_head_color(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='Color_Head_InceptionResnetV1'):

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='SAME'):
                print(inputs)
                net = slim.conv2d(inputs, 8, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 16, 3, scope='Conv2d_2b_3x3')
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                net = slim.conv2d(net, 16, 1, padding='VALID', scope='Conv2d_3b_1x1')
                net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_4a_3x3')
                net = slim.conv2d(net, 32, 3, stride=1, padding='VALID', scope='Conv2d_4b_3x3')

    return net


def block_at(inputs):
    mask = slim.conv2d(inputs, 16, 1, padding='VALID', scope='mask_1')
    mask = slim.conv2d(mask, 16, 3, stride=2, padding='VALID', scope='mask_2')
    mask = slim.max_pool2d(mask, 3, stride=2, padding='VALID', scope='mask_3')

    mask = slim.conv2d(mask, 16, 1, padding='VALID', scope='mask_4')
    mask = slim.conv2d(mask, 16, 3, stride=2, padding='VALID', scope='mask_5')
    mask = slim.max_pool2d(mask, 3, stride=2, padding='VALID', scope='mask_6')
    mask = slim.conv2d(mask, 1, 1, padding='VALID', scope='mask_7')

    save_mask = mask
    mask = slim.flatten(mask)
    mask = tf.nn.softmax(mask)

    mask = tf.reshape(mask, (
        -1, save_mask.get_shape()[1], save_mask.get_shape()[2], save_mask.get_shape()[3]))

    mask = upsample(mask, inputs.get_shape()[1:3])
    # net = tf.multiply(inputs, mask)
    return mask


def inception_resnet_v1_makemodels(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='Makemodels_InceptionResnetV1'):

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='SAME'):
                #--------#
                mask = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='mask_1')
                mask = slim.conv2d(mask, 32, 3, scope='mask_2')
                mask = slim.max_pool2d(mask, 3, stride=2, padding='VALID', scope='mask_3')

                mask = slim.conv2d(mask, 32, 1, padding='VALID', scope='mask_4')
                mask = slim.conv2d(mask, 32, 3, stride=2, padding='VALID', scope='mask_5')
                mask = slim.max_pool2d(mask, 3, stride=2, padding='VALID', scope='mask_6')
                mask = slim.conv2d(mask, 1, 1, padding='VALID', scope='mask_7')

                save_mask = mask
                mask = slim.flatten(mask)
                mask = tf.nn.softmax(mask)

                mask = tf.reshape(mask, (
                    -1, save_mask.get_shape()[1], save_mask.get_shape()[2], save_mask.get_shape()[3]))

                mask = upsample(mask, inputs.get_shape()[1:3])
                net = tf.multiply(inputs, mask)
                #--------#
                print(inputs)
                net = slim.conv2d(net, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, 3, scope='Conv2d_2b_3x3')
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')

                net = slim.conv2d(net, 64, 1, padding='VALID', scope='Conv2d_3b_1x1')
                net = slim.conv2d(net, 128, 3, stride=2, padding='VALID', scope='Conv2d_4a_3x3')
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_4a_3x3')

                # net = slim.conv2d(net, 128, 1, padding='VALID', scope='Conv2d_4b_1x1')
                # net = slim.conv2d(net, 128, 3, padding='VALID', scope='Conv2d_4b_3x3')

                print (net)
                net = slim.repeat(net, 3, block35, scale=0.17)
                print(net)

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 128, 128, 256, 256)
                    print(net)

                net = slim.repeat(net, 5, block17, scale=0.10)
                print(net)

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                    print(net)

                net = slim.repeat(net, 3, block8, scale=0.20)
                print(net)

                # net = block8(net, activation_fn=None)

                # net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool2D_m')
                #
                # net = slim.flatten(net)
                #
                # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                    scope='Dropout')
                #
                # net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                #                            scope='Bottleneck', reuse=False)

                # print(net)

                net = slim.dropout(net, dropout_keep_prob)
                net = slim.conv2d(net, bottleneck_layer_size, [1, 1], activation_fn=None, normalizer_fn=None,
                                    scope='conv2d_bottleneck_m')

                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool2D_m')
                net = tf.squeeze(net, [1, 2], name='squeeze_m')

    return net


def inception_resnet_v1_makemodels_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='Makemodels_InceptionResnetV1'):

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='SAME'):
                #--------#
                with tf.variable_scope('block_at_1'):
                    mask_inputs_1 = block_at(inputs)

                with tf.variable_scope('block_at_2'):
                    mask_inputs_2 = block_at(inputs)

                mask_inputs = tf.add(mask_inputs_2, mask_inputs_1, name="mask_inputs")
                net = tf.multiply(inputs, mask_inputs)
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_mask_1')

                with tf.variable_scope('block_at_3'):
                    mask_net_1 = block_at(net)

                with tf.variable_scope('block_at_4'):
                    mask_net_2 = block_at(net)

                mask_net = tf.add(mask_net_2, mask_net_1, name="mask_net")
                net = tf.multiply(net, mask_net)
                #--------#
                print(inputs)
                net = slim.conv2d(net, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, 3, scope='Conv2d_2b_3x3')
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')

                net = slim.conv2d(net, 64, 1, padding='VALID', scope='Conv2d_3b_1x1')
                net = slim.conv2d(net, 128, 3, stride=2, padding='VALID', scope='Conv2d_4a_3x3')
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_4a_3x3')

                # net = slim.conv2d(net, 128, 1, padding='VALID', scope='Conv2d_4b_1x1')
                # net = slim.conv2d(net, 128, 3, padding='VALID', scope='Conv2d_4b_3x3')

                print (net)
                net = slim.repeat(net, 3, block35, scale=0.17)
                print(net)

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 128, 128, 256, 256)
                    print(net)

                net = slim.repeat(net, 5, block17, scale=0.10)
                print(net)

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                    print(net)

                net = slim.repeat(net, 3, block8, scale=0.20)
                print(net)

                net = slim.dropout(net, dropout_keep_prob)
                net = slim.conv2d(net, bottleneck_layer_size, [1, 1], activation_fn=None, normalizer_fn=None,
                                    scope='conv2d_bottleneck_m')

                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool2D_m')
                net = tf.squeeze(net, [1, 2], name='squeeze_m')

    return net



def inception_resnet_v1_makemodels_v3(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='Makemodels_InceptionResnetV1'):

    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                stride=1, padding='SAME'):

                print(inputs)
                net = slim.conv2d(inputs, 64, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                print(net)

                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                print(net)

                net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_3a_3x3')
                print(net)

                net = slim.conv2d(net, 128, 1, padding='VALID', scope='Conv2d_3b_1x1')
                print(net)

                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID', scope='Conv2d_4a_3x3')
                # net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_4a_3x3')

                # net = slim.conv2d(net, 128, 1, padding='VALID', scope='Conv2d_4b_1x1')
                # net = slim.conv2d(net, 128, 3, padding='VALID', scope='Conv2d_4b_3x3')

                print (net)
                net = slim.repeat(net, 3, block35, scale=0.17)
                print(net)

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 128, 128, 256, 256)
                    print(net)

                net = slim.repeat(net, 5, block17, scale=0.10)
                print(net)

                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                    print(net)

                net = slim.repeat(net, 3, block8, scale=0.20)
                print(net)

                net = slim.dropout(net, dropout_keep_prob)
                net = slim.conv2d(net, bottleneck_layer_size, [1, 1], activation_fn=None, normalizer_fn=None,
                                    scope='conv2d_bottleneck_m')

                net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='AvgPool2D_m')
                net = tf.squeeze(net, [1, 2], name='squeeze_m')

    return net





def inference_combine_makemodel_vehicle(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        activation_fn=swish):

        models = inception_resnet_v1_makemodels_v3(images,
                                                is_training=phase_train,
                                                dropout_keep_prob=keep_probability,
                                                bottleneck_layer_size=bottleneck_layer_size,
                                                reuse=reuse)

        return models


