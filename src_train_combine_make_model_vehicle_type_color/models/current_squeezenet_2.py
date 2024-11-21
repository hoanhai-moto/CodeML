from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None,
                outputs_collections=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            outputs_collections=None):
            net = squeeze(inputs, squeeze_depth)
            outputs = expand(net, expand_depth)
            return outputs

def squeeze(inputs, num_outputs):
    return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

def expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')

    return tf.concat([e1x1, e3x3], 3)


def se_block(input_feature, name, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')

        scale = input_feature * excitation
        #scale = spatial_attn(input_feature, name)
    return scale


def block_tail(inputs, keep_probability, bottleneck_layer_size):
    fire6 = fire_module(inputs, 64, 256, scope='fire6')
    # print(net)
    fire7 = fire_module(fire6, 64, 256, scope='fire7')
    # print(net)
    net = tf.add(fire7, fire6, name="bypass67")
    print(net)
    # net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool7')
    # print (net)
    net = se_block(net, "se_block_4", ratio=8)
    print(net)

    fire8 = fire_module(net, 64, 256, scope='fire8')
    # print(net)
    fire9 = fire_module(fire8, 64, 256, scope='fire9')
    # print (net)
    net = tf.add(fire9, fire8, name="bypass89")
    print(net)
    net = se_block(net, "se_block_5", ratio=8)
    print(net)

    net = slim.dropout(net, keep_probability)
    net = slim.conv2d(net, bottleneck_layer_size, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv10')
    print(net)
    net = slim.avg_pool2d(net, net.get_shape()[1:3], scope='avgpool10')
    print(net)
    net = tf.squeeze(net, [1, 2], name='squeeze')
    print(net)
    return net


def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('squeezenet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                # net = slim.conv2d(images, 4, [3, 3], stride=1, scope='conv0')
                net = slim.conv2d(images, 32, [3, 3], stride=2, scope='conv1')
                print(net)
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                print(net)
                net = se_block(net, "se_block_1", ratio=8)
                print(net)

                fire2 = fire_module(net, 32, 128, scope='fire2')
                # print(net)
                fire3 = fire_module(fire2, 32, 128, scope='fire3')
                # print(net)
                net = tf.add(fire3, fire2, name="bypass23")
                print (net)
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
                print(net)
                net = se_block(net, "se_block_2", ratio=8)
                print(net)

                fire4 = fire_module(net, 48, 192, scope='fire4')
                # print(net)
                fire5 = fire_module(fire4, 48, 192, scope='fire5')
                # print(net)
                net = tf.add(fire5, fire4, name="bypass45")
                print(net)
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool5')
                print(net)
                net = se_block(net, "se_block_3", ratio=8)
                print(net)

                with tf.variable_scope('tail_vehicle_type'):
                    logits_vh_type = block_tail(net, keep_probability, bottleneck_layer_size)

                with tf.variable_scope('tail_color'):
                    logits_color = block_tail(net, keep_probability, bottleneck_layer_size)

    return logits_vh_type, logits_color
