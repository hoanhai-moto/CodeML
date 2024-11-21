import argparse
import os
import sys
from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import graph_util
import tensorflow as tf
import facenet
import models.inception_resnet_v1_ORG_modify_1 as icpnv1
import models.inception_resnet_v1_ORG_modify_1_reduce as icpnv1_vehicle_type
import models.squeezenet_10_attention as squeezenet_color
import numpy as np
import tensorflow.contrib.slim as slim

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def ycbcr_to_rgb(input, name=None):
    """
    Convert a YCbCr image to RGB format.

    Args:
        input: A 3-D or 4-D Tensor representing the image(s).
        name: Optional name for the operation.

    Returns:
        A Tensor converted to RGB format.
    """
    value = tf.cast(input, tf.float32)
    value = value - tf.constant([0, 128, 128], value.dtype)
    value = ypbpr_to_rgb(value)
    return tf.cast(value, input.dtype)

def ypbpr_to_rgb(input, name=None):
    """
    Convert a YPbPr image to RGB format.

    Args:
        input: A 3-D or 4-D Tensor representing the image(s).
        name: Optional name for the operation.

    Returns:
        A Tensor converted to RGB format.
    """
    input = tf.convert_to_tensor(input)
    assert input.dtype in (tf.float16, tf.float32, tf.float64)
    
    # Transformation kernel from YPbPr to RGB
    kernel = tf.constant(
        [
            [1.00000000e00, 0.0, 1.402],
            [1.00000000e00, -0.344136, -0.714136],
            [1.00000000e00, 1.772, 0.0],
        ],
        input.dtype,
    )
    return tf.tensordot(input, tf.transpose(kernel), axes=((-1,), (0,)))

def images_standardization(inputs):
    """
    Standardize input images by removing mean and scaling by standard deviation.

    Args:
        inputs: A 4-D Tensor of shape [batch_size, height, width, channels].

    Returns:
        A standardized Tensor of the same shape.
    """
    info_img = inputs.get_shape().as_list()  # Get input shape info
    shape_patch = tf.shape(inputs)  # Dynamic shape
    length_img = info_img[1] * info_img[2] * info_img[3]  # Total number of pixels
    h_img = info_img[1]  # Image height
    w_img = info_img[2]  # Image width
    channel = info_img[3]  # Number of channels (e.g., RGB)

    with tf.variable_scope("images_standardization", [inputs]):
        length_sqrt = np.sqrt(length_img)  # Square root of total pixels
        inputs_rhs = tf.reshape(inputs, (shape_patch[0], length_img))  # Flatten the images
        mean_gpu = tf.reduce_mean(inputs_rhs, axis=1)  # Compute mean
        mean_gpu = tf.expand_dims(mean_gpu, axis=1)  # Expand dimensions for broadcasting
        s_gpu = tf.sqrt(tf.divide(tf.reduce_sum(tf.square(inputs_rhs - mean_gpu), axis=1), length_img))  # Compute standard deviation
        adjusted_stddev = tf.maximum(s_gpu, 1.0 / length_sqrt)  # Ensure stddev is not too small
        adjusted_stddev = tf.expand_dims(adjusted_stddev, axis=1)  # Expand for broadcasting
        y = (inputs_rhs - mean_gpu) / adjusted_stddev  # Standardize
        y = tf.reshape(y, (shape_patch[0], h_img, w_img, channel))  # Reshape to original dimensions
        return y

def main(model_dir, output_file):
    """
    Main function to load model, process it, and save the frozen graph.

    Args:
        model_dir: Directory containing the model metagraph and checkpoint files.
        output_file: Path to save the frozen graph protobuf file.
    """
    with tf.Graph().as_default():  # Create a new TensorFlow graph
        config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})  # Configure session
        with tf.Session(config=config) as sess:  # Start a session
            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))  # Get model files

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)

            input_placeholder = tf.placeholder(tf.float32, shape=(None, 180, 264, 3), name='input')  # Define input placeholder

            # Model parameters
            num_class = 86
            nrof_class_make = 107 
            nrof_class_model = 1162
            nrof_class_vehicle_type = 11
            weight_decay = 5e-5
            keep_probability = 0.7
            phase_train_placeholder = False
            embedding_size = 1198
            
            # Build the model
            prelogits = icpnv1.inference(input_placeholder,
                                          keep_probability=keep_probability,
                                          bottleneck_layer_size=embedding_size,
                                          phase_train=phase_train_placeholder,
                                          weight_decay=weight_decay)
                            
            output_cnn_vehicle_type = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Mixed_7a/concat:0")  # Get intermediate tensor

            prelogits_vehicle_type = icpnv1_vehicle_type.inference_tail(output_cnn_vehicle_type,
                                                                        keep_probability=0.7,
                                                                        phase_train=phase_train_placeholder,
                                                                        bottleneck_layer_size=1198,
                                                                        weight_decay=weight_decay)

            output_cnn_color = tf.get_default_graph().get_tensor_by_name("input:0")  # Get input tensor for color model

            prelogits_color = squeezenet_color.inference(output_cnn_color,
                                                          keep_probability=0.7,
                                                          phase_train=phase_train_placeholder,
                                                          bottleneck_layer_size=512,
                                                          weight_decay=weight_decay)

            # Define logits for classification tasks
            logits_model = slim.fully_connected(prelogits,
                                                1045,
                                                activation_fn=None,
                                                scope='logits_model', reuse=False)

            logits_vehicle_type = slim.fully_connected(prelogits_vehicle_type,
                                                       nrof_class_vehicle_type,
                                                       activation_fn=None,
                                                       scope='classify_layer', reuse=False)

            logits_color = slim.fully_connected(prelogits_color,
                                                10,
                                                activation_fn=None,
                                                scope='classify_layer_color', reuse=False)

            # Apply softmax to logits
            tf.nn.softmax(logits_model, name='softmax_model')
            tf.nn.softmax(logits_vehicle_type, name='softmax_vehicle_type')
            tf.nn.softmax(logits_color, name='softmax_color')

            model_dir_exp = os.path.expanduser(model_dir)  # Expand user directory
            saver = tf.train.Saver()  # Create a saver to restore model
            saver.restore(sess, os.path.join(model_dir_exp, ckpt_file))  # Restore model weights

            input_graph_def = sess.graph.as_graph_def()  # Get the graph definition
            output_graph_def = freeze_graph_def(sess, input_graph_def, ['softmax_model', 'softmax_vehicle_type', "softmax_color"])  # Freeze the graph

            # Serialize and save the output graph
            with tf.gfile.GFile(output_file, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph: %s" % (len(output_graph_def.node), output_file))

def freeze_graph_def(sess, input_graph_def, output_node_names):
    """
    Freeze the graph by converting variables to constants.

    Args:
        sess: The current TensorFlow session.
        input_graph_def: The input graph definition.
        output_node_names: Names of the output nodes.

    Returns:
        A frozen graph definition with all variables converted to constants.
    """
    # Modify certain node operations for compatibility
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    # Collect names of important nodes for freezing
    whitelist_names = []
    for node in input_graph_def.node:
        if (node.name.startswith('InceptionResnetV1') or
            node.name.startswith('ICT_tail_vehicle_type') or
            node.name.startswith('squeezenet') or
            node.name.startswith('logits_make') or
            node.name.startswith('logits_model') or
            node.name.startswith('logits_vehicle_type') or
            node.name.startswith('classify_layer_color') or
            node.name.startswith('classify_layer') or
            node.name.startswith('softmax_color') or
            node.name.startswith('softmax_vehicle_type') or
            node.name.startswith('softmax_model')):
            whitelist_names.append(node.name)

    # Convert variables in the graph to constants
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names,
        variable_names_whitelist=whitelist_names)
    return output_graph_def

if __name__ == '__main__':
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description='Freeze TensorFlow model.')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the model metagraph and checkpoint files.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the frozen graph protobuf file.')

    args = parser.parse_args()  # Parse arguments
    main(args.model_dir, args.output_file)  # Call main with parsed arguments