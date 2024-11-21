from pickle import load
# import model_architectures as m_arch
import models.inception_resnet_v1 as irset
import models.inception_resnet_v1_ORG_modify_1 as icpnv1
import models.inception_resnet_Abao_old as icpnv1_Abao_old
import models.inception_resnet_v1_Bao as icpnv1_Abao

import models.inception_resnet_v1_ORG as icpnv1_full
import models.inception_v4 as icpnv4
import models.squeezenet_3 as squeeze3

import models.inception_resnet_v2 as icpnv2
import models.squeezenet as sqe
from models.resnet_utils import resnet_arg_scope

import test_module as test_phase
import ultils
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import argparse
import sys
import datetime

import math
from sklearn.utils import shuffle
import cv2
from sklearn.metrics import classification_report
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import tensorflow.contrib.slim as slim
from sklearn.model_selection import train_test_split, StratifiedKFold
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_max_epoch_from_file(file_max_epoch):
    """Loads the maximum epoch number from a file.

    Args:
        file_max_epoch: Path to the file containing the maximum epoch number.

    Returns:
        The maximum epoch number as an integer.
    """

    f = open(file_max_epoch, 'r')
    all_line = f.readlines()
    return int(all_line[0])


def read_mapping_model(mapping_file,model_dir= None):
    """Reads class mappings from a file and optionally saves them to a new file.

    Args:
        mapping_file: Path to the file containing the class mappings.
        model_dir: Optional directory to save the cleaned mappings.

    Returns:
        A list of cleaned class names.
    """

    with open(mapping_file) as f:
        class_Alan = f.readlines()

    class_Alan = [i.strip().lower().replace(" ","").replace("/","-")  for i in class_Alan]
    if model_dir != None :
        
        with open(os.path.join(model_dir , f'{model_dir}.txt'), 'w') as f:
            for item in class_Alan:
                f.write("%s\n" % item)
    return class_Alan   


def run_valiate(args, sess, nclass, val_data,model_dir,model_name,epoch):
    """Runs validation on a trained model and generates a classification report.

    Args:
        args: Arguments containing configuration parameters (e.g., batch size, image sizes, file paths).
        sess: TensorFlow session.
        nclass: Number of classes.  (Not used in the current code)
        val_data: Validation data as a tuple of (image paths, labels).
        model_dir: Directory containing the model.
        model_name: Name of the model. (Not used in the current code)
        epoch: Current epoch number.

    Returns:
        The  average recall from the classification report.
    """

    print("starting calc on validation data")
    
    mapping_list =read_mapping_model(args.file_mapping_model,model_dir = model_dir)
    mapping_maker_list =read_mapping_model(args.file_mapping_make,model_dir = model_dir)


    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    predict_placehoder_make_model = tf.get_default_graph().get_tensor_by_name("softmax:0")  # predict
      # predict

    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    predict = []
    predict_make = []
    nrof_images = len(val_data[0])
    nrof_batches_per_epoch = math.ceil(nrof_images / args.batch_size)
    print("nrof_images: ", nrof_images)
    print('nrof_batches_per_epoch: ', nrof_batches_per_epoch)



    for k in range(nrof_batches_per_epoch):
        start_index = k * args.batch_size
        end_index = min((k + 1) * args.batch_size, nrof_images)
        paths_batch = val_data[0][start_index:end_index]
        array_images = ultils.load_data(paths_batch,
                                        args.image_size_w,
                                        args.image_size_h,
                                        args.color)

        feed_dict = {
            images_placeholder: array_images,
            phase_train_placeholder: False
            # batch_size_placeholder : len(array_images)
        }

        predictions = sess.run(predict_placehoder_make_model, feed_dict=feed_dict)
        mapping_convert = np.array([mapping_list[i] for i in np.argmax(predictions, axis=1)])
        predict += list(mapping_convert)

        if (k + 1) % 1000 == 0:
            print('%dth batch on total %d was completed' % (k, nrof_batches_per_epoch))

    ground_truth = val_data[1].reshape(-1)
    predict = np.array(predict)
    ground_truth = np.array(ground_truth)
    predict = predict.reshape(-1)
    predict = [pr.replace("-","_") for pr in predict]
    ground_truth = [gr.replace("-","_") for gr in ground_truth]
    

    reports_model  = classification_report(ground_truth, predict)
    
    accuracy_model = np.sum(ground_truth == predict) / len(predict)
    
    reports = reports_model 

    with open(os.path.join(model_dir, f"report_classify_make_model_{epoch}.txt"), "w") as file_report:
            file_report.write(reports)

    return classification_report(ground_truth, predict,output_dict =True)['macro avg']['recall']

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final calculated output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)


        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
        output_test = tf.add(s_cos_t, cos_mt_temp)
        output_test = tf.identity(output_test, 'linear_logits')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')

        inv_mask = tf.subtract(1., mask, name='inverse_mask')


        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output, output_test


def center_loss(features, label, alfa, nrof_classes):
    """
       Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)

    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)

    centers = tf.scatter_sub(centers, label, diff)

    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers



def data_augumentation(image_aug, random_flip_left_right=True, random_brightness=True,
                       random_saturation=True, random_hue=True, random_contrast=True,
                       random_rotate=True):
    """Applies random data augmentation to an image tensor.

    Args:
        image_aug: The input image tensor.
        random_flip_left_right: Whether to apply random horizontal flips.
        random_brightness: Whether to apply random brightness adjustments.
        random_saturation: Whether to apply random saturation adjustments.
        random_hue: Whether to apply random hue adjustments.
        random_contrast: Whether to apply random contrast adjustments.
        random_rotate: Whether to apply random rotations.

    Returns:
        The augmented image tensor.
    """


    flag_condition = tf.random_uniform([7], 0, 2, dtype=tf.int32)
    angle = tf.random_uniform([1], -15, 15, dtype=tf.float32)

    if random_flip_left_right:
        image_aug = tf.cond(tf.equal(flag_condition[0], 1),
                            lambda: tf.image.random_flip_left_right(image_aug),
                            lambda: image_aug)

    if random_brightness:
        image_aug = tf.cond(tf.equal(flag_condition[1], 1),
                            lambda: tf.image.random_brightness(image_aug, max_delta=0.1),
                            lambda: image_aug)
    if random_saturation:
        image_aug = tf.cond(tf.equal(flag_condition[2], 1),
                            lambda: tf.image.random_saturation(image_aug, lower=0.75, upper=1.5),
                            lambda: image_aug)

    if random_hue:
        image_aug = tf.cond(tf.equal(flag_condition[3], 1),
                            lambda: tf.image.random_hue(image_aug, max_delta=0.15),
                            lambda: image_aug)
    if random_contrast:
        image_aug = tf.cond(tf.equal(flag_condition[4], 1),
                            lambda: tf.image.random_contrast(image_aug, lower=0.75, upper=1.5),
                            lambda: image_aug)
    if random_rotate:
        image_aug = tf.cond(tf.equal(flag_condition[5], 1),
                            lambda: tf.contrib.image.rotate(image_aug, angle * math.pi / 180, interpolation='BILINEAR'),
                            lambda: image_aug)

    image_aug = tf.contrib.image.rotate(image_aug, angle * math.pi / 180.0, interpolation='BILINEAR')

    image_aug = tf.image.per_image_standardization(image_aug)
    return image_aug


def create_flow_data(args, image_paths_placeholder, nrof_labels, labels_placeholder, batch_size_placeholder):
    """Creates a TensorFlow data flow graph for image preprocessing and batching.

    Args:
        args: Arguments containing configuration parameters (e.g., image sizes, color mode).
        image_paths_placeholder: Placeholder for image paths.
        nrof_labels: Number of labels.
        labels_placeholder: Placeholder for labels.
        batch_size_placeholder: Placeholder for batch size.

    Returns:
        A tuple containing the enqueue operation, image batch tensor, and label batch tensor.
    """

    input_queue = data_flow_ops.FIFOQueue(capacity=1000000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(1,), (nrof_labels,)],
                                          shared_name=None, name=None)

    enqueue_op = input_queue.enqueue_many([image_paths_placeholder,
                                           labels_placeholder], name='enqueue_op')

    nrof_preprocess_threads = 4

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_jpeg(file_contents, channels=3, dct_method="INTEGER_ACCURATE")
            image = tf.image.resize_images(image, [args.image_size_h, args.image_size_w])
            image = data_augumentation(image)
            # image.set_shape((args.image_size_h, args.image_size_w, 3))

            if(args.color != 'color'):
                image = tf.reduce_mean(image, axis=2)
                image = tf.expand_dims(image, axis=2)

            images.append(image)
        images_and_labels.append([images, label])

    image_batch, label_batch = tf.train.shuffle_batch_join(images_and_labels,
                                                           batch_size=batch_size_placeholder,
                                                           enqueue_many=False,
                                                           capacity=4 * nrof_preprocess_threads * args.batch_size,
                                                           allow_smaller_final_batch=True,
                                                           min_after_dequeue=0)

    return enqueue_op, image_batch, label_batch


def loss_compute(args,
                 logits_make,
                 logits_model,
                 label_make,
                 label_model,
                 learning_rate_placeholder,
                 nrof_class_make,
                 nrof_class_model,
                 global_step):
    """Computes the total loss for training a dual-output model.

    Args:
        args: Arguments containing configuration parameters (e.g., learning rate decay settings, smoothing).
        logits_make: Logits for the "make" output.
        logits_model: Logits for the "model" output.
        label_make: Labels for the "make" output.
        label_model: Labels for the "model" output.
        learning_rate_placeholder: Placeholder for the learning rate.
        nrof_class_make: Number of classes for the "make" output.
        nrof_class_model: Number of classes for the "model" output.
        global_step: Global step tensor.

    Returns:
        A tuple containing the learning rate, total loss, and regularization losses.
    """

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs * args.epoch_size,
                                               args.learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)


    if(args.smooth_label):
        one_hot_encoding = tf.one_hot(indices=tf.cast(label_make, tf.int32), depth=nrof_class_make)
        cross_entropy_make = tf.losses.softmax_cross_entropy(one_hot_encoding, logits_make, label_smoothing=0.1)
        cross_entropy_make = tf.reduce_mean(cross_entropy_make, name='cross_entropy_make')

        one_hot_encoding = tf.one_hot(indices=tf.cast(label_model, tf.int32), depth=nrof_class_model)
        cross_entropy_model = tf.losses.softmax_cross_entropy(one_hot_encoding, logits_model, label_smoothing=0.1)
        cross_entropy_model = tf.reduce_mean(cross_entropy_model, name='cross_entropy_model')
    else:
        cross_entropy_make = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_make,
                                                                                   logits=logits_make))

        cross_entropy_model = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label_model,
                                                                                    logits=logits_model))


    tf.add_to_collection('losses_make', cross_entropy_model)
    tf.add_to_collection('losses_model', cross_entropy_make)
    tf.add_to_collection('losses', cross_entropy_model)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_make + cross_entropy_model] + regularization_losses, name='total_loss')

    return learning_rate, total_loss, regularization_losses


def config_load_and_save_params(learning_rate_placeholder, total_loss, global_step, scope_var = None):
    """Configures the optimization and saving operations for the training process.

    Args:
        learning_rate_placeholder: Placeholder for the learning rate.
        total_loss: Total loss tensor.
        global_step: Global step tensor.
        scope_var: Optional scope for variables (not used in the current implementation).

    Returns:
        A tuple containing the general saver, the saver for InceptionResnetV1 variables,
        the training operation, and the summary operation.
    """

    set_A_vars =[v for v in tf.trainable_variables()
                if "InceptionResnetV1" in v.name
                ]
    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    saver_set_A = tf.train.Saver(set_A_vars, max_to_keep=3)
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate_placeholder)
    #optimizer = tf.train.MomentumOptimizer(learning_rate_placeholder, momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate_placeholder, beta1=0.9, beta2=0.999, epsilon=0.1)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate_placeholder, decay=0.9, momentum=0.9, epsilon=1.0)
    train_op = optimizer.minimize(total_loss, global_step=global_step)  #

    summary_op = tf.summary.merge_all()
    return saver, saver_set_A, train_op, summary_op


def training_on(args, log_dir,
                pretrained_model, saver_set_A,
                image_list, label_list,
                index_dequeue_op, enqueue_op, image_paths_placeholder,
                labels_placeholder, learning_rate_placeholder,
                phase_train_placeholder, batch_size_placeholder,
                global_step, total_loss, train_op, summary_op,
                regularization_losses, saver, model_dir,
                subdir, accuracy_training,
                val_infor, test_infor, nrof_classes):
    """
    Performs the training loop for the model.

    Args:
        args: Training configuration arguments.
        log_dir: Directory for saving logs.
        pretrained_model: Path to a pretrained model (optional).
        saver_inception: Saver for InceptionResnetV1 variables.
    Returns:
        Accuracy on the test set.
    """

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True)) #gpu_options=gpu_options,

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    good_acc = 0.0
    with sess.as_default():
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            saver_set_A.restore(sess, pretrained_model)

        print('Running training')
        epoch = 0
        max_nrof_epochs = args.max_nrof_epochs
        while epoch < max_nrof_epochs:
            step = sess.run(global_step, feed_dict=None)

            epoch = step // args.epoch_size

            train(args, sess, epoch,
                  image_list, label_list,
                  index_dequeue_op, enqueue_op, image_paths_placeholder,
                  labels_placeholder, learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                  global_step, total_loss, train_op, summary_op, summary_writer, regularization_losses,
                  args.learning_rate_schedule_file, accuracy_training)
            
            if((epoch != 0)):
            # if(epoch != 0):

                if(len(val_infor[0]) > 0):
                    acc_val = run_valiate(args, sess, nrof_classes, val_infor,model_dir, subdir,epoch)
                    print (acc_val)
                    if (acc_val > good_acc):
                        good_acc = acc_val
                        ultils.save_variables_and_metagraph(sess, saver, model_dir, subdir, step)

            if (epoch % 10):
                max_nrof_epochs_new = load_max_epoch_from_file("./learning_schedule/max_epoch.txt")
                if (max_nrof_epochs_new != 0):
                    max_nrof_epochs = max_nrof_epochs_new


        print('test set before close session: ')
        sess.close()
        acc_test = test_phase.evaluate_models(model_dir,
                                              test_infor,
                                              batch_size=args.batch_size,
                                              image_size_w=args.image_size_w,
                                              image_size_h=args.image_size_h,
                                              color=args.color,
                                              nclasses=nrof_classes)
        return acc_test


def define_placehoder(args, range_size, nrof_heads):
    """Defines the placeholders needed for the TensorFlow graph.

    Args:
        args: Training configuration arguments.
        range_size: Size of the range for the input producer.
        nrof_heads: Number of output heads (for multi-task learning).

    Returns:
        A tuple containing the index queue, index dequeue operation, and various
        placeholders for learning rate, batch size, training phase, image paths,
        and labels.
    """

    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                shuffle=True, seed=None, capacity=32)

    index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size,
                                                'index_dequeue')  # lay 1 lan chung nay thang ....
        
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

    labels_placeholder = tf.placeholder(tf.int64, shape=(None, nrof_heads), name='labels')

    return index_queue, index_dequeue_op, learning_rate_placeholder, \
           batch_size_placeholder, phase_train_placeholder,\
           image_paths_placeholder, labels_placeholder



def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file,
          accuracy_training=None):
    """
    Performs one epoch of training.

    Args:
        args: Training configuration arguments.
        sess: TensorFlow session.
        epoch: Current epoch number.

    Returns:
        The global step after the epoch.
    """

    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = ultils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]

    image_epoch = np.array(image_list)[index_epoch]
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    labels_array = np.squeeze(labels_array, axis=1)

    image_paths_array = np.expand_dims(np.array(image_epoch), 1)

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array,
                          labels_placeholder: labels_array})

    # Training loop
    train_time = 0

    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr,
                     phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}

        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str, accuracy_tr = sess.run([loss,
                                                                         train_op,
                                                                         global_step,
                                                                         regularization_losses,
                                                                         summary_op,
                                                                         accuracy_training],
                                                                        feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss, accuracy_tr = sess.run([loss,
                                                            train_op,
                                                            global_step,
                                                            regularization_losses,
                                                            accuracy_training],
                                                           feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\tacc %2.3f\tlr %2.8f' % (epoch,
                                                                                                 batch_number+1,
                                                                                                 args.epoch_size,
                                                                                                 duration,
                                                                                                 err,
                                                                                                 np.sum(reg_loss),
                                                                                                 accuracy_tr,
                                                                                                 lr))

        batch_number += 1
        train_time += duration



    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step


def main(args,
         train_data,
         val_data,
         test_data,
         nrof_class_make=100,
         nrof_class_model=100):
    # Get the current date and time for directory naming
    today = datetime.datetime.now()
    subdir = today.strftime('%Y%m%d_%H%M%S')  # Format for timestamp
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    # Create log directory if it doesn't exist
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    
    # Create model directory if it doesn't exist
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Get the source path of the current script
    src_path, _ = os.path.split(os.path.realpath(__file__))

    # Set a random seed for reproducibility
    np.random.seed(seed=777)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    
    # Load a pre-trained model if specified
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    nrof_labels = 2  # Number of labels (make and model)
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)  # Set TensorFlow random seed
        global_step = tf.Variable(0, trainable=False)  # Variable to track the global step

        image_list, label_list = train_data[0], train_data[1]  # Unpack training data

        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)  # Convert labels to tensor
        range_size = array_ops.shape(labels)[0]  # Get the number of labels

        # Define placeholders for input data
        index_queue, index_dequeue_op, learning_rate_placeholder, \
        batch_size_placeholder, phase_train_placeholder, \
        image_paths_placeholder, labels_placeholder = define_placehoder(args, range_size, nrof_labels)

        # Create data flow for training
        enqueue_op, image_batch, label_batch = create_flow_data(args,
                                                                image_paths_placeholder,
                                                                nrof_labels,
                                                                labels_placeholder,
                                                                batch_size_placeholder)

        # Squeeze and set identities for image and label batches
        image_batch = tf.squeeze(image_batch, [1], name='image_batch')
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        print('label_batch:', label_batch)

        # Separate labels for make and model
        labels_make = label_batch[:, 0]
        labels_model = label_batch[:, 1]

        # Print class and example information
        print('Total number of classes - model: %d' % nrof_class_model)
        print('Total number of classes - make: %d' % nrof_class_make)
        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')

        # Build the inference graph using a neural network architecture
        prelogits = icpnv1.inference(image_batch,
                                     keep_probability=args.keep_probabilties,
                                     bottleneck_layer_size=args.embedding_size,
                                     phase_train=phase_train_placeholder,
                                     weight_decay=args.weight_decay)

        # Define the output layers for make and model classifications
        logits_make = slim.fully_connected(prelogits,
                                           nrof_class_make,
                                           activation_fn=None,
                                           scope='logits_make', reuse=False)

        logits_model = slim.fully_connected(prelogits,
                                            nrof_class_model,
                                            activation_fn=None,
                                            scope='logits_model', reuse=False)

        # Calculate the total number of parameters in the model
        total_number_parameter = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("total_number_parameter ", total_number_parameter)

        # Compute loss and regularization
        learning_rate, total_loss, regularization_losses = loss_compute(args,
                                                                        logits_make,
                                                                        logits_model,
                                                                        labels_make,
                                                                        labels_model,
                                                                        learning_rate_placeholder,
                                                                        nrof_class_make,
                                                                        nrof_class_model,
                                                                        global_step)

        # Compute training probabilities and accuracy
        probilitys_train = tf.nn.softmax(logits_model, name='softmax')

        cast_label = tf.cast(labels_model, tf.int64)
        argx_max_train = tf.cast(tf.argmax(probilitys_train, 1), tf.int64)
        argx_max_train = tf.identity(argx_max_train, 'predict')
        accuracy_training = tf.reduce_mean(tf.cast(tf.equal(argx_max_train, cast_label), tf.float32))

        # Configure saving and loading parameters for the model
        saver, saver_set_A, train_op, summary_op = config_load_and_save_params(learning_rate_placeholder,
                                                                             total_loss,
                                                                             global_step, scope_var="InceptionResnetV1")

        # Start training process and evaluate the model
        acc_test = training_on(args,
                               log_dir,
                               pretrained_model, saver_set_A,
                               image_list, label_list,
                               index_dequeue_op, enqueue_op,
                               image_paths_placeholder, labels_placeholder,
                               learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder,
                               global_step, total_loss, train_op, summary_op,
                               regularization_losses, saver, model_dir, subdir,
                               accuracy_training=accuracy_training,
                               val_infor=val_data,
                               test_infor=test_data,
                               nrof_classes=nrof_class_model)

    return acc_test, model_dir  # Return test accuracy and model directory





def create_mapping(labels, name_file, mapping = None):
    """Creates a mapping from string labels to integer indices and saves it to a file.

    Args:
        labels: A list or numpy array of string labels.
        name_file: The name of the file to save the mapping to.
        mapping: An optional list specifying a predefined mapping.

    Returns:
        A numpy array of integer labels corresponding to the input string labels.
    """

    mapping_logits = {}
    if mapping == None : 
        unique_label = np.unique(labels)
    else :
        unique_label = np.array(mapping)
    # print(unique_label)
    # print(len(np.unique(labels)))
    
    
    for i in range(len(unique_label)):
        mapping_logits[unique_label[i]] = i
    
    # write_mapping2file(mapping_logits, name_file)
    
    label_digit = []
    for i, lb in enumerate(labels):
        label_digit.append(mapping_logits[lb])
    return np.array(label_digit)


def get_image_in_mapping(df,mapping_file,isVal = False):
    """Filters a DataFrame based on a model mapping file.

    Args:
        df: The input DataFrame containing image paths and labels.
        mapping_file: Path to the JSON file containing the model mapping.
        isVal: Boolean indicating whether this is for validation.

    Returns:
        If isVal is False: A tuple of (image_lists, labels_make, labels_model, mapping_make_model).
        If isVal is True: A tuple of (image_lists, labels_model).
        If the mapping_file is invalid or causes an error, it returns the original unfiltered data
        along with mapping_make_model set to None.
    """

    ### Read csv #####
    image_lists = df['path'].values
    labels_make = df['labels_make'].values
    labels_model = df['labels_model'].values

    if mapping_file == None:
        mapping_make_model = None 
        return image_lists , labels_make , labels_model ,mapping_make_model 

    elif mapping_file != None:
        mapping_make_model = read_mapping_model(mapping_file)
        df = df[df['labels_model'].isin(mapping_make_model)]
        
        image_lists = df['path'].values

        labels_make = df['labels_make'].values
        labels_model = df['labels_model'].values


    if isVal == False :
        return image_lists , labels_make , labels_model ,mapping_make_model 
    if isVal == True :
        return image_lists , labels_model


def read_csv_train(args ,path_csv,mapping_file):
    """Reads training data from a CSV file, creates mappings, and prepares labels.

    Args:
        args: Training arguments (not used in the current implementation).
        path_csv: Path to the CSV file.
        mapping_file: Path to the model mapping file.

    Returns:
        A tuple containing:
        - image_lists: A NumPy array of image paths.
        - word: A NumPy array of combined make and model labels.
        - nrof_class_make: The number of unique make labels.
        - nrof_class_model: The number of unique model labels.
        Returns None if there's an issue with the mapping file.
    """

    df = pd.read_csv(path_csv, encoding='utf-8')

    image_lists , labels_make , labels_model ,mapping_make_model = get_image_in_mapping(df,mapping_file)
    labels_make = create_mapping(labels_make, "../make_synword.csv")
    labels_model = create_mapping(labels_model, "../model_synword.csv",mapping_make_model)
    
    nrof_class_make = len(np.unique(labels_make))
    nrof_class_model = len(np.unique(labels_model))

    labels_make = np.expand_dims(labels_make, axis=1)
    labels_model = np.expand_dims(labels_model, axis=1)

    word = np.concatenate((labels_make, labels_model), axis=1)
    
    return image_lists, word, nrof_class_make, nrof_class_model

def read_csv_val(path_csv, mapping_file):
    """Reads validation data from a CSV file.

    Args:
        path_csv: Path to the CSV file.
        mapping_file: Path to the model mapping file.

    Returns:
        A tuple containing:
        - image_lists: A NumPy array of image paths.
        - labels_model: A NumPy array of model labels.
        Returns None if there's an issue reading the CSV or mapping file.
    """
    try:
        df = pd.read_csv(path_csv, encoding='utf-8')
        image_lists, labels_model = get_image_in_mapping(df, mapping_file, isVal=True)
        return image_lists, labels_model
    except FileNotFoundError:
        print(f"Error: CSV file not found: {path_csv}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def train_classify(args):
    """Trains a classification model.

    Args:
        args: Training arguments.
    """

    train_data = read_csv_train(args, args.file_training, args.file_mapping_model)
    if train_data is None:
        print("Error: Failed to read training data. Exiting.")
        return  # Exit early if training data reading failed

    image_lists_train, label_train, nrof_class_make, nrof_class_model = train_data


    print('nrof_class_make, nrof_class_model:', nrof_class_make, nrof_class_model)

    val_data = read_csv_val(args.file_val, args.file_mapping_model)
    if val_data is None:
        print("Error: Failed to read validation data. Exiting.")
        return  # Exit early if validation data reading failed

    image_lists_val, label_val = val_data


    train_infor = (image_lists_train, label_train)
    val_infor = (image_lists_val, label_val)
    test_infor = ([], [])  # Empty test data for now

    args.nrof_class_make = nrof_class_make
    args.nrof_class_model = nrof_class_model

    # Call main training function (assuming it's defined elsewhere)
    acc_test, model_dir = main(args, train_infor, val_infor, test_infor,
                               nrof_class_make, nrof_class_model)



import argparse
import sys

def parse_arguments(argv):
    # Create an ArgumentParser object to handle command line arguments
    parser = argparse.ArgumentParser()

    # Add argument for the path to the validation file
    parser.add_argument('--path_validate', type=str, help='Path to the validation file', default='')

    # Add argument for the path to the training data file
    parser.add_argument('--file_training', type=str, help='Path to the training data file', default="train_mm.csv")

    # Add argument for the path to the validation data file
    parser.add_argument('--file_val', type=str, help='Path to the validation data file', default="val_mm.csv")

    # Add argument for the path to the model mapping file
    parser.add_argument('--file_mapping_model', type=str, help='Path to model mapping file', default='1.0.06102021.172738_Medium_MakeModel_US_EU.txt')

    # Add argument for the path to the make mapping file
    parser.add_argument('--file_mapping_make', type=str, help='Path to make mapping file', default='../make_synword.csv')

    # Add argument for the base directory for logs
    parser.add_argument('--logs_base_dir', type=str, default="../log")

    # Add argument for the base directory for models
    parser.add_argument('--models_base_dir', type=str, default="/media/tienthanh2/TRAIN_DATA/media/HoanHai/make_model/inception_v1_full_add_VN")

    # Add argument to specify the color mode (e.g., grayscale or color)
    parser.add_argument('--color', type=str, help='Specify color mode', default='color')

    # Add argument for using label smoothing
    parser.add_argument('--smooth_label', type=bool, help='Use label smoothing', default=False)

    # Add argument for GPU memory usage limit
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on GPU memory usage', default=0.5)

    # Add argument to specify a pre-trained model to load before training
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrained model before training starts', default="")

    # Add argument for maximum number of training epochs
    parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run', default=5000)

    # Add argument for batch size (number of images processed at a time)
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch', default=64)

    # Add arguments for image dimensions (width and height)
    parser.add_argument('--image_size_w', type=int, help='Image width in pixels', default=264)
    parser.add_argument('--image_size_h', type=int, help='Image height in pixels', default=180)

    # Add argument for the size of the embedding layer
    parser.add_argument('--embedding_size', type=int, help='Size of the embedding layer', default=1198)

    # Add argument for number of batches per epoch
    parser.add_argument('--epoch_size', type=int, help='Number of batches per epoch', default=1000)

    # Add argument for the bottleneck layer size
    parser.add_argument('--bottleneck_layer_size', type=int, help='Size of the bottleneck layer', default=128)

    # Add argument for keeping probabilities during dropout
    parser.add_argument('--keep_probabilties', type=float, help='Keep probability during dropout', default=0.8)

    # Add argument for random image cropping
    parser.add_argument('--random_crop', help='Enable random cropping of images', action='store_true')

    # Add argument for random horizontal flipping of images
    parser.add_argument('--random_flip', help='Enable random horizontal flipping of training images', action='store_true')

    # Add argument for random rotations of images
    parser.add_argument('--random_rotate', help='Enable random rotations of training images', action='store_true')

    # Add argument for L2 weight regularization
    parser.add_argument('--weight_decay', type=float, help='L2 weight regularization coefficient', default=5e-5)

    # Add argument for selecting the optimization algorithm
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM', 'SGD'],
                        help='The optimization algorithm to use', default='SGD')

    # Add argument for the initial learning rate
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. Set to -1 for learning rate schedule', default=-1)

    # Add argument for the number of epochs between learning rate decay
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay', default=100)

    # Add argument for the learning rate decay factor
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor', default=1.0)

    # Add argument for exponential decay tracking of parameters
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters', default=0.9999)

    # Add argument for random seed for reproducibility
    parser.add_argument('--seed', type=int, help='Random seed', default=777)

    # Add argument for number of preprocessing threads
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of threads for data loading and augmentation', default=4)

    # Add argument to enable logging of histograms in TensorBoard
    parser.add_argument('--log_histograms', help='Enable logging of weight/bias histograms in TensorBoard', action='store_true')

    # Add argument for the file containing the learning rate schedule
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule when learning_rate is set to -1', 
                        default='./learning_schedule/learning_schedule_1.txt')

    # Parse and return the command line arguments
    return parser.parse_args(argv)

if __name__ == '__main__':
    # Call the training function with parsed arguments from the command line
    train_classify(parse_arguments(sys.argv[1:]))