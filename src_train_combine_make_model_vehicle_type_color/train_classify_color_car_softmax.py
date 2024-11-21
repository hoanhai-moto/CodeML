"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import sklearn.datasets as ds
import math
import csv
from tensorflow.python.ops import random_ops
from tensorflow.python import debug as tf_debug
import json
import pandas as pd
from sklearn.utils import shuffle
import cv2
from sklearn.metrics import classification_report
import multiprocessing
import tqdm
from pathlib import Path
import random



def rotate(images, angles, interpolation="BILINEAR", name=None):
    """Rotates image(s) counterclockwise by the given angle(s) in radians.

    This function rotates images by the given angles. It enlarges the image size to contain the entire rotated image.
    It also adds a random small horizontal offset.

    Args:
        images: A tensor of shape (num_images, height, width, channels)
            or (height, width, channels) or (height, width).
        angles: A scalar angle to rotate all images by, or a list of angles, one for each image.
        interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
        name: The name of the op.

    Returns:
        Image(s) with the same type and shape as `images`, rotated by the given
        angle(s).

    Raises:
        TypeError: If `image` is an invalid type.
        TypeError: If `angles` is an invalid type.
    """

    with ops.name_scope(name, "rotate"):
        image_or_images = ops.convert_to_tensor(images)
        if image_or_images.get_shape().ndims is None:
            raise TypeError("image_or_images rank must be statically known")
        elif len(image_or_images.get_shape()) == 2:
            images = image_or_images[None, :, :, None]
        elif len(image_or_images.get_shape()) == 3:
            images = image_or_images[None, :, :, :]
        elif len(image_or_images.get_shape()) == 4:
            images = image_or_images
        else:
            raise TypeError("Images should have rank between 2 and 4.")

        image_height = tf.cast(array_ops.shape(images)[1],
                               tf.float32)[None]
        image_width = tf.cast(array_ops.shape(images)[2],
                              tf.float32)[None]

        new_width = image_width * tf.cos(tf.abs(angles)) + image_height * tf.sin(tf.abs(angles))
        new_height = image_height * tf.cos(tf.abs(angles)) + image_width * tf.sin(tf.abs(angles))

        final_width = tf.cast(new_width, tf.int32)
        final_height = tf.cast(new_height, tf.int32)
        concat_hw = tf.concat([final_height, final_width], axis=0)

        with ops.name_scope(name, "angles_to_projective_transforms"):
            angle_or_angles = ops.convert_to_tensor(
                angles, name="angles", dtype=tf.float32)
            if len(angle_or_angles.get_shape()) == 0:
                angles = angle_or_angles[None]
            elif len(angle_or_angles.get_shape()) == 1:
                angles = angle_or_angles
            else:
                raise TypeError("Angles should have rank 0 or 1.")
            x_range = random_ops.random_uniform([], -15, 15)
            x_offset = (((new_width - 1) - (tf.cos(angles) *
                                            (new_width - 1) - tf.sin(angles) *
                                            (new_height - 1))) / 2.0) + (new_width * x_range * 0.01)
            y_offset = (((new_height - 1) - (tf.sin(angles) *
                                             (new_width - 1) + tf.cos(angles) *
                                             (new_height - 1))) / 2.0) - (new_height * tf.sin(tf.abs(angles)))
            num_angles = array_ops.shape(angles)[0]
            arr_rotate = array_ops.concat(
                values=[
                    tf.cos(angles)[:, None],
                    -tf.sin(angles)[:, None],
                    x_offset[:, None],
                    tf.sin(angles)[:, None],
                    tf.cos(angles)[:, None],
                    y_offset[:, None],
                    array_ops.zeros((num_angles, 2), tf.float32),
                ],
                axis=1)

        output = tf.contrib.image.transform(
            images,
            arr_rotate,
            output_shape=concat_hw,
            interpolation=interpolation)

        if image_or_images.get_shape().ndims is None:
            raise TypeError("image_or_images rank must be statically known")
        elif len(image_or_images.get_shape()) == 2:
            return output[0, :, :, 0]
        elif len(image_or_images.get_shape()) == 3:
            return output[0, :, :, :]
        else:
            return output


def combine_loss(embedding, labels, out_num, w_init=None, s=64., ma=1.0, mm=1.0, mb=0.0):
    """
    Combined Loss function for face recognition. This function implements a combination of different loss strategies
    including ArcFace, CosineFace, and additive margin variations.

    Args:
        embedding: The embedding tensor of shape (batch_size, embedding_size).
        labels: The ground truth labels of shape (batch_size,).
        out_num: The number of output classes.
        w_init: Initializer for the embedding weights.
        s: The scaling factor.
        ma: The multiplicative angular margin.
        mm: The additive angular margin.
        mb: The additive cosine margin.

    Returns:
        output: The final loss tensor.
        cos_t: The cosine similarity between embeddings and weights.
    """

    threshold = math.cos(math.pi - mm)
    with tf.variable_scope('arcface_loss', reuse=tf.AUTO_REUSE):
        # inputs and weights norm
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)

        weights_unit = tf.nn.l2_normalize(weights, axis=0)
        embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
        cos_t = tf.matmul(embedding_unit, weights_unit, name='cos_t')

        # cos_t = tf.matmul(embedding, weights, name='cos_t')

        if ma != 1.0 or mm != 0.0 or mb != 0.0:
            if ma == 1.0 and mm == 0.0:
                # cosine-face
                cos_mt = s * (cos_t - mb)
            else:
                t = tf.acos(cos_t)
                if ma != 1.0:
                    t = t * ma
                if mm > 0.0:
                    t = t + mm

                cos_mt = tf.cos(t)
                if mb > 0.0:
                    cos_mt = cos_mt - mb
                cos_mt = cos_mt * s

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s * (cos_t - mb)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(
            cos_mt_temp, mask), name='arcface_loss_output')
    return output, cos_t

def folder_structure(startpath,output_path,pretrained_model):
    """
    Generates a text file representing the folder structure of the training data.

    Args:
        startpath: The root directory of the training data.
        output_path: The directory to save the folder structure text file.
        pretrained_model: The name of the pretrained model used.
    
    Returns:
        list_folder_train: The string representation of the folder structure.
    """

    list_folder_train = ''
    list_folder_train += 'Pretrain model : '+pretrained_model +'\n' 
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        if level < 3 :
          list_folder_train += '{}{}/ \n'.format(indent, os.path.basename(root))         
        subindent = ' ' * 4 * (level + 1)
    with open(os.path.join(output_path,"FOLDER_TRAIN_STRUCTURE.txt"), "w") as text_file:
        text_file.write(list_folder_train)
    return list_folder_train 
       

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.8):
    """
    ArcFace loss function.

    Args:
        embedding: The embedding tensor of shape (batch_size, embedding_size).
        labels: The ground truth labels of shape (batch_size,).
        out_num: The number of output classes.
        w_init: Initializer for the embedding weights.
        s: The scaling factor.
        m: The additive angular margin.

    Returns:
        output: The final loss tensor.
    """

    cos_m = tf.cos(m)
    sin_m = tf.sin(m)
    mm = sin_m * m
    threshold = tf.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)

        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')

        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output



def read_lists(list_of_lists_file, is_shuffle=True):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    if is_shuffle:
        np.random.shuffle(listfile_labels)
    listfiles, labels = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels


def list_all_images(data_dir):
    images = []
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for file in files:
            if (file.split(".")[-1] in ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG"]):
                images.append(os.path.join(root, file))

    return images


def read_map_names(file_name):
    map_name = None
    with open(file_name, "r") as fr:
        map_name = json.load(fr)

    for key in map_name.keys():
        map_name[key] = int(map_name[key]) - 1

    return map_name


def gen_train_csv(data_list, map_names_file, outputs_dir):
    map_names = read_map_names(map_names_file)

    count_classes = map_names.copy()
    for k in count_classes.keys():
        count_classes[k] = 0

    total_samples = 0
    train = "path,label,type\n"
    rate = 10
    for data_dir in data_list:
        us_dir = os.path.join(data_dir, "US")
        mex_dir = os.path.join(data_dir, "MEX")
        can_dir = os.path.join(data_dir, "CAN")

        images_list_us = list_all_images(us_dir)
        images_list_mex = list_all_images(mex_dir)
        images_list_can = list_all_images(can_dir)

        count_classes["MX"] += len(images_list_mex)
        count_classes["CAN"] += len(images_list_can)

        for image_pth in images_list_us:
            cls = image_pth.split("/")[-2]
            if (cls in map_names.keys()):
                count_classes[cls] += 1
                if (total_samples % rate == 0):
                    train += image_pth + "," + str(map_names[cls]) + "," + "val" + "\n"
                else:
                    train += image_pth + "," + str(map_names[cls]) + "," + "train" + "\n"
                total_samples += 1

        for image_pth_mex in images_list_mex:
            if (total_samples % rate == 0):
                train += image_pth_mex + "," + "52" + "," + "val" + "\n"
            else:
                train += image_pth_mex + "," + "52" + "," + "train" + "\n"
            total_samples += 1

        for image_pth_can in images_list_can:
            if (total_samples % rate == 0):
                train += image_pth_can + "," + "53" + "," + "val" + "\n"
            else:
                train += image_pth_can + "," + "53" + "," + "train" + "\n"
            total_samples += 1

    train = train[:-1]
    with open(os.path.join(outputs_dir, "train.csv"), "w") as train_csv:
        train_csv.write(train)

    statistics = "state,num_images\n"
    for s, n in count_classes.items():
        statistics += s + "," + str(n) + "\n"

    statistics = statistics[:-1]
    with open(os.path.join(outputs_dir, "statistics_file.csv"), "w") as statistics_file:
        statistics_file.write(statistics)

    print("total samples for train: ", total_samples)
    return os.path.join(outputs_dir, "train.csv")


def gen_val_csv(data_list, map_names_file, outputs_dir):
    map_names = read_map_names(map_names_file)

    count_classes = map_names.copy()
    for k in count_classes.keys():
        count_classes[k] = 0

    total_samples = 0
    val = "path,label,type\n"
    rate = 7
    for data_dir in data_list:
        us_dir = os.path.join(data_dir, "US")
        mex_dir = os.path.join(data_dir, "MEX")
        can_dir = os.path.join(data_dir, "CAN")

        images_list_us = list_all_images(us_dir)
        images_list_mex = list_all_images(mex_dir)
        images_list_can = list_all_images(can_dir)

        count_classes["MX"] += len(images_list_mex)
        count_classes["CAN"] += len(images_list_can)

        for image_pth in images_list_us:
            cls = image_pth.split("/")[-2]
            if (cls in map_names.keys()):
                count_classes[cls] += 1
                val += image_pth + "," + str(map_names[cls]) + "," + "val" + "\n"
                total_samples += 1

        for image_pth_mex in images_list_mex:
            val += image_pth_mex + "," + "52" + "," + "val" + "\n"
            total_samples += 1

        for image_pth_can in images_list_can:
            val += image_pth_can + "," + "53" + "," + "val" + "\n"
            total_samples += 1

    val = val[:-1]
    with open(os.path.join(outputs_dir, "val.csv"), "w") as train_csv:
        train_csv.write(val)

    print("total samples for train: ", total_samples)
    return os.path.join(outputs_dir, "val.csv")


def get_id_of_data(type_data, type_labels):
    id = np.where(type_data == type_labels)[0]
    return id


def load_file_csv_noisetext(path_filecsv):
    df = pd.read_csv(path_filecsv, encoding='utf8')
    np.random.seed(777)
    df = shuffle(df)

    image_path = df['path'].values
    labels = df['label'].values
    type_data = df['type'].values

    image_path = np.array(image_path)
    labels = np.array(labels)
    type_data = np.array(type_data)

    return image_path, labels, type_data


def statis_number_samples_each_class(y_train, number_class):
    for i in range(number_class):
        idx = np.where(i == y_train)[0]
        print('class: ', i, " number samples: ", len(idx))


def get_data_train(filecsv_data):
    print('filecsv_data: ', filecsv_data)
    image_paths, labels, type_data = load_file_csv_noisetext(filecsv_data)
    n_classes = len(np.unique(np.array(labels)))
    print('number class: ', n_classes)

    train_id = get_id_of_data(type_data, "train")
    x_train_fold = image_paths[train_id]
    y_train_fold = labels[train_id]

    # statis_number_samples_each_class(y_train_fold, n_classes)

    test_id = get_id_of_data(type_data, "test")
    x_test_fold = image_paths[test_id]
    y_test_fold = labels[test_id]

    val_id = get_id_of_data(type_data, "val")
    x_val_fold = image_paths[val_id]
    y_val_fold = labels[val_id]

    y_train = y_train_fold
    y_valid = y_val_fold

    print("size train: ", len(x_train_fold))
    print("size validate: ", len(x_val_fold))
    print("size test: ", len(x_test_fold))

    train_infor = (x_train_fold, y_train)
    val_infor = (x_val_fold, y_valid)
    test_infor = (x_test_fold, y_test_fold)

    return train_infor, val_infor, test_infor

    # acc_test, model_dir = main(args, train_infor, val_infor, test_infor)


def get_data_val(filecsv_data):
    print('filecsv_data: ', filecsv_data)
    image_paths, labels, type_data = load_file_csv_noisetext(filecsv_data)
    n_classes = len(np.unique(np.array(labels)))
    print('number class: ', n_classes)

    val_id = get_id_of_data(type_data, "val")
    x_val_fold = image_paths[val_id]
    y_val_fold = labels[val_id]

    print("size validate: ", len(x_val_fold))
    val_infor = (x_val_fold, y_val_fold)

    return val_infor


def data_augumentation(image_aug, random_flip_left_right=True, random_brightness=True,
                       random_saturation=True, random_hue=True, random_contrast=True,
                       random_rotate=True):
    flag_condition = tf.random_uniform([2], 0, 2, dtype=tf.int32)
    angle = tf.random_uniform([1], -15, 15, dtype=tf.float32)


    #if(flag_condition[0] == 1):
    image_aug = tf.image.random_flip_left_right(image_aug)

    if (flag_condition[0] == 1):
        image_aug = tf.image.random_brightness(image_aug, max_delta=0.63)

    #if (flag_condition[2] == 1):
    #    image_aug = tf.image.random_saturation(image_aug, lower=0.75, upper=1.5)

    #if (flag_condition[3] == 1):
    #    image_aug = tf.image.random_hue(image_aug, max_delta=0.15)

    if (flag_condition[1] == 1):
        image_aug = tf.image.random_contrast(image_aug, lower=0.5, upper=1.8)

    #if (flag_condition[5] == 1):
    #    radian = angle * math.pi / 180
    #    image_aug = rotate(image_aug, radian)


    # if random_flip_left_right:
    #     image_aug = tf.cond(tf.equal(flag_condition[0], 1),
    #                         lambda: tf.image.random_flip_left_right(image_aug),
    #                         lambda: image_aug)

    # if random_brightness:
    #     image_aug = tf.cond(tf.equal(flag_condition[1], 1),
    #                         lambda: tf.image.random_brightness(image_aug, max_delta=0.1),
    #                         lambda: image_aug)
    # if random_saturation:
    #     image_aug = tf.cond(tf.equal(flag_condition[2], 1),
    #                         lambda: tf.image.random_saturation(image_aug, lower=0.75, upper=1.5),
    #                         lambda: image_aug)

    # if random_hue:
    #     image_aug = tf.cond(tf.equal(flag_condition[3], 1),
    #                         lambda: tf.image.random_hue(image_aug, max_delta=0.15),
    #                         lambda: image_aug)
    # if random_contrast:
    #     image_aug = tf.cond(tf.equal(flag_condition[4], 1),
    #                         lambda: tf.image.random_contrast(image_aug, lower=0.75, upper=1.5),
    #                         lambda: image_aug)
    # if random_rotate:
    #     random_angle = random_ops.random_uniform([], -15, 15)
    #     radian = random_angle * math.pi / 180
    #     image_aug = rotate(image_aug, radian)
        # image_aug = tf.cond(tf.equal(flag_condition[5], 1),
        #                     lambda: tf.contrib.image.rotate(image_aug, angle * math.pi / 180, interpolation='BILINEAR'),
        #                     lambda: image_aug)

    # image_aug = tf.image.per_image_standardization(image_aug)
    return image_aug

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[300, 300])

  return cropped_image

def create_flow_data(args, image_paths_placeholder, labels_placeholder, batch_size_placeholder, is_test=True):
    input_queue = data_flow_ops.FIFOQueue(capacity=1000000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(1,), (1,)],
                                          shared_name=None, name=None)

    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
    nrof_preprocess_threads = 4
    channel = 3
    

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        images_central_crop = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_jpeg(file_contents, channels=channel, dct_method='INTEGER_ACCURATE')
            image_central_crop=tf.image.central_crop(image, 0.5)
            #try:
            #   image_crop = random_crop(image)
            
            image_central_crop= tf.image.resize_images(image_central_crop, [args.image_size_h, args.image_size_w])
            #image_central_crop = data_augumentation(image_central_crop)
            image_central_crop= tf.image.per_image_standardization(image_central_crop)
            #   images.append(image_crop)

            #except:
            #   print('error')
            #   pass
            image = tf.image.resize_images(image, [args.image_size_h, args.image_size_w])
            

            #if(is_test == False):
            #image = data_augumentation(image)
            image = tf.image.per_image_standardization(image)
            #image = image/255
            #image_central_crop = image_central_crop/255

            images.append(image)
            images_central_crop.append(image_central_crop)
        images_and_labels.append([images, label])
        images_and_labels.append([images_central_crop, label])

    random.shuffle(images_and_labels)

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size_placeholder,
        shapes=[(args.image_size_h, args.image_size_w, channel), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * args.batch_size,
        allow_smaller_final_batch=True)

    return enqueue_op, image_batch, label_batch


def define_placehoder(args, range_size):
    index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                                                shuffle=True, seed=None, capacity=100)

    index_dequeue_op = index_queue.dequeue_many(args.batch_size * args.epoch_size, 'index_dequeue')

    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

    labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

    margin_placeholder = tf.placeholder(tf.float32, shape=(), name='margin')

    return index_queue, index_dequeue_op, learning_rate_placeholder, batch_size_placeholder, phase_train_placeholder, image_paths_placeholder, labels_placeholder, margin_placeholder
def list_all_img(data_dir):
    images = []
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for file in files:
            if(file.split(".")[-1] in ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG",'jfif']):
                images.append(os.path.join(root, file))

    return images

def read_folder_and_get_label(data_dir,lable_list):
    all_folder = os.listdir(data_dir)
    path_image = []
    images_and_labels = []
    for i, each_folder in enumerate(all_folder):
        all_imgs = list_all_img(os.path.join(data_dir, each_folder))
        for each_img in all_imgs:
            path_image.append(each_img)
            path_split=each_img.split('/')
            for path in path_split :
                if path in lable_list :
                    class_name = path
                    break 
                  
            # print(class_name)
            # print(lable_list.index(class_name))
          
            images_and_labels.append([each_img,lable_list.index(class_name)])

    random.shuffle(images_and_labels)
    path_image, labels = list(zip(*images_and_labels))
    path_image, labels = list(path_image), list(labels)
    return path_image, labels

    
def main(args):
    #train_csv_file = gen_train_csv(args.data_dir, args.map_names_file, args.outputs_dir)
   #train_data, val_data, test_data = get_data_train(train_csv_file)
    print("Start training model: ", args.model_def)
    network = importlib.import_module(args.model_def)
    
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    
    
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    Path(os.path.join(model_dir,"best_score")).mkdir(parents=True, exist_ok=True)
    list_folder_train = folder_structure(args.data_dir,model_dir,args.pretrained_model)
    # Write arguments to a text file
    # facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # datalist = ds.load_files(args.data_dir, load_content=False, shuffle=True)

        image_list, label_list = read_folder_and_get_label(args.data_dir,lable_list)
        test_image, label_test_list = read_folder_and_get_label(args.data_eval_dir,lable_list)
        nrof_classes = len(np.unique(np.array(label_list)))
        print('Size train : ',len(image_list))
        print('Size test : ',len(test_image))
        print('Num class : ',nrof_classes)

        # Create a queue that produces indices into the view_imgs_path and view_labels
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        args.num_class = nrof_classes
        
        # listfiles_train = datalist.filenames
        # labels_train = datalist.target

        index_queue, index_dequeue_op, learning_rate_placeholder, \
        batch_size_placeholder, phase_train_placeholder, \
        image_paths_placeholder, labels_placeholder, margin_placeholder = define_placehoder(args, range_size)

        enqueue_op, image_batch, label_batch = create_flow_data(args,
                                                                image_paths_placeholder,
                                                                labels_placeholder,
                                                                batch_size_placeholder)

        # enqueue_op_test, image_batch_test, label_batch_test = create_flow_data(args,
        #                                                                        image_paths_placeholder,
        #                                                                        labels_placeholder,
        #                                                                        batch_size_placeholder)

        tf.summary.image("CD", image_batch)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')
        # Build the inference graph

        prelogits, _ = network.inference(image_batch, keep_probability = args.keep_probability,
                                      phase_train=phase_train_placeholder,
                                      bottleneck_layer_size=512,
                                      weight_decay=args.weight_decay)

        logits = slim.fully_connected(prelogits, args.num_class, activation_fn=None,
                                    scope='classify_layer', reuse=False)


        # logits, cos_t = combine_loss(embedding=prelogits,
        #                       labels=label_batch,
        #                       w_init=slim.xavier_initializer(uniform=False),
        #                       out_num=args.num_class)
        print('#############PreLogit#############')
        print(prelogits)
        print('##################################')
        
        print('#############Logit#############')
        print(logits)
        print('##################################')

        # print('#############Cos_T#############')
        # print(cos_t)
        # print('##################################')


        for v in tf.trainable_variables():
            print (v)

        probilitys_train = tf.nn.softmax(logits, name='softmax')
        probilitys_train = tf.identity(probilitys_train, 'softmax_predict')

        cast_label = tf.cast(label_batch, tf.int64)
        argx_max_train = tf.cast(tf.argmax(probilitys_train, 1), tf.int64)
        argx_max_train = tf.identity(argx_max_train, 'predict')
        accuracy_training = tf.reduce_mean(tf.cast(tf.equal(argx_max_train, cast_label), tf.float32))

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)

        tf.summary.scalar('learning_rate', learning_rate)
        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                       logits=logits,
                                                                       name='cross_entropy_per_example')


        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, learning_rate,
                                 args.moving_average_decay, tf.global_variables(), args.log_histograms)

        set_A_vars = [v for v in tf.trainable_variables()
                      if "squeezenet" in v.name
                      and 'arcface_loss' not in v.name
                      and 'classify_layer' not in v.name]
                      # and 'squeezenet/Bottleneck' not in v.name]


        for layer in tf.trainable_variables():
           print(layer)     



        saver_set_A = tf.train.Saver(set_A_vars)
        # Create a saver
        saver = tf.train.Saver(max_to_keep=10)

        saver_best_score = tf.train.Saver(max_to_keep=None)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                log_device_placement=False,
                                                allow_soft_placement=True))

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        # tf.train.write_graph(sess.graph, log_dir, "model.pbtxt")

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                # saver.restore(sess, pretrained_model)
                saver_set_A.restore(sess, args.pretrained_model)

            # Training and validation loop
            print('Running training')
            print('Total parameter: ', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
            epoch = 0
            best_acc = 0.0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size


                # Train for one epoch

                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op,
                      image_paths_placeholder, labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses,
                      args.learning_rate_schedule_file, accuracy_training)

                    # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                #if epoch % 10 == 0 :
                
                val_acc,report,silver,gray = run_validate_arcloss(args, sess, nrof_classes)
                #print("recall ",recall_plate)
                # val_acc = run_validate(args, sess, nrof_classes, val_data)
                if val_acc > 0.7 and silver < 0.3 and gray < 0.3:

                   best_acc = val_acc
                   with open(os.path.join(os.path.join(model_dir,"best_score"), f"report_classify_state_{epoch}.txt"), "w") as file_report:
                         file_report.write(report)
                                          
                   save_variables_and_metagraph(sess,saver_best_score, summary_writer, os.path.join(model_dir,"best_score"), subdir, step)

                
                with open(os.path.join(model_dir, f"report_classify_state_{epoch}.txt"), "w") as file_report:
                         file_report.write(report)

                save_variables_and_metagraph(sess, saver, summary_writer,model_dir, subdir, step)

    return model_dir


def train(args, sess, epoch, listfiles_train, labels_train, index_dequeue_op, enqueue_op, image_paths_placeholder,
          labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file,
          accuracy_training):
    batch_number = 0
    step = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)


    index_epoch = sess.run(index_dequeue_op)
    print('index_epoch ', len(index_epoch))
    image_epoch = np.array(listfiles_train)[index_epoch]
    label_epoch = np.array(labels_train)[index_epoch]
    # Enqueue one epoch of image paths and labels
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        if batch_number % 100 == 0:
            err, _, step, reg_loss, summary_str, acc = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op, accuracy_training], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss, acc = sess.run([loss, train_op, global_step, regularization_losses, accuracy_training], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\tLR %.8f\tacc %.3f' %
              (epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss), lr, acc))
        batch_number += 1
        train_time += duration
    return step


def result_percent(writer, result_arr, step):
    error_rates = [0.1, 0.5, 1.0, 100.0]

    total_test = len(result_arr)
    true_arr = sorted([x for x in result_arr if x[0] == 1], key=lambda x: x[1], reverse=True)
    false_arr = sorted([x for x in result_arr if x[0] == 0], key=lambda x: x[1], reverse=True)
    threshold_arr = ['threshold']
    per_true_arr = ['%True']
    num_true_arr = ['True']
    num_false_arr = ['False']
    per_false_arr = ['%False']
    per_false_real_arr = ['%FalseReal']

    for error_rate in error_rates:
        num_fasle = int(error_rate * 0.01 * total_test)
        num_false_arr.append(num_fasle)
        per_false_real_arr.append(num_fasle / total_test)
        threshold = false_arr[:num_fasle][-1][1]
        num_true = len([x for x in true_arr if x[1] >= threshold])
        threshold_arr.append(threshold)
        num_true_arr.append(num_true)
        per_true_arr.append(num_true / total_test)

    per_false_arr.extend([str(x) for x in error_rates])
    writer.writerow('')
    writer.writerow([step])
    writer.writerow(per_false_arr)
    writer.writerow(per_false_real_arr)
    writer.writerow(num_false_arr)
    writer.writerow(threshold_arr)
    writer.writerow(per_true_arr)
    writer.writerow(num_true_arr)


def test(args, sess, image_batch, enqueue_op_test, image_paths_placeholder, labels_placeholder,
         phase_train_placeholder, batch_size_placeholder, softmax, step, log_dir):

    score_arr = []
    MAX_BATCH_SZ = args.batch_size


    listfiles_test, labels_test = read_folder_and_get_label(args.data_eval_dir,lable_list)

    total_image = len(listfiles_test)

    num_batches = int(math.ceil(len(listfiles_test) / MAX_BATCH_SZ))
    view_imgs_path = np.expand_dims(np.array(listfiles_test), 1)
    labels_array = np.expand_dims(np.array(labels_test), 1)
    sess.run(enqueue_op_test, {image_paths_placeholder: view_imgs_path, labels_placeholder: labels_array})

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    pbar = tqdm.tqdm(total=total_image, desc="extract features")
    predict = []
    print('Run validation...')
    for i in range(num_batches):
        start_offset = i * MAX_BATCH_SZ
        end_offset = min((i + 1) * MAX_BATCH_SZ, len(view_imgs_path))
        # image_input = sess.run(image_batch, feed_dict={batch_size_placeholder: end_offset - start_offset})
        feed_dict = {phase_train_placeholder: False,
                     batch_size_placeholder: args.batch_size}

        batch_results = sess.run(softmax, feed_dict=feed_dict)
        for item in batch_results:
            predict.append(item)
            pbar.update(1)
    pbar.close()

    predict = np.array(predict)
    predict = np.argmax(predict, axis=1).astype(np.int16)
    predict = predict.reshape(-1)

    ground_truth = val_data[1]
    ground_truth = np.array(ground_truth, dtype=np.int16)
    ground_truth = ground_truth.reshape(-1)

    state_names = read_map_names(args.map_names_file)
    dict_state_name = {}
    for k, v in state_names.items():
        dict_state_name[str(v)] = k

    labels_name = []
    for i in range(54):
        labels_name.append(dict_state_name[str(i)])
    
    predict1 = predict[predict != 7] 
    predict1 = predict[predict != 0]
    ground_truth1 = ground_truth[ground_truth!= 7] 
    ground_truth1 = ground_truth[ground_truth!= 0]
    reports = classification_report(ground_truth, predict, labels=np.arange(args.num_class), target_names=labels_name)
    with open(os.path.join(args.outputs_dir, "report_classify_state.txt"), "w") as file_report:
        file_report.write(reports)

    print(reports)

    return np.sum(ground_truth1 == predict1) / len(predict1)



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def load_data(image_paths, image_width, image_height, color):
    nrof_samples = len(image_paths)
    channel = 3
    if (color != 'color'):
        channel = 1
    images = np.zeros((nrof_samples, image_height, image_width, channel))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        # print (image_paths[i])
        img_rgb = cv2.resize(img, (image_width, image_height))

        if (channel == 1):
            img_rgb = np.mean(img_rgb, axis=-1)
            img_rgb = np.expand_dims(img_rgb, axis=-1)

        img_rgb = prewhiten(img_rgb)
        # img_rgb = img_rgb / 255.0
        images[i, :, :, :] = img_rgb
    return images

def decode_img(img_path, label):
    file_contents = tf.read_file(img_path)
    image = tf.image.decode_jpeg(file_contents, channels=3, dct_method="INTEGER_ACCURATE")
    return image, label

def normalization(image, label, w, h):
    #image=tf.image.central_crop(image, 0.6)
    image = tf.image.resize_images(image, [h, w])
    image = tf.image.per_image_standardization(image)
    #image = image/255
    return image, label

def get_trainer_preproc(args, image_paths, image_labels, is_training=True, prefetch_size=8):
    tf_data_thread_num = multiprocessing.cpu_count()
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    # if is_training:
    #     dataset = dataset.shuffle(len(image_paths))
    #     dataset = dataset.repeat(args.max_nrof_epochs)
    dataset = dataset.map(map_func=lambda x,y: decode_img(x, y), num_parallel_calls=tf_data_thread_num)
    dataset = dataset.map(map_func=lambda x,y: normalization(x, y, args.image_size_w, args.image_size_h),
                                    num_parallel_calls=tf_data_thread_num)
    dataset = dataset.batch(args.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def run_validate_arcloss(args, sess, nclass):

    print("starting calc on validation data")

    image_dirs ,image_labels = read_folder_and_get_label(args.data_eval_dir ,lable_list )
    total_image = len(image_dirs)
    val_dataset = get_trainer_preproc(args, image_dirs, image_labels, is_training=False)
    val_iterator = val_dataset.make_initializable_iterator()
    image_batch, _ = val_iterator.get_next()
    graph = tf.get_default_graph()
    input_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    output_placeholder = tf.get_default_graph().get_tensor_by_name("softmax:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")

    sess.run(val_iterator.initializer)
    pbar = tqdm.tqdm(total=total_image, desc="extract features")
    predict = []
    total_batch = int(math.ceil(total_image / args.batch_size))
    for batch_num in range(total_batch):
        image_np = sess.run(image_batch)
        feed_dict = {
            input_placeholder: image_np,
            phase_train_placeholder: False,
            # batch_size_placeholder: args.batch_size
        }
        batch_embs = sess.run(output_placeholder, feed_dict=feed_dict)
        for item in batch_embs:
            predict.append(item)
            pbar.update(1)
    pbar.close()


    predict = np.array(predict)
    predict = np.argmax(predict, axis=1).astype(np.int16)
    predict = predict.reshape(-1)
    
    ground_truth = image_labels
    
    ground_truth = np.array(ground_truth, dtype=np.int16)
    ground_truth = ground_truth.reshape(-1)
    

    
    reports = classification_report(ground_truth, predict,target_names=lable_list)
    with open(os.path.join(args.outputs_dir, "report_classify_state.txt"), "w") as file_report:
            file_report.write(reports)
        
    silver = classification_report(ground_truth, predict,target_names=lable_list , output_dict = True)['silver']['recall']
        
    gray = classification_report(ground_truth, predict,target_names=lable_list, output_dict = True)['gray']['recall']
    
    print(reports)

    return np.sum(ground_truth == predict) / len(predict),reports,silver,gray


def run_validate(args, sess, nclass, val_data):
    val_csv_file = gen_val_csv(args.data_val_dir, args.map_names_file, args.outputs_dir)
    val_data = get_data_val(val_csv_file)

    print("starting calc on validation data")
    image_dirs = val_data[0]
    total_image = len(image_dirs)
    image_labels = val_data[1]
    val_dataset = get_trainer_preproc(args, image_dirs, image_labels, is_training=False)
    val_iterator = val_dataset.make_initializable_iterator()
    image_batch, _ = val_iterator.get_next()
    graph = tf.get_default_graph()
    input_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    output_placeholder = tf.get_default_graph().get_tensor_by_name("softmax:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # batch_size_placeholder = tf.get_default_graph().get_tensor_by_name("batch_size:0")

    sess.run(val_iterator.initializer)
    pbar = tqdm.tqdm(total=total_image, desc="extract features")
    predict = []
    total_batch = int(math.ceil(total_image / args.batch_size))
    for batch_num in range(total_batch):
        image_np = sess.run(image_batch)
        feed_dict = {
            input_placeholder: image_np,
            phase_train_placeholder: False,
            # batch_size_placeholder: args.batch_size
        }
        batch_embs = sess.run(output_placeholder, feed_dict=feed_dict)
        for item in batch_embs:
            predict.append(item)
            pbar.update(1)
    pbar.close()



    predict = np.array(predict)
    predict = np.argmax(predict, axis=1).astype(np.int16)
    predict = predict.reshape(-1)

    ground_truth = val_data[1]
    ground_truth = np.array(ground_truth, dtype=np.int16)
    ground_truth = ground_truth.reshape(-1)

    state_names = read_map_names(args.map_names_file)
    dict_state_name = {}
    for k, v in state_names.items():
        dict_state_name[str(v)] = k

    labels_name = []
    for i in range(54):
        labels_name.append(dict_state_name[str(i)])

    reports = classification_report(ground_truth, predict, labels=np.arange(args.num_class), target_names=labels_name)
    with open(os.path.join(args.outputs_dir, "report_classify_state.txt"), "w") as file_report:
        file_report.write(reports)

    print (reports)

    return np.sum(ground_truth == predict) / len(predict)


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


class train_params:
    def __init__(self):
        self.logs_base_dir = '~/logs/bullet'
        self.models_base_dir = '~/models/bullet'
        self.gpu_memory_fraction = 0.8
        # Load a pretrained model before training starts.
        self.pretrained_model = ''
        # Path to the data directory containing aligned face patches. Multiple directories are separated with colon.
        self.data_dir = '~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned'
        self.data_val_dir = ""
        self.map_names_file = ""
        self.outputs_dir = ""
        self.image_size_w = 192
        self.image_size_h = 48
        self.color = "gray"
        self.num_class = 54
        self.margin_schedule_file = ""
        # Model definition. Points to a module containing the definition of the inference graph.
        self.model_def = 'models.nn4'
        # Number of epochs to run.
        self.max_nrof_epochs = 5000
        # Number of images to process in a batch.
        self.batch_size = 90
        # Number of batches per epoch.
        self.epoch_size = 1000
        # Dimensionality of the embedding.
        self.embedding_size = 128
        # Performs random cropping of training images. If false, the center image_size pixels from the training images are used.
        # If the size of the images in the data directory is equal to image_size no cropping is performed
        self.random_crop = True
        # Performs random horizontal flipping of training images.
        self.random_flip = True
        # Performs random rotations of training images.
        self.random_rotate = False
        # Keep probability of dropout for the fully connected layer(s).
        self.keep_probability = 1.0
        # L2 weight regularization.
        self.weight_decay = 0.0
        # The optimization algorithm to use
        # choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']
        self.optimizer = 'ADAM'
        # Initial learning rate. If set to a negative value a learning rate
        # schedule can be specified in the file "learning_rate_schedule.txt"
        self.learning_rate = 0.1
        # Number of epochs between learning rate decay.
        self.learning_rate_decay_epochs = 10
        # Learning rate decay factor.
        self.learning_rate_decay_factor = 0.9
        # Exponential decay for tracking of training parameters.
        self.moving_average_decay = 0.9999
        # Random seed.
        self.seed = 666
        # Number of preprocessing (data loading and augumentation) threads.
        self.nrof_preprocess_threads = 4
        # Enables logging of weight/bias histograms in tensorboard.
        self.log_histograms = 'store_true'
        # File containing the learning rate schedule that is used when learning_rate is set to to -1.
        self.learning_rate_schedule_file = 'data/learning_rate_schedule.txt'
        # File containing image data used for dataset filtering


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/bullet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/bullet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
                        default='')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.mv_inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float, help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


lable_list = ['gray','black','white','red','blue','yellow','green','silver','orange','brown']

args = train_params()

args.logs_base_dir = "/home/bahy/HoanHai/src_classification/log/Data_US_State"
args.models_base_dir = "/home/bahy/HoanHai/color_car/20210826_color_car_squeezenet_attention"
args.data_dir = "/media/bahy/data_ssd/bahy/color/20210825_train_color/train"  # input
#args.data_eval_dir = "/media/bahy/data_ssd/bahy/licenplate_noise/20210728_license_noise_classify_test/20210723_11h17_noise_plate_test"
args.data_eval_dir = "/media/bahy/data_ssd/bahy/color/20210825_train_color/val"
args.outputs_dir = "/home/bahy/HoanHai/src_classification/CD_State/Car_Plate_classify/train_lincense_plate_update/src_train_state"
#args.pretrained_model ='/home/bahy/HoanHai/color_car/20210826_color_car_squeezenet_10_merge_silver_gray/20210928-170019/best_score/model-20210928-170019.ckpt-4000'
args.image_size_w = 264
args.image_size_h = 180
args.model_def = "models.squeezenet_10_attention"
args.optimizer = "ADAM"
args.learning_rate = -1
args.max_nrof_epochs = 5000
args.keep_probability = 0.7
args.learning_rate_schedule_file = "/media/bahy1804/DATA_TRAIN/HoanHai/20211027_src_train_color/Color_classify/data/learning_rate_schedule_custom.txt"
args.weight_decay = 5e-4
args.batch_size = 64
args.epoch_size = 1000
#from pathlib import Path
#Path(args.models_base_dir).mkdir(parents=True, exist_ok=True)
#Path(args.logs_base_dir).mkdir(parents=True, exist_ok=True)
args.color = 'color'
main(args)
