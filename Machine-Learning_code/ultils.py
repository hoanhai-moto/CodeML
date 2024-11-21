import numpy as np
import pandas as pd
import cv2
import time
import os
from sklearn.utils import shuffle



mask = [[1, 2, 4],
        [128, 0, 8],
        [64, 32, 16]]
mask = np.array(mask)

def local_mean_binary_pattern(image):
    h, w  = image.shape
    padding_image = np.zeros((h + 2, w + 2))
    padding_image[1:h+1, 1:w+1] = image

    lmbp = np.zeros((h + 2, w + 2))
    for j in range(1, h):
        for i in range(1, w):
            mean_block = padding_image[j - 1: j + 2, i - 1: i + 2]
            mean_block = (mean_block > np.mean(mean_block)) * 1
            lmbp[j][i] = np.sum(mask * mean_block)

    hist, bins = np.histogram(lmbp.ravel(), 256, [0, 256])
    return hist


def save_variables_and_metagraph(sess, saver, model_dir, model_name, step):
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



def get_learning_rate_from_file(filename, epoch):
    all_line = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                all_line.append(line)
    learning_rate = -1
    for ln in all_line:
        par = ln.strip().split(':')
        e = int(par[0])
        lr = float(par[1])
        if e == epoch:
            learning_rate = lr

    if(learning_rate == -1):
        for ln in all_line:
            par = ln.strip().split(':')
            e = int(par[0])
            lr = float(par[1])
            if e <= epoch:
                learning_rate = lr

    return learning_rate

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def load_data(image_paths, image_width, image_height, color):
    nrof_samples = len(image_paths)
    channel = 3
    if(color != 'color'):
        channel = 1
    images = np.zeros((nrof_samples, image_height, image_width, channel))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        try :
            img_rgb = cv2.resize(img, (image_width, image_height))
        except:
            print("ERROR in image :" , image_paths[i])
            continue

        if(channel == 1):
            img_rgb = np.mean(img_rgb, axis=-1)
            img_rgb = np.expand_dims(img_rgb, axis=-1)

        img_rgb = prewhiten(img_rgb)
        #img_rgb = img_rgb/255.0
        images[i, :, :, :] = img_rgb
    return images

def make_fivecrop(img, size_crop):
    all_img = []
    img_resze = cv2.resize(img,(96, 96))
    h, w, c = img_resze.shape
    top_left = img_resze[: size_crop, : size_crop]
    all_img.append(top_left)


    top_right = img_resze[: size_crop, w - size_crop:]
    all_img.append(top_right)


    bottom_left = img_resze[h - size_crop:, : size_crop ]
    all_img.append(bottom_left)

    bottom_right = img_resze[h - size_crop: , w - size_crop:]
    all_img.append(bottom_right)

    ycenter = int(h /2)
    xcenter = int(w / 2)
    center_img = img_resze[ycenter - int(size_crop/2) : ycenter + int(size_crop /2),
                   xcenter - int(size_crop/2) : xcenter + int(size_crop /2)]
    all_img.append(center_img)
    return all_img


def load_data_tencrop(image_paths, image_width, image_height):
    images = np.zeros((11, image_height, image_width, 3))
    img = cv2.imread(image_paths, cv2.IMREAD_COLOR)
    all_img = make_fivecrop(img, size_crop = image_width)
    img_flip = cv2.flip(img, 1)
    all_img += make_fivecrop(img_flip, size_crop=image_width)
    img_rgb = cv2.resize(img, (image_width, image_height))
    all_img.append(img_rgb)

    for i in range(len(all_img)):
        img_rgb = prewhiten(all_img[i])
        images[i, :, :, :] = img_rgb

    return images


def load_data_gray(image_paths, image_width, image_height):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_height, image_width, 1))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
        img_rgb = cv2.resize(img, (image_width, image_height))

        img_rgb = np.mean(img_rgb, axis=2)
        img_rgb = np.expand_dims(img_rgb, axis=2)

        img_rgb = prewhiten(img_rgb)
        images[i, :, :, :] = img_rgb
    return images

def load_file_csv(path_filecsv):
    df = pd.read_csv(path_filecsv, encoding='utf8')
    df = shuffle(df)
    image_path = df['path'].values
    labels = df['label'].values
    image_path = np.array(image_path)
    labels = np.array(labels)
    return image_path, labels

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def load_and_resize(folder_in, folder_out):
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    all_class = os.listdir(folder_in)

    f = open(folder_out + "/" + "metadata.csv", 'w')
    f.write('path,label\n')


    for i in range(len(all_class)):
        path_read_img = os.path.join(folder_in, all_class[i])
        all_img = os.listdir(path_read_img)

        path_save_img = os.path.join(folder_out, all_class[i])
        if not os.path.exists(path_save_img):
            os.mkdir(path_save_img)


        for j in range(len(all_img)):
            img = cv2.imread(os.path.join(path_read_img, all_img[j]))
            img = cv2.resize(img, (32, 32))

            cv2.imwrite(os.path.join(path_save_img, all_img[j]), img)
            f.write(os.path.join(path_save_img, all_img[j]) + "," + str(i) + "\n")
    f.close()

