# coding: utf-8
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Concatenate
import numpy as np
import time
from scipy.misc import imread
import tensorflow as tf
import pickle
import keras
from ssd import SSD300
from SSD_RGBD3 import RGBD_SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from random import shuffle
from scipy.misc import imresize
import matplotlib.pyplot as plt
import cv2
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
from SSD_tester import calc_detection_prec_rec, calc_detection_ap
from keras.backend.tensorflow_backend import set_session
from depth_preprocess import hole_filling

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
 #       visible_device_list="0",
        allow_growth=True # True->必要になったら確保, False->全部
    )
)
set_session(tf.Session(config=config))

voc_classes = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser',
               'garbage_bin', 'lamp', 'monitor', 'night_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet']
NUM_CLASSES = len(voc_classes) + 1
rgb_input_shape = (300, 300, 3) #channel last
depth_input_shape = (300, 300, 1) #channel last
model = RGBD_SSD300(rgb_input_shape, depth_input_shape, num_classes=NUM_CLASSES)

priors = pickle.load(open('../pkls/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

rgb_gt = pickle.load(open('../pkls/SUNRGBD/RGB_v8.pkl', 'rb'))
depth_gt = pickle.load(open('../pkls/SUNRGBD/Depth_v8.pkl', 'rb'))
rgb_keys = sorted(rgb_gt.keys())
depth_keys = sorted(depth_gt.keys())
shuffle(rgb_keys)
shuffle(depth_keys)
num_train = int(round(0.8 * len(rgb_keys)))
rgb_train_keys = rgb_keys[:num_train]
depth_train_keys = depth_keys[:num_train]
num_val = int(round((len(rgb_keys) - num_train)/2))
rgb_val_keys = rgb_keys[num_train:]
rgb_val_keys = rgb_val_keys[:num_val]
depth_val_keys = depth_keys[num_train:]
depth_val_keys = depth_val_keys[:num_val]


with open('/data/jun/pkls/RGBD-3/rgb-v3.pkl','wb') as f:
    pickle.dump(rgb_keys, f)

with open('/data/jun/pkls/RGBD-3/depth-v3.pkl','wb') as f:
    pickle.dump(depth_keys, f)



class Generator(object):
    def __init__(self, rgb_gt, depth_gt, bbox_util,
                 batch_size, path_prefix,
                 rgb_train_keys, depth_train_keys, rgb_val_keys, depth_val_keys, rgb_image_size, depth_image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.rgb_gt = rgb_gt
        self.depth_gt = depth_gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.rgb_train_keys = np.array(rgb_train_keys)
        self.rgb_val_keys = np.array(rgb_val_keys)
        self.depth_train_keys = np.array(depth_train_keys)
        self.depth_val_keys = np.array(depth_val_keys)
        self.train_batches = len(rgb_train_keys)
        self.val_batches = len(rgb_val_keys)
        self.rgb_image_size = rgb_image_size
        self.depth_image_size = depth_image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, depth_img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            depth_img = depth_img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, depth_img, y
    
    def vertical_flip(self, img, depth_img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            depth_img = depth_img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, depth_img, y
    
    def random_sized_crop(self, img, depth_img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        depth_img = depth_img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, depth_img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                indices = np.random.permutation(len(self.rgb_train_keys))
                #shuffle(self.train_keys)
                rgb_keys = self.rgb_train_keys[indices]
                depth_keys = self.depth_train_keys[indices]
            else:
                indices = np.random.permutation(len(self.rgb_val_keys))
                #shuffle(self.train_keys)
                rgb_keys = self.rgb_val_keys[indices]
                depth_keys = self.depth_val_keys[indices]
                #shuffle(self.val_keys)
            rgb_inputs = []
            depth_inputs = []
            targets = []
            for rgb_key, depth_key in zip(rgb_keys, depth_keys):            
                rgb_img_path = self.path_prefix + rgb_key
                depth_img_path = self.path_prefix + depth_key
                rgb_img = imread(rgb_img_path).astype('float32')
                depth_img = imread(depth_img_path, mode='L').astype('float32')
                y = self.rgb_gt[rgb_key].copy()
                if train and self.do_crop:
                    rgb_img, depth_img, y = self.random_sized_crop(rgb_img, depth_img, y)
                rgb_img = imresize(rgb_img, self.rgb_image_size).astype('float32')
                depth_img = imresize(depth_img, self.depth_image_size).astype('float32')
                depth_img = depth_img / np.max(depth_img)
                depth_img = np.sqrt(depth_img)
                depth_img = np.array(depth_img*256, dtype=float)
                depth_img = hole_filling(depth_img)
                #depth_img = depth_img.astype('float32')
 #               depth_img = np.uint8(depth_img)
                #depth_img = cv2.Canny(depth_img, 70, 110)
                #depth_img = cv2.bilateralFilter(depth_img, d=5, sigmaColor=5, sigmaSpace=2)
                depth_img = np.expand_dims(depth_img, axis=2)
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        rgb_img = jitter(rgb_img)
                    if self.lighting_std:
                        rgb_img = self.lighting(rgb_img)
                    if self.hflip_prob > 0:
                        rgb_img, depth_img, y = self.horizontal_flip(rgb_img, depth_img, y)
                    if self.vflip_prob > 0:
                        rgb_img, depth_img, y = self.vertical_flip(rgb_img, depth_img, y)

                std = np.std(rgb_img)
                rgb_img -= np.mean(rgb_img)
                rgb_img /= std

                depth_std = np.std(depth_img)
                depth_img -= np.mean(depth_img)
                depth_img /= depth_std
                y = self.bbox_util.assign_boxes(y)
                rgb_inputs.append(rgb_img)
                depth_inputs.append(depth_img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_rgb_inp = np.array(rgb_inputs)
                    tmp_depth_inp = np.array(depth_inputs)
                    tmp_targets = np.array(targets)
                    #tmp_targets = np.concatenate((tmp_targets, tmp_targets), axis=1)
                    inputs = []
                    targets = []
                    yield [np.array(tmp_rgb_inp), np.array(tmp_depth_inp)], [tmp_targets]


path_prefix = '/data/jun/dataset/'
gen = Generator(rgb_gt, depth_gt, bbox_util, 16, path_prefix,
                rgb_train_keys,depth_train_keys, rgb_val_keys, depth_val_keys,
                (rgb_input_shape[0], rgb_input_shape[1]), (depth_input_shape[0], depth_input_shape[1]), do_crop=True)
names = ['train_loss', 'val_loss']
# In[9]:

#freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
#           'conv2_1', 'conv2_2', 'pool2']
#           'conv3_1', 'conv3_2', 'conv3_3', 'pool3']
# #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

#for L in model.layers:
#    if L.name in freeze:
#        L.trainable = False


# In[10]:


def write_log(callback, names, losses, batch_no):
    for name, value in zip(names, losses):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()



#base_lr = 1e-4 v13, v14 
base_lr = 1e-4
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss,)

log_path = '../tensor_log/estimation/RGBD-3/v3'
callback = TensorBoard(log_path)
callback.set_model(model)



nb_epoch = 100
batch_size = gen.batch_size
epoch_length  = gen.train_batches//gen.batch_size
#epoch_length  = 5
#epoch_length  = 1500
val_epoch_length = gen.val_batches//gen.batch_size
losses = np.zeros((epoch_length))
val_losses = np.zeros((val_epoch_length))
iter_num = 0
val_iter_num = 0
best_loss = np.Inf
best_val_loss = np.inf
start_time = time.time()
for epoch in range(nb_epoch):
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch + 1, nb_epoch))

    while True:
        X, Y = gen.generate(True).__next__()
        loss = model.train_on_batch(X, Y)
        losses[iter_num] = loss
        iter_num += 1
        progbar.update(iter_num, [('loss', np.mean(losses[:iter_num]))])

        if iter_num == epoch_length:
            results = []
            while True:
                val_X, val_Y  = gen.generate(False).__next__()
                val_loss = model.test_on_batch(val_X, val_Y)
                predict = model.predict_on_batch(val_X)
                results.append(predict)
                val_losses[val_iter_num] = val_loss
                val_iter_num += 1
                if val_epoch_length == val_iter_num:
                    val_iter_num = 0
                    break
            train_avg_loss = np.mean(losses)
            val_avg_loss = np.mean(val_losses)
            write_log(callback, names, [train_avg_loss, val_avg_loss], epoch)


            print('average loss: {train_loss:.4f}, validation loss: {val_loss:.4f}'.format(train_loss=train_avg_loss, val_loss=val_avg_loss))
            curr_loss = train_avg_loss
            curr_val_loss = val_avg_loss
            start_time = time.time()
            print('Elapsed time: {}'.format(time.time() - start_time))
            if curr_val_loss < best_val_loss:
                print('Total loss decreased from {} to {}, saving weights'.format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                model.save_weights('/data/jun/checkpoints/SUNRGBD/estimation/RGBD-3/v3/weights.{epoch:02d}-{val_loss:.2f}.hdf5'.format(epoch=epoch, val_loss=best_val_loss))
            iter_num = 0
            break


