

# coding: utf-8

# In[1]:

from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Concatenate
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import pickle
import keras
from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from random import shuffle
from scipy.misc import imresize
import matplotlib.pyplot as plt
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
  #      visible_device_list="0",
        allow_growth=True # True->必要になったら確保, False->全部
    )
)
set_session(tf.Session(config=config))




# In[2]:

# some constants
#voc_classes = ['bed', 'shelf', 'computer', 'plate', 'lamp', 'garbage_bin', 'sofa', 'sofa_chair', 'pillow',
#               'box', 'table', 'projector', 'door', 'monitor']
#voc_classes = ['bed', 'shelf', 'computer', 'plate', 'lamp', 'garbage_bin', 'sofa', 'chair', 'microwave',
#               'box', 'table', 'projector', 'door', 'monitor', 'piano', 'person', 'bench','stool', 'cup', 'pillow', 'recycle_bin', 'podium', 'window', 'cabinet', 'dresser', 'mirror', 'washer', 'container']

#voc_classes = ['bed', 'shelf', 'computer', 'plate', 'lamp', 'garbage_bin', 'sofa', 'chair', 'microwave',
#               'box', 'table', 'projector', 'door', 'monitor', 'person', 'bench', 'cup', 'pillow', 'recycle_bin', 'window', 'mirror']
voc_classes = ['bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser',
               'garbage_bin', 'lamp', 'monitor', 'night_stand', 'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet']
print(voc_classes)
NUM_CLASSES = len(voc_classes) + 1
input_shape = (300, 300, 3) #channel last
print('V5')
print('based on VGG16 pretrained model')


# In[3]:

# priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('../weights_SSD300.hdf5', by_name=True)
print(model.summary())
#model.load_weights('../checkpoints/SUNRGBD/weights-v3.16-3.50.hdf5')
#model.load_weights('../checkpoints/VOCB3DO/weights-v4.00-3.54.hdf5', by_name=True)
#model.load_weights('../checkpoints/vocdevkit/weights.14-2.02.hdf5', by_name=True)

priors = pickle.load(open('../pkls/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


# In[4]:

#model3= Concatenate([model, model2])


# In[5]:

gt = pickle.load(open('../pkls/SUNRGBD/RGB_v8.pkl', 'rb'))

keys = sorted(gt.keys())
#shuffle(keys)
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
num_val = int(round((len(keys) - num_train)/2))
val_keys = keys[num_train:]
val_keys = val_keys[:num_val]

#with open('/data/jun/pkls/RGB/v7.pkl','wb') as f:
#    pickle.dump(keys, f)


# In[6]:



# In[7]:

class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
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
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
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
        return img, new_targets
    
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield (preprocess_input(tmp_inp), tmp_targets)
                    


# In[8]:

path_prefix = '/data/jun/dataset/'
gen = Generator(gt, bbox_util, 32, path_prefix,
        train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=True)


# In[9]:


# In[10]:

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

#tb_cb = keras.callbacks.TensorBoard(log_dir="../tensor_log/estimation/RGB/v7/")
#callbacks = [keras.callbacks.ModelCheckpoint('/data/jun/checkpoints/SUNRGBD/estimation/prec/RGB/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#                                             verbose=1,
#                                            save_best_only=True,
#                                             save_weights_only=True),
#             keras.callbacks.LearningRateScheduler(schedule)]
            # ,tb_cb]


# In[11]:


base_lr = 4e-4
optim = keras.optimizers.Adam(lr=base_lr)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

# In[ ]:

nb_epoch = 100
history = model.fit_generator(gen.generate(True), gen.train_batches//gen.batch_size,
                              nb_epoch, verbose=1,
                              #callbacks=callbacks,
                              validation_data=gen.generate(False),
                              validation_steps=gen.batch_size,
                              workers=1)

