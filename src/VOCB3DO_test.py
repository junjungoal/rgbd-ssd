
# coding: utf-8

# ### Libraries 

# In[11]:

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
import matplotlib.pyplot as plt
from SSD_tester import calc_detection_prec_rec, calc_detection_ap


np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


# In[2]:

voc_classes = ['door_handle', 'book', 'mouse', 'pillow', 'bowl', 'phone', 'speaker', 'stapler', 'table', 
               'cup', 'monitor', 'keyboard', 'letter_tray', 'remote', 'paper_notebook', 'power_outlet', 
               'tabple_knife', 'soap', 'bookcase','chair', 'scissors', 'sofa', 'bottle']
NUM_CLASSES = len(voc_classes) + 1
input_shape = (300, 300, 3) #channel last


# ### Load the weight from the hdf5

# In[3]:

model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('../checkpoints/VOCB3DO/weights.52-3.87.hdf5', by_name=True)


# In[4]:

gt = pickle.load(open('../pkls/VOCB3DO.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

path_prefix = '../dataset/VOCB3DO/KinectColor/'
inputs = []
images = []
val_keys = sorted(val_keys)
for key in val_keys:
    img_path = path_prefix + key
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))


# In[5]:

#print(inputs.shape)


# In[26]:

priors = pickle.load(open('../pkls/prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)
preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)
results = np.array(results)


# In[27]:

gt_bboxes = []
gt_labels = []
for key in val_keys:
    index = np.where(gt[key][:, 4:] == 1)
    gt_bboxes.append(gt[key][:, :4])
    gt_labels.append(index[1].reshape(len(index[1]), 1))
gt_bboxes = np.array(gt_bboxes)
gt_labels = np.array(gt_labels)


# In[28]:

pred_labels = results[:, :, :1]
pred_scores = results[:, :, 1:2]
pred_bboxes = results[:, :, 2:]


# In[25]:




# In[10]:

prec, rec = calc_detection_prec_rec(pred_labels, pred_scores, pred_bboxes, gt_bboxes, gt_labels)


# In[15]:

ap = calc_detection_ap(prec, rec)
{'ap': ap, 'map': np.nanmean(ap)}


# In[ ]:

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, NUM_CLASSES)).tolist()
    print(img.shape)
    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * 300))
        ymin = int(round(top_ymin[i] * 300))
        xmax = int(round(top_xmax[i] * 300))
        ymax = int(round(top_ymax[i] * 300))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



