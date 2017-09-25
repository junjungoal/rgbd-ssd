import scipy.io
import numpy as np
import argparse

class MatPreprocessor(object):
    def __init__(self, mat_path, dataset_path):
        self.mat_path = mat_path
        self.dataset_path = dataset_path
        self.num_classes = 19
        self.rgb_data = dict()
        self.depth_data = dict()
        self.name_list = []
        self._preprocess_mat()

    def _preprocess_mat(self):
        width = 567
        height = 427
        matdata = scipy.io.loadmat(self.mat_path)
        bboxes_data = matdata['SUNRGBDMeta2DBB'][0]
        for data in bboxes_data:
            bounding_boxes = []
            one_hot_classes = []
            path = self.dataset_path + data[0][0]
            if len(data[1]) != 0:
                for klass in data[1][0]:
                    for bbox in klass[1]:
                        xmin = bbox[0] / width
                        ymin = bbox[1] / height
                        xmax = (bbox[2] + xmin) / width
                        ymax = (bbox[3] + ymin) / height
                    bounding_box = [xmin, ymin, xmax, ymax]
                    class_name = klass[2]
                    one_hot_class = self._to_one_hot(class_name)
                    if one_hot_class is None:
                        continue
                    bounding_boxes.append(bounding_box)
                    one_hot_classes.append(one_hot_class)
                if len(bounding_box) == 0:
                    continue
                image_data = np.hstack((bounding_boxes, one_hot_classes))
                rgb_image_name = data[3][0].replace('/n/fs/sun3d/data/', '')
                depth_image_name = data[2][0].replace('/n/fs/sun3d/data/', '')
                self.rgb_data[rgb_image_name] = image_data
                self.depth_data[depth_image_name] = image_data


    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'bed':
            one_hot_vector[0] = 1
        elif name == 'shelf':
            one_hot_vector[1] = 1
        elif name == 'desk':
            one_hot_vector[2] = 1
        elif name == 'plate':
            one_hot_vector[3] = 1
        elif name == 'lamp':
            one_hot_vector[4] = 1
        elif name == 'pillow':
            one_hot_vector[5] = 1
        elif name == 'chair':
            one_hot_vector[6] = 1
        elif name == 'garbage_bin':
            one_hot_vector[7] = 1
        elif name == 'door':
            one_hot_vector[8] = 1
        elif name == 'table':
            one_hot_vector[9] = 1
        elif name == 'mirror':
            one_hot_vector[10] = 1
        elif name == 'bench':
            one_hot_vector[11] = 1
        elif name == 'projector':
            one_hot_vector[12] = 1
        elif name == 'cabinet':
            one_hot_vector[13] = 1
        elif name == 'dresser':
            one_hot_vector[14] = 1
        elif name == 'sofa':
            one_hot_vector[15] = 1
        elif name == 'monitor':
            one_hot_vector[16] = 1
        elif name == 'person':
            one_hot_vector[17] = 1
        elif name == 'box':
            one_hot_vector[18] = 1
        else:
            print('unknown label: %s' %name)
 #           return None
        return one_hot_vector

parser = argparse.ArgumentParser(description='indicate specific image file path and mat file')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--mat_path', type=str)
args = parser.parse_args()

import pickle

preprocessed_data = MatPreprocessor(args.mat_path, args.dataset_path)
rgb_data = preprocessed_data.rgb_data
depth_data = preprocessed_data.depth_data

pickle.dump(rgb_data, open('../pkls/SUNRGBD/RGB.pkl', 'wb'))
pickle.dump(depth_data, open('../pkls/SUNRGBD/Depth.pkl', 'wb'))
