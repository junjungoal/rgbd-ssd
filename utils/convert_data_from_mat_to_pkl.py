import scipy.io
import numpy as np
import os
import argparse
#http://www.cs.cmu.edu/~ehsiao/datasets.html
# Washington
class MatPreprocessor(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.num_classes = 6
        self.rgb_data = dict()
        self.depth_data = dict()
        self.scenes = ['desk', 'table', 'kitchen_small', 'meeting_small', 'table_small']
        self.name_list =[]
        self._preprocess_mat()

    def _preprocess_mat(self):
        for scene in self.scenes:
            nb_dir = len(os.listdir(self.data_path + '/' + scene)) // 2
            for i in range(nb_dir):
                #nb_files = len(os.listdir(self.data_path + '/' + scene + '/' + scene + '_' + i])) / 2
                matdata = scipy.io.loadmat(self.data_path + '/' + scene + '/' + scene + '_' + str(i+1) + '.mat')
                bbox_mat = matdata['bboxes']
                bbox_array = np.array(bbox_mat)[0]
                filenames =  os.listdir(self.data_path)
                for j, bbox in enumerate(bbox_array):
                    # bboxes points are stored at index 0
                    # data type is stored at index 1
                    bounding_boxes = []
                    one_hot_classes = []
                    if len(bbox[0]) != 0:
                        for annotation in bbox[0]:
                            category = annotation[0][0]
                            top = (annotation[2][0][0]-10) / 480
                            bottom = (annotation[3][0][0]) / 480
                            left = (annotation[4][0][0]-25) / 600
                            right = (annotation[5][0][0]-15) / 600
                            bounding_box = [left, bottom, right, top]
                            one_hot_class = self._to_one_hot(category)
                            bounding_boxes.append(bounding_box)
                            one_hot_classes.append(one_hot_class)
                    """
                    1. get file name
                    """
                    bounding_boxes = np.asarray(bounding_boxes)
                    one_hot_classes = np.asarray(one_hot_classes)
                    if len(one_hot_classes) == 0:
                        continue
                    print(one_hot_classes)
                    image_data = np.hstack((bounding_boxes, one_hot_classes))
                    self.rgb_data[scene + '/' + scene + '_' + str(i+1) + '/' + scene + '_' + str(i+1) + '_' + str(j+1) + '.png'] = image_data
                    self.depth_data[scene + '/' + scene + '_' + str(i+1) + '/' + scene + '_' + str(i+1) + '_' + str(j+1) + '_depth.png'] = image_data


    def _to_one_hot(self, name):
        one_hot_vector = [0] * self.num_classes
        if name == 'soda_can':
            one_hot_vector[0] = 1
        elif name == 'coffee_mug':
            one_hot_vector[1] = 1
        elif name == 'cap':
            one_hot_vector[2] = 1
        elif name == 'flashlight':
            one_hot_vector[3] = 1
        elif name == 'bowl':
            one_hot_vector[4] = 1
        elif name == 'cereal_box':
            one_hot_vector[5] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

parser = argparse.ArgumentParser(description='indicate specific image file path and mat file')
parser.add_argument('--data_path', type=str, help='input the data path')

args = parser.parse_args()

import pickle

preprocessed_data = MatPreprocessor(args.data_path)
rgb_data = preprocessed_data.rgb_data
depth_data = preprocessed_data.depth_data

pickle.dump(rgb_data, open('../pkls/rgbd-scenes/RGB.pkl', 'wb'))
pickle.dump(depth_data, open('../pkls/rgbd-scenes/Depth.pkl', 'wb'))
