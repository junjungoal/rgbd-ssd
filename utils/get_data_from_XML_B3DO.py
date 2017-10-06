import numpy as np
import os
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 24
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == 'pen_or_pencil':
            one_hot_vector[0] = 1
        elif name == 'book':
            one_hot_vector[1] = 1
        elif name == 'mouse':
            one_hot_vector[2] = 1
        elif name == 'pillow':
            one_hot_vector[3] = 1
        elif name == 'table':
            one_hot_vector[4] = 1
        elif name == 'phone':
            one_hot_vector[5] = 1
        elif name == 'speaker':
            one_hot_vector[6] = 1
        elif name == 'stapler':
            one_hot_vector[7] = 1
        elif name == 'table':
            one_hot_vector[8] = 1
        elif name == 'cup':
            one_hot_vector[9] = 1
        elif name == 'monitor':
            one_hot_vector[10] = 1
        elif name == 'keyboard':
            one_hot_vector[11] = 1
        elif name == 'letter_tray':
            one_hot_vector[12] = 1
        elif name == 'remote':
            one_hot_vector[13] = 1
        elif name == 'paper_notebook':
            one_hot_vector[14] = 1
        elif name == 'poweroutlet':
            one_hot_vector[15] = 1
        elif name == 'towl':
            one_hot_vector[16] = 1
        elif name == 'table_knife':
            one_hot_vector[17] = 1
        elif name == 'soap':
            one_hot_vector[18] = 1
        elif name == 'bookcase':
            one_hot_vector[19] = 1
        elif name == 'chair':
            one_hot_vector[20] = 1
        elif name == 'bottle':
            one_hot_vector[21] = 1
        elif name == 'dish_or_place':
            one_hot_vector[22] = 1
        elif name == 'scissors':
            one_hot_vector[23] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

## example on how to use it
import pickle
data = XML_preprocessor('../dataset/VOCB3DO/Annotations/').data
pickle.dump(data,open('../pkls/VOCB3DO.pkl','wb'))

