"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import concatenate, average
from keras.models import Model
from keras.applications.vgg16 import VGG16

from ssd_layers import Normalize
from ssd_layers import PriorBox


def RGBD_SSD300(rgb_input_shape, depth_input_shape, num_classes=21):
    vgg16 = VGG16(weights='imagenet', include_top=False)
    weights = vgg16.get_weights()

    depth_input_layer = Input(shape=depth_input_shape)
    rgb_input_layer = Input(shape=rgb_input_shape)
    # Block 1
    rgb_conv1_1 = Conv2D(64, (3, 3),
                     name='rgb_conv1_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[0], weights[1]]
                     )(rgb_input_layer)

    depth_conv1_1 = Conv2D(64, (3, 3),
                     name='depth_conv1_1',
                     padding='same',
                     activation='relu',
                     #weights=[weights[0], weights[1]]
                     )(depth_input_layer)

    rgb_conv1_2 = Conv2D(64, (3, 3),
                     name='rgb_conv1_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[2], weights[3]]
                     )(rgb_conv1_1)
    depth_conv1_2 = Conv2D(64, (3, 3),
                     name='depth_conv1_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[2], weights[3]]
                     )(depth_conv1_1)

    rgb_pool1 = MaxPooling2D(name='rgb_pool1',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same', )(rgb_conv1_2)
    depth_pool1 = MaxPooling2D(name='depth_pool1',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same', )(depth_conv1_2)

    # Block 2
    rgb_conv2_1 = Conv2D(128, (3, 3),
                     name='rgb_conv2_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[4], weights[5]]
                     )(rgb_pool1)

    depth_conv2_1 = Conv2D(128, (3, 3),
                     name='depth_conv2_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[4], weights[5]]
                     )(depth_pool1)


    rgb_conv2_2 = Conv2D(128, (3, 3),
                     name='rgb_conv2_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[6], weights[7]]
                     )(rgb_conv2_1)

    depth_conv2_2 = Conv2D(128, (3, 3),
                     name='depth_conv2_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[6], weights[7]]
                     )(depth_conv2_1)

    rgb_pool2 = MaxPooling2D(name='rgb_pool2',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(rgb_conv2_2)

    depth_pool2 = MaxPooling2D(name='depth_pool2',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(depth_conv2_2)

    # Block 3
    rgb_conv3_1 = Conv2D(256, (3, 3),
                     name='rgb_conv3_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[8], weights[9]]
                     )(rgb_pool2)

    depth_conv3_1 = Conv2D(256, (3, 3),
                     name='depth_conv3_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[8], weights[9]]
                     )(depth_pool2)

    rgb_conv3_2 = Conv2D(256, (3, 3),
                     name='rgb_conv3_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[10], weights[11]]
                     )(rgb_conv3_1)

    depth_conv3_2 = Conv2D(256, (3, 3),
                     name='conv3_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[10], weights[11]]
                     )(depth_conv3_1)


    rgb_conv3_3 = Conv2D(256, (3, 3),
                     name='rgb_conv3_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[12], weights[13]]
                     )(rgb_conv3_2)

    depth_conv3_3 = Conv2D(256, (3, 3),
                     name='depth_conv3_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[12], weights[13]]
                     )(depth_conv3_2)

    rgb_pool3 = MaxPooling2D(name='rgb_pool3',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(rgb_conv3_3)
    depth_pool3 = MaxPooling2D(name='depth_pool3',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(depth_conv3_3)
    # Block 4
    rgb_conv4_1 = Conv2D(512, (3, 3),
                     name='rgb_conv4_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[14], weights[15]]
                     )(rgb_pool3)

    depth_conv4_1 = Conv2D(512, (3, 3),
                     name='depth_conv4_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[14], weights[15]]
                     )(depth_pool3)

    rgb_conv4_2 = Conv2D(512, (3, 3),
                     name='rgb_conv4_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[16], weights[17]]
                     )(rgb_conv4_1)
    depth_conv4_2 = Conv2D(512, (3, 3),
                     name='depth_conv4_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[16], weights[17]]
                     )(depth_conv4_1)
    rgb_conv4_3 = Conv2D(512, (3, 3),
                     name='rgb_conv4_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[18], weights[19]]
                     )(rgb_conv4_2)
    depth_conv4_3 = Conv2D(512, (3, 3),
                     name='depth_conv4_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[18], weights[19]]
                     )(depth_conv4_2)
    rgb_pool4 = MaxPooling2D(name='rgb_pool4',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(rgb_conv4_3)

    depth_pool4 = MaxPooling2D(name='depth_pool4',
                         pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same')(depth_conv4_3)
    # Block 5
    rgb_conv5_1 = Conv2D(512, (3, 3),
                     name='rgb_conv5_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[20], weights[21]]
                     )(rgb_pool4)

    depth_conv5_1 = Conv2D(512, (3, 3),
                     name='depth_conv5_1',
                     padding='same',
                     activation='relu',
                     weights=[weights[20], weights[21]]
                     )(depth_pool4)

    rgb_conv5_2 = Conv2D(512, (3, 3),
                     name='rgb_conv5_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[22], weights[23]]
                     )(rgb_conv5_1)
    depth_conv5_2 = Conv2D(512, (3, 3),
                     name='depth_conv5_2',
                     padding='same',
                     activation='relu',
                     weights=[weights[22], weights[23]]
                     )(depth_conv5_1)
    rgb_conv5_3 = Conv2D(512, (3, 3),
                     name='rgb_conv5_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[24], weights[25]]
                     )(rgb_conv5_2)
    depth_conv5_3 = Conv2D(512, (3, 3),
                     name='depth_conv5_3',
                     padding='same',
                     activation='relu',
                     weights=[weights[24], weights[25]]
                     )(depth_conv5_2)
    rgb_pool5 = MaxPooling2D(name='rgb_pool5',
                         pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(rgb_conv5_3)
    depth_pool5 = MaxPooling2D(name='depth_pool5',
                         pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same')(depth_conv5_3)


    # FC6
    rgb_fc6 = Conv2D(1024, (3, 3),
                 name='rgb_fc6',
                 dilation_rate=(6, 6),
                 padding='same',
                 activation='relu'
                 )(rgb_pool5)

    depth_fc6 = Conv2D(1024, (3, 3),
                 name='depth_fc6',
                 dilation_rate=(6, 6),
                 padding='same',
                 activation='relu'
                 )(depth_pool5)
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    rgb_fc7 = Conv2D(1024, (1, 1),
                 name='rgb_fc7',
                 padding='same',
                 activation='relu'
                 )(rgb_fc6)

    depth_fc7 = Conv2D(1024, (1, 1),
                 name='depth_fc7',
                 padding='same',
                 activation='relu'
                 )(depth_fc6)
    # x = Dropout(0.5, name='drop7')(x)

    # Block 6
    rgb_conv6_1 = Conv2D(256, (1, 1),
                     name='rgb_conv6_1',
                     padding='same',
                     activation='relu')(rgb_fc7)
    depth_conv6_1 = Conv2D(256, (1, 1),
                     name='depth_conv6_1',
                     padding='same',
                     activation='relu')(depth_fc7)
    rgb_conv6_2 = Conv2D(512, (3, 3),
                     name='rgb_conv6_2',
                     strides=(2, 2),
                     padding='same',
                     activation='relu')(rgb_conv6_1)

    depth_conv6_2 = Conv2D(512, (3, 3),
                     name='depth_conv6_2',
                     strides=(2, 2),
                     padding='same',
                     activation='relu')(depth_conv6_1)

    # Block 7
    rgb_conv7_1 = Conv2D(128, (1, 1),
                     name='rgb_conv7_1',
                     padding='same',
                     activation='relu')(rgb_conv6_2)
    depth_conv7_1 = Conv2D(128, (1, 1),
                     name='depth_conv7_1',
                     padding='same',
                     activation='relu')(depth_conv6_2)

    rgb_conv7_1z = ZeroPadding2D(name='rgb_conv7_1z')(rgb_conv7_1)
    depth_conv7_1z = ZeroPadding2D(name='depth_conv7_1z')(depth_conv7_1)
    rgb_conv7_2 = Conv2D(256, (3, 3),
                     name='rgb_conv7_2',
                     padding='valid',
                     strides=(2, 2),
                     activation='relu')(rgb_conv7_1z)
    depth_conv7_2 = Conv2D(256, (3, 3),
                     name='depth_conv7_2',
                     padding='valid',
                     strides=(2, 2),
                     activation='relu')(depth_conv7_1z)

    # Block 8
    rgb_conv8_1 = Conv2D(128, (1, 1),
                     name='rgb_conv8_1',
                     padding='same',
                     activation='relu')(rgb_conv7_2)
    depth_conv8_1 = Conv2D(128, (1, 1),
                     name='depth_conv8_1',
                     padding='same',
                     activation='relu')(depth_conv7_2)
    rgb_conv8_2 = Conv2D(256, (3, 3),
                     name='rgb_conv8_2',
                     padding='same',
                     strides=(2, 2),
                     activation='relu')(rgb_conv8_1)

    depth_conv8_2 = Conv2D(256, (3, 3),
                     name='depth_conv8_2',
                     padding='same',
                     strides=(2, 2),
                     activation='relu')(depth_conv8_1)
    # Last Pool
    rgb_pool6 = GlobalAveragePooling2D(name='rgb_pool6')(rgb_conv8_2)
    depth_pool6 = GlobalAveragePooling2D(name='depth_pool6')(depth_conv8_2)

    # Prediction from conv4_3
    num_priors = 3
    img_size = (rgb_input_shape[1], rgb_input_shape[0])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)

    rgb_depth_conv4_3 = concatenate([rgb_conv4_3, depth_conv4_3], axis=3)
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(rgb_depth_conv4_3)
    conv4_3_norm_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                                   name='conv4_3_norm_mbox_loc',
                                   padding='same')(conv4_3_norm)
    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
    conv4_3_norm_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                                    name=name,
                                    padding='same')(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    conv4_3_norm_mbox_priorbox = PriorBox(img_size, 30.0,
                                          name='conv4_3_norm_mbox_priorbox',
                                          aspect_ratios=[2],
                                          variances=[0.1, 0.1, 0.2, 0.2])(conv4_3_norm)

    # Prediction from fc7
    num_priors = 6
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    rgb_depth_fc7 = concatenate([rgb_fc7, depth_fc7], axis=3)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                           padding='same',
                           name=name)(rgb_depth_fc7)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)

    fc7_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                          name='fc7_mbox_loc',
                          padding='same')(rgb_depth_fc7)
    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
    fc7_mbox_priorbox = PriorBox(img_size, 60.0,
                                 name='fc7_mbox_priorbox',
                                 max_size=114.0,
                                 aspect_ratios=[2, 3],
                                 variances=[0.1, 0.1, 0.2, 0.2]
                                 )(rgb_depth_fc7)

    # Prediction from conv6_2
    num_priors = 6
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    rgb_depth_conv6_2 = concatenate([rgb_conv6_2, depth_conv6_2], axis=3)
    conv6_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                               padding='same',
                               name=name)(rgb_depth_conv6_2)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
    conv6_2_mbox_loc = Conv2D(num_priors * 4, (3, 3,),
                              name='conv6_2_mbox_loc',
                              padding='same')(rgb_depth_conv6_2)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
    conv6_2_mbox_priorbox = PriorBox(img_size, 114.0,
                                     max_size=168.0,
                                     aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2],
                                     name='conv6_2_mbox_priorbox')(rgb_depth_conv6_2)
    # Prediction from conv7_2
    num_priors = 6
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)

    rgb_depth_conv7_2 = concatenate([rgb_conv7_2, depth_conv7_2], axis=3)
    conv7_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                               padding='same',
                               name=name)(rgb_depth_conv7_2)
    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)
    conv7_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                              padding='same',
                              name='conv7_2_mbox_loc')(rgb_depth_conv7_2)
    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)
    conv7_2_mbox_priorbox = PriorBox(img_size, 168.0,
                                     max_size=222.0,
                                     aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2],
                                     name='conv7_2_mbox_priorbox')(rgb_depth_conv7_2)
    # Prediction from conv8_2
    num_priors = 6
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    rgb_depth_conv8_2 = concatenate([rgb_conv8_2, depth_conv8_2], axis=3)
    conv8_2_mbox_conf = Conv2D(num_priors * num_classes, (3, 3),
                               padding='same',
                               name=name)(rgb_depth_conv8_2)
    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)
    conv8_2_mbox_loc = Conv2D(num_priors * 4, (3, 3),
                              padding='same',
                              name='conv8_2_mbox_loc')(rgb_depth_conv8_2)
    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)
    conv8_2_mbox_priorbox = PriorBox(img_size, 222.0,
                                     max_size=276.0,
                                     aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2],
                                     name='conv8_2_mbox_priorbox')(rgb_depth_conv8_2)

    # Prediction from pool6
    num_priors = 6
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 512)
    else:
        target_shape = (512, 1, 1)
    rgb_depth_pool6 = concatenate([rgb_pool6, depth_pool6], axis=1)
    pool6_mbox_loc_flat = Dense(num_priors * 4, name='pool6_mbox_loc_flat', activation='tanh')(rgb_depth_pool6)
    pool6_mbox_conf_flat = Dense(num_priors * num_classes, name=name, activation='tanh')(rgb_depth_pool6)
    pool6_reshaped = Reshape(target_shape,
                             name='pool6_reshaped')(rgb_depth_pool6)
    pool6_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                                   variances=[0.1, 0.1, 0.2, 0.2],
                                   name='pool6_mbox_priorbox')(pool6_reshaped)
    # Gather all predictions
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat,
                            fc7_mbox_loc_flat,
                            conv6_2_mbox_loc_flat,
                            conv7_2_mbox_loc_flat,
                            conv8_2_mbox_loc_flat,
                            pool6_mbox_loc_flat],
                           axis=1,
                           name='mbox_loc')
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat,
                             fc7_mbox_conf_flat,
                             conv6_2_mbox_conf_flat,
                             conv7_2_mbox_conf_flat,
                             conv8_2_mbox_conf_flat,
                             pool6_mbox_conf_flat],
                            axis=1,
                            name='mbox_conf')
    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox,
                                 fc7_mbox_priorbox,
                                 conv6_2_mbox_priorbox,
                                 conv7_2_mbox_priorbox,
                                 conv8_2_mbox_priorbox,
                                 pool6_mbox_priorbox],
                                axis=1,
                                name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,
                               mbox_conf,
                               mbox_priorbox],
                              axis=2,
                              name='predictions')
    model = Model(inputs=[rgb_input_layer, depth_input_layer], outputs=predictions)
    return model
