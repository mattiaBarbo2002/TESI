#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:49:08 2023

@author: lbergamasco
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D, ConvLSTM2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Multiply, UpSampling3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers,metrics
from tensorflow.keras.applications import VGG16,ResNet50
from tensorflow.keras.layers import Add
from tensorflow.keras.losses import cosine_similarity

#from classification_models_3D.tfkeras import Classifiers

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def ca_block2D(inputs, ratio=4):
        
    """
	    Channel Attention Module exploiting the inter-channel relationship of features.
    """
    #inputs_changed_dims = tf.transpose(inputs, [0,1,2,4,3])  
    filters = int(inputs.shape[3])
    avg_pool = K.mean(inputs, axis=[1, 2,4], keepdims=False)
    avg_pool = tf.expand_dims(avg_pool,axis=1)
    #avg_pool_shape = avg_pool.shape
#    print(avg_pool.shape)
    max_pool = K.max(inputs, axis=[1, 2,4], keepdims=False)
    max_pool = tf.expand_dims(max_pool,axis=1)
    #max_pool_shape = max_pool.shape
#    print(max_pool.shape)

    x1 = tf.keras.layers.Conv1D(filters // ratio, kernel_size = 1, activation='relu')(avg_pool) 
    x1 = tf.keras.layers.Conv1D(filters, kernel_size = 1, activation=None)(x1)
#    print(K.int_shape(x1))

    x2 = tf.keras.layers.Conv1D(filters // ratio, kernel_size = 1, activation='relu')(max_pool) 
    x2 = tf.keras.layers.Conv1D(filters, kernel_size = 1, activation=None)(x2) 
#    print(K.int_shape(x2))

    x =  tf.keras.layers.Add()([x1, x2])
    x =  tf.keras.layers.Activation('sigmoid')(x)
#    print(K.int_shape(x))

#print(f"Attention, inputs: {inputs.shape}, x vector: {x.shape}")
    x = tf.expand_dims(x,axis=3)
    x = tf.expand_dims(x,axis=1)
    #x = tf.transpose(x, [0,1,2,4,3])
#    print(K.int_shape(x))

    outputs =  tf.keras.layers.Multiply()([inputs, x])
    return outputs

def sa_block(inputs):
    kernel_size = (7,7,1)
    print("SA", inputs.shape)
    avg_pool = K.mean(inputs, axis=(-1,-2), keepdims=True)
    max_pool = K.max(inputs, axis=(-1,-2), keepdims=True)
    x = Concatenate()([avg_pool, max_pool])
    x = Conv3D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    outputs = Multiply()([inputs, x])
    return outputs

def cbam_block(inputs):
    x = ca_block2D(inputs)
    x = sa_block(x)
    return x

def ca_block(inputs, ratio=8):
        
    """
	    Channel Attention Module exploiting the inter-channel relationship of features.
    """
    #inputs_changed_dims = tf.transpose(inputs, [0,1,2,4,3])  
#    filters = inputs.shape[-1]
    x = tf.expand_dims(inputs,axis=2)
#    print(K.int_shape(x))
#    x = tf.keras.layers.Conv1D(filters // ratio, kernel_size = 1, activation='relu')(inputs) 
    x = tf.keras.layers.Conv1D(1, kernel_size = 3, activation=None,padding='same')(x)

    x =  tf.keras.layers.Activation('sigmoid')(x)
#    print(K.int_shape(x))
    x = tf.squeeze(x,[2])
#    print(K.int_shape(x))

    outputs =  tf.keras.layers.Multiply()([inputs, x])
    return outputs

def MultiModal_3DCNN_3branch(input_shape1=None, input_shape2=None,input_shape3=None, weight_decay=0.01, 
                     n_layers=3, batch_momentum=0.9, n_classes=5, learning_rate=1e-4, weights = [0.5,0.5]):
    
    input1 = Input(shape=input_shape1,name='input1')
    input2 = Input(shape=input_shape2,name='input2')
    input3 = Input(shape=input_shape2,name='input3')
    
    for i in range(n_layers):
        if i == 0:
            x1 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(input1)
            
            x2 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(input2)
            
            x3 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(input3)
        elif i > 5:
            x1 = Conv3D(filters=512, kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x1)
            
            x2 = Conv3D(filters=512, kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x2)
            
            x3 = Conv3D(filters=512, kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x3)
        else:
            x1 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x1)
            
            x2 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x2)
            
            x3 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x3)
            
        x1 = BatchNormalization()(x1)
        x1 = ca_block2D(x1)
        
        x2 = BatchNormalization()(x2)
        x2 = ca_block2D(x2)
        
        x3 = BatchNormalization()(x3)
        x3 = ca_block2D(x3)
        
    x1 = K.mean(x1,axis=[1,2,4],keepdims=False)
    x2 = K.mean(x2,axis=[1,2,4],keepdims=False)
    x3 = K.mean(x3,axis=[1,2,4],keepdims=False)
    
    x = Concatenate()([x1,x2,x3])
    x = ca_block(x)
    
    x = tf.expand_dims(x,axis=1)
    
    output = layers.Conv1D(n_classes,1,name='output',kernel_regularizer=l2(weight_decay),activation='softmax',padding='same')(x)
    output = tf.squeeze(output,[1])
#    print(K.int_shape(output))
    model = Model([input1,input2,input3],output)
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=weighted_categorical_crossentropy(weights),metrics=[metrics.categorical_accuracy])
    model.summary()
    
    model_vgg = VGG16(weights='imagenet',include_top=False)
    model_vgg.summary()
    index_layer_model = [3,4,5,48,49,50,93,94,95,138,139,140,183,184,185,228,229,230,273,274,275]
    index_layer_model_vgg = [1,2,4,5,7,8,9]
    for i in range(len(index_layer_model)):
        i_vgg = i//3
        weights_2d, bias = model_vgg.layers[index_layer_model_vgg[i_vgg]].get_weights()
        layer3d = model.layers[index_layer_model[i]]
#        print(layer3d.name)
#        print(weights_2d.shape)
#        print(layer3d.get_weights()[0].shape)
#        print(bias.shape)
        # replicate them along z axis
        inflated_weights = []
        for z in range(3):
            inflated_weights.append(weights_2d)
        weights_3d = np.stack(tuple(inflated_weights), axis=2)
#        print(weights_3d.shape)
        #adapt the input
        if layer3d.get_weights()[0].shape[3] < weights_3d.shape[3]:
            weights_3d = weights_3d[:,:,:,:layer3d.get_weights()[0].shape[3],:]
        elif layer3d.get_weights()[0].shape[3] > weights_3d.shape[3]:
            diff = layer3d.get_weights()[0].shape[3] - weights_3d.shape[3]
            for i_in in range(diff):
                weights_3d = np.concatenate([weights_3d,np.expand_dims(weights_3d[:,:,:,0,:],axis=3)],axis=3)
        #adapt output
        if layer3d.get_weights()[0].shape[-1] < weights_3d.shape[-1]:
            weights_3d = weights_3d[:,:,:,:,:layer3d.get_weights()[0].shape[-1]]
            bias = bias[:layer3d.get_weights()[1].shape[0]]
        elif layer3d.get_weights()[0].shape[-1] > weights_3d.shape[-1]:
            diff = layer3d.get_weights()[0].shape[-1] - weights_3d.shape[-1]
#            print(diff)
            for i_in in range(diff):
                weights_3d = np.pad(weights_3d,[(0,0),(0,0),(0,0),(0,0),(0,1)],mode='symmetric')
            bias = np.pad(bias,(0,diff),mode='symmetric')
#        print(weights_3d.shape)
        layer3d.set_weights([weights_3d, bias])
    return model

def MultiModal_3DCNN_2branch(input_shape1=None, input_shape2=None,input_shape3=None, weight_decay=0.01, 
                     n_layers=3, batch_momentum=0.9, n_classes=5, learning_rate=1e-4, weights=[0.5,0.5]):
    
    input1 = Input(shape=input_shape1,name='input1')
    input2 = Input(shape=input_shape2,name='input2')
    
    for i in range(n_layers):
        if i == 0:
            x1 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(input1)
            
            x2 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(input2)
            
        elif i > 5:
            x1 = Conv3D(filters=512, kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x1)
            
            x2 = Conv3D(filters=512, kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x2)

        else:
            x1 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x1)
            
            x2 = Conv3D(filters=16*(2**i), kernel_size=3, strides = (2,2,1), activation="relu",
                        padding = "same", kernel_regularizer=l2(weight_decay))(x2)
            
            
        x1 = BatchNormalization()(x1)
        x1 = ca_block2D(x1)
        
        x2 = BatchNormalization()(x2)
        x2 = ca_block2D(x2)
        
    x1 = K.mean(x1,axis=[1,2,4],keepdims=False)
    x2 = K.mean(x2,axis=[1,2,4],keepdims=False)
    
    x = Concatenate()([x1,x2])
    x = ca_block(x)
    
    x = tf.expand_dims(x,axis=1)
    
    output = layers.Conv1D(n_classes,1,name='output',kernel_regularizer=l2(weight_decay),activation='softmax',padding='same')(x)
    output = tf.squeeze(output,[1])
#    print(K.int_shape(output))
    model = Model([input1,input2],output)
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=weighted_categorical_crossentropy(weights),metrics=[metrics.categorical_accuracy])
    model.summary()
    
    model_vgg = VGG16(weights='imagenet',include_top=False)
    model_vgg.summary()
    index_layer_model = [2,3,32,33,62,63,92,93,122,123,152,153,182,183]
    index_layer_model_vgg = [1,2,4,5,7,8,9]
    for i in range(len(index_layer_model)):
        i_vgg = i//2
        weights_2d, bias = model_vgg.layers[index_layer_model_vgg[i_vgg]].get_weights()
        layer3d = model.layers[index_layer_model[i]]
        #print(layer3d.name)
#        print(weights_2d.shape)
#        print(layer3d.get_weights()[0].shape)
#        print(bias.shape)
        # replicate them along z axis
        inflated_weights = []
        for z in range(3):
            inflated_weights.append(weights_2d)
        weights_3d = np.stack(tuple(inflated_weights), axis=2)
#        print(weights_3d.shape)
        #adapt the input
        if layer3d.get_weights()[0].shape[3] < weights_3d.shape[3]:
            weights_3d = weights_3d[:,:,:,:layer3d.get_weights()[0].shape[3],:]
        elif layer3d.get_weights()[0].shape[3] > weights_3d.shape[3]:
            diff = layer3d.get_weights()[0].shape[3] - weights_3d.shape[3]
            for i_in in range(diff):
                weights_3d = np.concatenate([weights_3d,np.expand_dims(weights_3d[:,:,:,0,:],axis=3)],axis=3)
        #adapt output
        if layer3d.get_weights()[0].shape[-1] < weights_3d.shape[-1]:
            weights_3d = weights_3d[:,:,:,:,:layer3d.get_weights()[0].shape[-1]]
            bias = bias[:layer3d.get_weights()[1].shape[0]]
        elif layer3d.get_weights()[0].shape[-1] > weights_3d.shape[-1]:
            diff = layer3d.get_weights()[0].shape[-1] - weights_3d.shape[-1]
#            print(diff)
            for i_in in range(diff):
                weights_3d = np.pad(weights_3d,[(0,0),(0,0),(0,0),(0,0),(0,1)],mode='symmetric')
            bias = np.pad(bias,(0,diff),mode='symmetric')
#        print(weights_3d.shape)
        layer3d.set_weights([weights_3d, bias])
    return model

def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2, 1), 
               batch_momentum=0.99, dilation_rate=(1,1,1)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 4
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block
        bn_name_base = 'bn' + str(stage) + block

        x = Conv3D(nb_filter1, (1, 1, 1), strides=strides,
                          name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv3D(nb_filter2, kernel_size, padding='same', dilation_rate=dilation_rate,
                          name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        x = Activation('relu')(x)

        x = Conv3D(nb_filter3, (1, 1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv3D(nb_filter3, (1, 1, 1), strides=strides,
                                 name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f

def MultiModal_3DCNN_2branch_v2(input_shape1=None, input_shape2=None, weight_decay=0.01, 
                                n_classes=5, learning_rate=1e-4, weights=[0.5,0.5]):
    
    input1 = Input(shape=input_shape1,name='input1')
    input2 = Input(shape=input_shape2,name='input2')
    
    x1 = Conv3D(16, (7, 7, 3), strides=(2, 2, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input1)
    x1 = BatchNormalization(axis=4, name='bn_conv1', momentum=0.99)(x1)
    x1 = Activation('relu')(x1)
    
    x2 = Conv3D(16, (7, 7, 3), strides=(2, 2, 1), padding='same', name='conv2', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv2', momentum=0.99)(x2)
    x2 = Activation('relu')(x2)
    
    #stage 1
    x1 = conv_block(kernel_size=3,filters=[16,16,64],stage=1,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[16,16,64],stage=1,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
    #stage 2
    x1 = conv_block(kernel_size=3,filters=[16,16,64],stage=2,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[16,16,64],stage=2,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
    #stage 3
    x1 = conv_block(kernel_size=3,filters=[32,32,128],stage=3,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[32,32,128],stage=3,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
    #stage 4
    x1 = conv_block(kernel_size=3,filters=[32,32,128],stage=4,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[32,32,128],stage=4,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
    #stage 5
    x1 = conv_block(kernel_size=3,filters=[64,64,256],stage=5,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[64,64,256],stage=5,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
    #stage 6
    x1 = conv_block(kernel_size=3,filters=[64,64,256],stage=6,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    x1 = ca_block2D(x1)
            
    x2 = conv_block(kernel_size=3,filters=[64,64,256],stage=6,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x2 = ca_block2D(x2)
    
#    #stage 7
#    x1 = conv_block(kernel_size=3,filters=[64,64,256],stage=7,block='opt',strides=(2, 2, 1),
#                    weight_decay=weight_decay)(x1)
#    x1 = ca_block2D(x1)
#            
#    x2 = conv_block(kernel_size=3,filters=[64,64,256],stage=7,block='SAR',strides=(2, 2, 1),
#                    weight_decay=weight_decay)(x2)
#    x2 = ca_block2D(x2)
        
    x1 = K.mean(x1,axis=[1,2,4],keepdims=False)
    x2 = K.mean(x2,axis=[1,2,4],keepdims=False)
    
    x = Concatenate()([x1,x2])
    x = ca_block(x)
    
    x = tf.expand_dims(x,axis=1)
    
    output = layers.Conv1D(n_classes,1,name='output',kernel_regularizer=l2(weight_decay),activation='softmax',padding='same')(x)
    output = tf.squeeze(output,[1])
#    print(K.int_shape(output))
    model = Model([input1,input2],output)
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=weighted_categorical_crossentropy(weights),metrics=[metrics.categorical_accuracy])
    model.summary()
    
    model_resnet = ResNet50(weights='imagenet',include_top=False)
    model_resnet.summary()
    index_layer_model = [2,3,8,9,14,15,20,22,21,23,58,59,64,65,70,72,71,73,108,109,114,115,120,122,121,123,158,159,164,165,170,
                         172,171,173,208,209,214,215,220,222,221,223,258,259,264,265,270,272,271,273]
    index_layer_model_resnet = [2,7,10,14,13,19,22,25,13,29,32,35,13,39,42,46,45,51,54,57,45,61,64,67,45]
    for i in range(len(index_layer_model)):
        i_resnet = i//2
        weights_2d, bias = model_resnet.layers[index_layer_model_resnet[i_resnet]].get_weights()
        layer3d = model.layers[index_layer_model[i]]
        print(layer3d.name)
        print(weights_2d.shape)
        print(layer3d.get_weights()[0].shape)
#        print(bias.shape)
        # replicate them along z axis
#        if layer3d.get_weights()[0].shape[2] == 1:
#            weights_3d = np.expand_dims(weights_2d,axis=2)
#        else:
        inflated_weights = []
        for z in range(layer3d.get_weights()[0].shape[2]):
            inflated_weights.append(weights_2d)
        weights_3d = np.stack(tuple(inflated_weights), axis=2)
#        print(weights_3d.shape)
        #adapt the input
        if layer3d.get_weights()[0].shape[3] < weights_3d.shape[3]:
            weights_3d = weights_3d[:,:,:,:layer3d.get_weights()[0].shape[3],:]
        elif layer3d.get_weights()[0].shape[3] > weights_3d.shape[3]:
            diff = layer3d.get_weights()[0].shape[3] - weights_3d.shape[3]
            for i_in in range(diff):
                weights_3d = np.concatenate([weights_3d,np.expand_dims(weights_3d[:,:,:,0,:],axis=3)],axis=3)
        #adapt output
        if layer3d.get_weights()[0].shape[-1] < weights_3d.shape[-1]:
            weights_3d = weights_3d[:,:,:,:,:layer3d.get_weights()[0].shape[-1]]
            bias = bias[:layer3d.get_weights()[1].shape[0]]
        elif layer3d.get_weights()[0].shape[-1] > weights_3d.shape[-1]:
            diff = layer3d.get_weights()[0].shape[-1] - weights_3d.shape[-1]
#            print(diff)
            for i_in in range(diff):
                weights_3d = np.pad(weights_3d,[(0,0),(0,0),(0,0),(0,0),(0,1)],mode='symmetric')
            bias = np.pad(bias,(0,diff),mode='symmetric')
        print(weights_3d.shape)
        layer3d.set_weights([weights_3d, bias])
    return model

def masked_mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(mask(y_true) - mask(y_pred)), axis=-1)`

   
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    summed_y_true = tf.reduce_sum(y_true, axis=[-1])
    mask = tf.greater(summed_y_true,0)
    masked_y_true = y_true[mask]
    masked_y_pred = y_pred[mask]
    
    masked_y_pred = tf.cast(masked_y_pred, "float32")
    masked_y_true = tf.cast(masked_y_true, "float32")
    return K.mean(tf.math.squared_difference(masked_y_pred, masked_y_true), axis=-1)
    
def masked_root_mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    `loss = mean(square(mask(y_true) - mask(y_pred)), axis=-1)`

   
    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    summed_y_true = tf.reduce_sum(y_true, axis=[-1])
    mask = tf.greater(summed_y_true,0)
    masked_y_true = y_true[mask]
    masked_y_pred = y_pred[mask]
    
    masked_y_pred = tf.cast(masked_y_pred, "float32")
    masked_y_true = tf.cast(masked_y_true, "float32")
    return tf.sqrt(K.mean(tf.math.squared_difference(masked_y_pred, masked_y_true), axis=-1))

def masked_cosine_similarity(y_true,y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    summed_y_true = tf.reduce_sum(y_true, axis=[-1])
    mask = tf.greater(summed_y_true,0)
    masked_y_true = y_true[mask]
    masked_y_pred = y_pred[mask]
    return cosine_similarity(masked_y_true,masked_y_pred)

def UNet3D_S2(input_shape1=None, weight_decay=0.01, learning_rate=1e-4):
    
    input2 = Input(shape=input_shape1,name='input2')
    
    x2 = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv1', momentum=0.99)(x2)
    x2 = Activation('relu')(x2) #256x256x16x16
    
    #stage 0
    x20 = conv_block(kernel_size=3,filters=[16,16,64],stage=0,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x20 = cbam_block(x20)#128x128x16x64
    
    #stage 1
#    x1 = conv_block(kernel_size=3,filters=[16,16,64],stage=1,block='opt',strides=(2, 2, 1),
#                    weight_decay=weight_decay)(x1)
#    x1 = ca_block2D(x1)
            
    x21 = conv_block(kernel_size=3,filters=[32,32,128],stage=1,block='SAR',strides=(2, 2, 2),
                    weight_decay=weight_decay)(x20)
    x21 = cbam_block(x21)#64x64x8x128
    
    #stage 2
#    x1 = conv_block(kernel_size=3,filters=[16,16,64],stage=2,block='opt',strides=(2, 2, 1),
#                    weight_decay=weight_decay)(x1)
#    x1 = ca_block2D(x1)
            
    x22 = conv_block(kernel_size=3,filters=[64,64,256],stage=2,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x21)
    x22 = cbam_block(x22)#32x32x4x256
    x22 = UpSampling3D(size=(2,2,1))(x22)
    
#    #stage 3
##    x1 = conv_block(kernel_size=3,filters=[32,32,128],stage=3,block='opt',strides=(2, 2, 1),
##                    weight_decay=weight_decay)(x1)
##    x1 = ca_block2D(x1)
#            
#    x23 = conv_block(kernel_size=3,filters=[32,32,128],stage=3,block='SAR',strides=(2, 2, 2),
#                    weight_decay=weight_decay)(x22)
#    x23 = cbam_block(x23)#16x16x8x128
#    
#    #stage 4
##    x1 = conv_block(kernel_size=3,filters=[32,32,128],stage=4,block='opt',strides=(2, 2, 1),
##                    weight_decay=weight_decay)(x1)
##    x1 = ca_block2D(x1)
#            
#    x24 = conv_block(kernel_size=3,filters=[32,32,128],stage=4,block='SAR',strides=(2, 2, 2),
#                    weight_decay=weight_decay)(x23)
#    x24 = cbam_block(x24)#8x8x4x128
#    x24 = UpSampling3D(size=(2,2,2))(x24)
     
    #stage 5
#    x25 = Concatenate(axis=-1)([x24,x23])#16x16x8x256
#    x25 = conv_block(kernel_size=3,filters=[32,32,128],stage=5,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x25)
#    x25 = cbam_block(x25)#16x16x8x128
#    x25 = UpSampling3D(size=(2,2,2))(x25)
#    
#    #stage6
#    x26 = Concatenate(axis=-1)([x25,x22])#32x32x16x192
#    x26 = conv_block(kernel_size=3,filters=[16,16,64],stage=6,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x26)
#    x26 = cbam_block(x26)#32x32x16x64
#    x26 = UpSampling3D(size=(2,2,1))(x26)
    
    #stage 7
    x27 = Concatenate(axis=-1)([x22,x21])#64x64x8x384
    x27 = conv_block(kernel_size=3,filters=[32,32,128],stage=7,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x27)
    x27 = cbam_block(x27)#64x64x16x64
    x27 = UpSampling3D(size=(2,2,2))(x27)
    
    #stage 8
    x28 = Concatenate(axis=-1)([x27,x20])#128x128x16x192
    x28 = conv_block(kernel_size=3,filters=[16,16,64],stage=8,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x28)
    x28 = cbam_block(x28)
    x28 = UpSampling3D(size=(2,2,1))(x28)
    
    x_out = Concatenate(axis=-1)([x28,x2])
    x_out = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv_out', kernel_regularizer=l2(weight_decay))(x_out)
    x_out = BatchNormalization(axis=4, name='bn_conv_out', momentum=0.99)(x_out)
    x_out = Activation('relu')(x_out) #256x256x16x16
    
    out = Conv3D(input_shape1[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output', kernel_regularizer=l2(weight_decay))(x_out)
    model = Model(input2,out)
    
    opt = optimizers.Adam(lr=learning_rate,epsilon=8e-8)
    model.compile(opt,loss=masked_cosine_similarity)
    model.summary()
    
    return model

def UNet3D_S1(input_shape1=None, weight_decay=0.01, learning_rate=1e-4):
    
    input2 = Input(shape=input_shape1,name='input2')
    
    x2 = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv1', momentum=0.99)(x2)
    x2 = Activation('relu')(x2) #256x256x16x16
    
    #stage 0
    x20 = conv_block(kernel_size=3,filters=[16,16,64],stage=0,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x20 = cbam_block(x20)#128x128x16x64
    
    #stage 1
    x21 = conv_block(kernel_size=3,filters=[32,32,128],stage=1,block='SAR',strides=(2, 2, 2),
                    weight_decay=weight_decay)(x20)
    x21 = cbam_block(x21)#64x64x8x128
    
    #stage 2
    x22 = conv_block(kernel_size=3,filters=[64,64,256],stage=2,block='SAR',strides=(2, 2, 2),
                    weight_decay=weight_decay)(x21)
    x22 = cbam_block(x22)#32x32x4x256
    x22 = UpSampling3D(size=(2,2,2))(x22)
    
    #stage 7
    x27 = Concatenate(axis=-1)([x22,x21])#64x64x8x384
    x27 = conv_block(kernel_size=3,filters=[32,32,128],stage=7,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x27)
    x27 = cbam_block(x27)#64x64x16x64
    x27 = UpSampling3D(size=(2,2,2))(x27)
    
    #stage 8
    x28 = Concatenate(axis=-1)([x27,x20])#128x128x16x192
    x28 = conv_block(kernel_size=3,filters=[16,16,64],stage=8,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x28)
    x28 = cbam_block(x28)
    x28 = UpSampling3D(size=(2,2,1))(x28)
    
    x_out = Concatenate(axis=-1)([x28,x2])
    x_out = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv_out', kernel_regularizer=l2(weight_decay))(x_out)
    x_out = BatchNormalization(axis=4, name='bn_conv_out', momentum=0.99)(x_out)
    x_out = Activation('relu')(x_out) #256x256x16x16
    
    out = Conv3D(input_shape1[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output', kernel_regularizer=l2(weight_decay))(x_out)
    model = Model(input2,out)
    
    opt = optimizers.Adam(lr=learning_rate,epsilon=8e-8)
    model.compile(opt,loss=masked_cosine_similarity)
    model.summary()
    
    return model

def UNet3D_LSTM(input_shape1=None, weight_decay=0.01, learning_rate=1e-4):
    
    input2 = Input(shape=input_shape1,name='input2')
    
    x2 = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv1', momentum=0.99)(x2)
    x2 = Activation('relu')(x2) #256x256x16x16
    
    #stage 0
    x20 = conv_block(kernel_size=3,filters=[16,16,64],stage=0,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    x20 = cbam_block(x20)#128x128x16x64
    
    #stage 1
    x21 = conv_block(kernel_size=3,filters=[32,32,128],stage=1,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x20)
    x21 = cbam_block(x21)#64x64x16x128
    
    #stage 2
    x22 = conv_block(kernel_size=3,filters=[64,64,256],stage=2,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x21)
    x22 = cbam_block(x22)#32x32x16x256
    
    #stage 3
    x23 = conv_block(kernel_size=3,filters=[64,64,256],stage=3,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x22)
    x23 = cbam_block(x23)#16x16x16x256
    x23 = tf.transpose(x23,[0,3,1,2,4])#16x16x16x256
    x23 = ConvLSTM2D(filters=64,kernel_size=3,padding='same',return_sequences=True)(x23)
    x23 = tf.transpose(x23,[0,2,3,1,4])#16x16x16x64
    x23 = UpSampling3D(size=(2,2,1))(x23)#32x32x16x64
    
    #stage 6
    x26 = Concatenate(axis=-1)([x23,x22])#32x32x16x320
    x26 = conv_block(kernel_size=3,filters=[64,64,256],stage=6,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x26)
    x26 = cbam_block(x26)#64x64x16x64
    x26 = UpSampling3D(size=(2,2,1))(x26)
    
    #stage 7
    x27 = Concatenate(axis=-1)([x26,x21])#64x64x8x384
    x27 = conv_block(kernel_size=3,filters=[32,32,128],stage=7,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x27)
    x27 = cbam_block(x27)#64x64x16x64
    x27 = UpSampling3D(size=(2,2,1))(x27)
    
    #stage 8
    x28 = Concatenate(axis=-1)([x27,x20])#128x128x16x192
    x28 = conv_block(kernel_size=3,filters=[16,16,64],stage=8,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x28)
    x28 = cbam_block(x28)
    x28 = UpSampling3D(size=(2,2,1))(x28)
    
    x_out = Concatenate(axis=-1)([x28,x2])
    x_out = Conv3D(16, (7, 7, 3), strides=(1, 1, 1), padding='same', name='conv_out', kernel_regularizer=l2(weight_decay))(x_out)
    x_out = BatchNormalization(axis=4, name='bn_conv_out', momentum=0.99)(x_out)
    x_out = Activation('relu')(x_out) #256x256x16x16
    
    out = Conv3D(input_shape1[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output', kernel_regularizer=l2(weight_decay))(x_out)
    model = Model(input2,out)
    
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=masked_mean_squared_error)
    model.summary()
    
    return model

def UNet3D_LSTMv2(input_shape1=None, weight_decay=0.01, learning_rate=1e-4,cbam=True):
    
    input2 = Input(shape=input_shape1,name='input2')
    
    x2 = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv1', momentum=0.99)(x2)
    x2 = Activation('relu')(x2) #256x256x16x16
    
    #stage 0
    x20 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=0,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    if cbam:
        x20 = cbam_block(x20)#128x128x16x64
    
    #stage 1
    x21 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=1,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x20)
    if cbam:
        x21 = cbam_block(x21)#64x64x16x128
    
    #stage 2
    x22 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=2,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x21)
    if cbam:
        x22 = cbam_block(x22)#32x32x16x256
    
    #stage 3
    x23 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=3,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x22)
    if cbam:
        x23 = cbam_block(x23)#16x16x16x256
    x23 = tf.transpose(x23,[0,3,1,2,4])#16x16x16x256
    x23 = ConvLSTM2D(filters=64,kernel_size=3,padding='same',return_sequences=True)(x23)
    x23 = tf.transpose(x23,[0,2,3,1,4])#16x16x16x64
    x23 = UpSampling3D(size=(2,2,1))(x23)#32x32x16x64
    
    #stage 6
    x26 = Concatenate(axis=-1)([x23,x22])#32x32x16x320
    x26 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=6,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x26)
    if cbam:
        x26 = cbam_block(x26)#64x64x16x64
    x26 = UpSampling3D(size=(2,2,1))(x26)
    
    #stage 7
    x27 = Concatenate(axis=-1)([x26,x21])#64x64x8x384
    x27 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=7,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x27)
    if cbam:
        x27 = cbam_block(x27)#64x64x16x64
    x27 = UpSampling3D(size=(2,2,1))(x27)
    
    #stage 8
    x28 = Concatenate(axis=-1)([x27,x20])#128x128x16x192
    x28 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=8,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x28)
    if cbam:
        x28 = cbam_block(x28)
    x28 = UpSampling3D(size=(2,2,1))(x28)
    
    x_out = Concatenate(axis=-1)([x28,x2])
    x_out = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv_out', kernel_regularizer=l2(weight_decay))(x_out)
    x_out = BatchNormalization(axis=4, name='bn_conv_out', momentum=0.99)(x_out)
    x_out = Activation('relu')(x_out) #256x256x16x16
    
    out = Conv3D(input_shape1[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output', kernel_regularizer=l2(weight_decay))(x_out)
    model = Model(input2,out)
    
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=masked_root_mean_squared_error)
    print('CBAM')
    
    return model

def XNet3D_LSTMv2(input_shape1=None, input_shape2=None, weight_decay=0.01, learning_rate=1e-4,cbam=True):
    
    input1 = Input(shape=input_shape1,name='input1')
    input2 = Input(shape=input_shape2,name='input2')
    
    x1 = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv_opt_in', kernel_regularizer=l2(weight_decay))(input1)
    x1 = BatchNormalization(axis=4, name='bn_conv_opt_in', momentum=0.99)(x1)
    x1 = Activation('relu')(x1) #256x256x16x16
    
    x2 = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv_sar_in', kernel_regularizer=l2(weight_decay))(input2)
    x2 = BatchNormalization(axis=4, name='bn_conv_sar_in', momentum=0.99)(x2)
    x2 = Activation('relu')(x2) #256x256x16x16
    
    #stage 0
    x10 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=0,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x1)
    if cbam:
        x10 = cbam_block(x10)#128x128x16x64
        
    x20 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=0,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x2)
    if cbam:
        x20 = cbam_block(x20)#128x128x16x64
    
    #stage 1
    x11 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=1,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x10)
    if cbam:
        x11 = cbam_block(x11)#64x64x16x128
        
    x21 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=1,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x20)
    if cbam:
        x21 = cbam_block(x21)#64x64x16x128
    
    #stage 2
    x12 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=2,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x11)
    if cbam:
        x12 = cbam_block(x12)#32x32x16x256
        
    x22 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=2,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x21)
    if cbam:
        x22 = cbam_block(x22)#32x32x16x256
    
    #stage 3
    x13 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=3,block='opt',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x12)
    if cbam:
        x13 = cbam_block(x13)#16x16x16x256
    x13 = tf.transpose(x13,[0,3,1,2,4])#16x16x16x256
    
    x23 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=3,block='SAR',strides=(2, 2, 1),
                    weight_decay=weight_decay)(x22)
    if cbam:
        x23 = cbam_block(x23)#16x16x16x256
    x23 = tf.transpose(x23,[0,3,1,2,4])#16x16x16x256
    conc = tf.concat([x13,x23],axis=-1)
    conc = ConvLSTM2D(filters=64,kernel_size=3,padding='same',return_sequences=True)(conc)
    conc = tf.transpose(conc,[0,2,3,1,4])#16x16x16x64
    conc = UpSampling3D(size=(2,2,1))(conc)#32x32x16x64
    
    #stage 4
    x14 = Concatenate(axis=-1)([conc,x12])#32x32x16x320
    x14 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=4,block='opt',strides=(1,1,1),weight_decay=weight_decay)(x14)
    if cbam:
        x14 = cbam_block(x14)#64x64x16x64
    x14 = UpSampling3D(size=(2,2,1))(x14)
    
    x24 = Concatenate(axis=-1)([conc,x22])#32x32x16x320
    x24 = conv_block(kernel_size=(3,3,1),filters=[64,64,256],stage=4,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x24)
    if cbam:
        x24 = cbam_block(x24)#64x64x16x64
    x24 = UpSampling3D(size=(2,2,1))(x24)
    
    #stage 5
    x15 = Concatenate(axis=-1)([x14,x11])#64x64x8x384
    x15 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=5,block='opt',strides=(1,1,1),weight_decay=weight_decay)(x15)
    if cbam:
        x15 = cbam_block(x15)#64x64x16x64
    x15 = UpSampling3D(size=(2,2,1))(x15)
    
    x25 = Concatenate(axis=-1)([x24,x21])#64x64x8x384
    x25 = conv_block(kernel_size=(3,3,1),filters=[32,32,128],stage=5,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x25)
    if cbam:
        x25 = cbam_block(x25)#64x64x16x64
    x25 = UpSampling3D(size=(2,2,1))(x25)
    
    #stage 6
    x16 = Concatenate(axis=-1)([x15,x10])#128x128x16x192
    x16 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=6,block='opt',strides=(1,1,1),weight_decay=weight_decay)(x16)
    if cbam:
        x16 = cbam_block(x16)
    x16 = UpSampling3D(size=(2,2,1))(x16)
    
    x26 = Concatenate(axis=-1)([x25,x20])#128x128x16x192
    x26 = conv_block(kernel_size=(3,3,1),filters=[16,16,64],stage=6,block='SAR',strides=(1,1,1),weight_decay=weight_decay)(x26)
    if cbam:
        x26 = cbam_block(x26)
    x26 = UpSampling3D(size=(2,2,1))(x26)
    
    x_opt = Concatenate(axis=-1)([x16,x1])
    x_opt = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv_opt_out', kernel_regularizer=l2(weight_decay))(x_opt)
    x_opt = BatchNormalization(axis=4, name='bn_conv_opt_out', momentum=0.99)(x_opt)
    x_opt = Activation('relu')(x_opt) #256x256x16x16
    
    x_sar = Concatenate(axis=-1)([x26,x2])
    x_sar = Conv3D(16, (7, 7, 1), strides=(1, 1, 1), padding='same', name='conv_sar_out', kernel_regularizer=l2(weight_decay))(x_sar)
    x_sar = BatchNormalization(axis=4, name='bn_conv_sar_out', momentum=0.99)(x_sar)
    x_sar = Activation('relu')(x_sar) #256x256x16x16
    
    opt_out = Conv3D(input_shape1[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output_opt', kernel_regularizer=l2(weight_decay))(x_opt)
    sar_out = Conv3D(input_shape2[-1], (1, 1, 1), strides=(1, 1, 1), padding='same', name='output_sar', kernel_regularizer=l2(weight_decay))(x_sar)
    model = Model([input1,input2],[opt_out,sar_out])
    
    opt = optimizers.Adam(lr=learning_rate,epsilon=1e-8)
    model.compile(opt,loss=masked_root_mean_squared_error)
    
    return model
    