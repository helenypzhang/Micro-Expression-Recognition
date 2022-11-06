#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution3D, MaxPooling3D
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import tensorflow as tf
K.set_image_data_format('channels_first')
image_rows, image_columns, image_depth = 64, 64, 18
flow_rows, flow_columns, flow_depth = 144, 120, 16


# In[10]:


def discriminator():
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    
    return model
# tensorflow.keras.layers.Layer
class reverseLayer(keras.layers.Layer):
    def __init__(self, high_value = 1.0):
        super(reverseLayer, self).__init__()
        self.iter_num = 0
        self.alpha = 15
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'iter_num': self.iter_num,
            'alpha': self.alpha,
            'low': self.low,
            'high': self.high,
            'max': self.max_iter,
        })
        
        return config

    def call(self, input):
        
        @tf.custom_gradient
        def custom_op(x):
            result = x * 1.0
            self.iter_num += 1
            def custom_grad(dy):
                self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha *
                                                                      self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
                return -self.coeff * dy
            return result, custom_grad
        
        return custom_op(input)
              
def origin():
    model = Sequential()
    model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,  kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

    return model
    
def DS():
    inputA = keras.Input(shape =(1, image_rows,image_columns,image_depth))
    inputB = keras.Input(shape = (1, flow_rows, flow_columns, flow_depth))

    #first branch (image sequence)
    x = Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu')(inputA)
    x = MaxPooling3D(pool_size=(3, 3, 3))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = keras.Model(inputs=inputA, outputs=x)

    #second branch (flow sequence)
    y = Convolution3D(32, (3, 3, 6), strides = (1,1,2), input_shape=(1, flow_rows, flow_columns, flow_depth), activation='relu')(inputB)
    y = MaxPooling3D(pool_size=(3, 3, 3))(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    y = keras.Model(inputs=inputB, outputs=y)

    combined = keras.layers.Concatenate(axis = -1)([x.output, y.output])

    z = Dense(128, kernel_initializer='normal', activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(3, kernel_initializer='normal')(z)
    z = Activation('softmax')(z)

    model = keras.Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
    
    return model

def DS_domain():
    inputA = keras.Input(shape =(1, image_rows,image_columns,image_depth))
    inputB = keras.Input(shape = (1, flow_rows, flow_columns, flow_depth))
    

    #first branch (image sequence)
    x = Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu')(inputA)
    x = MaxPooling3D(pool_size=(3, 3, 3))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = keras.Model(inputs=inputA, outputs=x)

    #second branch (flow sequence)
    y = Convolution3D(32, (3, 3, 6), strides = (1,1,2), input_shape=(1, flow_rows, flow_columns, flow_depth), activation='relu')(inputB)
    y = MaxPooling3D(pool_size=(3, 3, 3))(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    y = keras.Model(inputs=inputB, outputs=y)

    combined = keras.layers.Concatenate(axis = -1)([x.output, y.output])
    
    reversedL = reverseLayer()
    
    reversedFeatures = reversedL(combined)

    output_domain = discriminator()(reversedFeatures)

    z = Dense(128, kernel_initializer='normal', activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(3, kernel_initializer='normal')(z)
    z = Activation('softmax')(z)
    
    xent = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)

    model = keras.Model(inputs=[x.input, y.input], outputs=[z, output_domain])
    model.compile(loss = {'activation':'categorical_crossentropy', 'sequential':'binary_crossentropy'},
                  loss_weights = {'activation':1., 'sequential':0.5},
                  optimizer = 'SGD', metrics = ['accuracy'])
    
    return model


# In[11]:





# In[ ]:




