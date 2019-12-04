#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:17:56 2019

@author: dbaby
"""

'''
Keras Implementation of NICE layers with triangular-jacobian
'''
import keras
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Add, Subtract, Lambda, Input, concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf

def _relunetwork_model(odd_dim, hidden_dim, num_layers, 
        layer_id):
    '''
    Pass part of the input data (odd or even part) through
    the ReLU network
    '''
    x_in = Input(shape=(odd_dim,))
    x_out = x_in
    kernel_init = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    #print ("Input Dim is " + str(input_dim))
    for k in range(num_layers):
        x_out = Dense(hidden_dim, activation='relu',
                kernel_initializer=kernel_init)(x_out)
    x_out = Dense(odd_dim)(x_out)
    modelname = 'relublock_' + str(layer_id)
    relublockmodel = Model(x_in, x_out, name=modelname)
    return relublockmodel


class ScalingLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ScalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(ScalingLayer, self).build(input_shape)  

    def call(self, x):
        return tf.multiply(x, K.exp(self.kernel))
        
    def inverse (self):
        scale = K.exp(-self.kernel)
        return Lambda(lambda x: scale * x)

    def compute_output_shape(self, input_shape):
        return input_shape

def NICEModel(odd_dim, even_dim, numblocks=4, 
        numrelulayers=6, reluhiddendim=1024):
    '''
    Create the NICE model in Keras
    '''
    x_odd = Input(shape=(odd_dim,))
    x_even = Input(shape=(even_dim,))

    # scaling layer
    scale_odd = ScalingLayer(output_dim=K.int_shape(x_odd),
                                name = 'scaling_layer_odd')
    scale_even = ScalingLayer(output_dim=K.int_shape(x_even),
                                name = 'scaling_layer_even')

    relublocks = [] # list containing relu block layers
    for i in range(numblocks):
        _relublock = _relunetwork_model(odd_dim, reluhiddendim, 
                                      numrelulayers, i+1)
        relublocks.append(_relublock)
        odd_dim, even_dim = even_dim, odd_dim
   
    # Construct the forward model
    h_odd = x_odd
    h_even = x_even
    for i in range(numblocks):
        h_even = Add()([h_even, relublocks[i](h_odd)])
        h_odd, h_even = h_even, h_odd    
    # add scaling layer
    h_odd = scale_odd(h_odd)
    h_even = scale_even(h_even)
    h_merge = concatenate([h_odd, h_even])
    nice_forward = Model([x_odd, x_even], h_merge)
    nice_forward.summary()
    
    # Construct the inverse model
    x_odd = Input(shape=(odd_dim,))
    x_even = Input(shape=(even_dim,))
    h_odd = x_odd
    h_even = x_even
    # invert the scaling layers
    h_odd = scale_odd.inverse()(h_odd)
    h_even = scale_even.inverse()(h_even)
    for i in range(numblocks):
        h_even = Subtract()([h_even, relublocks[i](h_odd)])
        h_odd, h_even = h_even, h_odd
        
    nice_inverse = Model([x_odd, x_even], [h_odd, h_even])
    
    # plot models
    plot_model(nice_forward, to_file='nice_forward.png', 
                show_shapes=True, expand_nested=True)
    plot_model(nice_inverse, to_file='nice_inverse.png', 
                show_shapes=True, expand_nested=True)

    return nice_forward, nice_inverse








