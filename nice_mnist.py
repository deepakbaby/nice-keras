#!/usr/bin/env python3
'''
NICE implementation in Keras
MNIST example
'''

from layers import NICEModel
import keras.backend as K
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import imageio
import matplotlib.pyplot as plt
from keras.callbacks.callbacks import EarlyStopping

def reconstruct(x_odd, x_even):
    final_len =  x_odd.shape[1] + x_even.shape[1]
    x_recon = np.zeros((x_odd.shape[0], final_len))
    for k in range(x_odd.shape[1]):
        x_recon[:,2*k] = x_odd[:,k]
        x_recon[:, (2*k) + 1] = x_even[:,k]
    return x_recon

def ss_loss (y_true, y_pred):
    '''
    Scaling loss
    '''
    scale_weights_odd = nice_forward.get_layer('scaling_layer_odd').trainable_weights
    scale_weights_even = nice_forward.get_layer('scaling_layer_even').trainable_weights
    return K.sum(scale_weights_odd, axis=1) + K.sum(scale_weights_even, axis=1)

def loglike_loss_logistic(y_true, y_pred):
    '''
    log likelihood loss
    '''
    logistic_loglike = - K.softplus(y_pred) - K.softplus(-1 * y_pred)
    return K.sum(logistic_loglike, axis=1)


def nice_loss (y_true, y_pred):
    _ss_loss = ss_loss (y_true, y_pred)
    _log_loss = loglike_loss_logistic (y_true, y_pred)
    return - K.mean (_ss_loss + _log_loss, axis=0)


if __name__ == "__main__":
    
    TRAIN =  True
    SAMPLE = not True
    
     # network parameters
    numcouplinglayers = 4
    numrelublocks = 6
    relublockhiddendim = 1000 
    epochs = 5000
    batch_size = 2048
    savefilename = 'nice_mnist.h5'
    
    image_size = 28
    original_dim = image_size * image_size
    odd_dim = original_dim // 2
    even_dim = original_dim - odd_dim
    input_shape = (original_dim, )
    
    nice_forward, nice_inverse = NICEModel(odd_dim, even_dim, 
                                           numblocks=numcouplinglayers, 
                                           numrelulayers=numrelublocks, 
                                           reluhiddendim=relublockhiddendim)
    optimizer  = Adam(lr=1e-3)
    nice_forward.compile(optimizer = optimizer, loss=nice_loss, metrics = [loglike_loss_logistic, ss_loss])
    
    if TRAIN:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = np.reshape(x_train, [-1, original_dim]).astype('float32')
        rand_dequantise = np.random.uniform(low=0., high=1., size=x_train.shape).astype('float32')
        x_train += rand_dequantise
        x_train = x_train / 255.
        
        x_test = np.reshape(x_test, [-1, original_dim]).astype('float32')
        rand_dequantise = np.random.uniform(low=0., high=1., size=x_test.shape)
        x_test += rand_dequantise
        x_test = x_test.astype('float32') / 255.
    
        # Take odd and even parts of the data
        x_train_odd = x_train[:,0::2]
        x_train_even = x_train[:,1::2]
        
        es = EarlyStopping(monitor='loss', patience=100) 
        nice_forward.fit([x_train_odd, x_train_even], x_train, epochs=epochs, 
                batch_size=2048, verbose=2, callbacks=[es])
        
        nice_forward.save_weights(savefilename)
        
    if SAMPLE:
        
        nice_inverse.load_weights(savefilename)
        
        n = 15
        digit_size = image_size
        figure = np.zeros((digit_size * n, digit_size * n))
    
        for i in range(n):
            for j in range(n):
                z_sample = np.array(np.random.logistic(1, original_dim))
                h_odd = z_sample[:, :odd_dim]
                h_even = z_sample[:, odd_dim:]
                [xo, xe] = nice_inverse.predict([h_odd,h_even])
                x_inv = reconstruct(xo, xe)
                digit = x_inv[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
        figure = np.clip(figure*255, 0, 255)
        imageio.imwrite('test.png', figure)
        plt.imshow(figure)
        plt.show()        
