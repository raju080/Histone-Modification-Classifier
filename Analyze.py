import sys
import os
import random
import subprocess
import numpy as np
import pandas as pd
import h5py
import logging
import matplotlib.pyplot as plt
from itertools import combinations

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from keras import backend as K

import sklearn
from sklearn.model_selection import train_test_split

from Preprocess import Preprocessor
from Model import Model
from ConvolutionLayer import ConvolutionLayer



# output file to save the model after training
# model_file = 'TrainedModels/E116/model' + total_seq_postfix + '_' + model_type + crossval_seq_postfix + '_' + str(model_number) + '.h5'

# model_file = 'TrainedModels/E116/model_20000_cnn_lstm.h5'

siamese_network_file = 'TrainedModels/E116/siamese.h5'
embedding_model_file = 'TrainedModels/E116/model_20000_meuseum.h5'
vanilla_cnn_model_file = 'TrainedModels/E116/model_2000_vanilla_cnn_5.h5'

def get_weights_print_stats(layer):
    W = layer.get_weights()
    print(len(W))
    for w in W:
        print(w.shape)
    return W

def hist_weights(weights, file, bins=500):
    if len(weights.shape) == 1:
        plt.hist(np.ndarray.flatten(weights), bins=bins)
        plt.savefig(file)
        return
    for weight in weights:
            plt.hist(np.ndarray.flatten(weight), bins=bins)
            plt.savefig(file)


def check_weights(model_file):
    model = load_model(model_file, custom_objects={'ConvolutionLayer': ConvolutionLayer}, compile=False)
    weights = [layer.get_weights() for layer in model.layers]
    print('total layers: ', len(model.layers))
    # for layer in model.layers:
    #     print('\n\nLayer: {}\n'.format(layer.name))
    #     # if layer.name == 'conv1d':
    #     #     arr = np.array(layer.get_weights())[1]
    #     #     print(arr.shape)
    #     #     print(arr)
    #     layer = model.get_layer(layer.name)
    #     w = get_weights_print_stats(layer)
    #     hist_weights(w, 'debug/'+model_file.split('/')[-1]+layer.name+'.png')

    # checking the embedding weights
    emb_weights = np.array(model.get_layer('model').get_weights())
    # print(len(emb_weights))
    for i in range(len(emb_weights)):
        print('layer: ', i)
        arr = emb_weights[i]
        # print(len(emb_weights[i]))
        print(arr.shape)
        hist_weights(arr, 'debug/'+model_file.split('/')[-1]+str(i)+'.png')




check_weights(siamese_network_file)

