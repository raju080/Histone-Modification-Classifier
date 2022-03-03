import numpy as np
from numpy import newaxis
import pandas as pd
import sys
import math
import os
import json
import csv
import pandas
import random
import scipy
import matplotlib.pyplot as plt
import logging

from scipy.stats import spearmanr, pearsonr

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe

# specifically for model visualization
import keras
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, concatenate, ReLU, Activation, LeakyReLU, TimeDistributed, Reshape, Bidirectional, LSTM, GRU
from tensorflow.keras import backend as K, regularizers, activations
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Lambda
from tensorflow import keras

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.utils import shuffle

import keras.layers as kl
# from bpnet.losses import multinomial_nll


from ConvolutionLayer import ConvolutionLayer

# Reproducibility
# seed = random.randint(1, 1000)
seed = 527
np.random.seed(seed)
tf.random.set_seed(seed)



class Model:
    def __init__(self, filters, kernel_size, pool_type, regularizer, activation_type, epochs, batch_size, model_file):
        """initialize basic parameters"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        logging.basicConfig(filename=model_file+'log.log', level=logging.INFO)

        # Pringting model parameters
        self.parameterPrint()

    def parameterPrint(self):
        logging.info(
            "\n=========================== Parameters ===================================")
        logging.info("# of Filters: " + str(self.filters))
        logging.info("Kernel size: " + str(self.kernel_size))
        logging.info("Pool type: " + self.pool_type)
        logging.info("regularizer: " + self.regularizer)
        logging.info("activation_type: " + self.activation_type)
        logging.info("epochs: " + str(self.epochs))
        logging.info("batch_size: " + str(self.batch_size))
        logging.info(
            "============================================================================\n")

    def create_meuseum_model(self, dims, compile=True, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res = K.sum(K.square(y_true-y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))

        def auroc(y_true, y_pred):
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

        # To build this model with the functional API,
        # you would start by creating an input node:
        # dims[1] = sequence length
        # dims[2] = 4   (one hot encode for ACGT)
        forward_input = keras.Input(shape=(dims[1], dims[2]), name='forward')
        reverse_input = keras.Input(shape=(dims[1], dims[2]), name='reverse')

        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', input_shape=(dims[1],dims[2]), use_bias = False)
        # with trainable = False
        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, kernel_initializer = my_init, data_format='channels_last', input_shape=(dims[1],dims[2]), use_bias = False, trainable=False)
        first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size,
                                       alpha=alpha, beta=beta, bkg_const=bkg_const, data_format='channels_last', use_bias=True)

        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)

        concat = concatenate([fw, bw], axis=1, name='concat')
        pool_size_input = 4
        concat_relu = ReLU()(concat)
        # concat_relu = Activation(activation=activations.relu)(concat)
        # concat_relu = Activation(activation=activations.linear)(concat)
        #concat = Dense(1, activation= 'sigmoid')(concat)

        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=pool_size_input, name='maxpool')(concat_relu)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(
                pool_size=pool_size_input, name='avgpool')(concat_relu)
        elif self.pool_type == 'Custom':
            def out_shape(input_shape):
                shape = list(input_shape)
                shape[0] = 10
                return tuple(shape)
            #model.add(Lambda(top_k, arguments={'k': 10}))

            def top_k(inputs, k):
                # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
                inputs2 = tf.transpose(inputs, [0, 2, 1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0, 2, 1])

            pool_layer = Lambda(top_k, arguments={'k': 500})(concat_relu)
            # pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
            # pool_layer = MaxPooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'Custom_sum':
            # apply relu function before custom_sum functions
            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
                return new_vals
            pool_layer = Lambda(summed_up, name='custompool')(concat_relu)
            # pool_layer = MaxPooling1D(pool_size=2)(pool_layer)
            # pool_layer = Lambda(summed_up)(concat)
        else:
            raise NameError('Set the pooling layer name correctly')

        flat = Flatten(name='flat')(pool_layer)

        after_flat = Dense(64, name='dense')(flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            # outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, name='softmax', kernel_initializer='normal', kernel_regularizer=regularizers.l1(
                0.001), activation='softmax')(after_flat)
        elif self.regularizer == 'L_2':
            #outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, name='softmax', kernel_initializer='normal', kernel_regularizer=regularizers.l2(
                0.001), activation='softmax')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')

        # weight_forwardin_0=model.layers[0].get_weights()[0]
        # print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=outputs)

        if compile:
            #model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
            model.compile(loss='binary_crossentropy',
                        optimizer='adam', metrics=['accuracy'])
            # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])
        # logging.info(model.summary())
        model.summary(print_fn=lambda x: logging.info(x))
        return model

    def create_basic_model(self, sequence_shape, compile=True):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        def top_k(inputs, k):
            # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
            inputs2 = tf.transpose(inputs, [0, 2, 1])
            new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
            # transform back to (None, 10, 512)
            return tf.transpose(new_vals, [0, 2, 1])

        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = ConvolutionLayer(filters=self.filters//2, kernel_size=9, strides=1,
                                        input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer = ReLU()(concat_layer)
        # relu_layer = Activation(activation=activations.tanh)(concat_layer)
        # pool_layer = Lambda(sumUp)(relu_layer)
        pool_layer = MaxPooling1D(pool_size=2)(relu_layer)

        conv_layer_2 = Conv1D(filters=self.filters,
                              kernel_size=12, strides=1)(pool_layer)
        relu_layer_2 = ReLU()(conv_layer_2)

        # pool_layer = Lambda(sumUp)(relu_layer_2)
        # pool_layer = Lambda(top_k, arguments={'k': 10})(relu_layer_2)
        pool_layer = MaxPooling1D(pool_size=2)(relu_layer_2)
        flat_layer = Flatten()(pool_layer)

        dense_layer = Dense(64, activation=activations.relu)(flat_layer)

        # dense_layer = Dense(32, activation=activations.relu)(dense_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer)

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)

        if compile:
            model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        model.summary(print_fn=lambda x: logging.info(x))

        return model

    def create_Vanilla_CNN_model(self, sequence_shape, compile=True, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # input layers
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        # input layers
        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=1,
                                        input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer = ReLU()(concat_layer)

        pool_layer = Lambda(sumUp)(relu_layer)
        # pool_layer = Lambda(top_k, arguments={'k': 10})(relu_layer)
        # pool_layer = MaxPooling1D(pool_size=4)(relu_layer)
        flat_layer = Flatten()(pool_layer)

        dense_layer = Dense(32, activation=activations.relu)(flat_layer)

        # dense_layer = Dense(32, activation=activations.relu)(dense_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer)

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)

        if compile:
            model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        model.summary(print_fn=lambda x: logging.info(x))

        return model

    def create_Multi_CNN2_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        pool_size_input = 4
        # input layers
        input_layer = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]))

        conv_layer_1 = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size,
                                        activation=self.activation_type, input_shape=(sequence_shape[1], sequence_shape[2]))(input_layer)

        relu_layer_1 = ReLU()(conv_layer_1)
        conv_layer_2 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
                              activation=self.activation_type)(relu_layer_1)
        relu_layer_2 = ReLU()(conv_layer_2)

        # conv_layer_3 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
        #                       activation=self.activation_type)(relu_layer_3)
        # relu_layer_3 = ReLU()(conv_layer_3)

        # conv_layer_4 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
        #                       activation=self.activation_type)(relu_layer_3)
        # relu_layer_4 = ReLU()(conv_layer_4)

        # pool_layer = Lambda(sumUp)(relu_layer_4)

        pool_layer = Lambda(sumUp)(relu_layer_2)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(64, activation=self.activation_type)(flat_layer)

        dense_layer2 = Dense(32, activation=self.activation_type)(dense_layer1)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer2)

        model = keras.Model(
            inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model

    def create_Multi_CNN4_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                              activation=self.activation_type, input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer_1 = ReLU()(concat_layer)
        #
        pool_layer_1 = MaxPooling1D(pool_size=2)(relu_layer_1)
        conv_layer_2 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
                              activation=self.activation_type)(pool_layer_1)
        relu_layer_2 = ReLU()(conv_layer_2)
        pool_layer_2 = MaxPooling1D(pool_size=2)(relu_layer_2)
        conv_layer_3 = Conv1D(filters=self.filters//4, kernel_size=self.kernel_size,
                              activation=self.activation_type)(pool_layer_2)
        relu_layer_3 = ReLU()(conv_layer_3)
        pool_layer_3 = MaxPooling1D(pool_size=2)(relu_layer_3)
        conv_layer_4 = Conv1D(filters=self.filters//8, kernel_size=self.kernel_size,
                              activation=self.activation_type)(pool_layer_3)
        relu_layer_4 = ReLU()(conv_layer_4)

        #
        # conv_layer_2 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
        #                       activation=self.activation_type)(relu_layer_1)
        # relu_layer_2 = ReLU()(conv_layer_2)
        # conv_layer_3 = Conv1D(filters=self.filters//4, kernel_size=self.kernel_size,
        #                       activation=self.activation_type)(relu_layer_2)
        # relu_layer_3 = ReLU()(conv_layer_3)
        # conv_layer_4 = Conv1D(filters=self.filters//8, kernel_size=self.kernel_size,
        #                       activation=self.activation_type)(relu_layer_3)
        # relu_layer_4 = ReLU()(conv_layer_4)
        #

        pool_layer = Lambda(sumUp)(relu_layer_4)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(64, activation=self.activation_type)(flat_layer)
        dense_layer2 = Dense(32, activation=self.activation_type)(dense_layer1)
        dense_layer3 = Dense(32, activation=self.activation_type)(dense_layer2)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer3)

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model

    def create_bpnet_model(self, sequence_shape):
        tasks = ['Oct4', 'Sox2', 'Nanog', 'Klf4']

        # body
        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = Conv1D(filters=64, kernel_size=25,
                              activation='relu', padding='same', input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        x = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        # input = kl.Input(shape=(1000, 4))
        # x = kl.Conv1D(64, kernel_size=25, padding='same', activation='relu')(input)
        for i in range(1, 10):
            conv_x = kl.Conv1D(64, kernel_size=3, padding='same',
                               activation='relu', dilation_rate=2**i)(x)
            x = kl.add([conv_x, x])
        bottleneck = x

        # heads
        # outputs = []
        # for task in tasks:
        # # profile shape head
        #     px = kl.Reshape((-1, 1, 64))(bottleneck)
        #     px = kl.Conv2DTranspose(2, kernel_size=(25, 1), padding='same')(px)
        #     outputs.append(kl.Reshape((-1, 2))(px))
        #     # total counts head
        #     cx = kl.GlobalAvgPool1D()(bottleneck)
        #     outputs.append(kl.Dense(2)(cx))

        cx = kl.GlobalAvgPool1D()(bottleneck)
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(cx)

        # model = keras.models.Model([input], outputs)
        # model.compile(keras.optimizers.Adam(lr=0.004), loss=[multinomial_nll, 'mse'] * len(tasks), loss_weights=[1, 10] * len(tasks))
        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def create_LSTM_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')
        logging.info("Forward shape----------------" + str(forward_input))
        logging.info("Reverse shape----------------" + str(reverse_input))
        model = Sequential()
        n_features = 512
        n_neurons = 5
        first_layer = LSTM(n_neurons, return_sequences=True, input_shape=(
            sequence_shape[1], sequence_shape[2], n_features))
        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)
        x = concatenate([fw, bw], axis=1)
        # x = Dense(n_features,activation="sigmoid")(x)
        second_layer = LSTM(
            n_neurons*2, input_shape=(x.shape[1], x.shape[2], n_features/2))
        x = second_layer(x)
        flat = Flatten()(x)
        # model.add(flat)
        after_flat = Dense(32, activation=self.activation_type)(flat)
        # model.add(after_flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(
                0.001), activation='sigmoid')(after_flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
                0.001), activation='sigmoid')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')
        # model.add(outputs)

        # weight_forwardin_0=model.layers[0].get_weights()[0]
        # print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=outputs)

        model.compile(loss='mean_squared_error',
                      optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model

    def create_CNNLSTM_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals

        def top_k(inputs, k):
            # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
            inputs2 = tf.transpose(inputs, [0, 2, 1])
            new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
            # transform back to (None, 10, 512)
            return tf.transpose(new_vals, [0, 2, 1])

        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = ConvolutionLayer(filters=self.filters//2, kernel_size=self.kernel_size, strides=1,
                                        input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer = ReLU()(concat_layer)

        pool_layer = MaxPooling1D(pool_size=2)(relu_layer)
        conv_layer_2 = Conv1D(filters=self.filters,
                              kernel_size=12, strides=1)(pool_layer)
        
        relu_layer_2 = ReLU()(conv_layer_2)
        # pool_layer = Lambda(sumUp)(relu_layer_2)
        # pool_layer = Lambda(top_k, arguments={'k': 4})(relu_layer_2)
        pool_layer = MaxPooling1D(pool_size=2)(relu_layer_2)
        pool_layer = Reshape((1,pool_layer.shape[1],pool_layer.shape[2]))(pool_layer)
        print(pool_layer.shape)

        # flat = Flatten()(pool_layer)
        # flat = Bidirectional(LSTM(256))(flat)

        flat = TimeDistributed(Flatten())(pool_layer)
        flat = Bidirectional(LSTM(512))(flat)

        after_flat = Dense(64, activation=self.activation_type)(flat)
        # after_flat = Dense(32, activation=self.activation_type)(after_flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(
                0.001), activation='softmax')(after_flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
                0.001), activation='softmax')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=outputs)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])
        model.summary(print_fn=lambda x: logging.info(x))
        return model

    def create_deepHistone_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # batch size = 20

        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = Conv1D(filters=128, kernel_size=9, padding='same', input_shape=(
            sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        # concat_relu = ReLU()(concat)
        # DNA dense block
        # basic block
        batch_norm_layer_1 = BatchNormalization(
            epsilon=1e-05, momentum=0.1)(concat)
        relu_layer_1 = ReLU()(batch_norm_layer_1)
        conv_layer_1 = Conv1D(filters=128, kernel_size=9)(relu_layer_1)
        # basic block
        batch_norm_layer_1 = BatchNormalization(
            epsilon=1e-05, momentum=0.1)(conv_layer_1)
        relu_layer_1 = ReLU()(batch_norm_layer_1)
        conv_layer_1 = Conv1D(filters=128, kernel_size=9)(relu_layer_1)
        # basic block
        batch_norm_layer_1 = BatchNormalization(
            epsilon=1e-05, momentum=0.1)(conv_layer_1)
        relu_layer_1 = ReLU()(batch_norm_layer_1)
        conv_layer_1 = Conv1D(filters=128, kernel_size=9)(relu_layer_1)

        # pooling block
        batch_norm_layer_1 = BatchNormalization(
            epsilon=1e-05, momentum=0.1)(conv_layer_1)
        relu_layer_1 = ReLU()(batch_norm_layer_1)
        conv_layer_1 = Conv1D(filters=256, kernel_size=9)(relu_layer_1)
        pool_layer = MaxPooling1D(pool_size=4, strides=4)(conv_layer_1)

        # DNA dense block 2
        # basic block
        # batch_norm_layer_1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(pool_layer)
        # relu_layer_1 = ReLU()(batch_norm_layer_1)
        # conv_layer_1 = Conv1D(filters=256, kernel_size=9)(relu_layer_1)
        # # basic block
        # batch_norm_layer_1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv_layer_1)
        # relu_layer_1 = ReLU()(batch_norm_layer_1)
        # conv_layer_1 = Conv1D(filters=256, kernel_size=9)(relu_layer_1)
        # # basic block
        # batch_norm_layer_1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv_layer_1)
        # relu_layer_1 = ReLU()(batch_norm_layer_1)
        # conv_layer_1 = Conv1D(filters=256, kernel_size=9)(relu_layer_1)

        # # pooling block
        # batch_norm_layer_1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv_layer_1)
        # relu_layer_1 = ReLU()(batch_norm_layer_1)
        # conv_layer_1 = Conv1D(filters=512, kernel_size=9)(relu_layer_1)
        # pool_layer = MaxPooling1D(pool_size=4, strides=4)(conv_layer_1)

        # pool_layer = Lambda(sumUp)(relu_layer_2)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(925, activation=self.activation_type)(flat_layer)
        batch_norm_layer_2 = BatchNormalization(
            epsilon=1e-05, momentum=0.1)(dense_layer1)
        relu_layer_2 = ReLU()(batch_norm_layer_2)
        # dense_layer2 = Dense(32, activation=self.activation_type)(dense_layer1)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(relu_layer_2)

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model


    def createSiameseNetwork(self, embedding_network, sequence_shape):
        def euclidean_distance(vects):
            """Find the Euclidean distance between two vectors.
            Arguments:
                vects: List containing two tensors of same length.
            Returns:
                Tensor containing euclidean distance
                (as floating point value) between vectors.
            """
            x, y = vects
            sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
            return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
            # dist = tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
            # # dist -= tf.math.reduce_mean(dist)
            # return dist

        def loss(margin=1):
            """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

            Arguments:
                margin: Integer, defines the baseline for distance for which pairs
                        should be classified as dissimilar. - (default is 1).

            Returns:
                'constrastive_loss' function with data ('margin') attached.
            """

            # Contrastive loss = mean( (1-true_value) * square(prediction) +
            #                         true_value * square( max(margin-prediction, 0) ))
            def contrastive_loss(y_true, y_pred):
                """Calculates the constrastive loss.

                Arguments:
                    y_true: List of labels, each label is of type float32.
                    y_pred: List of predictions of same length as of y_true,
                            each label is of type float32.

                Returns:
                    A tensor containing constrastive loss as floating point value.
                """
                y_true = tf.cast(y_true, tf.float32)
                square_pred = tf.math.square(y_pred)
                margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
                return tf.math.reduce_mean(
                    (1 - y_true) * square_pred + (y_true) * margin_square
                    # (1 - y_true) * margin_square + (y_true) * square_pred
                )

            return contrastive_loss

        def contrastive_loss(y_true, y_pred, margin=1):
            """Calculates the constrastive loss.

            Arguments:
                y_true: List of labels, each label is of type float32.
                y_pred: List of predictions of same length as of y_true,
                        each label is of type float32.

            Returns:
                A tensor containing constrastive loss as floating point value.
            """
            y_true = tf.cast(y_true, tf.float32)
            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            return tf.math.reduce_mean(
                (1 - y_true) * square_pred + (y_true) * margin_square
                # (1 - y_true) * margin_square + (y_true) * square_pred
            )
        

        forward_input1 = keras.Input(
            shape=(sequence_shape[2], sequence_shape[3]), name='forward1')
        reverse_input1 = keras.Input(
            shape=(sequence_shape[2], sequence_shape[3]), name='reverse1')
        input_1 = [forward_input1, reverse_input1]

        forward_input2 = keras.Input(
            shape=(sequence_shape[2], sequence_shape[3]), name='forward2')
        reverse_input2 = keras.Input(
            shape=(sequence_shape[2], sequence_shape[3]), name='reverse2')
        input_2 = [forward_input2, reverse_input2]

        tower_1 = embedding_network(input_1)
        tower_2 = embedding_network(input_2)
        # euclidian distance
        # merge_layer = Lambda(euclidean_distance, name='euclidian_dist')([tower_1, tower_2])
        # normal_layer = BatchNormalization(name='normal')(merge_layer)
        # output_layer = Dense(1, name='sigmoid', activation="sigmoid")(normal_layer)
        # dense
        merge_layer = concatenate([tower_1, tower_2], axis=1)
        normal_layer = BatchNormalization(name='normal')(merge_layer)
        dense = Dense(128, activation=self.activation_type)(normal_layer)
        output_layer = Dense(1, name='sigmoid', activation="sigmoid")(dense)

        siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
        # siamese.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        siamese.compile(loss=loss(margin=1), optimizer='adam', metrics=['accuracy'])
        # siamese.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])

        return siamese
        


    def trainModel(self, model, processed_dict, seed=0, model_file=None):
        # print maximum length without truncation
        # np.set_printoptions(threshold=sys.maxsize)

        print('\n================ Training model ==================')

        fw_fasta = processed_dict["forward"]
        rc_fasta = processed_dict["reverse"]
        readout = processed_dict["readout"]

        print("Input size: " + str(fw_fasta.shape))

        if seed == 0:
            # seed = random.randint(1,1000)
            seed = 527

        x1_train, x1_test, y1_train, y1_test = train_test_split(
            fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            rc_fasta, readout, test_size=0.1, random_state=seed)

        # Copy the original target values for later uses
        y1_train_orig = y1_train.copy()
        y1_test_orig = y1_test.copy()

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        # Change it to categorical values
        y1_train = keras.utils.to_categorical(y1_train, 2)
        y1_test = keras.utils.to_categorical(y1_test, 2)

        print("\n=========================== Before model.fit ===================================")
        # Early stopping
        # checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,
                                 verbose=0, mode='auto', baseline=None, restore_best_weights=True)

        callbacks_list = [earlystop]
        # train the data
        history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs,
                  batch_size=self.batch_size, validation_split=0.1, verbose=1, callbacks=callbacks_list)
        # model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs,
        #   batch_size=self.batch_size, validation_split=0.1, verbose=2)
        print("=========================== After model.fit ===================================\n")
        # Save the entire model as a SavedModel.
        # model.save('my_model')
        # Save weights only: later used in self.filter_importance()
        # model.save_weights('./my_checkpoint')

        # save each convolution learned filters as txt file
        """
        motif_weight = model.get_weights()
        motif_weight = np.asarray(motif_weight[0])
        for i in range(int(self.filters)):
            x = motif_weight[:,:,i]
            berd = np.divide(np.exp(100*x), np.transpose(np.expand_dims(np.sum(np.exp(100*x), axis = 1), axis = 0), [1,0]))
            np.savetxt(os.path.join('./motif_files', 'filter_num_%d'%i+'.txt'), berd)
        """

        # logging.info(history.history)

        

        print("\n=========================== Predictions ===================================\n")

        ##########################################################
        # Prediction on train data
        pred_train = model.predict({'forward': x1_train, 'reverse': x2_train})
        # See which label has the highest confidence value
        predictions_train = np.argmax(pred_train, axis=1)

        true_pred, false_pred = 0, 0
        for count, value in enumerate(predictions_train):
            if y1_train_orig[count] == predictions_train[count]:
                true_pred += 1
            else:
                false_pred += 1

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        train_auc_score = sklearn.metrics.roc_auc_score(
            y1_train_orig, predictions_train)
        train_accuracy = true_pred/len(predictions_train)
        # train_accuracy = history.history['accuracy']

        ##########################################################
        # Prediction on test data
        pred_test = model.predict({'forward': x1_test, 'reverse': x2_test})
        # See which label has the highest confidence value
        predictions_test = np.argmax(pred_test, axis=1)

        true_pred, false_pred = 0, 0
        for count, value in enumerate(predictions_test):
            if y1_test_orig[count] == predictions_test[count]:
                true_pred += 1
            else:
                false_pred += 1

        test_auc_score = sklearn.metrics.roc_auc_score(
            y1_test_orig, predictions_test)
        test_accuracy = true_pred/len(predictions_test)
        # test_accuracy = history.history['val_accuracy']
        print("========================================================================\n")

        results = {
            "train_auc_score": train_auc_score,
            "train_accuracy": train_accuracy,
            "test_auc_score": test_auc_score,
            "test_accuracy": test_accuracy,
            "history": history,
            "seed": seed,
        }

        return results


    def trainSiameseNetwork(self, model, processed_pair_dict, seed=0, model_file=None):
        # print maximum length without truncation
        # np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = processed_pair_dict["forward"]
        rc_fasta = processed_pair_dict["reverse"]
        readout = processed_pair_dict["readout"]

        print("Input size: " + str(fw_fasta.shape))

        if seed == 0:
            # seed = random.randint(1,1000)
            seed = 527

        x1_train, x1_test, y1_train, y1_test = train_test_split(
            fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            rc_fasta, readout, test_size=0.1, random_state=seed)

        # Copy the original target values for later uses
        # y1_train_orig = y1_train.copy()
        # y1_test_orig = y1_test.copy()

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        # Change it to categorical values
        # y1_train = keras.utils.to_categorical(y1_train, 2)
        # y1_test = keras.utils.to_categorical(y1_test, 2)

        print("\n=========================== Before model.fit ===================================")
        # Early stopping
        # checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,
                                 verbose=0, mode='auto', baseline=None, restore_best_weights=False)

        callbacks_list = [earlystop]
        forward1 = np.transpose(x1_train, [1, 0, 2, 3])[0]
        reverse1 = np.transpose(x2_train, [1, 0, 2, 3])[0]
        forward2 = np.transpose(x1_train, [1, 0, 2, 3])[1]
        reverse2 = np.transpose(x2_train, [1, 0, 2, 3])[1]

        print('forward1 shape: ' + str(forward1.shape))
        print('reverse1 shape: ' + str(reverse1.shape))
        print('forward2 shape: ' + str(forward2.shape))
        print('reverse2 shape: ' + str(reverse2.shape))
        # y1_train_reshaped = np.array(y1_train).reshape(-1,1)
        # print('y shape: ', y1_train_reshaped.shape)

        # train the data
        history = model.fit([{'forward1': forward1, 'reverse1': reverse1}, {'forward2': forward2, 'reverse2': reverse2}], y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=2, callbacks=callbacks_list)
        # model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs,
        #   batch_size=self.batch_size, validation_split=0.1, verbose=2)
        print("=========================== After model.fit ===================================\n")
        # Save the entire model as a SavedModel.
        # model.save('my_model')
        # Save weights only: later used in self.filter_importance()
        # model.save_weights('./my_checkpoint')
        # print(model.layers)
        # print()


        # save each convolution learned filters as txt file
        """
        motif_weight = model.get_weights()
        motif_weight = np.asarray(motif_weight[0])
        for i in range(int(self.filters)):
            x = motif_weight[:,:,i]
            berd = np.divide(np.exp(100*x), np.transpose(np.expand_dims(np.sum(np.exp(100*x), axis = 1), axis = 0), [1,0]))
            np.savetxt(os.path.join('./motif_files', 'filter_num_%d'%i+'.txt'), berd)
        """

        # print(history.history)

        # logging.info("\n=========================== Predictions ===================================\n")
        print("\n=========================== Predictions ===================================\n")

        ##########################################################
        


        # Prediction on train data
        predictions = model.predict([{'forward1': forward1, 'reverse1': reverse1}, {'forward2': forward2, 'reverse2': reverse2}])
        # See which label has the highest confidence value

        print('siamese predictions:')
        print(predictions[:20])
        # print(predictions[:50])
        n, bins, patches = plt.hist(predictions, bins=50)
        plt.savefig('dist.png')
        mn, mx, mean = np.min(predictions), np.max(predictions), np.mean(predictions)
        print('predictions: min', mn, 'max', mx,'mean', mean)

        predictions_train = []
        for d in predictions:
            if d > .5:
                predictions_train.append(1)
            else:
                predictions_train.append(0)


        print('y_train :')
        print(y1_train[:50])
        print('predictions train:')
        print(predictions_train[:50])

        # true_pred, false_pred = 0, 0
        # for count, value in enumerate(predictions_train):
        #     if y1_train[count] == predictions_train[count]:
        #         true_pred += 1
        #     else:
        #         false_pred += 1
        predictions_train = np.array(predictions_train)
        true_pred = np.sum(y1_train == predictions_train)

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        train_auc_score = sklearn.metrics.roc_auc_score(
            y1_train, predictions_train)
        train_accuracy = true_pred/len(predictions_train)
        # train_accuracy = history.history['accuracy']
        print('train accuracy: ', train_accuracy)

        ##########################################################
        # Prediction on test data
        forward1 = np.transpose(x1_test, [1, 0, 2, 3])[0]
        reverse1 = np.transpose(x2_test, [1, 0, 2, 3])[0]
        forward2 = np.transpose(x1_test, [1, 0, 2, 3])[1]
        reverse2 = np.transpose(x2_test, [1, 0, 2, 3])[1]
        predictions = model.predict({'forward1': forward1, 'reverse1': reverse1, 'forward2': forward2, 'reverse2': reverse2})
        # See which label has the highest confidence value
        predictions_test = []
        for d in predictions:
            if d > .5:
                predictions_test.append(1)
            else:
                predictions_test.append(0)

        predictions_test = np.array(predictions_test)

        # true_pred, false_pred = 0, 0
        # for count, value in enumerate(predictions_test):
        #     if y1_test[count] == predictions_test[count]:
        #         true_pred += 1
        #     else:
        #         false_pred += 1
        true_pred = np.sum(y1_test == predictions_test)

        test_auc_score = sklearn.metrics.roc_auc_score(
            y1_test, predictions_test)
        test_accuracy = true_pred/len(predictions_test)
        print('test accuracy: ', test_accuracy)
        # test_accuracy = history.history['val_accuracy']
        print("========================================================================\n")

        results = {
            "train_auc_score": train_auc_score,
            "train_accuracy": train_accuracy,
            "test_auc_score": test_auc_score,
            "test_accuracy": test_accuracy,
            "history": history,
            "seed": seed,
        }

        return results


    def trainModelOneInputLayer(self, model, processed_dict, seed=0):
        # print maximum length without truncation
        # np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = processed_dict["forward"]
        rc_fasta = processed_dict["reverse"]
        readout = processed_dict["readout"]

        print("Input size: " + str(fw_fasta.shape))

        if seed == 0:
            # seed = random.randint(1,1000)
            seed = 527

        input_data = np.concatenate((fw_fasta, rc_fasta), axis=0)
        output_data = np.concatenate((readout, readout), axis=0)

        train_input_data, test_input_data, train_output_data, test_output_data = train_test_split(
            input_data, output_data, test_size=0.1, random_state=seed)

        # x1_train, x1_test, y1_train, y1_test = train_test_split(
        #     fw_fasta, readout, test_size=0.1, random_state=seed)
        # # split for reverse complemenet sequences
        # x2_train, x2_test, y2_train, y2_test = train_test_split(
        #     rc_fasta, readout, test_size=0.1, random_state=seed)

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        # train_input_data = np.concatenate((x1_train, x2_train), axis=0)
        # train_output_data = np.concatenate((y1_train, y2_train), axis=0)
        # test_input_data = np.concatenate((x1_test, x2_test), axis=0)
        # test_output_data = np.concatenate((y1_test, y2_test), axis=0)
        # input_data = {'forward': x1_train, 'reverse': x2_train}

        # Change it to categorical values
        train_output_data_categorical = keras.utils.to_categorical(
            train_output_data, 2)

        print("\n=========================== Before model.fit ===================================")
        # train the data
        model.fit(train_input_data, train_output_data_categorical, epochs=self.epochs,
                  batch_size=self.batch_size, validation_split=0.1, verbose=1)
        print("=========================== After model.fit ===================================\n")
        # Save the entire model as a SavedModel.
        # model.save('my_model')
        # Save weights only: later used in self.filter_importance()
        # model.save_weights('./my_checkpoint')

        # save each convolution learned filters as txt file
        """
        motif_weight = model.get_weights()
        motif_weight = np.asarray(motif_weight[0])
        for i in range(int(self.filters)):
            x = motif_weight[:,:,i]
            berd = np.divide(np.exp(100*x), np.transpose(np.expand_dims(np.sum(np.exp(100*x), axis = 1), axis = 0), [1,0]))
            np.savetxt(os.path.join('./motif_files', 'filter_num_%d'%i+'.txt'), berd)
        """

        print("\n=========================== Predictions ===================================\n")

        ##########################################################
        # Prediction on train data
        pred_train = model.predict(train_input_data)
        # See which label has the highest confidence value
        predictions_train = np.argmax(pred_train, axis=1)

        true_pred, false_pred = 0, 0
        for count, value in enumerate(predictions_train):
            if train_output_data[count] == predictions_train[count]:
                true_pred += 1
            else:
                false_pred += 1

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        train_auc_score = sklearn.metrics.roc_auc_score(
            train_output_data, predictions_train)
        train_accuracy = true_pred/len(predictions_train)
        print('train-set auc score is: ' + str(train_auc_score))
        print('train-set accuracy is: ' + str(train_accuracy))

        ##########################################################
        # Prediction on test data
        pred_test = model.predict(test_input_data)
        # See which label has the highest confidence value
        predictions_test = np.argmax(pred_test, axis=1)

        true_pred, false_pred = 0, 0
        for count, value in enumerate(predictions_test):
            if test_output_data[count] == predictions_test[count]:
                true_pred += 1
            else:
                false_pred += 1

        test_auc_score = sklearn.metrics.roc_auc_score(
            test_output_data, predictions_test)
        test_accuracy = true_pred/len(predictions_test)
        print('\ntest-set auc score is: ' + str(test_auc_score))
        print('test-set accuracy is: ' + str(test_accuracy))
        print("========================================================================\n")

        results = {
            "train_auc_score": train_auc_score,
            "train_accuracy": train_accuracy,
            "test_auc_score": test_auc_score,
            "test_accuracy": test_accuracy,
            "seed": seed,
        }

        return results

    def trainModelWithHardwareSupport(self, model, processed_dict, with_gpu=False):
        if with_gpu == True:
            device_name = tf.test.gpu_device_name()
            print(device_name)
            if device_name != '/device:GPU:0':
                print(
                    '\n\nThis error most likely means that this notebook is not '
                    'configured to use a GPU.  Change this in Notebook Settings via the '
                    'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
            raise SystemError('GPU device not found')

            with tf.device('/device:GPU:0'):
                trainModel(model, processed_dict)

        else:
            trainModel(model, processed_dict)


    def get_model(self, model_type, dims):
        if model_type == 'meuseum':
            return self.create_meuseum_model(dims)
        elif model_type == 'basic':
            return self.create_basic_model(dims)
        elif model_type == 'cnn_lstm':
            return self.create_CNNLSTM_model(dims)


    def cross_val(self, model_type, processed_dict, fold, seed):
        fw_fasta = processed_dict["forward"]
        rc_fasta = processed_dict["reverse"]
        readout = processed_dict["readout"]

        forward_shuffle, readout_shuffle = shuffle(
            fw_fasta, readout, random_state=seed)
        reverse_shuffle, readout_shuffle = shuffle(
            rc_fasta, readout, random_state=seed)

        # save the information of 10 folds auc scores
        train_auc_scores = []
        test_auc_scores = []
        train_accs = []
        test_accs = []

        # Provides train/test indices to split data in train/test sets.
        kFold = StratifiedKFold(n_splits=fold)
        # ln = np.zeros(len(readout_shuffle))
        print("\n============================= Cross val ================================\n")
        print('fold: ' + str(fold) + ' seed: ' + str(seed))
        for train_indexes, test_indexes in kFold.split(forward_shuffle, readout_shuffle):
            print("\n\n---------- fold -------------")
            fwd_train = forward_shuffle[train_indexes]
            fwd_test = forward_shuffle[test_indexes]
            rc_train = reverse_shuffle[train_indexes]
            rc_test = reverse_shuffle[test_indexes]
            y_train = readout_shuffle[train_indexes]
            y_test = readout_shuffle[test_indexes]

            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            y_train_orig = y_train.copy()
            y_test_orig = y_test.copy()
            y_train = keras.utils.to_categorical(y_train, 2)
            y_test = keras.utils.to_categorical(y_test, 2)
            # Early stopping
            model = self.get_model(model_type, fw_fasta.shape)
            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs,
                                batch_size=self.batch_size, validation_split=0.0, validation_data=({'forward': fwd_test, 'reverse': rc_test}, y_test), verbose=2, callbacks=[earlystop])

            # Without early stopping
            # model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

            # pred_train = model.predict({'forward': x1_test, 'reverse': x2_test})

            pred_train = model.predict(
                {'forward': fwd_train, 'reverse': rc_train})
            predictions_train = np.argmax(pred_train, axis=1)

            true_pred = 0
            false_pred = 0
            for count, value in enumerate(predictions_train):
                if y_train_orig[count] == predictions_train[count]:
                    true_pred += 1
                else:
                    false_pred += 1

            auc_score = sklearn.metrics.roc_auc_score(
                y_train_orig, predictions_train)
            print('train-set auc score is: ' + str(auc_score))
            train_auc_scores.append(auc_score)
            train_accs.append(true_pred/(true_pred+false_pred))

            ##########################################################
            # Apply on test set
            pred_test = model.predict(
                {'forward': fwd_test, 'reverse': rc_test})
            predictions_test = np.argmax(pred_test, axis=1)

            true_pred = 0
            false_pred = 0
            for count, value in enumerate(predictions_test):
                if y_test_orig[count] == predictions_test[count]:
                    true_pred += 1
                else:
                    false_pred += 1

            auc_score = sklearn.metrics.roc_auc_score(
                y_test_orig, predictions_test)
            #auc_score = sklearn.metrics.roc_auc_score(y_test, pred)
            print('test-set auc score is: ' + str(auc_score))
            test_auc_scores.append(auc_score)
            test_accs.append(true_pred/(true_pred+false_pred))

        # logging.info('auc scores: ' + str(test_auc_scores))
        print('Mean train auc_scores of 5-fold cv is ' +
              str(np.mean(train_auc_scores)))
        
        # logging.info('accuracies: ', str(test_accs))
        print('Mean test auc_scores of 10-fold cv is ' +
              str(np.mean(test_auc_scores)))

        results = {
            "train_auc_score": np.mean(train_auc_scores),
            "train_accuracy": np.mean(train_accs),
            "test_auc_score": np.mean(test_auc_scores),
            "test_accuracy": np.mean(test_accs),
            "seed": seed,
        }

        return results
