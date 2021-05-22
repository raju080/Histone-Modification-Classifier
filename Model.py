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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU, Add
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.utils import shuffle

from ConvolutionLayer import ConvolutionLayer

# Reproducibility
# seed = random.randint(1, 1000)
seed = 527
np.random.seed(seed)
tf.random.set_seed(seed)


class Model:
    def __init__(self, filters, kernel_size, pool_type, regularizer, activation_type, epochs, batch_size):
        """initialize basic parameters"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size

        # Pringting model parameters
        self.parameterPrint()

    def parameterPrint(self):
        print(
            "\n=========================== Parameters ===================================")
        print("# of Filters: " + str(self.filters))
        print("Kernel size: " + str(self.kernel_size))
        print("Pool type: " + self.pool_type)
        print("regularizer: " + self.regularizer)
        print("activation_type: " + self.activation_type)
        print("epochs: " + str(self.epochs))
        print("batch_size: " + str(self.batch_size))
        print(
            "============================================================================\n")


    def create_meuseum_model(self, dims, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
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
        first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation_type,
                                       alpha=alpha, beta=beta, bkg_const=bkg_const, data_format='channels_last', use_bias=True)

        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)

        concat = concatenate([fw, bw], axis=1)
        print("Concat shape-----------------", concat.shape)
        pool_size_input = concat.shape[1]
        print("pool_size_input--------------", pool_size_input)
        concat_relu = ReLU()(concat)
        #concat = Dense(1, activation= 'sigmoid')(concat)

        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=pool_size_input)(concat)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(pool_size=pool_size_input)(concat)
        elif self.pool_type == 'Custom':
            def out_shape(input_shape):
                shape = list(input_shape)
                print(input_shape)
                shape[0] = 10
                return tuple(shape)
            #model.add(Lambda(top_k, arguments={'k': 10}))

            def top_k(inputs, k):
                # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
                print(inputs.shape)
                inputs2 = tf.transpose(inputs, [0, 2, 1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0, 2, 1])

            pool_layer = Lambda(top_k, arguments={'k': 2})(concat_relu)
            # pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
            pool_layer = MaxPooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'Custom_sum':
            # apply relu function before custom_sum functions
            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
                return new_vals
            pool_layer = Lambda(summed_up)(concat_relu)
            # pool_layer = MaxPooling1D(pool_size=2)(pool_layer)
            # pool_layer = Lambda(summed_up)(concat)
        else:
            raise NameError('Set the pooling layer name correctly')

        flat = Flatten()(pool_layer)

        after_flat = Dense(32)(flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            # outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(
                0.001), activation='softmax')(after_flat)
        elif self.regularizer == 'L_2':
            #outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
                0.001), activation='softmax')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')

        # weight_forwardin_0=model.layers[0].get_weights()[0]
        # print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=outputs)

        #model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model


    def create_basic_model(self, sequence_shape):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        # input layers
        forward_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='forward')
        reverse_input = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]), name='reverse')

        conv_layer_1 = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size,
                                        activation=self.activation_type, input_shape=(sequence_shape[1], sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer = ReLU()(concat_layer)

        conv_layer_2 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
                              activation=self.activation_type)(relu_layer)
        relu_layer_2 = ReLU()(conv_layer_2)

        pool_layer = Lambda(sumUp)(relu_layer_2)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(32, activation=self.activation_type)(flat_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer1)

        model = keras.Model(
            inputs=[forward_input, reverse_input], outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model

    def create_Vanilla_CNN_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # input layers
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        # input layers
        input_layer = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]))

        conv_layer_1 = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size,
                                        activation=self.activation_type, input_shape=(sequence_shape[1], sequence_shape[2]))(input_layer)
    
        relu_layer = ReLU()(conv_layer_1)
        pool_layer = Lambda(sumUp)(relu_layer)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(32, activation=self.activation_type)(flat_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer1)

        model = keras.Model(
            inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model




    def create_Multi_CNN_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
            return new_vals
        pool_size_input = 4
        # input layers
        input_layer = keras.Input(
            shape=(sequence_shape[1], sequence_shape[2]))

        conv_layer_1 = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size,
                                        activation=self.activation_type, input_shape=(sequence_shape[1], sequence_shape[2]))(input_layer)

        pool_layer_1 = MaxPooling1D(pool_size=pool_size_input)(conv_layer_1)
        relu_layer_1 = ReLU()(pool_layer_1)

        conv_layer_2 = Conv1D(filters=self.filters//2, kernel_size=self.kernel_size,
                              activation=self.activation_type)(relu_layer_1)
        pool_layer_2 = MaxPooling1D(pool_size=pool_size_input)(conv_layer_2)
        relu_layer_2 = ReLU()(pool_layer_2)

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

        dense_layer2 = Dense(32, activation=self.activation_type)(flat_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
            0.001), activation='softmax')(dense_layer1)

        model = keras.Model(
            inputs=input_layer, outputs=output_layer)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model


    def create_LSTM_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # input layers
        forward_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'forward')
        reverse_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'reverse')
        print("Forward shape----------------",forward_input)
        print("Reverse shape----------------",reverse_input)
        model = Sequential()
        n_features = 512
        n_neurons = 5
        first_layer = LSTM(n_neurons, return_sequences =True, input_shape=(sequence_shape[1],sequence_shape[2],n_features))
        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)
        x = concatenate([fw, bw], axis=1)
        # x = Dense(n_features,activation="sigmoid")(x)
        second_layer = LSTM(n_neurons*2, input_shape=(x.shape[1],x.shape[2],n_features/2))
        x = second_layer(x)
        flat = Flatten()(x)
        # model.add(flat)
        after_flat = Dense(32, activation=self.activation_type)(flat)
        # model.add(after_flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= 'sigmoid')(after_flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'sigmoid')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')
        # model.add(outputs)

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(inputs=[forward_input, reverse_input], outputs=outputs)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model

    def create_CNNLSTM_model(self, sequence_shape, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # input layers
        forward_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'forward')
        reverse_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'reverse')
        print("Forward shape----------------",forward_input)
        print("Reverse shape----------------",reverse_input)
        model = Sequential()

        first_layer = Conv1D(filters=self.filters/2, kernel_size=self.kernel_size, activation=self.activation_type, input_shape=(sequence_shape[1],sequence_shape[2]))
        print(first_layer)
        first_layer = TimeDistributed(first_layer)
        print(first_layer)
        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)
        print("after Conv1D fw shape----------------",fw.shape)
        print("after Conv1D bw shape----------------",bw.shape)
        concat = concatenate([fw, bw], axis=1)
        print("Concat shape-----------------",concat.shape)
        # pool_size_input = concat.shape[1]
        #model.add(first_layer)
        # pool_size_input = 5
        pool_size_input = 2
        if self.pool_type == 'Max':
            pool_layer = TimeDistributed(MaxPooling1D(pool_size=pool_size_input))(concat)
        elif self.pool_type == 'Ave':
            pool_layer = TimeDistributed(AveragePooling1D(pool_size=pool_size_input))(concat)
        else:
            raise NameError('Set the pooling layer name correctly')
        # model.add(pool_layer)
        print("After Maxpooling shape-----------------",pool_layer.shape)
        second_layer = TimeDistributed(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation_type, input_shape=(pool_layer.shape[1],pool_layer.shape[2])))
        x = second_layer(pool_layer)
        print("after Conv1D x shape----------------",x.shape)
        # x = MaxPooling1D(pool_size=x.shape[1])(x)
        x = TimeDistributed(MaxPooling1D(pool_size=5))(x)
        print("after MaxPooling x shape----------------",x.shape)

        # third_layer = TimeDistributed(Conv1D(filters=self.filters*2, kernel_size=self.kernel_size, activation=self.activation_type, input_shape=(x.shape[1],x.shape[2])))
        # x = third_layer(x)
        # print("after Conv1D x shape----------------",x.shape)
        # x = TimeDistributed(MaxPooling1D(pool_size=x.shape[1]))(x)
        # print("after MaxPooling x shape----------------",x.shape)

        flat = TimeDistributed(Flatten())(x)
        # flat = LSTM(4)(flat)
        # model.add(flat)
        after_flat = Dense(32, activation=self.activation_type)(flat)
        # model.add(after_flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= 'sigmoid')(after_flat)
        elif self.regularizer == 'L_2':
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'sigmoid')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')
        # model.add(outputs)

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(inputs=[forward_input, reverse_input], outputs=outputs)

        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model


    def trainModel(self, model, processed_dict, seed=0):
        # print maximum length without truncation
        # np.set_printoptions(threshold=sys.maxsize)

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
        callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        # train the data
        model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=2, callbacks=[callback])
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
        print('train-set auc score is: ' + str(train_auc_score))
        print('train-set accuracy is: ' + str(train_accuracy))

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

        x1_train, x1_test, y1_train, y1_test = train_test_split(
            fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            rc_fasta, readout, test_size=0.1, random_state=seed)

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        train_input_data = np.concatenate((x1_train, x2_train), axis=0)
        train_output_data = np.concatenate((y1_train, y2_train), axis=0)
        test_input_data = np.concatenate((x1_test, x2_test), axis=0)
        test_output_data = np.concatenate((y1_test, y2_test), axis=0)
        # input_data = {'forward': x1_train, 'reverse': x2_train}

        # Change it to categorical values
        train_output_data_categorical = keras.utils.to_categorical(train_output_data, 2)

        print("\n=========================== Before model.fit ===================================")
        # Early stopping
        callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        # train the data
        model.fit(train_input_data, train_output_data_categorical, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, verbose=2, callbacks=[callback])
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


    def cross_val(self, model, processed_dict):
        fw_fasta = processed_dict["forward"]
        rc_fasta = processed_dict["reverse"]
        readout = processed_dict["readout"]

        #if self.activation_type == 'linear':
        #    readout = np.log2(readout)
        #    readout = np.ndarray.tolist(readout)

        forward_shuffle, readout_shuffle = shuffle(fw_fasta, readout, random_state=seed)
        reverse_shuffle, readout_shuffle = shuffle(rc_fasta, readout, random_state=seed)
        readout_shuffle = np.array(readout_shuffle)
        #readout_shuffle2 = np.array(readout_shuffle2)
        # initialize metrics to save values
        metrics = []

        # save the information of 10 folds auc scores
        train_auc_scores = []
        test_auc_scores = []

        # Provides train/test indices to split data in train/test sets.
        kFold = StratifiedKFold(n_splits=5)
        ln = np.zeros(len(readout_shuffle))
        print("readout shuffle length---------",len(readout_shuffle))
        for train, test in kFold.split(ln, ln):
            model = None
            #model, model2 = self.create_model(processed_dict)
            # print("----------",train,test)
            fwd_train = forward_shuffle[train]
            fwd_test = forward_shuffle[test]
            rc_train = reverse_shuffle[train]
            rc_test = reverse_shuffle[test]
            y_train = readout_shuffle[train]
            y_test = readout_shuffle[test]
            #y_train = readout_shuffle2[train]
            #y_test = readout_shuffle2[test]

            # fwd_train = np.asarray(fwd_train)
            # fwd_test = np.asarray(fwd_test)
            # rc_train = np.asarray(rc_train)
            # rc_test = np.asarray(rc_test)
            # y1_train = np.asarray(y1_train)
            # y1_test = np.asarray(y1_test)
            # y2_train = np.asarray(y2_train)
            # y2_test = np.asarray(y2_test)

            # # change from list to numpy array
            # y1_train = np.asarray(y1_train)
            # y1_test = np.asarray(y1_test)
            # y2_train = np.asarray(y2_train)
            # y2_test = np.asarray(y2_test)

            # # Copy the original target values for later uses
            # y1_train_orig = y1_train.copy()
            # y1_test_orig = y1_test.copy()

            # # if we want to merge two training dataset
            # # comb = np.concatenate((y1_train, y2_train))

            # ## Change it to categorical values
            # y1_train = keras.utils.to_categorical(y1_train, 2)
            # y1_test = keras.utils.to_categorical(y1_test, 2)
            # model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            y_train_orig = y_train.copy()
            y_test_orig = y_test.copy()
            y_train = keras.utils.to_categorical(y_train, 2)
            y_test = keras.utils.to_categorical(y_test, 2)
            # Early stopping
            callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0, callbacks = [callback],verbose=2)

            # Without early stopping
            # model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

            # pred_train = model.predict({'forward': x1_test, 'reverse': x2_test})

            pred_train = model.predict({'forward': fwd_train, 'reverse': rc_train})
            predictions_train = np.argmax(pred_train, axis=1)
            print(y_train_orig[0:10])
            print(predictions_train[0:10])

            true_pred = 0
            false_pred = 0
            for count, value in enumerate(predictions_train):
                if y_train_orig[count] == predictions_train[count]:
                    true_pred += 1
                else:
                    false_pred += 1

            print('Total number of train-set predictions is: ' + str(len(y_train)))
            print('Number of correct train-set predictions is: ' + str(true_pred))
            print('Number of incorrect train-set predictions is: ' + str(false_pred))
            auc_score = sklearn.metrics.roc_auc_score(y_train_orig, predictions_train)
            print('train-set auc score is: ' + str(auc_score))
            print('train-set seed number is: ' + str(seed))
            # auc_score = sklearn.metrics.roc_auc_score(y_train, pred_train)
            # print('train-set auc score is: ' + str(auc_score))
            # print('train-set seed number is: ' + str(seed))
            train_auc_scores.append(auc_score)

            ##########################################################
            #Apply on test set
            pred_test = model.predict({'forward': fwd_test, 'reverse': rc_test})
            predictions_test = np.argmax(pred_test, axis=1)
            print(y_test_orig[0:10])
            print(predictions_test[0:10])

            true_pred = 0
            false_pred = 0
            for count, value in enumerate(predictions_test):
                if y_test_orig[count] == predictions_test[count]:
                    true_pred += 1
                else:
                    false_pred += 1

            print('Total number of test-set predictions is: ' + str(len(y_test)))
            print('Number of correct test-set predictions is: ' + str(true_pred))
            print('Number of incorrect test-set predictions is: ' + str(false_pred))
            auc_score = sklearn.metrics.roc_auc_score(y_test_orig, predictions_test)
            #auc_score = sklearn.metrics.roc_auc_score(y_test, pred)
            print('test-set auc score is: ' + str(auc_score))
            print('test-set seed number is: ' + str(seed))
            test_auc_scores.append(auc_score)

        print('seed number = %d' %seed)
        print(train_auc_scores)
        print('Mean train auc_scores of 10-fold cv is ' + str(np.mean(train_auc_scores)))
        print(test_auc_scores)
        print('Mean test auc_scores of 10-fold cv is ' + str(np.mean(test_auc_scores)))