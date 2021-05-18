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

#Reproducibility
seed = random.randint(1,1000)

np.random.seed(seed)
tf.random.set_seed(seed)

class ConvolutionLayer(Conv1D):
    def __init__(self, 
                filters,
                kernel_size,
                data_format='channels_last',
                alpha=100, 
                beta=0.01, 
                bkg_const=[0.25, 0.25, 0.25, 0.25],
                padding='valid',
                activation="relu",
                use_bias=False,
                kernel_initializer='glorot_uniform',
                __name__ = 'ConvolutionLayer',
                **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.bkg_const = bkg_const
        self.run_value = 1

    def call(self, inputs):

      ## shape of self.kernel is (12, 4, 512)
      ##the type of self.kernel is <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

        # print("self.run value is", self.run_value)
        if self.run_value > 2:

            x_tf = self.kernel  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            # self.alpha = 10
            self.beta = 1/self.alpha
            bkg = tf.constant(self.bkg_const)
            bkg_tf = tf.cast(bkg, tf.float32)
            # filt_list = tf.map_fn(lambda x: tf.math.scalar_mul(self.beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(self.alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis = 1), axis = 1)), tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(self.alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis = 1), axis = 1))), axis = 1)), axis = 1)), tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)
            filt_list = tf.map_fn(
                lambda x: tf.math.scalar_mul(
                    self.beta,
                    tf.subtract(
                        tf.subtract(
                            tf.subtract(
                                tf.math.scalar_mul(self.alpha, x),
                                tf.expand_dims(
                                    tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1
                                ),
                            ),
                            tf.expand_dims(
                                tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.subtract(
                                                tf.math.scalar_mul(self.alpha, x),
                                                tf.expand_dims(
                                                    tf.math.reduce_max(
                                                        tf.math.scalar_mul(self.alpha, x), axis=1
                                                    ),
                                                    axis=1,
                                                ),
                                            )
                                        ),
                                        axis=1,
                                    )
                                ),
                                axis=1,
                            ),
                        ),
                        tf.math.log(
                            tf.reshape(
                                tf.tile(bkg_tf, [tf.shape(x)[0]]),
                                [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                            )
                        ),
                    ),
                ),
                x_tf,
            )

            # filt_list = tf.math.scalar_mul(
            #     self.beta,
            #     tf.subtract(
            #         tf.subtract(
            #             tf.subtract(
            #                 tf.math.scalar_mul(self.alpha, x),
            #                 tf.expand_dims(
            #                     tf.math.reduce_max(
            #                         tf.math.scalar_mul(self.alpha, x),
            #                         axis=1
            #                     ),
            #                     axis=1
            #                 )
            #             ),
            #             tf.expand_dims(
            #                 tf.math.log(
            #                     tf.math.reduce_sum(
            #                         tf.math.exp(
            #                             tf.subtract(
            #                                 tf.math.scalar_mul(self.alpha, x),
            #                                 tf.expand_dims(
            #                                     tf.math.reduce_max(
            #                                         tf.math.scalar_mul(self.alpha, x),
            #                                         axis=1
            #                                     ),
            #                                     axis=1
            #                                 )
            #                             )
            #                         ),
            #                         axis=1
            #                     )
            #                 ),
            #                 axis=1
            #             )
            #         ),
            #         tf.math.log(
            #             tf.reshape(
            #                 tf.tile(
            #                     bkg_tf,
            #                     [ tf.shape(x)[0] ]
            #                 ),
            #                 [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ]
            #             )
            #         )
            #     )
            # )
            #print("type of output from map_fn is", type(filt_list)) ##type of output from map_fn is <class 'tensorflow.python.framework.ops.Tensor'>   shape of output from map_fn is (10, 12, 4)
            #print("shape of output from map_fn is", filt_list.shape)
            #transf = tf.reshape(filt_list, [12, 4, self.filters]) ##12, 4, 512
            transf = tf.transpose(filt_list, [1, 2, 0])
            ##type of transf is <class 'tensorflow.python.framework.ops.Tensor'>
            outputs = self._convolution_op(inputs, transf) ## type of outputs is <class 'tensorflow.python.framework.ops.Tensor'>

        else:
            outputs = self._convolution_op(inputs, self.kernel)


        self.run_value += 1
        return outputs

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
        print("\n=========================== Parameters ===================================")
        print("# of Filters: " + str(self.filters))
        print("Kernel size: " + str(self.kernel_size))
        print("Pool type: " + self.pool_type)
        print("regularizer: " + self.regularizer)
        print("activation_type: " + self.activation_type)
        print("epochs: " + str(self.epochs))
        print("batch_size: " + str(self.batch_size))
        print("============================================================================\n")

    def create_meuseum_model(self, dims, alpha=100, beta=0.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))


        def auroc(y_true, y_pred):
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

        # To build this model with the functional API,
        # you would start by creating an input node:
        # dims[1] = sequence length
        # dims[2] = 4   (one hot encode for ACGT)
        forward_input = keras.Input(shape=(dims[1],dims[2]), name = 'forward')
        reverse_input = keras.Input(shape=(dims[1],dims[2]), name = 'reverse')

        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', input_shape=(dims[1],dims[2]), use_bias = False)
        ## with trainable = False
        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, kernel_initializer = my_init, data_format='channels_last', input_shape=(dims[1],dims[2]), use_bias = False, trainable=False)
        first_layer = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation_type, alpha=alpha, beta=beta, bkg_const=bkg_const, data_format='channels_last', use_bias = True)

        fw = first_layer(forward_input)
        bw = first_layer(reverse_input)

        concat = concatenate([fw, bw], axis=1)
        print("Concat shape-----------------",concat.shape)
        pool_size_input = concat.shape[1]
        print("pool_size_input--------------",pool_size_input)
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
                inputs2 = tf.transpose(inputs, [0,2,1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0,2,1])

            pool_layer = Lambda(top_k, arguments={'k': 2})(concat_relu)
            # pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
            pool_layer = MaxPooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'Custom_sum':
            ## apply relu function before custom_sum functions
            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis = 1, keepdims = True)
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
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= 'softmax')(after_flat)
        elif self.regularizer == 'L_2':
            #outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'softmax')(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        # print("creating the model")
        model = keras.Model(inputs=[forward_input, reverse_input], outputs=outputs)

        #model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model


    def create_basic_model(self, sequence_shape):
        def sumUp(inputs):
            new_vals = tf.math.reduce_sum(inputs, axis = 1, keepdims = True)
            return new_vals
        # input layers
        forward_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'forward')
        reverse_input = keras.Input(shape=(sequence_shape[1],sequence_shape[2]), name = 'reverse') 

        conv_layer_1 = ConvolutionLayer(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation_type, input_shape=(sequence_shape[1],sequence_shape[2]))
        conv_layer_1_fw = conv_layer_1(forward_input)
        conv_layer_1_rs = conv_layer_1(reverse_input)

        concat_layer = concatenate([conv_layer_1_fw, conv_layer_1_rs], axis=1)
        relu_layer = ReLU()(concat_layer)

        conv_layer_2 = Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation_type)(relu_layer)
        relu_layer_2 = ReLU()(conv_layer_2)

        pool_layer = Lambda(sumUp)(relu_layer_2)
        flat_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(32, activation=self.activation_type)(flat_layer)

        # Binary classification with 2 output neurons
        output_layer = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation= 'softmax')(dense_layer1)

        model = keras.Model(inputs=[forward_input, reverse_input], outputs=output_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

        return model
    
    def runModelWithHardwareSupport(self, model, processed_dict, with_gpu=False):

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
                runModel(model, processed_dict)
        
        else: 
            runModel(model, processed_dict)


    def runModel(self, model, processed_dict, seed=0):
        # print maximum length without truncation
        # np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = processed_dict["forward"]
        rc_fasta = processed_dict["reverse"]
        readout = processed_dict["readout"]

        if seed == 0:
            # seed = random.randint(1,1000)
            seed = 527

        x1_train, x1_test, y1_train, y1_test = train_test_split(fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # Copy the original target values for later uses
        y1_train_orig = y1_train.copy()
        y1_test_orig = y1_test.copy()

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        ## Change it to categorical values
        y1_train = keras.utils.to_categorical(y1_train, 2)
        y1_test = keras.utils.to_categorical(y1_test, 2)

        print("\n=========================== Before model.fit ===================================")
        # train the data
        model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs,  batch_size=self.batch_size, validation_split=0.1, verbose=2)
        print("=========================== After model.fit ===================================\n")
        ## Save the entire model as a SavedModel.
        ##model.save('my_model')
        # Save weights only: later used in self.filter_importance()
        #model.save_weights('./my_checkpoint')

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
        # print('Total number of train-set predictions is: ' + str(len(y1_train_orig)))
        # print('Number of correct train-set predictions is: ' + str(true_pred))
        # print('Number of incorrect train-set predictions is: ' + str(false_pred))

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        train_auc_score = sklearn.metrics.roc_auc_score(y1_train_orig, predictions_train)
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
        # print('Total number of test-set predictions is: ' + str(len(y1_test_orig)))
        # print('Number of correct test-set predictions is: ' + str(true_pred))
        # print('Number of incorrect test-set predictions is: ' + str(false_pred))

        test_auc_score = sklearn.metrics.roc_auc_score(y1_test_orig, predictions_test)
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
