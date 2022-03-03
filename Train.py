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

# constants
epigenome = 'E116'
histone = 'H3K27ac'
seed = 527
total_seq_postfix = ""
cross_val_fold = 5
crossval_seq_postfix = '' if cross_val_fold == 1 else '_fold'+str(cross_val_fold)
# types: basic, meuseum, vanilla_cnn, multi_cnn2, multi_cnn4, bpnet, cnn_lstm, deephistone
model_type = 'cnn_lstm'


# E116

output_file_name = 'all_results.csv'
model_number = sum(1 for line in open(output_file_name))

# output file to save the model after training
model_file = 'TrainedModels/E116/model' + total_seq_postfix + '_' + model_type + crossval_seq_postfix + '_' + str(model_number) + '.h5'

# model_file = 'TrainedModels/E116/model_20000_cnn_lstm.h5'

siamese_network_file = 'TrainedModels/E116/siamese.h5'
# embedding_model_file = 'TrainedModels/E116/model_20000_meuseum.h5'
embedding_model_file = 'TrainedModels/E116/model_20000_basic_5.h5'
# embedding_model_file = 'TrainedModels/E116/model_20000_cnn_lstm.h5'
# embedding_model_file = 'TrainedModels/E116/siamese.h5_embedding.h5'

input_bed_file = 'E116/E116-H3K27ac/E116-H3K27ac.narrowPeak'
pos_seq_file = 'E116/E116-H3K27ac/E116-H3K27ac_modified' + total_seq_postfix + '.fa'
neg_seq_file = 'E116/E116-H3K27ac/E116-H3K27ac_negSet' + total_seq_postfix + '.fa'

forward_seq_file = 'E116/E116-H3K27ac/forward' + total_seq_postfix + '.npy'
reverse_seq_file = 'E116/E116-H3K27ac/reverse' + total_seq_postfix + '.npy'
readout_file = 'E116/E116-H3K27ac/readout' + total_seq_postfix + '.npy'

ref_qtl_file = 'E116/E116-H3K27ac/haQTLDeephistone_ref.fa'
alt_qtl_file = 'E116/E116-H3K27ac/haQTLDeephistone_alt.fa'
forward_qtl_ref_file = 'E116/E116-H3K27ac/forwardQTLref.npy'
reverse_qtl_ref_file = 'E116/E116-H3K27ac/reverseQTLref.npy'
readout_qtl_ref_file = 'E116/E116-H3K27ac/readoutQTLref.npy'
forward_qtl_alt_file = 'E116/E116-H3K27ac/forwardQTLalt.npy'
reverse_qtl_alt_file = 'E116/E116-H3K27ac/reverseQTLalt.npy'
readout_qtl_alt_file = 'E116/E116-H3K27ac/readoutQTLalt.npy'


total_pair_data = 2000

forward_pair_data_seq_file = 'E116/E116-H3K27ac/forwardPairData' + str(total_pair_data) + '.npy'
reverse_pair_data_seq_file = 'E116/E116-H3K27ac/reversePairData' + str(total_pair_data) + '.npy'
pair_data_readout_file = 'E116/E116-H3K27ac/readoutPairData' + str(total_pair_data) + '.npy'


# file names for 4 classes: signal pos neg, qtl ref alt
# forward_seq_file = 'E116/E116-H3K27ac/forward.npy'
# reverse_seq_file = 'E116/E116-H3K27ac/reverse.npy'
# readout_file = 'E116/E116-H3K27ac/readout.npy'

# signal_pos_seq_file = 'E116/E116-H3K27ac/signal_pos.npy'
# signal_neg_seq_file = 'E116/E116-H3K27ac/signal_neg.npy'
# qtl_ref_seq_file = 'E116/E116-H3K27ac/qtl_ref.npy'
# qtl_alt_seq_file = 'E116/E116-H3K27ac/qtl_alt.npy'


# train_forward_seq_file = 'E116/E116-H3K27ac/forward' + total_seq_postfix + '_train.npy'
# train_reverse_seq_file = 'E116/E116-H3K27ac/reverse' + total_seq_postfix + '_train.npy'
# train_readout_file = 'E116/E116-H3K27ac/readout' + total_seq_postfix + '_train.npy'

# test_forward_seq_file = 'E116/E116-H3K27ac/forward' + total_seq_postfix + '_test.npy'
# test_reverse_seq_file = 'E116/E116-H3K27ac/reverse' + total_seq_postfix + '_test.npy'
# test_readout_file = 'E116/E116-H3K27ac/readout' + total_seq_postfix + '_test.npy'







# E118

# output file to save the model after training
# model_file = 'TrainedModels/E118/model' + total_seq_postfix + '_' + model_type + crossval_seq_postfix + '.h5'


# model_file = 'TrainedModels/model_20000_multi_cnn4_withPooling_e12_f256.h5'

# input_bed_file = 'Dataset/E118-H3K27ac.narrowPeak'
# pos_seq_file = 'Dataset/E118-H3K27ac_modified' + total_seq_postfix + '.fa'
# neg_seq_file = 'Dataset/E118-H3K27ac_negSet' + total_seq_postfix + '.fa'

# forward_seq_file = 'Dataset/forward' + total_seq_postfix + '.npy'
# reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '.npy'
# readout_file = 'Dataset/readout' + total_seq_postfix + '.npy'

# forward_seq_file = 'Dataset/forwardQTLref.npy'
# reverse_seq_file = 'Dataset/reverseQTLref.npy'
# readout_file = 'Dataset/readoutQTLref.npy'

# forward_seq_file = 'Dataset/forwardQTLalt.npy'
# reverse_seq_file = 'Dataset/reverseQTLalt.npy'
# readout_file = 'Dataset/readoutQTLalt.npy'

# train_forward_seq_file = 'Dataset/forward' + total_seq_postfix + '_train.npy'
# train_reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '_train.npy'
# train_readout_file = 'Dataset/readout' + total_seq_postfix + '_train.npy'

# test_forward_seq_file = 'Dataset/forward' + total_seq_postfix + '_test.npy'
# test_reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '_test.npy'
# test_readout_file = 'Dataset/readout' + total_seq_postfix + '_test.npy'


# initialization
logging.basicConfig(filename=model_file+'log.log', level=logging.INFO)
log_file = open('log.log', 'a')



def processInputData(pos_seq_file, neg_seq_file, forward_seq_file, reverse_seq_file, readout_file):
    preprocessor = Preprocessor()
    processed_dict = preprocessor.oneHotEncode(
        pos_seq_file=pos_seq_file, neg_seq_file=neg_seq_file)
    np.save(forward_seq_file, processed_dict['forward'])
    np.save(reverse_seq_file, processed_dict['reverse'])
    np.save(readout_file, processed_dict['readout'])


def readInputData(forward_seq_file, reverse_seq_file, readout_file):
    processed_dict = {}
    processed_dict['forward'] = np.load(forward_seq_file)
    processed_dict['reverse'] = np.load(reverse_seq_file)
    processed_dict['readout'] = np.load(readout_file)
    return processed_dict


def splitTrainTestData(forward_seq_file, reverse_seq_file, readout_file):
    processed_dict = readInputData(
        forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_dict['forward']
    reverse = processed_dict['reverse']
    readout = processed_dict['readout']

    x1_train, x1_test, y1_train, y1_test = train_test_split(
        forward, readout, test_size=0.1, random_state=seed)
    # split for reverse complemenet sequences
    x2_train, x2_test, y2_train, y2_test = train_test_split(
        reverse, readout, test_size=0.1, random_state=seed)

    np.save(train_forward_seq_file, x1_train)
    np.save(train_reverse_seq_file, x2_train)
    np.save(train_readout_file, y1_train)

    np.save(test_forward_seq_file, x1_test)
    np.save(test_reverse_seq_file, x2_test)
    np.save(test_readout_file, y1_test)


def preparePairData(forward_seq_file, reverse_seq_file, readout_file, forwardQTLref_file, reverseQTLref_file, forwardQTLalt_file, reverseQTLalt_file, forward_pair_data_seq_file, reverse_pair_data_seq_file, pair_data_readout_file, pair_data_size=2000):
    processed_data = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_data['forward']
    reverse = processed_data['reverse']
    readout = processed_data['readout']

    forward_pos_sequences = []
    reverse_pos_sequences = []
    forward_neg_sequences = []
    reverse_neg_sequences = []
    for i in range(len(readout)):
        if (readout[i] == 0):
            forward_neg_sequences.append(forward[i])
            reverse_neg_sequences.append(reverse[i])
        elif (readout[i] == 1):
            forward_pos_sequences.append(forward[i])
            reverse_pos_sequences.append(reverse[i])
    forward_pos_sequences = np.array(forward_pos_sequences)
    reverse_pos_sequences = np.array(reverse_pos_sequences)
    forward_neg_sequences = np.array(forward_neg_sequences)
    reverse_neg_sequences = np.array(reverse_neg_sequences)

    forward_qtl_ref_sequences = np.load(forwardQTLref_file)
    reverse_qtl_ref_sequences = np.load(reverseQTLref_file)
    forward_qtl_alt_sequences = np.load(forwardQTLalt_file)
    reverse_qtl_alt_sequences = np.load(reverseQTLalt_file)

    # forward_classes = [forward_pos_sequences, forward_neg_sequences, forward_qtl_ref_sequences, forward_qtl_alt_sequences]
    # reverse_classes = [reverse_pos_sequences, reverse_neg_sequences, reverse_qtl_ref_sequences, reverse_qtl_alt_sequences]
    # without qtl
    forward_classes = [forward_pos_sequences, forward_neg_sequences]
    reverse_classes = [reverse_pos_sequences, reverse_neg_sequences]
    forward_pair_data = []
    reverse_pair_data = []
    readout_pair_data = []
    size_per_class = pair_data_size//(len(forward_classes)*len(forward_classes))

    # ############### different approach ###############
    pair_index_comb = list(combinations(range(6000),2))

    for i in range(len(forward_classes)):
        for j in range(len(forward_classes)):
            print('pairs for class: ', i, j)
            pair_idxs = random.sample(pair_index_comb, size_per_class)
            print(pair_idxs[:10])
            a_forward = forward_classes[i]
            a_reverse = reverse_classes[i]
            b_forward = forward_classes[j]
            b_reverse = reverse_classes[j]
            if (i%2) == (j%2):
              label = 0
            else:
              label = 1
            for idx1, idx2 in pair_idxs:
                forward_pair_data.append((a_forward[idx1], b_forward[idx2]))
                reverse_pair_data.append((a_reverse[idx1], b_reverse[idx2]))
                readout_pair_data.append(label)

    # for i in range(len(forward_classes)):
    #     for j in range(len(reverse_classes)):
    #         A_forward = forward_classes[i]
    #         A_reverse = reverse_classes[i]
    #         B_forward = forward_classes[j]
    #         B_reverse = reverse_classes[j]
    #         for k in range(size_per_class):
    #             idx1 = random.randint(0, len(A_forward)-1)
    #             idx2 = random.randint(0, len(B_forward)-1)
    #             if ((i%2) == (j%2)):
    #                 label = 0
    #             else:
    #                 label = 1
    #             forward_pair_data.append((A_forward[idx1], B_forward[idx2]))
    #             reverse_pair_data.append((A_reverse[idx1], B_reverse[idx2]))
    #             readout_pair_data.append(label)
                

    forward_pair_data = np.array(forward_pair_data)
    reverse_pair_data = np.array(reverse_pair_data)
    readout_pair_data = np.array(readout_pair_data)

    np.save(forward_pair_data_seq_file, forward_pair_data)
    np.save(reverse_pair_data_seq_file, reverse_pair_data)
    np.save(pair_data_readout_file, readout_pair_data)





# get dictionary from text file
def readParameters(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split()
            dict[key] = val
    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])
    return dict


def plotCurve(history):
    # plot
    # print('plotting curve')
    # print(history.history)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.legend(['train acc', 'val acc'], loc='upper left')
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy & loss')
    plt.xlabel('epoch')
    plt.ylabel('acc & loss')
    plt.legend(['train acc', 'val acc', 'train loss', 'val loss'], loc='upper left')
    # plt.show()
    plt.savefig(model_file.split('.')[0] + '.png')


def createAndTrainBasicModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the basic model
    basic_model = model.create_basic_model(processed_data["forward"].shape)
    basic_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(basic_model, processed_data, seed, model_file)
        plotCurve(results['history'])
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)

    # results = model.trainModelWithHardwareSupport(basic_model, processed_data, with_gpu=True)
    basic_model.save(model_file)

    return results


def createAndTrainVanillaCNN(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the basic model
    cnn_model = model.create_Vanilla_CNN_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(cnn_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMultiCNN2(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file)
    # creating the basic model
    cnn_model = model.create_Multi_CNN2_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == '') :
        results = model.trainModel(cnn_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMultiCNN4(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the basic model
    cnn_model = model.create_Multi_CNN4_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(cnn_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMeuseumModel(processed_data, parameters_dict, model_file, alpha=100, beta=.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the meuseum model
    meuseum_model = model.create_meuseum_model(processed_data["forward"].shape, alpha, beta, bkg_const)
    # meuseum_model = load_model('TrainedModels/E116/model_20000_meuseum.h5', custom_objects={'ConvolutionLayer': ConvolutionLayer}, compile=True)
    print(meuseum_model.summary(), file=log_file)
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(meuseum_model, processed_data, seed, model_file)
        plotCurve(results['history'])
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(meuseum_model, processed_data, with_gpu=True)
    print('saving model: ', model_file, file=log_file)
    meuseum_model.save(model_file)

    return results


def createAndTrainBpnetModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the bpnet model
    bpnet_model = model.create_bpnet_model(processed_data["forward"].shape)
    bpnet_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(bpnet_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(bpnet_model, processed_data, with_gpu=True)
    bpnet_model.save(model_file)

    return results


def createAndTrainDeepHistoneModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the bpnet model
    dhistone_model = model.create_deepHistone_model(
        processed_data["forward"].shape)
    dhistone_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(dhistone_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold, seed)
    # results = model.trainModelWithHardwareSupport(dhistone_model, processed_data, with_gpu=True)
    dhistone_model.save(model_file)

    return results


def createAndTrainCNNLstmModel(processed_data, parameters_dict, model_file):
    global model_type
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)
    # creating the basic model
    basic_model = model.create_CNNLSTM_model(processed_data["forward"].shape)
    basic_model.summary()
    # running the model with the processed data
    if (crossval_seq_postfix == ''):
        results = model.trainModel(basic_model, processed_data, seed, model_file)
    else:
        results = model.cross_val(model_type, processed_data, cross_val_fold,  seed)

    # results = model.trainModelWithHardwareSupport(basic_model, processed_data, with_gpu=True)
    basic_model.save(model_file)

    return results



def createAndTrainSiameseNetwork(processed_pair_data, parameters_dict, embedding_model_file, siamese_model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
                  activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"], model_file=model_file, log_file=log_file)

    embedding_model = load_model(embedding_model_file, custom_objects={'ConvolutionLayer': ConvolutionLayer}, compile=False)
    embedding_model.layers.pop()
    # embedding_model_saved.summary()
    # embedding_model = model.create_basic_model((1, 1000, 4), compile=False)
    # embedding_model = model.create_meuseum_model((1, 1000, 4), compile=False)
    # embedding_model = model.create_Vanilla_CNN_model((1, 1000, 4), compile=False)
    # embedding_model.summary()
    # embedding_model.set_weights(embedding_model_saved.get_weights())
    # creating the basic model
    siamese = model.createSiameseNetwork(embedding_model, processed_pair_data["forward"].shape)
    siamese.summary()
    keras.utils.plot_model(siamese, siamese_model_file+'_architecture.png', show_shapes=True)
    keras.utils.plot_model(embedding_model, siamese_model_file+'_embedding_architecture.png', show_shapes=True)
    # running the model with the processed data
    results = model.trainSiameseNetwork(siamese, processed_pair_data, seed, siamese_model_file)
    # print(results)

    # results = model.trainModelWithHardwareSupport(basic_model, processed_data, with_gpu=True)
    embedding_model.save(siamese_model_file+'_embedding.h5')
    siamese.save(siamese_model_file)

    return results



def saveResults(parameters_dict, results, fis): 
    global epigenome
    global histone
    input_length = total_seq_postfix[1:]
    global model_type
    global model_file
    global output_file_name

    computed_cnt = -1
    if os.path.isfile(output_file_name):
        try:
            df = pd.read_csv(output_file_name)
            computed_cnt = len(df)
        except IOError:
            computed_cnt = -1
    else:
        computed_cnt = -1

    with open(output_file_name, "a") as output_file:
        if (computed_cnt<0):
            output_file.write("Model, Model File, Epigenome ID,Histone marker,Input length,Epochs,Batch size,Filters,Kernel size,Pool type,Activation,Regularizer,Alpha,Beta,Seed,Train auc score,Train accuracy,Test auc score,Test accuracy,Cross validation,FIS 1,FIS 2,Observation\n")

        output_text = "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{:.5f},{:.5f},{}\n".format(
                                                    model_type,
                                                    model_file,
                                                    epigenome,
                                                    histone,
                                                    input_length,
                                                    parameters_dict['epochs'],
                                                    parameters_dict['batch_size'],
                                                    parameters_dict['filters'],
                                                    parameters_dict['kernel_size'],
                                                    parameters_dict['pool_type'],
                                                    parameters_dict['activation_type'],
                                                    parameters_dict['regularizer'],
                                                    results['seed'],
                                                    results['train_auc_score'],
                                                    results['train_accuracy'],
                                                    results['test_auc_score'],
                                                    results['test_accuracy'],
                                                    0,
                                                    fis[0],
                                                    fis[1],
                                                    None,
                                                )
        # print(output_text)
        output_file.write(output_text)
        output_file.flush()


def createAndTrainModel(processed_data, parameters_dict, model_type):
    if model_type == 'basic':
        results = createAndTrainBasicModel(
            processed_data, parameters_dict, model_file)
    elif model_type == 'meuseum':
        results = createAndTrainMeuseumModel(
            processed_data, parameters_dict, model_file)
    elif model_type == 'vanilla_cnn':
        results = createAndTrainVanillaCNN(
            processed_data, parameters_dict, model_file)
    elif model_type == 'multi_cnn2':
        results = createAndTrainMultiCNN2(
            processed_data, parameters_dict, model_file)
    elif model_type == 'multi_cnn4':
        results = createAndTrainMultiCNN4(
            processed_data, parameters_dict, model_file)
    elif model_type == 'bpnet':
        results = createAndTrainBpnetModel(
            processed_data, parameters_dict, model_file)
    elif model_type == 'deephistone':
        results = createAndTrainDeepHistoneModel(
            processed_data, parameters_dict, model_file)
    elif model_type == 'cnn_lstm':
        results = createAndTrainCNNLstmModel(
            processed_data, parameters_dict, model_file)

    return results


def testModel(model_file, model_type, forward_seq_file, reverse_seq_file, readout_file):
    model = load_model(model_file, custom_objects={
                       'ConvolutionLayer': ConvolutionLayer})
    model.summary()
    processed_dict = readInputData(
        forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_dict['forward']
    reverse = processed_dict['reverse']
    readout = processed_dict['readout']
    # print("Input size forward: " + str(processed_dict['forward'].shape))
    # print("Input size reverse: " + str(processed_dict['reverse'].shape))
    # print("Input size readout: " + str(processed_dict['readout'].shape))
    # Prediction on test data
    if model_type == 'vanilla_cnn' or model_type == 'multi_cnn2':
        test_input_data = np.concatenate((forward, reverse), axis=0)
        test_output_data = np.concatenate((readout, readout), axis=0)
    else:
        test_input_data = {'forward': forward, 'reverse': reverse}
        test_output_data = readout
    print("\n=========================== Prediction ===================================\n")
    pred_test = model.predict(test_input_data)
    # See which label has the highest confidence value
    predictions_test = np.argmax(pred_test, axis=1)

    # print("test input size: " + str(test_input_data.shape))
    # print("test output size: " + str(test_output_data.shape))
    print("prediction output size: " + str(predictions_test.shape))

    true_pred, false_pred = 0, 0
    for count, value in enumerate(predictions_test):
        if test_output_data[count] == predictions_test[count]:
            true_pred += 1
        else:
            false_pred += 1

    # test_auc_score = sklearn.metrics.roc_auc_score(
        # test_output_data, predictions_test)
    test_auc_score = 0
    test_accuracy = true_pred/len(predictions_test)
    # print('\ntest-set auc score is: ' + str(test_auc_score))
    print('test-set accuracy is: ' + str(test_accuracy))
    print("========================================================================\n")
    return test_auc_score, test_accuracy


# pair data test

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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

def testModelPairData(model_file, forward_seq_file, reverse_seq_file, readout_file):
    # embedding model
    model = load_model(embedding_model_file, custom_objects={
                       'ConvolutionLayer': ConvolutionLayer})

    processed_dict = readInputData(
        forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_dict['forward'][:200]
    reverse = processed_dict['reverse'][:200]
    readout = processed_dict['readout'][:200]
    # print("Input size forward: " + str(processed_dict['forward'].shape))
    # print("Input size reverse: " + str(processed_dict['reverse'].shape))
    # print("Input size readout: " + str(processed_dict['readout'].shape))
    # Prediction on test data
    forward1 = np.transpose(forward, [1, 0, 2, 3])[0]
    reverse1 = np.transpose(reverse, [1, 0, 2, 3])[0]
    forward2 = np.transpose(forward, [1, 0, 2, 3])[1]
    reverse2 = np.transpose(reverse, [1, 0, 2, 3])[1]
    print("\n=========================== Prediction ===================================\n")
    pred_tower1 = model.predict({'forward': forward1, 'reverse': reverse1})
    pred_tower2 = model.predict({'forward': forward2, 'reverse': reverse2})
    # See which label has the highest confidence value
    # print(pred_tower1)
    # print(pred_tower2)
    dist = euclidean_distance([pred_tower1, pred_tower2])
    # dist = BatchNormalization()(dist)
    print(dist[:20])
    # n, bins, patches = plt.hist(dist)
    # # print(n, bins, patches)
    # plt.show()
    # plt.savefig('dist.png')
    # hist, bins = np.histogram(dist)
    # print('\ndist distribution: ')
    # print(hist, bins)
    out = np.array(sigmoid(dist))
    print(out[:20])
    # plt.plot(hist, bins)
    # plt.show()
    # plt.savefig('dist.png')
    # print(out)
    # hist, bins = np.histogram(out)
    n, bins, patches = plt.hist(dist, bins=50)
    # print(n, bins, patches)
    plt.savefig('dist.png')
    mn, mx, mean = np.min(dist), np.max(dist), np.mean(dist)
    print(mn, mx, mean)
    mn, mx, mean = np.min(out), np.max(out), np.mean(out)
    print(mn, mx, mean)
    # print('\noutput distribution: ')
    # print(hist, bins)
    predictions_test = []
    for d in out:
        if d>.5:
            predictions_test.append(1)
        else:
            predictions_test.append(0)
    predictions_test = np.array(predictions_test)
    print('predictions: ', predictions_test[:20])
    print('readout: ', readout[:20])
    # predictions_test1 = np.argmax(pred_tower1, axis=1)
    # predictions_test2 = np.argmax(pred_tower2, axis=1)
    # print(predictions_test1)
    # print(predictions_test2)
    # print(readout)

    # print("test input size: " + str(test_input_data.shape))
    # print("test output size: " + str(test_output_data.shape))
    # print("prediction output size: " + str(predictions_test.shape))

    # true_pred, false_pred = 0, 0
    # for count, value in enumerate(predictions_test):
    #     if test_output_data[count] == predictions_test[count]:
    #         true_pred += 1
    #     else:
    #         false_pred += 1

    # # test_auc_score = sklearn.metrics.roc_auc_score(
    #     # test_output_data, predictions_test)
    # test_auc_score = 0
    # test_accuracy = true_pred/len(predictions_test)
    # # print('\ntest-set auc score is: ' + str(test_auc_score))
    # print('test-set accuracy is: ' + str(test_accuracy))
    # print("========================================================================\n")
    # return test_auc_score, test_accuracy


def testModelOnQTLs(model_file, model_type, forward_seq_ref_file, reverse_seq_ref_file,  forward_seq_alt_file, reverse_seq_alt_file):
    model = load_model(model_file, custom_objects={
                       'ConvolutionLayer': ConvolutionLayer})
    model.summary()
    forward_ref_sequences = np.load(forward_seq_ref_file)
    reverse_ref_sequences = np.load(reverse_seq_ref_file)
    forward_alt_sequences = np.load(forward_seq_alt_file)
    reverse_alt_sequences = np.load(reverse_seq_alt_file)

    test_input_ref_data = {'forward': forward_ref_sequences, 'reverse': reverse_ref_sequences}

    test_input_alt_data = {'forward': forward_alt_sequences, 'reverse': reverse_alt_sequences}
    
    pred_test_ref = model.predict(test_input_ref_data)
    pred_test_alt = model.predict(test_input_alt_data)

    diff_arr, diff_arr2 = [], []
    for i in range(len(pred_test_ref)):
        diff_arr.append(np.abs(pred_test_ref[i][1] - pred_test_alt[i][1]))
        diff_arr2.append(np.abs(pred_test_ref[i][0] - pred_test_alt[i][0]))
    print(diff_arr[:10])
    print(np.average(diff_arr))

    # fig = plt.figure(figsize =(10, 7))

    # plt.plot(diff_arr[:20])
    plt.boxplot(diff_arr)
    # plt.title('model accuracy')
    # plt.legend(['train acc', 'val acc'], loc='upper left')
    # plt.show()

    plt.title('qtl diff plot')
    plt.xlabel('qtls')
    plt.ylabel('fis')
    plt.legend(['train acc'], loc='upper left')
    plt.savefig(model_file.split('.')[0] + '_fis_qtl.png')
    return np.average(diff_arr), np.average(diff_arr2)


def hyperParameterTuner(processed_data):
    epigenome = "E116"
    histone = "H3K4me3"

    alpha_vals = [100, 10, 500, 1000]
    epoches = [20, 15, 30, 25, 40]
    batch_sizes = [512, 256, 128, 64]
    filters = [512, 256, 128, 64, 1024]
    kernel_sizes = [16, 12, 9, 32]
    pool_types = ["Max", "Custom", "Custom_sum"]
    regularizers = ["L_1", "L_2"]
    activation_types = ["linear", "relu", "sigmoid"]

    bkg_const=[0.25, 0.25, 0.25, 0.25]

    output_file_name = "all_outputs_"+input_length+"seqs.csv"
    computed_cnt = 0
    if os.path.isfile(output_file_name):
        try:
            df = pd.read_csv(output_file_name)
            computed_cnt = len(df)
        except IOError:
            computed_cnt = -1
    else:
        computed_cnt = -1

    with open(output_file_name, "a") as output_file:
        if (computed_cnt<0):
            output_file.write("Index,Epigenome ID,Histone marker,Input length,Epochs,Batch size,Filters,Kernel size,Pool type,Activation,Regularizer,Alpha,Beta,Seed,Train auc score,Train accuracy,Test auc score,Test accuracy,Cross validation,Observation\n")
        index = 1
        for alpha in alpha_vals:
            beta = 1/alpha
            for epoch in epoches:
                for b_size in batch_sizes:
                    for fltr in filters:
                        for k_size in kernel_sizes:
                            for pool_type in pool_types:
                                for activation_type in activation_types:
                                    for regularizer in regularizers:
                                        parameters_dict = {
                                            "epochs": epoch,
                                            "batch_size": b_size,
                                            "filters": fltr,
                                            "kernel_size": k_size,
                                            "pool_type": pool_type,
                                            "activation_type": activation_type,
                                            "regularizer": regularizer,
                                        }
                                        # check the number of computed loops
                                        if (index <= computed_cnt):
                                            pass
                                        else:
                                            results = createAndTrainMeuseumModel(processed_data, parameters_dict, alpha, beta, bkg_const)
                                            # print(results)
                                            output_text = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n".format(
                                                index,
                                                epigenome,
                                                histone,
                                                input_length,
                                                epoch,
                                                b_size,
                                                fltr,
                                                k_size,
                                                pool_type,
                                                activation_type,
                                                regularizer,
                                                alpha,
                                                beta,
                                                results['seed'],
                                                results['train_auc_score'],
                                                results['train_accuracy'],
                                                results['test_auc_score'],
                                                results['test_accuracy'],
                                                0,
                                                None,
                                            )
                                            # print(output_text)
                                            output_file.write(output_text)
                                            output_file.flush()
                                        index = index+1




def main():
    global forward_seq_file, reverse_seq_file, readout_file
    global model_number, model_file, model_type
    # Modify the bed file keeping qvalue>=4
    # preprocessor = Preprocessor()
    # preprocessor.modifyBedFile(input_bed_file)

    # Generate the negative sequences
    # command = 'Rscript bed_to_null_seq.R'
    # os.system(command)

    # Convert bed to fa
    # subprocess.call(['sh', './bed_to_fa.sh'])

    # Shrink fa
    # subprocess.call(['sh', './fa_shrink.sh'])

    # Preprocess data
    # Preprocess the pos and neg fasta file
    processInputData(pos_seq_file, neg_seq_file, forward_seq_file, reverse_seq_file, readout_file)
    # Preprocess qtl data
    processInputData(ref_qtl_file, '', forward_qtl_ref_file, reverse_qtl_ref_file, readout_qtl_ref_file)
    processInputData(alt_qtl_file, '', forward_qtl_alt_file, reverse_qtl_alt_file, readout_qtl_alt_file)

    # Split train and test data
    # splitTrainTestData(forward_seq_file, reverse_seq_file, readout_file)

    # Prepare pair dataset
    # processed_data = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    # preparePairData(forward_seq_file, reverse_seq_file, readout_file, forward_qtl_ref_file, reverse_qtl_ref_file, forward_qtl_alt_file, reverse_qtl_alt_file, forward_pair_data_seq_file, reverse_pair_data_seq_file, pair_data_readout_file, total_pair_data)
    # testModelPairData(siamese_network_file, forward_pair_data_seq_file, reverse_pair_data_seq_file, pair_data_readout_file)
    
    # Run model on processed data
    processed_data = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    print("Input size: " + str(processed_data['forward'].shape), file=log_file)
    # Run the model once with the parameters
    parameter_file = 'parameters.txt'
    parameters_dict = readParameters(parameter_file)
    # results = createAndTrainModel(processed_data, parameters_dict, model_type)
    # print(results)
    # fis1, fis2 = testModelOnQTLs(model_file, model_type, forward_qtl_ref_file, reverse_qtl_ref_file, forward_qtl_alt_file, reverse_qtl_alt_file)
    # print('fis: ', fis1, fis2)
    # saveResults(parameters_dict, results, (fis1, fis2))

    # tuner
    for model_type in ['basic', 'meuseum', 'cnn_lstm']:
      print('\n\n-------------------itr----------------\n', file=log_file)
      model_number = sum(1 for line in open(output_file_name))
      model_file = 'TrainedModels/E116/model' + total_seq_postfix + '_' + model_type + crossval_seq_postfix + '_' + str(model_number) + '.h5'
      print('model file: ', model_file, file=log_file)
      prev_model_file = model_file
      results = createAndTrainModel(processed_data, parameters_dict, model_type)
      print('results: ', results, file=log_file)
      fis1, fis2 = testModelOnQTLs(prev_model_file, model_type, forward_qtl_ref_file, reverse_qtl_ref_file, forward_qtl_alt_file, reverse_qtl_alt_file)
      print('fis: ', fis1, fis2, file=log_file)
      saveResults(parameters_dict, results, (fis1, fis2))

    # Run a model on siamese network
    # processed_data = readInputData(forward_pair_data_seq_file, reverse_pair_data_seq_file, pair_data_readout_file)
    # parameter_file = 'parameters.txt'
    # parameters_dict = readParameters(parameter_file)
    # createAndTrainSiameseNetwork(processed_data, parameters_dict, embedding_model_file, siamese_network_file)

    # Prediction on the test data
    # testModel(model_file, model_type, test_forward_seq_file, test_reverse_seq_file, test_readout_file)
    # testModel(model_file, model_type, forward_seq_file, reverse_seq_file, readout_file)
    # testModelOnQTLs(model_file, model_type, 'E116/E116-H3K27ac/forwardQTLref.npy', 'E116/E116-H3K27ac/reverseQTLref.npy', 'E116/E116-H3K27ac/forwardQTLalt.npy', 'E116/E116-H3K27ac/reverseQTLalt.npy')

    # test on siamese network
    # testModelPairData(siamese_network_file, forward_pair_data_seq_file, reverse_pair_data_seq_file, pair_data_readout_file)

if __name__ == "__main__":
    sys.exit(main())
