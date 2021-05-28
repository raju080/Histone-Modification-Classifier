import sys
import os
import subprocess
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras.models import save_model, load_model

import sklearn
from sklearn.model_selection import train_test_split

from Preprocess import Preprocessor
from Model import Model
from ConvolutionLayer import ConvolutionLayer

# constants
seed = 527
total_seq_postfix = "_200"
# types: basic, meuseum, vanilla_cnn, multi_cnn2, multi_cnn4, bpnet, deephistone
model_type = 'deephistone'
# output file to save the model after training
model_file = 'TrainedModels/model'+ total_seq_postfix + '_' + model_type + '.h5'

# model_file = 'TrainedModels/model_20000_multi_cnn4_withPooling_e12_f256.h5'

input_bed_file = 'Dataset/E118-H3K27ac.narrowPeak'
pos_seq_file = 'Dataset/E118-H3K27ac_modified' + total_seq_postfix + '.fa'
neg_seq_file = 'Dataset/E118-H3K27ac_negSet' + total_seq_postfix + '.fa'

forward_seq_file = 'Dataset/forward' + total_seq_postfix + '.npy'
reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '.npy'
readout_file = 'Dataset/readout' + total_seq_postfix + '.npy'

train_forward_seq_file = 'Dataset/forward' + total_seq_postfix + '_train.npy'
train_reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '_train.npy'
train_readout_file = 'Dataset/readout' + total_seq_postfix + '_train.npy'

test_forward_seq_file = 'Dataset/forward' + total_seq_postfix + '_test.npy'
test_reverse_seq_file = 'Dataset/reverse' + total_seq_postfix + '_test.npy'
test_readout_file = 'Dataset/readout' + total_seq_postfix + '_test.npy'


def processInputData(pos_seq_file, neg_seq_file):
    preprocessor = Preprocessor()
    processed_dict = preprocessor.oneHotEncode(
        pos_seq_file=pos_seq_file, neg_seq_file=neg_seq_file) 
    np.save(forward_seq_file, processed_dict['forward'])
    np.save(reverse_seq_file, processed_dict['reverse'])
    np.save(readout_file, processed_dict['readout'])
    print(processed_dict)


def readInputData(forward_seq_file, reverse_seq_file, readout_file):
    processed_dict = {}
    processed_dict['forward'] = np.load(forward_seq_file)    
    processed_dict['reverse'] = np.load(reverse_seq_file)
    processed_dict['readout'] = np.load(readout_file)
    return processed_dict



def splitTrainTestData(forward_seq_file, reverse_seq_file, readout_file):
    processed_dict = readInputData(forward_seq_file, reverse_seq_file, readout_file)
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


def createAndTrainBasicModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the basic model
    basic_model = model.create_basic_model(processed_data["forward"].shape)
    basic_model.summary()
    # running the model with the processed data
    results = model.trainModel(basic_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(basic_model, processed_data, with_gpu=True)
    basic_model.save(model_file)

    return results


def createAndTrainVanillaCNN(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the basic model
    cnn_model = model.create_Vanilla_CNN_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    results = model.trainModelOneInputLayer(cnn_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMultiCNN2(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the basic model
    cnn_model = model.create_Multi_CNN2_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    results = model.trainModelOneInputLayer(cnn_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMultiCNN4(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the basic model
    cnn_model = model.create_Multi_CNN4_model(processed_data["forward"].shape)
    cnn_model.summary()
    # running the model with the processed data
    results = model.trainModel(cnn_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(cnn_model, processed_data, with_gpu=True)
    cnn_model.save(model_file)

    return results


def createAndTrainMeuseumModel(processed_data, parameters_dict, model_file, alpha=100, beta=.01, bkg_const=[0.25, 0.25, 0.25, 0.25]):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the meuseum model
    meuseum_model = model.create_meuseum_model(processed_data["forward"].shape, alpha, beta, bkg_const)
    meuseum_model.summary()
    # running the model with the processed data
    results = model.trainModel(meuseum_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(meuseum_model, processed_data, with_gpu=True)
    meuseum_model.save(model_file)

    return results


def createAndTrainBpnetModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the bpnet model
    bpnet_model = model.create_bpnet_model(processed_data["forward"].shape)
    bpnet_model.summary()
    # running the model with the processed data
    results = model.trainModel(bpnet_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(bpnet_model, processed_data, with_gpu=True)
    bpnet_model.save(model_file)

    return results


def createAndTrainDeepHistoneModel(processed_data, parameters_dict, model_file):
    # initiate a model with the specified parameters
    # seed = random.randint(1,1000)
    seed = 527
    model = Model(filters=parameters_dict["filters"], kernel_size=parameters_dict["kernel_size"], pool_type=parameters_dict["pool_type"], regularizer=parameters_dict["regularizer"],
            activation_type=parameters_dict["activation_type"], epochs=parameters_dict["epochs"], batch_size=parameters_dict["batch_size"])
    # creating the bpnet model
    dhistone_model = model.create_deepHistone_model(processed_data["forward"].shape)
    dhistone_model.summary()
    # running the model with the processed data
    results = model.trainModel(dhistone_model, processed_data, seed)
    # results = model.trainModelWithHardwareSupport(dhistone_model, processed_data, with_gpu=True)
    dhistone_model.save(model_file)

    return results


def createAndTrainModel(processed_data, parameters_dict, model_type):
    if model_type=='basic':
        results = createAndTrainBasicModel(processed_data, parameters_dict, model_file)
    elif model_type=='meuseum':
        results = createAndTrainMeuseumModel(processed_data, parameters_dict, model_file)
    elif model_type=='vanilla_cnn':
        results = createAndTrainVanillaCNN(processed_data, parameters_dict, model_file)
    elif model_type=='multi_cnn2':
        results = createAndTrainMultiCNN2(processed_data, parameters_dict, model_file)
    elif model_type=='multi_cnn4':
        results = createAndTrainMultiCNN4(processed_data, parameters_dict, model_file)
    elif model_type=='bpnet':
        results = createAndTrainBpnetModel(processed_data, parameters_dict, model_file)
    elif model_type=='deephistone':
        results = createAndTrainDeepHistoneModel(processed_data, parameters_dict, model_file)

    return results


def testModel(model_file, model_type, forward_seq_file, reverse_seq_file, readout_file):
    model = load_model(model_file, custom_objects={'ConvolutionLayer': ConvolutionLayer})
    model.summary()
    processed_dict = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_dict['forward']
    reverse = processed_dict['reverse']
    readout = processed_dict['readout']
    print("Input size: " + str(processed_dict['forward'].shape))
    # Prediction on test data
    if model_type=='basic' or model_type=='meuseum' or model_type=='multi_cnn4' or model_type=='deephistone' or model_type=='bpnet':
        test_input_data = {'forward': forward, 'reverse': reverse}
        test_output_data = readout
    elif model_type=='vanilla_cnn' or model_type=='multi_cnn2':
        test_input_data = np.concatenate((forward, reverse), axis=0)
        test_output_data = np.concatenate((readout, readout), axis=0)
    else:
        test_input_data = np.concatenate((forward, reverse), axis=0)
        test_output_data = np.concatenate((readout, readout), axis=0)
    print("\n=========================== Prediction ===================================\n")
    pred_test = model.predict(test_input_data)
    # See which label has the highest confidence value
    predictions_test = np.argmax(pred_test, axis=1)

    # print("test input size: " + str(test_input_data.shape))
    # print("test output size: " + str(test_output_data.shape))
    # print("prediction output size: " + str(predictions_test.shape))

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
    return test_auc_score, test_accuracy



# def hyperParameterTuner(processed_data):
#     epigenome = "E116"
#     histone = "H3K4me3"

#     alpha_vals = [100, 10, 500, 1000]
#     epoches = [20, 15, 30, 25, 40]
#     batch_sizes = [512, 256, 128, 64]
#     filters = [512, 256, 128, 64, 1024]
#     kernel_sizes = [16, 12, 9, 32]
#     pool_types = ["Max", "Custom", "Custom_sum"]
#     regularizers = ["L_1", "L_2"]
#     activation_types = ["linear", "relu", "sigmoid"]

#     bkg_const=[0.25, 0.25, 0.25, 0.25]

#     output_file_name = "all_outputs_"+input_length+"seqs.csv"
#     computed_cnt = 0
#     if os.path.isfile(output_file_name):
#         try:
#             df = pd.read_csv(output_file_name)
#             computed_cnt = len(df)
#         except IOError:
#             computed_cnt = -1
#     else:
#         computed_cnt = -1 

#     with open(output_file_name, "a") as output_file:
#         if (computed_cnt<0):
#             output_file.write("Index,Epigenome ID,Histone marker,Input length,Epochs,Batch size,Filters,Kernel size,Pool type,Activation,Regularizer,Alpha,Beta,Seed,Train auc score,Train accuracy,Test auc score,Test accuracy,Cross validation,Observation\n")
#         index = 1
#         for alpha in alpha_vals:
#             beta = 1/alpha
#             for epoch in epoches:
#                 for b_size in batch_sizes:
#                     for fltr in filters:
#                         for k_size in kernel_sizes:
#                             for pool_type in pool_types:
#                                 for activation_type in activation_types:
#                                     for regularizer in regularizers:
#                                         parameters_dict = {
#                                             "epochs": epoch,
#                                             "batch_size": b_size,
#                                             "filters": fltr,
#                                             "kernel_size": k_size,
#                                             "pool_type": pool_type,
#                                             "activation_type": activation_type,
#                                             "regularizer": regularizer,
#                                         }
#                                         # check the number of computed loops
#                                         if (index <= computed_cnt):
#                                             pass
#                                         else:
#                                             results = createAndTrainMeuseumModel(processed_data, parameters_dict, alpha, beta, bkg_const)
#                                             # print(results)
#                                             output_text = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n".format(
#                                                 index,
#                                                 epigenome, 
#                                                 histone, 
#                                                 input_length, 
#                                                 epoch,
#                                                 b_size,
#                                                 fltr,
#                                                 k_size,
#                                                 pool_type,
#                                                 activation_type,
#                                                 regularizer,
#                                                 alpha,
#                                                 beta,
#                                                 results['seed'],
#                                                 results['train_auc_score'],
#                                                 results['train_accuracy'],
#                                                 results['test_auc_score'],
#                                                 results['test_accuracy'],
#                                                 0,
#                                                 None,
#                                             )
#                                             # print(output_text)
#                                             output_file.write(output_text)
#                                             output_file.flush()
#                                         index = index+1






def main():
    ### Modify the bed file keeping qvalue>=4
    # preprocessor = Preprocessor()
    # preprocessor.modifyBedFile(input_bed_file)


    ### Generate the negative sequences
    # command = 'Rscript bed_to_null_seq.R'
    # os.system(command)


    ### Convert bed to fa
    # subprocess.call(['sh', './bed_to_fa.sh'])


    ### Shrink fa
    # subprocess.call(['sh', './fa_shrink.sh'])


    ### Preprocess the pos and neg fasta file
    # processInputData(pos_seq_file, neg_seq_file)


    ### Split train and test data
    # splitTrainTestData(forward_seq_file, reverse_seq_file, readout_file)


    ### Run model on processed data
    processed_data = readInputData(train_forward_seq_file, train_reverse_seq_file, train_readout_file)
    print("Input size: " + str(processed_data['forward'].shape))
    # processed_data = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    # run the model once with the parameters
    parameter_file = 'parameters.txt'
    parameters_dict = readParameters(parameter_file)

    createAndTrainModel(processed_data, parameters_dict, model_type)

    ### Prediction on the test data
    # testModel(model_file, model_type, test_forward_seq_file, test_reverse_seq_file, test_readout_file)
    
    

if __name__ == "__main__":
    sys.exit(main())
