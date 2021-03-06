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

model_file = 'TrainedModels/model'+ total_seq_postfix + '.h5'


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


def createAndTrainBasicModel(processed_data, parameters_dict):
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


def createAndTrainMeuseumModel(processed_data, parameters_dict, alpha=1000, beta=.001, bkg_const=[0.25, 0.25, 0.25, 0.25]):
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


def testModel(model_file, forward_seq_file, reverse_seq_file, readout_file):
    model = load_model(model_file, custom_objects={'ConvolutionLayer': ConvolutionLayer})
    processed_dict = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    forward = processed_dict['forward']
    reverse = processed_dict['reverse']
    readout = processed_dict['readout']
    # Prediction on test data
    print("\n=========================== Prediction ===================================\n")
    pred_test = model.predict({'forward': forward, 'reverse': reverse})
    # See which label has the highest confidence value
    predictions_test = np.argmax(pred_test, axis=1)

    true_pred, false_pred = 0, 0
    for count, value in enumerate(predictions_test):
        if readout[count] == predictions_test[count]:
            true_pred += 1
        else:
            false_pred += 1
    # print('Total number of test-set predictions is: ' + str(len(y1_test_orig)))
    # print('Number of correct test-set predictions is: ' + str(true_pred))
    # print('Number of incorrect test-set predictions is: ' + str(false_pred))

    test_auc_score = sklearn.metrics.roc_auc_score(
        readout, predictions_test)
    test_accuracy = true_pred/len(predictions_test)
    print('\ntest-set auc score is: ' + str(test_auc_score))
    print('test-set accuracy is: ' + str(test_accuracy))
    print("========================================================================\n")


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
    # processed_data = readInputData(train_forward_seq_file, train_reverse_seq_file, train_readout_file)
    # # processed_data = readInputData(forward_seq_file, reverse_seq_file, readout_file)
    # # run the model once with the parameters
    # parameter_file = 'parameters.txt'
    # parameters_dict = readParameters(parameter_file)
    # # results = createAndTrainBasicModel(processed_data, parameters_dict)
    # results = createAndTrainMeuseumModel(processed_data, parameters_dict)


    ### Prediction on the test data
    testModel(model_file, test_forward_seq_file, test_reverse_seq_file, test_readout_file)
    
    

if __name__ == "__main__":
    sys.exit(main())
