from os import read
import sys
import time
import pandas as pd
import numpy as np


seed = 527
total_seq_postfix = "_5000"
# types: basic, meuseum, vanilla_cnn, multi_cnn
model_type = 'vanilla_cnn'
# output file to save the model after training
model_file = 'TrainedModels/model'+ total_seq_postfix + '_' + model_type + '.h5'

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



def readInputData(forward_seq_file, reverse_seq_file, readout_file):
    processed_dict = {}
    processed_dict['forward'] = np.load(forward_seq_file)    
    processed_dict['reverse'] = np.load(reverse_seq_file)
    processed_dict['readout'] = np.load(readout_file)
    return processed_dict

def readBedFile(fileName):
    df = pd.read_csv(
        fileName, sep='\t', skiprows=1, header=[1, 2, 3, 4, 5, 6])
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
              'thickStart', 'thickEnd', 'itemRgb', 'count', 'percentMeth']
    df.columns = header[:len(df.columns)]
    return df


def writeDataFrameToBed(file_path, df):
    df.to_csv(file_path, header=None,
              index=None, sep='\t')

def readQTLs(file_path):
    dfs = pd.read_excel(file_path, sheet_name=None)
    qtl_df = dfs.get('SuppTable_3_delrosario')
    print(qtl_df)


def processForDeepHistone(inputFile):
    pass


def main():
    # neg_input_bed_file = 'Dataset/E118-H3K27ac_negSet.bed'
    # df = readBedFile(neg_input_bed_file)
    # df['chromEnd'] = df['chromEnd']+1
    # writeDataFrameToBed(neg_input_bed_file, df)

    # qtl_file = 'Dataset/41592_2015_BFnmeth3326_MOESM187_ESM.xlsx'
    # readQTLs(qtl_file)
    processed_data = readInputData(test_forward_seq_file, test_reverse_seq_file, test_readout_file)
    print(processed_data)



if __name__ == "__main__":
    sys.exit(main())
