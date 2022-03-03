from os import read
import os
import sys
import time
import pandas as pd
import numpy as np
# import myvariant
# from biothings_client import get_client
import random
from tensorflow.keras.models import save_model, load_model
from ConvolutionLayer import ConvolutionLayer

# from sklearn.model_selection import StratifiedKFold

# data = [random.random() for _ in range(20)]
# output = [random.randint(0, 2) for _ in range(20)]
# print(len(data))
# kfold = StratifiedKFold(n_splits=5)
# for train_idx, test_idx in kfold.split(data, output):
#     print('---- itr ----')
#     print('train idx: ', train_idx)
#     print('test tdx: ', test_idx)

# sys.exit()

# pip3 install myvariant

seed = 527
total_seq_postfix = "_2000"
# types: basic, meuseum, vanilla_cnn, multi_cnn
model_type = 'vanilla_cnn'
# output file to save the model after training
model_file = 'TrainedModels/model'+ total_seq_postfix + '_' + model_type + '.h5'

input_bed_file = 'E116/E116-H3K27ac/E116-H3K27ac.narrowPeak'
pos_seq_file = 'E116/E116-H3K27ac/E116-H3K27ac_modified' + total_seq_postfix + '.fa'
neg_seq_file = 'E116/E116-H3K27ac/E116-H3K27ac_negSet' + total_seq_postfix + '.fa'

forward_seq_file = 'E116/E116-H3K27ac/forward' + total_seq_postfix + '.npy'
reverse_seq_file = 'E116/E116-H3K27ac/reverse' + total_seq_postfix + '.npy'
readout_file = 'E116/E116-H3K27ac/readout' + total_seq_postfix + '.npy'

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

def readQTLs(input_file_path):
    dfs = pd.read_excel(input_file_path, sheet_name=None)
    qtl_df = dfs.get('SuppTable_3_delrosario')
    qtl_df.index = qtl_df['#rsid']
    qtl_df['start'] = qtl_df['pos']-500
    qtl_df['end'] = qtl_df['pos']+500
    qtl_df['ref'] = np.array(['X']*len(qtl_df))
    qtl_df['alt'] = np.array(['X']*len(qtl_df))
    qtl_df['refaltidx'] = qtl_df['#rsid'].copy()
    qtl_df = qtl_df[qtl_df['#rsid'].apply(lambda row: (row[0]!='N' and '.' not in row))]

    mv = myvariant.MyVariantInfo()
    # mv = get_client('variant')
    rsid = list(qtl_df['#rsid'])
    qrs = mv.querymany(rsid, scopes='dbsnp.rsid', fields='dbsnp')

    for qr in qrs:
        try:
            id = qr['dbsnp']['rsid']
            refbp = qr['dbsnp']['ref']
            altbp = qr['dbsnp']['alt']
            qtl_df.at[id, 'ref'] = refbp
            qtl_df.at[id, 'alt'] = altbp
            qtl_df.at[id, 'refaltidx'] = '>' + qtl_df.at[id, 'chr'] + ':' + str(qtl_df.at[id, 'start']) + '-' + str(qtl_df.at[id, 'end'])
        except:
            refbp = 'X'
            altbp = 'X'

        
    # ref_list = []
    # alt_list = []
    # for qr in qrs:
    #     try:
    #         ref_list.append(qr['dbsnp']['ref'])
    #         alt_list.append(qr['dbsnp']['alt'])
    #     except:
    #         ref_list.append('X')
    #         alt_list.append('X')
    # print(ref_list[:10])
    # qtl_df['ref'] = np.array(ref_list)
    # qtl_df['alt'] = np.array(alt_list)
    
    qtl_df = qtl_df[qtl_df['ref'].apply(lambda r: r!='X')]

    out_df = qtl_df[['chr', 'start', 'end']]
    ref_alt = qtl_df[['refaltidx', 'ref', 'alt']]

    output_file_path = input_file_path.split(".")[0] + "_ref.bed"
    refalt_file_path = input_file_path.split(".")[0] + "_snprefalt.txt"
    out_df.to_csv(output_file_path, sep='\t', index=False, header=False)
    ref_alt.to_csv(refalt_file_path, index=False)
    # print(mv.querymany(['rs9927825'], scopes='dbsnp.rsid',
        #   fields='dbsnp'))
    

def createAltFaQTLs(input_file_path):
    refalt_file_path = input_file_path.split(".")[0] + "_snprefalt.txt"
    ref_alt_df = pd.read_csv(refalt_file_path, index_col=0)
    output_file_path = input_file_path.split(".")[0] + "_alt.fa"
    input_file_path = input_file_path.split(".")[0] + "_ref.fa"
    c = 0
    i = 0
    
    with open(input_file_path, 'r') as fin, open(output_file_path, 'w') as fout:
        id = ''
        for line in fin:
            if line[0]=='>':
                id = line[:-1]
                fout.write(line)
            else:
                if line[499].capitalize() == ref_alt_df.at[id, 'ref'].capitalize():
                    line = line[:499] + ref_alt_df.at[id, 'alt'].capitalize() + line[500:]
                    fout.write(line)
                    c = c+1
                i = i+1
    print(len(ref_alt_df))
    print(c)



def createAltSequences(fa_file_path):
    output_file_path = fa_file_path.split(".")[0] + "_alt.fa"
    if os.path.isfile(fa_file_path):
            with open(fa_file_path, "r") as fin, open(output_file_path) as fout:
                for line in fin:
                    if line[0] == ">":
                        fout.write(line)
                    else:
                        all_seq.append(line.strip().upper().replace(
                            "N", random.choice(bases)))





def testModel(model, processed_data):
    pass


def main():
    # neg_input_bed_file = 'Dataset/E118-H3K27ac_negSet.bed'
    # df = readBedFile(neg_input_bed_file)
    # df['chromEnd'] = df['chromEnd']+1
    # writeDataFrameToBed(neg_input_bed_file, df)

    # qtl_file = 'Dataset/haQTLDeephistone.xlsx'
    # readQTLs(qtl_file)
    # createAltFaQTLs(qtl_file)
    
    
    # processed_data = readInputData(test_forward_seq_file, test_reverse_seq_file, test_readout_file)
    # print(processed_data)
    model = load_model('TrainedModels/E116/model_20000_cnn_lstm.h5', custom_objects={
                       'ConvolutionLayer': ConvolutionLayer})
    model.summary()



if __name__ == "__main__":
    sys.exit(main())
