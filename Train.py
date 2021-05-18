import sys
import numpy as np
from Preprocess import Preprocessor


# constants
forward_seq_file = 'Dataset/forward.npy'
reverse_seq_file = 'Dataset/reverse.npy'
readout_file = 'Dataset/readout.npy'



def processInputData(pos_seq_file, neg_seq_file):
    preprocessor = Preprocessor()
    processed_dict = preprocessor.oneHotEncode(
        pos_seq_file=pos_seq_file, neg_seq_file=neg_seq_file)
    np.save(forward_seq_file, processed_dict['forward'])
    np.save(reverse_seq_file, processed_dict['reverse'])
    np.save(readout_file, processed_dict['readout'])
    print(processed_dict)


def readInputData():
    processed_dict = {}
    processed_dict['forward'] = np.load(forward_seq_file)    
    processed_dict['reverse'] = np.load(reverse_seq_file)
    processed_dict['readout'] = np.load(readout_file)
    return processed_dict


def main():
    input_bed_file = 'Dataset/E118-H3K27ac.narrowPeak'
    pos_seq_file = 'Dataset/E118-H3K27ac.fa'
    neg_seq_file = 'Dataset/E118-H3K27ac_negSet.fa'

    processInputData(pos_seq_file, neg_seq_file)
    processed_dict = readInputData()
    print(processed_dict)
    

if __name__ == "__main__":
    sys.exit(main())
