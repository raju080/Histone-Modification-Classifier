from os import read
import os
import sys
import random
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Preprocessor:
    def __init__(self):
        pass


    def modifyBedFile(self, input_file_name):
        output_file_name = input_file_name.split(".")[0] + "_modified.bed"

        with open(input_file_name, "r") as in_file_ref, open(output_file_name, "w") as output_file_ref:
            for line in in_file_ref:
                fields = line.strip().split("\t")
                peak_offset = int(fields[9])
                seq_start = int(fields[1])
                fields[1] = str(seq_start + peak_offset - 500)
                fields[2] = str(seq_start + peak_offset + 500)

                if len(fields) >= 10:
                    if (
                        float(fields[8]) >= 4
                        and fields[0] != "chrX"
                        and fields[0] != "chrY"
                    ):
                        output_file_ref.write("\t".join(fields[:6]) + "\n")


    def readFaIntoList(self, fa_file):
        all_seq = []
        bases = ["A", "T", "G", "C"]
        if os.path.isfile(fa_file):
            with open(fa_file, "r") as f:
                for line in f:
                    if line[0] == ">":
                        pass
                    else:
                        all_seq.append(line.strip().upper().replace(
                            "N", random.choice(bases)))
        return all_seq


    def readFaIntoListMultiLine(self, fa_file):
        all_seq = []
        bases = ["A", "T", "G", "C"]
        with open(fa_file, "r") as f:
            seq = ""
            for line in f:
                line = line.strip()
                line = line.upper()
                if line[0] == ">":
                    if len(seq) > 0:
                        all_seq.append(seq)
                    seq = ""
                else:
                    # all_seq.append(line.replace('N', random.choice(bases)))
                    # readout.append(1.0)
                    seq = seq + line.replace("N", random.choice(bases))

            if len(seq) > 0:
                all_seq.append(seq)
                seq = ""
        return all_seq


    def readInputFiles(self, pos_seq_file, neg_seq_file):
        pos_seq_list = self.readFaIntoList(pos_seq_file)
        neg_seq_list = self.readFaIntoList(neg_seq_file)
        # pos_seq_list = self.readFaIntoListMultiLine(pos_seq_file)
        # neg_seq_list = self.readFaIntoListMultiLine(neg_seq_file)
        readout = []
        for _ in range(len(pos_seq_list)):
            readout.append(1.0)
        for _ in range(len(neg_seq_list)):
            readout.append(0.0)
        all_seq = []
        all_seq.extend(pos_seq_list)
        all_seq.extend(neg_seq_list)
        return all_seq, readout


    # augment the samples with reverse complement
    def findReverseComplements(self, sequences):
        def rc_comp(seq):
            rc_dict = {"A": "T", "C": "G", "G": "C", "T": "A"}
            rc_seq = "".join([rc_dict[c] for c in seq[::-1]])
            return rc_seq

        all_sequences = []
        for i in range(len(sequences)):
            all_sequences.append(rc_comp(sequences[i]))
        return all_sequences


    def augment(self, pos_seq_file, neg_seq_file):
        fw_fa, readout = self.readInputFiles(pos_seq_file, neg_seq_file)
        rc_fa = self.findReverseComplements(fw_fa)
        dict = {"forward": fw_fa, "readout": readout, "reverse": rc_fa}
        return dict


    def oneHotEncode(self, pos_seq_file, neg_seq_file):
        self.processed_data = {}
        # The LabelEncoder encodes a sequence of bases as a sequence of
        # integers.
        integer_encoder = LabelEncoder()
        # The OneHotEncoder converts an array of integers to a sparse matrix where
        # each row corresponds to one possible value of each feature.
        one_hot_encoder = OneHotEncoder(categories="auto")

        # reads the fasta files, generates forward sequenses, reverse complements and readouts
        dict = self.augment(pos_seq_file, neg_seq_file)

        # print(dict)

        forward = []
        reverse = []

        # some sequences do not have entire 'ACGT'
        temp_seq_list = []
        for sequence in dict["forward"]:
            new_seq = "ACGT" + sequence
            temp_seq_list.append(new_seq)

        for sequence in temp_seq_list:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            forward.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        max_length, min_length = self.find_max_min_length_of_seq(forward)

        if max_length != min_length:  # checks if all seqs are of same length
            for i in range(len(forward)):
                while len(forward[i]) < max_length:
                    forward[i] = np.vstack((forward[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        for i in range(len(forward)):
            forward[i] = forward[i][4:]

        forward = np.stack(forward)

        # some sequences do not have entire 'ACGT'
        temp_seq_list = []
        for sequence in dict["reverse"]:
            new_seq = "ACGT" + sequence
            temp_seq_list.append(new_seq)

        # print("Started one hot encoding reverse")
        for sequence in temp_seq_list:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            reverse.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        max_length, min_length = self.find_max_min_length_of_seq(reverse)

        if max_length != min_length:  # checks if all seqs are of same length
            for i in range(len(reverse)):
                while len(reverse[i]) < max_length:
                    reverse[i] = np.vstack((reverse[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        for i in range(len(reverse)):
            reverse[i] = reverse[i][4:]

        reverse = np.stack(reverse)

        self.processed_data["forward"] = forward
        self.processed_data["reverse"] = reverse
        self.processed_data["readout"] = np.stack(dict["readout"])

        return self.processed_data


    def find_max_min_length_of_seq(self, lst):
        max_length, min_length = -1, 10000000
        for seq in lst:
            if len(seq) > max_length:
                max_length = len(seq)
            elif len(seq) < min_length:
                min_length = len(seq)
        return max_length, min_length
