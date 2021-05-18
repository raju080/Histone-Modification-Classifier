import sys


class Preprocessor:

  def __init__(self):
    pass

  def modifyBedFile(self, input_file_name):
    train_file_name = input_file_name.split('.')[0] + '_train.bed'
    test_file_name = input_file_name.split('.')[0] + '_test.bed'

    with open(input_file_name, 'r') as in_file_ref:
      with open(train_file_name, 'w') as train_file_ref:
        with open(test_file_name, 'w') as test_file_ref:
          for line in in_file_ref:
            fields = line.strip().split('\t')
            peak_offset = int(fields[9])
            seq_start = int(fields[1])
            seq_end = int(fields[2])
            fields[1] = str(seq_start+peak_offset-500)
            fields[2] = str(seq_start+peak_offset+500)
            
            if len(fields)>=10:
              if float(fields[7])>=10 and fields[0]!='chrX' and fields[0]!='chrY':
                if fields[0]=='chr1':
                  test_file_ref.write('\t'.join(fields[:6]) + '\n')
                else:
                  train_file_ref.write('\t'.join(fields[:6]) + '\n')
              


