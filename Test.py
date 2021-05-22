from os import read
import sys
import time
import pandas as pd
import numpy as np


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


def main():
    # neg_input_bed_file = 'Dataset/E118-H3K27ac_negSet.bed'
    # df = readBedFile(neg_input_bed_file)
    # df['chromEnd'] = df['chromEnd']+1
    # writeDataFrameToBed(neg_input_bed_file, df)

    qtl_file = 'Dataset/41592_2015_BFnmeth3326_MOESM187_ESM.xlsx'
    readQTLs(qtl_file)


if __name__ == "__main__":
    sys.exit(main())
