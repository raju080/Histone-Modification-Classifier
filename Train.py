import sys
from Preprocess import Preprocessor 


def main():
  input_bed_file = 'Dataset/E118-H3K27ac.narrowPeak'

  preprocessor = Preprocessor()
  preprocessor.modifyBedFile(input_bed_file)


if __name__ == "__main__":
    sys.exit(main())