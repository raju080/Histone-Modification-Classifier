# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("GenomicRanges")
# BiocManager::install("rtracklayer")
# BiocManager::install("BSgenome")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg19.masked")
# # BiocManager::install("BSgenome.Hsapiens.UCSC.hg18.masked")

# install.packages(c('ROCR','kernlab','seqinr'))
# install.packages("gkmSVM")

library("gkmSVM")
genNullSeqs(
  "Dataset/E118-H3K27ac_train.bed", 
  genomeVersion='hg19', 
  outputBedFN = 'Dataset/E118-H3K27ac_negSet_train.bed', 
  outputPosFastaFN = 'Dataset/E118-H3K27ac_posSet_train.fa',
  outputNegFastaFN = 'Dataset/E118-H3K27ac_negSet_train.fa', 
  xfold = 1, 
  repeat_match_tol = 0.02, 
  GC_match_tol = 0.02, 
  length_match_tol = 0.02, 
  batchsize = 5000, 
  nMaxTrials = 100, 
  genome = NULL)