if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("GenomicRanges")
BiocManager::install("rtracklayer")
BiocManager::install("BSgenome")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19.masked")
# BiocManager::install("BSgenome.Hsapiens.UCSC.hg18.masked")

install.packages(c('ROCR','kernlab','seqinr'))
install.packages("gkmSVM")

library("gkmSVM")

genNullSeqs(
  "E116/E116-H3K27ac/E116-H3K27ac_modified.bed", 
  genomeVersion='hg19', 
  outputBedFN = 'E116/E116-H3K27ac/E116-H3K27ac_negSet.bed', 
  outputPosFastaFN = 'E116/E116-H3K27ac/E116-H3K27ac_posSet.fa',
  outputNegFastaFN = 'E116/E116-H3K27ac/E116-H3K27ac_negSet.fa', 
  xfold = 1, 
  repeat_match_tol = 0.02, 
  GC_match_tol = 0.02, 
  length_match_tol = 0.02, 
  batchsize = 5000, 
  nMaxTrials = 200, 
  genome = NULL)

#genNullSeqs(
#  "Dataset/E118-H3K27ac_modified.bed", 
#  genomeVersion='hg19', 
#  outputBedFN = 'Dataset/E118-H3K27ac_negSet.bed', 
#  outputPosFastaFN = 'Dataset/E118-H3K27ac_posSet.fa',
#  outputNegFastaFN = 'Dataset/E118-H3K27ac_negSet.fa', 
#  xfold = 1, 
#  repeat_match_tol = 0.02, 
#  GC_match_tol = 0.02, 
#  length_match_tol = 0.02, 
#  batchsize = 5000, 
#  nMaxTrials = 150, 
#  genome = NULL)