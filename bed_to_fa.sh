sudo apt install bedtools

ref_genome=/root/gdrive/MyDrive/Thesis/Epigenetic-Modification/hg19.fa

for f in $(find . -name '*.bed')
do
	dir="$(dirname $f)"
	file="$(basename $f)"
	extension="${file##*.}"
	filename="${file%.*}"
	output_file="${dir}/${filename}.fa"
	# if [ -n `grep '[.]*negSet.bed' $f` ]; then
	# 	echo "$f"
	# else 
	# 	echo "null"
	# fi
	# echo $(grep 'negSet' $f)

  # echo "$f"
  # echo "$output_file"
	bedtools getfasta -fi $ref_genome -bed $f -fo $output_file
done


# for d in $(find . )
# do 
#   if [ -d "$d" ]; then
#     cd $d
    
#     cd -
#   fi
# done