#awk -v RS='>' 'NR>1 { gsub("\n", ";", $0); sub(";$", "", $0); print ">"$0 }' E116-H3K4me3_negSet.fa | head -n 2000 | tr ';' '\n' > E116-H3K4me3_negSet_2k.fa
size=5000

for f in $(find . -name '*modified.fa')
do
	dir=`dirname "$f"`
	file=`basename "$f"`
	extension="${file##*.}"
	filename="${file%.*}"
	output_file="${filename}_${size}.${extension}"
	cd $dir
	awk -v RS='>' 'NR>1 { gsub("\n", ";", $0); sub(";$", "", $0); print ">"$0 }' $file | 			head -n $size | tr ';' '\n' > $output_file
	cd -
done

for f in $(find . -name '*negSet.fa')
do
	dir=`dirname "$f"`
	file=`basename "$f"`
	extension="${file##*.}"
	filename="${file%.*}"
	output_file="${filename}_${size}.${extension}"
	cd $dir
	awk -v RS='>' 'NR>1 { gsub("\n", ";", $0); sub(";$", "", $0); print ">"$0 }' $file | 			head -n $size | tr ';' '\n' > $output_file
	cd -
done