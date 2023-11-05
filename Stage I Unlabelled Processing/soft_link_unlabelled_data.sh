# create soft link
cd BC5CDR-chem-IOB
ln -s ./unlabeled/chem_weak.txt weak.txt
cd ../BC5CDR-disease-IOB
ln -s ./unlabeled/disease_weak.txt weak.txt
cd ../NCBI-disease-IOB
ln -s ./unlabeled/disease_weak.txt weak.txt