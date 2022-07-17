#! /bin/bash

path="/mnt/e/CMIP5_ForM_noTauu"
cd $path
cd tos 
for fname in `ls`
do 
echo $fname 
cdo sellonlatbox,190,240,-5,5 $fname ../nino34_reg/$fname
cdo fldmean ../nino34_reg/$fname ../nino34_regm/$fname
cdo ymonmean ../nino34_regm/$fname ../nino34_regmm/$fname

cdo sub ../nino34_regm/$fname ../nino34_regmm/$fname ../nino34_rega/$fname

done