#! /bin/bash

path="/mnt/e/CMIP5_ForM_noTauu"
gridPath="/home/zilumeng/PythonPP20/ENSO_Deeplearning/data/tg_grid.txt"
cd $path
cd tos 
for fname in `ls`
do 
echo $fname 
cdo remapbil,$gridPath $fname ../tos_reg/$fname
cdo ymonmean ../tos_reg/$fname ../tos_reg_m/$fname
cdo sub ../tos_reg/$fname ../tos_reg_m/$fname ../tos_reg_a/$fname

done

