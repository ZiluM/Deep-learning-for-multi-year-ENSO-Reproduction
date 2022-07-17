#! /bin/bash

path="/mnt/e/CMIP5_ForM_noTauu"
gridPath="/home/zilumeng/PythonPP20/ENSO_Deeplearning/data/tg_grid.txt"
cd $path
cd zos 
for fname in `ls`
do 
echo $fname 
cdo remapbil,$gridPath $fname ../zos_reg/$fname
cdo ymonmean ../zos_reg/$fname ../zos_reg_m/$fname
cdo sub ../zos_reg/$fname ../zos_reg_m/$fname ../zos_reg_a/$fname
done

