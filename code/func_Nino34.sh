#! /bin/bash


# data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/tos"
# data_path="/mnt/e/CMIP5_ForM_noTauu/tos"
data_path="/mnt/e/New_proc_El_Nino/SST"
# data_path="/mnt/e/CMIP5_ForM_noTauu/zos"
gridPath="/home/zilumeng/PythonPP20/ENSO_Deeplearning/data/tg_grid.txt"

# end_data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/nino341"
end_data_path="/mnt/e/CMIP5_ForM_noTauu/nino341"

garb_path="/mnt/e/CMIP5_ForM_noTauu/garb2"

function GetNino34() {

    local fname=${1}

    echo ${fname}
    cdo sellonlatbox,190,240,-5,5 $fname ${garb_path}/nino34_reg_${fname}
    cdo fldmean  ${garb_path}/nino34_reg_${fname}  ${garb_path}/nino34_regm_${fname}
    cdo ymonmean ${garb_path}/nino34_regm_${fname} ${garb_path}/nino34_regmm_${fname}
    cdo sub ${garb_path}/nino34_regm_${fname} ${garb_path}/nino34_regmm_${fname} ${end_data_path}/${fname}

    
}

cd $data_path
for fn in `ls`
do 

GetNino34 ${fn}

done