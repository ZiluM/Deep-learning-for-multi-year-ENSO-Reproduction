#! /bin/bash
# ===================OBS sst==============================
# data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/tos"
# ===================OBS ssh==============================
data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/zos"
# ===================cmip5 sst==============================
# data_path="/mnt/e/CMIP5_ForM_noTauu/tos"
# ===================cmip6 sst==============================
# data_path="/mnt/e/New_proc_El_Nino/SST"
# ===================cmip6 ssh==============================
# data_path="/mnt/e/New_proc_El_Nino/SSH"
# ===================cmip5 ssh==============================
# data_path="/mnt/e/CMIP5_ForM_noTauu/zos"
# ===================sst end==============================
# end_data_path="/mnt/e/CMIP5_ForM_noTauu/tos1"
# ===================ssh end==============================
# end_data_path="/mnt/e/CMIP5_ForM_noTauu/zos1"
# ===================sst obs end==============================
# end_data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/tos1"
# ===================ssh obs end==============================
end_data_path="/mnt/e/CMIP5_ForM_noTauu/OBS/zos1"
# ===================sst garb==============================
# garb_path="/mnt/e/CMIP5_ForM_noTauu/garb"
# ===================ssh garb==============================
garb_path="/mnt/e/CMIP5_ForM_noTauu/garb1" # for zos


# ===================grid txt==============================
gridPath="/home/zilumeng/PythonPP20/ENSO_Deeplearning/data/tg_grid.txt"

function GetTos() {

    local fname=${1}

    echo ${fname}
    cdo remapbil,${gridPath} $fname ${garb_path}/reg_${fname}
    cdo ymonmean ${garb_path}/reg_${fname} ${garb_path}/reg_mean_${fname}
    cdo sub ${garb_path}/reg_${fname} ${garb_path}/reg_mean_${fname} ${end_data_path}/${fname}

    
}

cd $data_path
for fn in `ls`
do 
    echo $fn
    GetTos $fn
done

