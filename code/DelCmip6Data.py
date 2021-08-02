"""
DelCmip6Data.py
此脚本用来聚合、整理cmip6数据，准备训练
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 下载的CMIP6位置
loc = "./Cmip6"

# 使用 xarray 把分散的几个文件merge起来
FileList = os.listdir(loc)
Toslist = []
Zoslist = []
for FName in FileList:
    ModeName, varName = FName.split("_")[0:2]
    if ModeName == "GFDL-ESM4":
        if varName == "tos":
            Toslist.append(xr.open_dataset(loc + '/' + FName)["tos"])
        else:
            Zoslist.append(xr.open_dataset(loc + '/' + FName)["zos"])
# 把这两个聚合在一起，形成一个大的Data array
TosArray = xr.concat(Toslist, dim="time")
ZosArray = xr.concat(Zoslist, dim="time")
# 更改时间 , 需要统一时间，方便后期使用。
TimeRange = pd.date_range(start="18500101", end="20091201", freq="MS")
TosArray["time"] = TimeRange
ZosArray["time"] = TimeRange
# 获得Nino3.4区温区 tos 为 sst

TosA = TosArray.groupby("time.month") - TosArray.groupby("time.month").mean()
Nino34I = TosA.loc[:, -5:5, 190:240].mean(dim=["lat", "lon"])
# 计算3月滑动平均
Nino34I = Nino34I.rolling(time=3, center=True).mean()

# 插值成需要的网格
lat = np.arange(-55, 60.1, 5, )
lon = np.arange(0, 360, 5)
TosInterped = TosArray.interp(lat=lat, lon=lon)
ZosInterped = ZosArray.interp(lat=lat, lon=lon)
# 计算异常值
TosAInterped = TosInterped.groupby("time.month") - TosInterped.groupby("time.month").mean()
ZosAInterped = ZosInterped.groupby("time.month") - ZosInterped.groupby("time.month").mean()
# 保存，用于训练
Nino34ID = xr.Dataset({"Nino34I": Nino34I})
Nino34ID.to_netcdf("./TrainData/Cmip6Nino34I.nc")

TosAD = xr.Dataset({"TosA": TosAInterped})
ZosAD = xr.Dataset({"ZosA": ZosAInterped})
TosAD.to_netcdf("./TrainData/TosA.nc")
ZosAD.to_netcdf("./TrainData/ZosA.nc")

