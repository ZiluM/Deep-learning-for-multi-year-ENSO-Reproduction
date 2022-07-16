import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

sst = xr.open_dataset("ersstMerge.nc")["sst"]

# 获取nino3.4指数
ssta = sst.groupby("time.month") - sst.groupby("time.month").mean()
Nino34I = ssta.loc[:, 0, -5:5, 190:240].mean(dim=["lat", "lon"])
Nino34I = Nino34I.rolling(time=3, center=True).mean()
# Nino34I.plot()
# plt.show()

# 进行插值,插值到论文需要的 5*5
lat = np.arange(-55, 60.1, 5)
lon = np.arange(0, 360, 5)
sst2 = sst.interp(lat=lat, lon=lon, method="linear")
# 绘图看看

# 计算anomaly
ssta2 = sst2.groupby("time.month") - sst2.groupby("time.month").mean()
ssta2 = ssta2[:, 0]
ssta2[-1000].plot()
print(ssta2.shape)
plt.show()
# ssta[100].plot()
# plt.show()
print(ssta2.max().item())
Nino34IDataset = xr.Dataset({"nino34": Nino34I})
sstaDataset = xr.Dataset({"ssta": ssta2})
Nino34IDataset.to_netcdf("./TrainData/ersstv5Nino34.nc")
sstaDataset.to_netcdf("./TrainData/ersstv5ssta.nc")
