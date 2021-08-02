"""
Delsoda.py
此脚本用来处理SODA
下载链接：http://iridl.ldeo.columbia.edu/SOURCES/.CARTON-GIESE/.SODA/.v2p2p4/.ssh/data.nc
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fLoc = r"./soda/ssh.nc"
ssh = xr.open_dataset(fLoc, decode_times=False)["ssh"]
print(ssh)
# 统一时间
ssh["time"] = pd.date_range("18710101", "20081201", freq="MS")

# 出来看看
print(ssh)
ssh[100].plot()
plt.show()

# 插值到文章需要的格式
lat = np.arange(-55, 60.1, 5)
lon = np.arange(0, 360, 5)
ssh1 = ssh.interp(lat=lat, lon=lon, method="linear")
ssh1[100].plot()
plt.show()
print(ssh1)

# 计算距平值
ssha = ssh1.groupby("time.month") - ssh1.groupby("time.month").mean()
ssha[100].plot()
plt.show()

# 保存数据
sshaD = xr.Dataset({"ssha": ssha})
sshaD.to_netcdf("./TrainData/SODAssha.nc")
