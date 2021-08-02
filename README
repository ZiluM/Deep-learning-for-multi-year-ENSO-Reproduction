# 用Python从头到尾复现一篇Nature的工作 #1 数据下载及预处理

作者：Vector

邮箱： mzll1202@163.com

QQ：1192684038

## 前言

本篇文章将从**数据下载、处理、神经网络训练、画图**四个大步骤叙说笔者在复现 **Deep learning for multi-year ENSO forecasts**这篇文章的工作。所涉及Python库有 wget , matplotlib , numpy ,xarray , **pytorch** 等一系列在深度学习以及气象数据处理中经常使用的函数库，希望这篇文章能够对大家有所帮助。笔者也只是大学二年级的本科生，做这些东西也只是凭借个人兴趣，水平低下、错误频出也是常有的事情，请大家见谅。

简单介绍一下这篇文章，这篇文章主要是用 **sea surface temp 和 sea surface height 来预测 Nino3.4区的海温**（一个厄尔尼诺指数）。此文使用的神经网络、数据的处理都不是很复杂，适合作为气象神经网络入门的第一个尝试性工作。

本文是复现工作的第一篇文章，主要讲解 数据下载及预处理。

![image-20210709160838531](.\image-20210709160838531.png)

## 本文简介

**看完这篇博文，你将了解 Python 下载CMIP数据、下载SODA、ERSSTV5、GODAS以及相关的预处理，比如统一时间、插值、距平值计算、滑动平均计算。**

## 数据下载与预处理

由于神经网络预训练数据需要cmip模式数据，训练、验证时需要观测数据，因此我们首先对需要数据进行下载。

![image-20210725171614710](.\image-20210725171614710.png)

1、CMIP数据

对于使用的CMIP数据，本文并没有使用论文中使用的CMIP5数据，而是使用CMIP6数据。

对于数据的下载，可以直接百度搜索CMIP6然后选择所需的Label下载即可。如下图所示,**变量**选择zos,tos分别对应(SSH,SST)。

![image-20210725175350546](./image-20210725175350546.png)





![image-20210725175503375](.\image-20210725175503375.png)

![image-20210725175537367](.\image-20210725175537367.png)

选择你喜欢的模式数据下载。我这里作为范例，选择**GFDL-ESM4**的数据下载，写一个Python脚本作为示范

```Python
"""
DownCmip6.py
这个脚本用来下载 Cmip6 GFDL_ESM4的 zos, tos数据
"""

import wget
ini = r"https://esgdata.gfdl.noaa.gov/thredds/fileServer/gfdl_dataroot4/CMIP/" + \
      r"NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Omon/%s/gr/v20190726/%s_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_%s-%s.nc"
#
for var in ["tos", "zos"]:
    for year in range(1850, 2019, 20):
        time1 = str(year) + str(0) + str(1)
        time2 = str(year + 19) + str(12)
        NeedUrl = ini % (var, var, time1, time2)
        print(NeedUrl)
        fileName = "GFDL-ESM4_%s_%s-%s.nc" % (var, time1, time2)
        print(fileName)
        wget.download(NeedUrl, './Cmip6/' + fileName)
```

仔细观察下载的URL，你会发现：https://esgdata.gfdl.noaa.gov/thredds/fileServer/gfdl_dataroot4/CMIP/后面跟的 <u>NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Omon/tos/gr/v20190726/tos_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_185001-186912.nc</u>
从前往后依次是 机构名称、模式名称、实验名称、variant label、变量频率、变量名、网格、版本、文件名。我们根据上述规律，使用wget就可以很简单的下载数据了。

接下来是处理CMIP数据,为了统一语言，我使用python中的xarray来处理、merge文件。缺点是很慢，优点是易学。下面的脚本中，merge nc文件的主要函数是concat,需要输入一系列网格相同的Dataarray，然后在time维度上进行统一。非常建议统一时间，以免后期出幺蛾子。

可以看到我计算距平值使用的语句是 **TosA = TosArray.groupby("time.month") - TosArray.groupby("time.month").mean()**，这是计算距平值常用的语法。

**Nino34I = Nino34I.rolling(time=3, center=True).mean()**使用这个语句来生成三月滑动平均。

插值使用 **TosArray.interp(lat=lat, lon=lon)**，输入指定的网格和维度即可，默认为线性插值，我们这里插值成5*5的网格。

对于保存nc文件，需要使用**TosAD = xr.Dataset({"TosA": TosAInterped})**来将Dataarray转化为Dataset,然后使用**TosAD.to_netcdf("./TrainData/TosA.nc")**进行保存。

```python
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

```

2.在分析资料

第一个需要的在分析资料是 **ERSSTV5**,这个直接百度搜索即可。但是可以看到是有许多文件的，我们同样用wget+分析链接的方式下载。

```python
"""
DownLersstv5.py
用来下载ersst v5的数据
"""

import wget

for year in range(1870, 2020):
    for month in range(1, 13):
        month = str(month).zfill(2)
        url = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.{}{}.nc".format(year, month)
        file = wget.download(url, out=r"./ersstv5D")
        print(file)
```

同样的，我们使用xarray来merge下载的多个nc文件，并且保存。

```python
"""
本 脚本用来 merge ersstv5数据
"""

import xarray as xr
import os
import pandas as pd

fileLoc = "./ersstv5D"
fList = os.listdir(fileLoc)

# 对下载的nc文件进行合并
ncList = []
for FileName in fList:
    nc = xr.open_dataset(fileLoc + r"/" + FileName)["sst"]
    ncList.append(nc)
# 使用 concat 进行合并，不过会很慢
sst1 = xr.concat(ncList, dim="time")
# print(sst1)
# print(sst1.shape)
# 画出来看看
# sst1[100].plot()
# plt.show()

# 统一时间格式并保存
Time = pd.date_range(start="18700101", end="20191201", freq="MS")
sst1["time"] = Time
print(sst1)

SstSave = xr.Dataset({"sst": sst1})
SstSave.to_netcdf("ersstMerge.nc")
```

第二个是GODAS，相同的方式使用wget下载。

```python
"""
DownloadGODAS.py
这个脚本用来下载GODAS SSH数据
"""
import wget

ini = "ftp://ftp2.psl.noaa.gov/Datasets/godas/sshg.%s.nc"
for year in range(1980, 2020):
    file = wget.download(ini % year, "./GODAS/GODAS%sSSH.nc" % year)
    print(file)
```

然后是merge

```python
"""
MergeGODAS.py
用来merge GODAS
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

Dirloc = "./GODAS"
DirList = os.listdir(Dirloc)
GODASL = []
for loc in DirList:
    Rloc = Dirloc + "/" + loc
    SSH = xr.open_dataset(Rloc)["sshg"]
    GODASL.append(SSH)

SSH = xr.concat(GODASL, dim="time")
Time = pd.date_range("19800101", "20191230", freq="MS")
SSH["time"] = Time

lat = np.arange(-55, 60.1, 5)
lon = np.arange(0, 360, 5)

SSH = SSH.interp(lat=lat, lon=lon)
SSHA = SSH.groupby("time.month") - SSH.groupby("time.month").mean()

SSHADataset = xr.Dataset({"ssha": SSHA})
SSHADataset.to_netcdf("./ValidationData/GODASssha.nc")

```

然后是SODA，SODA和前面几个数据一样，可以直接百度

```python
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

```

处理完之后，你将会得到这几个文件用于模型的训练、验证。

![image-20210725182552087](.\image-20210725182552087.png)

## 结语

本文介绍了数据的预处理，希望大家能够有所收获。本工作主要完成于2020年秋天、冬天，据此已经过去半年多，有些错误、水平低下的地方在所难免，希望大家多多包涵。

