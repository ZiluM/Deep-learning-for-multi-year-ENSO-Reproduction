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
