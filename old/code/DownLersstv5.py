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
