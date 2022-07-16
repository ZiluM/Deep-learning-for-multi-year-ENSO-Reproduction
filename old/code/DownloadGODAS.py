"""
DownloadGODAS.py
这个脚本用来下载GODAS SSH数据
"""
import wget

ini = "ftp://ftp2.psl.noaa.gov/Datasets/godas/sshg.%s.nc"
for year in range(1980, 2020):
    file = wget.download(ini % year, "./GODAS/GODAS%sSSH.nc" % year)
    print(file)


