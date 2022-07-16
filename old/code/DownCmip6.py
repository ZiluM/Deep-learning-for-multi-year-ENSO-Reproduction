"""
DownCmip6.py
这个脚本用来下载 Cmip6 GFDL_ESM4的 zos, tos数据
"""

import wget


# gn = xr.open_dataset(r"D:\EdgeDownload\tos_Omon_GFDL-ESM4_historical_r1i1p1f1_gn_185001-186912.nc")
# # print(gn)
# gr = xr.open_dataset(r"D:\EdgeDownload\tos_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_185001-186912.nc")
# print(gr)
# 下载gr

# exampleUrl = r"https://esgdata.gfdl.noaa.gov/thredds/fileServer/gfdl_dataroot4/CMIP/" + \
#              r"NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Omon/tos/gr/v20190726/tos_Omon_GFDL" \
#              r"-ESM4_historical_r1i1p1f1_gr_185001-186912.nc "
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

