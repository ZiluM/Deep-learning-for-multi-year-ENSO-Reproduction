"""
一个简简单单的训练脚本，训练名叫 Exp1
"""
import torch
from DataLoaderFunc import ENSODataset
from torch.utils.data import DataLoader, ConcatDataset
from CNNClass import ConvNetwork
import TrainFuncVal as TFV
import FuncPlot
import os

# 忽略 matplotlib与torch的一个加载错误。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 初始化网络
Net = ConvNetwork(30, 30)

# 加载数据
DS = ENSODataset("CMIP")
DS1 = ENSODataset("OBSTrain")
# 生成DataLoader
DL = DataLoader(ConcatDataset((DS, DS1)), batch_size=400, shuffle=True)

# 开始训练与验证
TFV.trainFunc(Net, DL, 5, saveName="Exp1", optim=torch.optim.Adam(Net.parameters()))

# 如果需要迁移学习 ， 可以 将Dataset分开加载
# 画出结果
FuncPlot.trainPlot("Exp1")
